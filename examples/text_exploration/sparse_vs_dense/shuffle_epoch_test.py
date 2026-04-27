#!/usr/bin/env python3
"""ARB-139 Phase 1: shuffle / multi-epoch fairness test for SSH.

Tests SSH (sigmoid-bounded, single-table, modulated) at fixed corpus size
(1M unique tokens) with different ordering / epoch regimes:

    1ep_inorder    1 epoch, no shuffle (current default)
    1ep_shuffled   1 epoch, shuffled once at start
    3ep_shuffled   3 epochs, reshuffled per epoch
    5ep_shuffled   5 epochs, reshuffled per epoch (matches word2vec effort)

The 5ep variant is the closest to word2vec's training regime, isolating
the algorithmic difference from the data-presentation difference.

Output: data/runs/arb139/shuffle_epoch_test.csv
Resumable via CSV check.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from examples.text_exploration.sparse_vs_dense.data import (
    build_vocab,
    encode_tokens,
    load_simlex,
    load_text8,
)
from examples.text_exploration.sparse_vs_dense.evaluation import (
    evaluate_capacity,
    evaluate_simlex,
)
from examples.text_exploration.sparse_vs_dense.sparse_skipgram_hebbian_modulated_baseline import (
    ModulatedSSHEmbeddings,
    _TRAIN_FN,
    _build_unigram_cdf,
)


def run_one(
    *,
    label: str,
    n_epochs: int,
    shuffle_first: bool,
    base_tokens: list[int],
    id_to_token: list[str],
    simlex,
    n_dims: int = 1024,
    k_active: int = 40,
    window: int = 5,
    n_neg: int = 5,
    lr_pos: float = 0.05,
    lr_neg: float = 0.05,
    ema_alpha: float = 0.0,
    sigmoid_bounded: bool = True,
    seed: int = 0,
) -> dict:
    print(
        f"\n--- {label}: epochs={n_epochs} shuffle_first={shuffle_first} "
        f"sigmoid_bounded={sigmoid_bounded} ---"
    )
    rng = np.random.default_rng(seed)
    V = len(id_to_token)
    base_tids = np.asarray(base_tokens, dtype=np.int64)

    A_center = (rng.standard_normal((V, n_dims)) * 0.01).astype(np.float32)
    A_context = A_center  # single-table
    A_ema_center = A_center.copy()
    A_ema_context = A_ema_center

    cdf = _build_unigram_cdf(base_tids, V, 0.75)

    e_center_buf = np.empty(k_active, dtype=np.int64)
    e_context_buf = np.empty(k_active, dtype=np.int64)
    e_neg_buf = np.empty(k_active, dtype=np.int64)

    t0 = time.monotonic()
    for epoch in range(n_epochs):
        # Shuffle decision: first epoch only if shuffle_first; later epochs
        # always reshuffled (multi-epoch is meaningless without it).
        if epoch == 0 and not shuffle_first:
            tids_epoch = base_tids.copy()
        else:
            tids_epoch = rng.permutation(base_tids).astype(np.int64)

        n_pairs = len(tids_epoch) * 2 * window
        n_negs = n_pairs * n_neg + 1024
        neg_uniform = rng.random(n_negs)
        negs_buf = np.searchsorted(cdf, neg_uniform).astype(np.int64)

        _TRAIN_FN(
            A_center, A_context,
            A_ema_center, A_ema_context,
            tids_epoch, cdf,
            negs_buf, e_center_buf, e_context_buf, e_neg_buf,
            n_dims, k_active, window, n_neg,
            float(lr_pos), float(lr_neg), 0.0, True,
            float(ema_alpha), False, bool(sigmoid_bounded),
        )

    train_dt = time.monotonic() - t0

    extract_table = A_ema_center if ema_alpha > 0 else A_center
    sdrs: dict[str, np.ndarray] = {}
    for w in range(V):
        top_k_idx = np.argpartition(-extract_table[w], k_active)[:k_active]
        code = np.zeros(n_dims, dtype=np.bool_)
        code[top_k_idx] = True
        sdrs[id_to_token[w]] = code

    emb = ModulatedSSHEmbeddings(sdrs)
    s = evaluate_simlex(emb, simlex)
    cap = evaluate_capacity(emb, seed=seed, sample=300)

    row = {
        "label": label,
        "n_epochs": n_epochs,
        "shuffle_first": shuffle_first,
        "sigmoid_bounded": sigmoid_bounded,
        "ema_alpha": ema_alpha,
        "tokens_per_epoch": len(base_tids),
        "total_presentations": n_epochs * len(base_tids),
        "simlex_spearman": s["spearman"],
        "simlex_n": s["n_pairs"],
        "cap_mean_sim": cap["mean_pairwise_sim"],
        "cap_collision_frac": cap["high_collision_frac"],
        "cap_eff_dim": cap["eff_dim"],
        "train_dt_s": train_dt,
    }
    print(
        f"  simlex={s['spearman']:+.3f}  coll={cap['high_collision_frac']:.3f}  "
        f"ed={cap['eff_dim']:.1f}  train={train_dt:.1f}s"
    )
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-tokens", type=int, default=1_000_000)
    p.add_argument("--vocab-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--csv",
        type=str,
        default="data/runs/arb139/shuffle_epoch_test.csv",
    )
    args = p.parse_args()

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    done_labels: set[str] = set()
    if csv_path.exists():
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                existing.append(row)
                done_labels.add(row["label"])

    print("Loading 1M tokens of text8 ...")
    tokens = load_text8(max_tokens=args.n_tokens)
    token_to_id, id_to_token = build_vocab(tokens, vocab_size=args.vocab_size)
    vocab_set = set(id_to_token)
    simlex = load_simlex(vocab=vocab_set)
    base_tokens = encode_tokens(tokens, token_to_id)

    plan = [
        # (label, n_epochs, shuffle_first)
        ("1ep_inorder",  1, False),
        ("1ep_shuffled", 1, True),
        ("3ep_shuffled", 3, True),
        ("5ep_shuffled", 5, True),
    ]

    all_rows: list[dict] = list(existing)
    for label, n_epochs, shuffle_first in plan:
        if label in done_labels:
            print(f"--- skipping: {label} (already in CSV) ---")
            continue
        row = run_one(
            label=label,
            n_epochs=n_epochs,
            shuffle_first=shuffle_first,
            base_tokens=base_tokens,
            id_to_token=id_to_token,
            simlex=simlex,
            seed=args.seed,
        )
        all_rows.append(row)
        keys = sorted({k for r in all_rows for k in r})
        with csv_path.open("w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"  [partial] wrote {csv_path} ({len(all_rows)} rows)")

    print(f"\nDone. Wrote {csv_path}.")


if __name__ == "__main__":
    main()
