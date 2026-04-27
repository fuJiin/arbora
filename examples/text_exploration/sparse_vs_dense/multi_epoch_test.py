#!/usr/bin/env python3
"""ARB-139: multi-epoch SSH test.

Tests whether multi-epoch training (with shuffled token order between
epochs) reduces SSH's trajectory chaos. If averaging across stochastic
realizations converges to a stable point, multi-epoch should produce
much higher and more consistent SimLex than single-pass on equivalent
total token budget.

Compares:
  1 epoch × 1M tokens  (baseline, single-pass)
  2 epochs × 500k tokens
  4 epochs × 250k tokens
  10 epochs × 100k tokens

Each variant trains on the SAME total tokens (1M token-presentations)
but with different epoch structure. seed=0, single_table+modulation+EMA(0.001).

Output: data/runs/arb139/multi_epoch_test.csv
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


def train_multi_epoch(
    base_token_ids: np.ndarray,
    *,
    n_epochs: int,
    id_to_token: list[str],
    n_dims: int,
    k_active: int,
    window: int,
    n_neg: int,
    lr_pos: float,
    lr_neg: float,
    ema_alpha: float,
    seed: int,
) -> tuple[ModulatedSSHEmbeddings, dict]:
    """Train SSH with `n_epochs` shuffled passes over the base token slice."""
    rng = np.random.default_rng(seed)
    V = len(id_to_token)

    A_center = (rng.standard_normal((V, n_dims)) * 0.01).astype(np.float32)
    A_context = A_center  # single-table
    A_ema_center = A_center.copy()
    A_ema_context = A_ema_center

    cdf = _build_unigram_cdf(base_token_ids, V, 0.75)

    e_center_buf = np.empty(k_active, dtype=np.int64)
    e_context_buf = np.empty(k_active, dtype=np.int64)
    e_neg_buf = np.empty(k_active, dtype=np.int64)

    t_train = time.monotonic()
    n_total_pairs_processed = 0

    for epoch in range(n_epochs):
        # Shuffle token order each epoch — core mechanism for averaging
        # over stochastic trajectories.
        if epoch == 0:
            tids_epoch = base_token_ids.copy()
        else:
            tids_epoch = rng.permutation(base_token_ids).astype(np.int64)

        # Pre-sample negatives for this epoch.
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
            float(ema_alpha), False, False,
        )
        n_total_pairs_processed += n_pairs

    train_dt = time.monotonic() - t_train

    extract_table = A_ema_center if ema_alpha > 0 else A_center
    sdrs: dict[str, np.ndarray] = {}
    for w in range(V):
        top_k_idx = np.argpartition(-extract_table[w], k_active)[:k_active]
        code = np.zeros(n_dims, dtype=np.bool_)
        code[top_k_idx] = True
        sdrs[id_to_token[w]] = code

    return ModulatedSSHEmbeddings(sdrs), {
        "n_epochs": n_epochs,
        "tokens_per_epoch": len(base_token_ids),
        "total_token_presentations": n_epochs * len(base_token_ids),
        "n_pairs_processed": n_total_pairs_processed,
        "train_dt_s": train_dt,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-dims", type=int, default=1024)
    p.add_argument("--k-active", type=int, default=40)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--n-neg", type=int, default=5)
    p.add_argument("--lr-pos", type=float, default=0.05)
    p.add_argument("--lr-neg", type=float, default=0.05)
    p.add_argument("--ema-alpha", type=float, default=0.001)
    p.add_argument(
        "--csv",
        type=str,
        default="data/runs/arb139/multi_epoch_test.csv",
    )
    args = p.parse_args()

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Load full 1M tokens up front (we'll slice this for different epoch sizes).
    print("Loading 1M tokens of text8 ...")
    tokens = load_text8(max_tokens=1_000_000)
    token_to_id, id_to_token = build_vocab(tokens, vocab_size=args.vocab_size)
    vocab_set = set(id_to_token)
    simlex = load_simlex(vocab=vocab_set)
    full_token_ids = np.asarray(encode_tokens(tokens, token_to_id), dtype=np.int64)

    plan = [
        # (label, n_epochs, tokens_per_epoch)
        ("1 epoch x 1M",     1, 1_000_000),
        ("2 epochs x 500k",  2,   500_000),
        ("4 epochs x 250k",  4,   250_000),
        ("10 epochs x 100k", 10,  100_000),
    ]

    rows: list[dict] = []
    for label, n_epochs, n_tok in plan:
        print(f"\n--- {label} (total presentations = {n_epochs * n_tok:,}) ---")
        base_tids = full_token_ids[:n_tok]
        emb, stats = train_multi_epoch(
            base_tids,
            n_epochs=n_epochs,
            id_to_token=id_to_token,
            n_dims=args.n_dims,
            k_active=args.k_active,
            window=args.window,
            n_neg=args.n_neg,
            lr_pos=args.lr_pos,
            lr_neg=args.lr_neg,
            ema_alpha=args.ema_alpha,
            seed=args.seed,
        )
        s = evaluate_simlex(emb, simlex)
        cap = evaluate_capacity(emb, seed=args.seed, sample=300)
        row = {
            "label": label,
            "n_epochs": n_epochs,
            "tokens_per_epoch": n_tok,
            "total_presentations": n_epochs * n_tok,
            "ema_alpha": args.ema_alpha,
            "simlex_spearman": s["spearman"],
            "simlex_n": s["n_pairs"],
            "cap_mean_sim": cap["mean_pairwise_sim"],
            "cap_collision_frac": cap["high_collision_frac"],
            "cap_eff_dim": cap["eff_dim"],
            "train_dt_s": stats["train_dt_s"],
        }
        rows.append(row)
        keys = sorted({k for r in rows for k in r})
        with csv_path.open("w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(
            f"  simlex={s['spearman']:+.3f}  coll={cap['high_collision_frac']:.3f}  "
            f"ed={cap['eff_dim']:.1f}  train={stats['train_dt_s']:.1f}s"
        )

    print(f"\nDone. Wrote {csv_path}.")


if __name__ == "__main__":
    main()
