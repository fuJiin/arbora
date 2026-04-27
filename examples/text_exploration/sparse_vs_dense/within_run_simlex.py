#!/usr/bin/env python3
"""ARB-139: within-run SimLex curve diagnostic.

Trains modulated-SSH (single-table + EMA) on text8 1M tokens with
checkpoints every 50k tokens. At each checkpoint, evaluate SimLex on the
current state. Writes CSV with one row per checkpoint.

Goal: distinguish trajectory chaos (SimLex bumpy WITHIN a single run) from
cross-run noise (smooth within, jumpy across runs of different size).

Output: data/runs/arb139/within_run_simlex.csv

Wall: ~12 min (one 1M run + cheap eval at each checkpoint).
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


def extract_sdrs(
    A: np.ndarray, k_active: int, id_to_token: list[str]
) -> dict[str, np.ndarray]:
    """Top-k of each row of A → boolean SDRs."""
    sdrs: dict[str, np.ndarray] = {}
    n_dims = A.shape[1]
    for w in range(A.shape[0]):
        top_k_idx = np.argpartition(-A[w], k_active)[:k_active]
        code = np.zeros(n_dims, dtype=np.bool_)
        code[top_k_idx] = True
        sdrs[id_to_token[w]] = code
    return sdrs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-tokens", type=int, default=1_000_000)
    p.add_argument("--vocab-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-stride", type=int, default=50_000)
    p.add_argument("--n-dims", type=int, default=1024)
    p.add_argument("--k-active", type=int, default=40)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--n-neg", type=int, default=5)
    p.add_argument("--lr-pos", type=float, default=0.05)
    p.add_argument("--lr-neg", type=float, default=0.05)
    p.add_argument("--ema-alpha", type=float, default=0.01)
    p.add_argument(
        "--csv",
        type=str,
        default="data/runs/arb139/within_run_simlex.csv",
    )
    args = p.parse_args()

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.n_tokens:,} tokens of text8 ...")
    tokens = load_text8(max_tokens=args.n_tokens)
    token_to_id, id_to_token = build_vocab(tokens, vocab_size=args.vocab_size)
    vocab_set = set(id_to_token)
    simlex = load_simlex(vocab=vocab_set)
    token_ids = np.asarray(encode_tokens(tokens, token_to_id), dtype=np.int64)
    V = len(id_to_token)
    N = len(token_ids)

    rng = np.random.default_rng(args.seed)

    # Allocate state once (single-table mode for simplicity).
    A_center = (rng.standard_normal((V, args.n_dims)) * 0.01).astype(np.float32)
    A_context = A_center  # single-table alias
    A_ema_center = A_center.copy()
    A_ema_context = A_ema_center

    cdf = _build_unigram_cdf(token_ids, V, 0.75)

    # Scratch buffers.
    e_center_buf = np.empty(args.k_active, dtype=np.int64)
    e_context_buf = np.empty(args.k_active, dtype=np.int64)
    e_neg_buf = np.empty(args.k_active, dtype=np.int64)

    # Iterate chunks.
    rows: list[dict] = []
    chunk_starts = list(range(0, N, args.checkpoint_stride))
    print(
        f"Running {len(chunk_starts)} chunks of {args.checkpoint_stride:,} tokens "
        f"each (extracting from A_ema_center if ema_alpha>0, else A_center)."
    )

    extract_table = A_ema_center if args.ema_alpha > 0 else A_center
    t_total = time.monotonic()

    for ck_i, chunk_start in enumerate(chunk_starts):
        chunk_end = min(chunk_start + args.checkpoint_stride, N)
        chunk_tids = token_ids[chunk_start:chunk_end]
        if len(chunk_tids) == 0:
            continue

        # Pre-sample negatives for this chunk.
        chunk_pairs = len(chunk_tids) * 2 * args.window
        n_negs = chunk_pairs * args.n_neg + 1024
        neg_uniform = rng.random(n_negs)
        negs_buf = np.searchsorted(cdf, neg_uniform).astype(np.int64)

        t_train = time.monotonic()
        _TRAIN_FN(
            A_center,
            A_context,
            A_ema_center,
            A_ema_context,
            chunk_tids,
            cdf,
            negs_buf,
            e_center_buf,
            e_context_buf,
            e_neg_buf,
            args.n_dims,
            args.k_active,
            args.window,
            args.n_neg,
            float(args.lr_pos),
            float(args.lr_neg),
            0.0,  # decay
            True,  # modulate
            float(args.ema_alpha),
            False,  # subtract_mean
            False,  # sigmoid_bounded
        )
        train_dt = time.monotonic() - t_train

        # Extract + evaluate.
        t_eval = time.monotonic()
        sdrs = extract_sdrs(extract_table, args.k_active, id_to_token)
        emb = ModulatedSSHEmbeddings(sdrs)
        s = evaluate_simlex(emb, simlex)
        cap = evaluate_capacity(emb, seed=args.seed, sample=300)
        eval_dt = time.monotonic() - t_eval

        n_so_far = chunk_end
        row = {
            "checkpoint_idx": ck_i,
            "n_tokens": n_so_far,
            "simlex_spearman": s["spearman"],
            "simlex_n": s["n_pairs"],
            "cap_mean_sim": cap["mean_pairwise_sim"],
            "cap_collision_frac": cap["high_collision_frac"],
            "cap_eff_dim": cap["eff_dim"],
            "train_dt_s": train_dt,
            "eval_dt_s": eval_dt,
        }
        rows.append(row)
        print(
            f"  ck={ck_i:>3d}  n={n_so_far:>9,}  "
            f"simlex={s['spearman']:+.3f}  coll={cap['high_collision_frac']:.3f}  "
            f"ed={cap['eff_dim']:5.1f}  train={train_dt:5.1f}s  eval={eval_dt:.1f}s"
        )
        # Persist after each checkpoint.
        keys = sorted({k for r in rows for k in r})
        with csv_path.open("w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(f"\nDone in {time.monotonic() - t_total:.1f}s. Wrote {csv_path}.")


if __name__ == "__main__":
    main()
