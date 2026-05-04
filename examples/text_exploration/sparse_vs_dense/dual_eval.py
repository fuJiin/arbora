#!/usr/bin/env python3
"""ARB-139: dual-eval diagnostic — Jaccard-on-binary vs cosine-on-raw-accumulator.

Trains modulated SSH (sigmoid-bounded, single-table, in-order) at
several corpus sizes and evaluates SimLex two ways:

    binary:     Jaccard over top-k(A_w) — current default, what downstream
                consumers actually see
    continuous: cosine over raw A_w — diagnostic eval of the underlying
                continuous representation

If the two diverge sharply (continuous smooth, binary jumpy), the bumps
we see are an eval-side artifact of discrete readout + Spearman-over-
Jaccard, not algorithm instability. If both bounce, the algorithm has
real instability we need to address structurally.

Output: data/runs/arb139/dual_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from examples.text_exploration.sparse_vs_dense.data import (
    build_vocab,
    encode_tokens,
    load_simlex,
    load_text8,
)
from examples.text_exploration.sparse_vs_dense.evaluation import evaluate_simlex
from examples.text_exploration.sparse_vs_dense.sparse_skipgram_hebbian_modulated_baseline import (
    train_sparse_skipgram_hebbian_modulated,
)


def run_one(*, n_tokens: int, vocab_size: int, seed: int) -> dict:
    print(f"\n--- n_tokens={n_tokens:,} seed={seed} ---")
    tokens = load_text8(max_tokens=n_tokens)
    token_to_id, id_to_token = build_vocab(tokens, vocab_size=vocab_size)
    vocab_set = set(id_to_token)
    simlex = load_simlex(vocab=vocab_set)
    token_ids = encode_tokens(tokens, token_to_id)

    t0 = time.monotonic()
    emb, _stats = train_sparse_skipgram_hebbian_modulated(
        token_ids,
        id_to_token=id_to_token,
        n_dims=1024,
        k_active=40,
        window=5,
        n_neg=5,
        lr_pos=0.05,
        lr_neg=0.05,
        modulate=True,
        decay=0.0,
        single_table=True,
        ema_alpha=0.0,
        sigmoid_bounded=True,
        seed=seed,
    )
    train_s = time.monotonic() - t0

    # Eval 1: Jaccard on binary
    s_binary = evaluate_simlex(emb, simlex)
    # Eval 2: cosine on raw accumulator
    s_continuous = evaluate_simlex(emb.continuous, simlex)

    row = {
        "n_tokens": n_tokens,
        "seed": seed,
        "simlex_binary_jaccard": s_binary["spearman"],
        "simlex_continuous_cosine": s_continuous["spearman"],
        "delta": s_continuous["spearman"] - s_binary["spearman"],
        "simlex_n": s_binary["n_pairs"],
        "train_s": train_s,
    }
    print(
        f"  binary(Jaccard)={s_binary['spearman']:+.3f}  "
        f"continuous(cosine)={s_continuous['spearman']:+.3f}  "
        f"Δ={row['delta']:+.3f}  | train={train_s:.1f}s"
    )
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n-tokens",
        type=int,
        nargs="+",
        default=[100_000, 500_000, 1_000_000],
    )
    p.add_argument(
        "--csv",
        type=str,
        default="data/runs/arb139/dual_eval.csv",
    )
    args = p.parse_args()

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    done_sizes: set[int] = set()
    if csv_path.exists():
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                existing.append(row)
                done_sizes.add(int(row["n_tokens"]))

    all_rows: list[dict] = list(existing)
    for n in args.n_tokens:
        if n in done_sizes:
            print(f"--- skipping: n_tokens={n:,} (already in CSV) ---")
            continue
        row = run_one(n_tokens=n, vocab_size=args.vocab_size, seed=args.seed)
        all_rows.append(row)
        keys = sorted({k for r in all_rows for k in r})
        with csv_path.open("w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)

    print(f"\nDone. Wrote {csv_path}.")


if __name__ == "__main__":
    main()
