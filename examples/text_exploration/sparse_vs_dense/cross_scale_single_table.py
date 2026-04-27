#!/usr/bin/env python3
"""ARB-139: cross-scale test of single-table modulated SSH.

Runs at 1M, 5M, 10M tokens (100k and 500k already done in smoke test).
seed=0, decay=0 (modulation alone is the simplest variant; we already
showed decay tuning is fragile across scales).

Output: data/runs/arb139/cross_scale_single_table.csv

Total wall-clock estimate: ~3 hours sequential, ~5 hours if running in
parallel with another job due to CPU contention.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from examples.text_exploration.sparse_vs_dense.data import (
    build_vocab,
    encode_tokens,
    load_analogy,
    load_simlex,
    load_text8,
)
from examples.text_exploration.sparse_vs_dense.evaluation import (
    evaluate_analogy,
    evaluate_bundling_capacity,
    evaluate_capacity,
    evaluate_simlex,
)
from examples.text_exploration.sparse_vs_dense.sparse_skipgram_hebbian_modulated_baseline import (
    train_sparse_skipgram_hebbian_modulated,
)


def run_one(*, n_tokens: int, vocab_size: int, seed: int) -> dict:
    print(f"\n--- single-table  n_tokens={n_tokens:,}  seed={seed} ---")
    tokens = load_text8(max_tokens=n_tokens)
    token_to_id, id_to_token = build_vocab(tokens, vocab_size=vocab_size)
    vocab_set = set(id_to_token)
    simlex = load_simlex(vocab=vocab_set)
    analogy = load_analogy(vocab=vocab_set)
    token_ids = encode_tokens(tokens, token_to_id)

    t0 = time.monotonic()
    emb, stats = train_sparse_skipgram_hebbian_modulated(
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
        seed=seed,
    )
    train_s = time.monotonic() - t0

    s = evaluate_simlex(emb, simlex)
    a = evaluate_analogy(emb, analogy)
    cap = evaluate_capacity(emb, seed=seed)
    bundle = evaluate_bundling_capacity(emb, seed=seed)

    row = {
        "single_table": True,
        "decay": 0.0,
        "n_tokens": n_tokens,
        "seed": seed,
        "simlex_spearman": s["spearman"],
        "simlex_n": s["n_pairs"],
        "analogy_top1": a["top1"],
        "analogy_n": a["n_entries"],
        "cap_mean_sim": cap["mean_pairwise_sim"],
        "cap_collision_frac": cap["high_collision_frac"],
        "cap_eff_dim": cap["eff_dim"],
        "bundling_capacity": bundle["capacity_estimate"],
        "active_per_word_mean": stats["active_per_word_mean"],
        "train_s": train_s,
    }
    print(
        f"  simlex={s['spearman']:+.3f} analogy={a['top1']:.3f} "
        f"coll={cap['high_collision_frac']:.3f} ed={cap['eff_dim']:.1f} "
        f"bundle_k*={bundle['capacity_estimate']} | train={train_s:.1f}s"
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
        default=[1_000_000, 5_000_000, 10_000_000],
    )
    p.add_argument(
        "--csv",
        type=str,
        default="data/runs/arb139/cross_scale_single_table.csv",
    )
    args = p.parse_args()

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load any existing rows and skip already-completed sizes.
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
        print(f"  [partial] wrote {csv_path} ({len(all_rows)} rows)")

    print(f"\nDone. Wrote {csv_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
