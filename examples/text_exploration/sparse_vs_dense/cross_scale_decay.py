#!/usr/bin/env python3
"""ARB-139: cross-scale test of modulated SSH with vs without decay=3e-4.

Test whether decay=3e-4 (the 1M-tokens optimum) is universally good or
scale-dependent. Runs (decay, n_tokens) combinations on text8 vocab=5000,
seed=0. 1M is skipped — we already have decay=0 (0.073) and decay=3e-4
(0.098) from `decay_sweep_1M.csv`.

Output: data/runs/arb139/cross_scale_decay.csv

Total wall-clock estimate: ~5.5 hours
  100k × 2: ~2 min
  500k × 2: ~10 min
  5M × 2:   ~110 min
  10M × 2:  ~210 min
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


def run_one(decay: float, *, n_tokens: int, vocab_size: int, seed: int) -> dict:
    print(f"\n--- decay={decay:g}  n_tokens={n_tokens:,}  seed={seed} ---")
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
        decay=decay,
        seed=seed,
    )
    train_s = time.monotonic() - t0

    s = evaluate_simlex(emb, simlex)
    a = evaluate_analogy(emb, analogy)
    cap = evaluate_capacity(emb, seed=seed)
    bundle = evaluate_bundling_capacity(emb, seed=seed)

    row = {
        "decay": decay,
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
        "--csv",
        type=str,
        default="data/runs/arb139/cross_scale_decay.csv",
    )
    args = p.parse_args()

    # (decay, n_tokens) plan — interleaved by size so partial results give
    # a complete-shape view (rather than all 100k first then all 10M last).
    plan = []
    for n in [100_000, 500_000, 5_000_000, 10_000_000]:
        for decay in [0.0, 3e-4]:
            plan.append((decay, n))

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Load any existing rows so we can resume after a kill.
    existing: list[dict] = []
    done_keys: set[tuple[float, int]] = set()
    if csv_path.exists():
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                existing.append(row)
                done_keys.add((float(row["decay"]), int(row["n_tokens"])))

    all_rows: list[dict] = list(existing)
    for decay, n_tokens in plan:
        if (decay, n_tokens) in done_keys:
            print(
                f"--- skipping: decay={decay:g} n_tokens={n_tokens:,} (already in CSV) ---"
            )
            continue
        row = run_one(
            decay,
            n_tokens=n_tokens,
            vocab_size=args.vocab_size,
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

    print(f"\nDone. Wrote {csv_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
