#!/usr/bin/env python3
"""ARB-139: decay rate sweep for modulated SSH at 1M tokens.

Hypothesis: 1e-4 was too aggressive at 1M because each word gets ~2x more
update applications than at 500k, so total decay-shrinkage per word is too
large. If decay needs to scale inversely with corpus size, the optimal rate
at 1M should be ~5e-5 (half of 500k's optimum).

Sweep: [0, 1e-5, 3e-5, 5e-5, 1e-4, 3e-4]
Output: data/runs/arb139/decay_sweep_1M.csv
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
        f"  simlex={s['spearman']:+.3f} "
        f"coll={cap['high_collision_frac']:.3f} "
        f"ed={cap['eff_dim']:.1f} "
        f"bundle_k*={bundle['capacity_estimate']} "
        f"| train={train_s:.1f}s"
    )
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-tokens", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vocab-size", type=int, default=5000)
    p.add_argument(
        "--decays",
        type=float,
        nargs="+",
        default=[0.0, 1e-5, 3e-5, 5e-5, 1e-4, 3e-4],
    )
    p.add_argument(
        "--csv",
        type=str,
        default="data/runs/arb139/decay_sweep_1M.csv",
    )
    args = p.parse_args()

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for decay in args.decays:
        row = run_one(
            decay,
            n_tokens=args.n_tokens,
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
