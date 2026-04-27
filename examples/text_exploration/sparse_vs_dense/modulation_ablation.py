#!/usr/bin/env python3
"""ARB-139 modulation ablation: vanilla SSH vs modulated SSH vs mod+decay.

Six runs total — three variants × two corpus sizes (500k, 1M tokens) —
on text8 vocab=5000, seed=0. Writes one CSV with all rows.

Usage:
    uv run python -m examples.text_exploration.sparse_vs_dense.modulation_ablation \\
        --csv data/runs/arb139/modulation_ablation.csv

Variants:
    vanilla         lr_pos=0.05  lr_neg=0.02  modulate=False  decay=0
                    (matches the original sparse_skipgram_hebbian_baseline)
    modulated       lr_pos=0.05  lr_neg=0.05  modulate=True   decay=0
                    (symmetric rates rebalanced by surprise modulator)
    mod+decay       lr_pos=0.05  lr_neg=0.05  modulate=True   decay=1e-4
                    (Oja-like uniform decay on top of modulation)
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


VARIANTS = [
    # (name, lr_pos, lr_neg, modulate, decay)
    ("vanilla", 0.05, 0.02, False, 0.0),
    ("modulated", 0.05, 0.05, True, 0.0),
    ("mod+decay", 0.05, 0.05, True, 1e-4),
]


def run_variant(
    variant_name: str,
    lr_pos: float,
    lr_neg: float,
    modulate: bool,
    decay: float,
    *,
    n_tokens: int,
    vocab_size: int,
    seed: int,
) -> dict:
    print(f"\n--- variant={variant_name}  n_tokens={n_tokens:,}  seed={seed} ---")
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
        lr_pos=lr_pos,
        lr_neg=lr_neg,
        modulate=modulate,
        decay=decay,
        seed=seed,
    )
    train_s = time.monotonic() - t0

    s = evaluate_simlex(emb, simlex)
    a = evaluate_analogy(emb, analogy)
    cap = evaluate_capacity(emb, seed=seed)
    bundle = evaluate_bundling_capacity(emb, seed=seed)

    row = {
        "variant": variant_name,
        "n_tokens": n_tokens,
        "seed": seed,
        "lr_pos": lr_pos,
        "lr_neg": lr_neg,
        "modulate": modulate,
        "decay": decay,
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
        f"  simlex={s['spearman']:+.3f} (n={s['n_pairs']}) "
        f"analogy={a['top1']:.3f} (n={a['n_entries']}) "
        f"| coll={cap['high_collision_frac']:.3f} "
        f"ed={cap['eff_dim']:.1f} bundle_k*={bundle['capacity_estimate']} "
        f"| train={train_s:.1f}s"
    )
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-tokens", type=int, nargs="+", default=[500_000, 1_000_000])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vocab-size", type=int, default=5000)
    p.add_argument(
        "--csv",
        type=str,
        default="data/runs/arb139/modulation_ablation.csv",
    )
    args = p.parse_args()

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for n in args.n_tokens:
        for variant_name, lr_pos, lr_neg, modulate, decay in VARIANTS:
            row = run_variant(
                variant_name,
                lr_pos,
                lr_neg,
                modulate,
                decay,
                n_tokens=n,
                vocab_size=args.vocab_size,
                seed=args.seed,
            )
            all_rows.append(row)
            # Write CSV after every row so a crash doesn't lose work.
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
