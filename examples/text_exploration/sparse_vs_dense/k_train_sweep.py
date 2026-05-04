#!/usr/bin/env python3
"""ARB-139 plateau diagnostic Phase B: train SSH with varied (k, D).

The k_eval sweep showed the trained accumulator at k_train=40, D=1024
has discriminative structure that's better extracted with k_eval=160.
Now: does training at higher k change the underlying structure?

Two configs at fixed corpus size (1M default), seed=0:

  - k=160, D=1024 — denser readout, same dim space (16% active)
  - k=160, D=4096 — preserves the original 4% sparsity ratio

Comparison at k_eval = k_train (matched readout) so we're testing
training-time behavior, not just readout aperture.

Output: data/runs/arb139/k_train_sweep.csv
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
    evaluate_capacity,
    evaluate_simlex,
)
from examples.text_exploration.sparse_vs_dense.sparse_skipgram_hebbian_modulated_baseline import (
    train_sparse_skipgram_hebbian_modulated,
)


def run_one(*, n_tokens: int, n_dims: int, k_active: int, seed: int) -> dict:
    label = f"D={n_dims}, k={k_active} ({100 * k_active / n_dims:.1f}% sparse)"
    print(f"\n--- training SSH @ {label}, n_tokens={n_tokens:,}, seed={seed} ---")
    tokens = load_text8(max_tokens=n_tokens)
    token_to_id, id_to_token = build_vocab(tokens, vocab_size=5000)
    vocab_set = set(id_to_token)
    simlex = load_simlex(vocab=vocab_set)
    analogy = load_analogy(vocab=vocab_set)
    token_ids = encode_tokens(tokens, token_to_id)

    t0 = time.monotonic()
    emb, _stats = train_sparse_skipgram_hebbian_modulated(
        token_ids,
        id_to_token=id_to_token,
        n_dims=n_dims,
        k_active=k_active,
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

    s = evaluate_simlex(emb, simlex)
    a = evaluate_analogy(emb, analogy)
    cap = evaluate_capacity(emb, seed=seed, sample=300)

    row = {
        "n_tokens": n_tokens,
        "n_dims": n_dims,
        "k_active": k_active,
        "sparsity_pct": 100.0 * k_active / n_dims,
        "seed": seed,
        "simlex_jaccard": s["spearman"],
        "simlex_n": s["n_pairs"],
        "analogy_top1": a["top1"],
        "analogy_n": a["n_entries"],
        "cap_mean_sim": cap["mean_pairwise_sim"],
        "cap_collision_frac": cap["high_collision_frac"],
        "cap_eff_dim": cap["eff_dim"],
        "train_s": train_s,
    }
    print(
        f"  simlex(jaccard)={s['spearman']:+.4f}  analogy={a['top1']:.4f}  "
        f"coll={cap['high_collision_frac']:.3f}  ed={cap['eff_dim']:.1f}  "
        f"train={train_s:.1f}s"
    )
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n-tokens",
        type=int,
        default=1_000_000,
        help="Corpus size for the sweep.",
    )
    p.add_argument(
        "--csv",
        type=str,
        default="data/runs/arb139/k_train_sweep.csv",
    )
    args = p.parse_args()

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip configs already in CSV.
    existing: list[dict] = []
    done_keys: set[tuple[int, int, int, int]] = set()
    if csv_path.exists():
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                existing.append(row)
                done_keys.add(
                    (
                        int(row["n_tokens"]),
                        int(row["n_dims"]),
                        int(row["k_active"]),
                        int(row["seed"]),
                    )
                )

    plan = [
        # (n_dims, k_active)
        (1024, 40),  # baseline (already known)
        (1024, 160),  # denser readout, same D
        (4096, 160),  # preserved 4% sparsity, 4× D
    ]

    all_rows: list[dict] = list(existing)
    for n_dims, k_active in plan:
        key = (args.n_tokens, n_dims, k_active, args.seed)
        if key in done_keys:
            print(f"--- skip: D={n_dims}, k={k_active} (already in CSV) ---")
            continue
        row = run_one(
            n_tokens=args.n_tokens,
            n_dims=n_dims,
            k_active=k_active,
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
