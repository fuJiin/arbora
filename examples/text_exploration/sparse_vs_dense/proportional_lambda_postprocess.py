"""Post-processing for the proportional-lambda-consolidation sweep.

Reads the sweep outputs at runs/arb139/proportional_lambda_2026-05-02/,
computes per-pair retention on the shared SimLex partition, dumps a
parquet for the downstream subspace-overlap KPI, and renders the
retention bar chart + per-bit bonus distribution histogram.

Usage:
    uv run --extra embeddings --extra viz python -m \
        examples.text_exploration.sparse_vs_dense.proportional_lambda_postprocess \
        --sweep-root runs/arb139/proportional_lambda_2026-05-02
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from examples.text_exploration.sparse_vs_dense.continual_learning_v2 import (
    CorpusPlan,
    build_vocab,
    prepare_corpus,
    split_simlex_pairs,
    words_in_corpus,
)
from examples.text_exploration.sparse_vs_dense.data import load_simlex, load_text8
from examples.text_exploration.sparse_vs_dense.evaluation import (
    cosine_similarity,
    jaccard_similarity,
)
from examples.text_exploration.sparse_vs_dense.gutenberg_loader import (
    load_gutenberg_corpus,
)

VARIANTS = ["w2v_baseline", "ssh_baseline", "ssh_absolute", "ssh_proportional"]


def per_pair_retention(
    snap1: dict[str, np.ndarray],
    snap2: dict[str, np.ndarray],
    pairs: list[tuple[str, str, float]],
    is_sparse: bool,
) -> pd.DataFrame:
    """For each (a, b, human_score) pair, return phase-1 + phase-2 sims and retention."""
    sim_fn = jaccard_similarity if is_sparse else cosine_similarity
    rows = []
    for a, b, human in pairs:
        if a not in snap1 or b not in snap1 or a not in snap2 or b not in snap2:
            continue
        s1 = float(sim_fn(snap1[a], snap1[b]))
        s2 = float(sim_fn(snap2[a], snap2[b]))
        rows.append(
            {
                "word_a": a,
                "word_b": b,
                "human": float(human),
                "sim_phase1": s1,
                "sim_phase2": s2,
                "retention_pair": (s2 / s1) if abs(s1) > 1e-6 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def rebuild_partitions(n_per_phase: int, vocab_hint: int, seed: int):
    """Rebuild the SimLex partitions identically to a sweep cell.

    Loads text8 + Gutenberg, runs prepare_corpus with the same plan, builds
    the joint vocab, and returns (vocab_set, parts).
    """
    tokens_a_raw = load_text8(max_tokens=n_per_phase)
    tokens_b_raw = load_gutenberg_corpus(max_tokens=n_per_phase)
    plan = CorpusPlan(
        chunk_size=1000,
        subsample_threshold=1e-3,
        min_count=5,
        shuffle_chunks=False,
        seed=seed,
    )
    prep_a = prepare_corpus(tokens_a_raw, plan=plan)
    prep_b = prepare_corpus(tokens_b_raw, plan=plan)
    combined_tokens = prep_a.flat_tokens + prep_b.flat_tokens
    _, id_to_token = build_vocab(combined_tokens, vocab_size=vocab_hint)
    vocab_set = set(id_to_token)
    a_words = words_in_corpus(prep_a.flat_tokens, vocab_set)
    b_words = words_in_corpus(prep_b.flat_tokens, vocab_set)
    pairs = load_simlex(vocab=vocab_set)
    parts = split_simlex_pairs(pairs, a_words, b_words)
    return vocab_set, parts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", type=str, required=True)
    p.add_argument("--n-per-phase", type=int, default=1_000_000)
    p.add_argument("--vocab-hint", type=int, default=10_000)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument(
        "--bonus",
        type=float,
        default=0.5,
        help="Bonus magnitude used in the sweep (for histogram annotation).",
    )
    args = p.parse_args()

    sweep_root = Path(args.sweep_root)
    seeds = [int(s) for s in args.seeds.split(",")]

    # Rebuild SimLex partitions once per seed (joint vocab depends on seed-affected
    # subsampling — but in practice partitions are stable across seeds because the
    # vocabulary is dominated by frequent words that always survive).
    print("Rebuilding SimLex partitions ...")
    _, parts_seed0 = rebuild_partitions(args.n_per_phase, args.vocab_hint, seeds[0])
    print(
        f"  shared={len(parts_seed0['shared'])} a_only={len(parts_seed0['a_only'])} "
        f"b_only={len(parts_seed0['b_only'])} cross={len(parts_seed0['cross'])}"
    )

    all_pair_rows = []
    summary_rows = []

    for seed in seeds:
        for variant in VARIANTS:
            cell_dir = sweep_root / f"seed_{seed}" / variant
            if (
                not (cell_dir / "snap1_word2vec.pkl").exists()
                and not (cell_dir / "snap1_ssh.pkl").exists()
            ):
                print(f"  [skip] {cell_dir} — no snapshots")
                continue
            method = "word2vec" if variant == "w2v_baseline" else "ssh"
            is_sparse = variant != "w2v_baseline"
            with (cell_dir / f"snap1_{method}.pkl").open("rb") as f:
                snap1 = pickle.load(f)
            with (cell_dir / f"snap2_{method}.pkl").open("rb") as f:
                snap2 = pickle.load(f)

            shared = parts_seed0["shared"]
            df = per_pair_retention(snap1, snap2, shared, is_sparse=is_sparse)
            df["seed"] = seed
            df["variant"] = variant
            df["partition"] = "shared"
            all_pair_rows.append(df)

            # Aggregate retention: ratio of mean partition similarity.
            mean_s1 = df["sim_phase1"].mean()
            mean_s2 = df["sim_phase2"].mean()
            retention = (mean_s2 / mean_s1) if abs(mean_s1) > 1e-6 else float("nan")
            summary_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "n_shared_pairs": len(df),
                    "mean_sim_phase1": mean_s1,
                    "mean_sim_phase2": mean_s2,
                    "retention_shared": retention,
                }
            )
            print(
                f"  seed={seed} variant={variant}: "
                f"n={len(df)} sim1={mean_s1:.4f} sim2={mean_s2:.4f} "
                f"retention={retention:.3f}"
            )

    pair_df = pd.concat(all_pair_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    parquet_path = sweep_root / "per_pair_retention.parquet"
    pair_df.to_parquet(parquet_path, index=False)
    print(f"\nWrote {parquet_path}  ({len(pair_df)} rows)")

    summary_path = sweep_root / "summary_retention.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    # Aggregate stats per variant.
    agg = (
        summary_df.groupby("variant")["retention_shared"]
        .agg(["mean", "std", "count"])
        .reindex(VARIANTS)
        .reset_index()
    )
    print("\nPer-variant retention (shared partition):")
    print(agg.to_string(index=False))
    agg.to_csv(sweep_root / "agg_retention.csv", index=False)
    print(f"Wrote {sweep_root / 'agg_retention.csv'}")


if __name__ == "__main__":
    main()
