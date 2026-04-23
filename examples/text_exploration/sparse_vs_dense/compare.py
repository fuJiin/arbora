#!/usr/bin/env python3
"""Sparse-vs-dense word embedding comparison driver (ARB-139).

Trains both gensim Skip-gram (dense) and word-level T1 (sparse binary)
on the same text8 subset with the same vocab, then evaluates both on
SimLex-999 and Google analogy. Reports a side-by-side comparison.

Usage:
    uv run python -m examples.text_exploration.sparse_vs_dense.compare
    uv run python -m examples.text_exploration.sparse_vs_dense.compare \
        --max-tokens 1000000 --vocab-size 5000

Sample-efficiency mode runs the comparison at multiple training-token
counts and emits a CSV for plotting curves.
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
    evaluate_partial_cue,
    evaluate_simlex,
)
from examples.text_exploration.sparse_vs_dense.t1_word import train_t1_word
from examples.text_exploration.sparse_vs_dense.word2vec_baseline import train_word2vec


def run_one(
    *,
    n_tokens: int,
    vocab_size: int,
    seed: int,
    skip: list[str],
    w2v_epochs: int,
    t1_epochs: int,
    t1_kwargs: dict | None = None,
) -> list[dict]:
    """Train + eval both architectures on a slice of text8. Returns rows."""
    print(f"\n=== Run: n_tokens={n_tokens} vocab={vocab_size} seed={seed} ===")
    tokens = load_text8(max_tokens=n_tokens)
    token_to_id, id_to_token = build_vocab(tokens, vocab_size=vocab_size)
    vocab_set = set(id_to_token)
    simlex = load_simlex(vocab=vocab_set)
    analogy = load_analogy(vocab=vocab_set)
    print(
        f"  {len(tokens):,} tokens, vocab={len(id_to_token)}, "
        f"simlex={len(simlex)}, analogy={len(analogy)}"
    )

    rows: list[dict] = []

    if "word2vec" not in skip:
        t0 = time.monotonic()
        emb_w2v, stats = train_word2vec(
            tokens,
            vocab=id_to_token,
            vector_size=100,
            window=5,
            min_count=5,
            epochs=w2v_epochs,
            seed=seed,
        )
        s = evaluate_simlex(emb_w2v, simlex)
        a = evaluate_analogy(emb_w2v, analogy)
        cap = evaluate_capacity(emb_w2v, seed=seed)
        rows.append(
            {
                "model": "word2vec",
                "seed": seed,
                "n_tokens": n_tokens,
                "vocab_size": vocab_size,
                "simlex_spearman": s["spearman"],
                "simlex_pearson": s["pearson"],
                "simlex_n": s["n_pairs"],
                "analogy_top1": a["top1"],
                "analogy_n": a["n_entries"],
                "cap_mean_sim": cap["mean_pairwise_sim"],
                "cap_collision_frac": cap["high_collision_frac"],
                "cap_eff_dim": cap["eff_dim"],
                "cap_n_words": cap["n_words"],
                "elapsed_s": stats["elapsed_s"],
                "wall_s": time.monotonic() - t0,
            }
        )
        print(
            f"  word2vec: simlex={s['spearman']:.3f} (n={s['n_pairs']}) "
            f"analogy={a['top1']:.3f} (n={a['n_entries']}) "
            f"| cap: mean_sim={cap['mean_pairwise_sim']:.3f} "
            f"coll={cap['high_collision_frac']:.2f} ed={cap['eff_dim']:.1f} "
            f"({stats['elapsed_s']:.1f}s)"
        )

    if "t1" not in skip:
        t0 = time.monotonic()
        token_ids = encode_tokens(tokens, token_to_id)
        emb_t1, stats = train_t1_word(
            token_ids,
            id_to_token=id_to_token,
            epochs=t1_epochs,
            seed=seed,
            region_kwargs=t1_kwargs or {},
            log_every=0,
        )
        s = evaluate_simlex(emb_t1, simlex)
        a = evaluate_analogy(emb_t1, analogy)
        cap = evaluate_capacity(emb_t1, seed=seed)
        pc = evaluate_partial_cue(emb_t1, simlex, seed=seed)
        rows.append(
            {
                "model": "t1_sparse",
                "seed": seed,
                "n_tokens": n_tokens,
                "vocab_size": vocab_size,
                "simlex_spearman": s["spearman"],
                "simlex_pearson": s["pearson"],
                "simlex_n": s["n_pairs"],
                "analogy_top1": a["top1"],
                "analogy_n": a["n_entries"],
                "cap_mean_sim": cap["mean_pairwise_sim"],
                "cap_collision_frac": cap["high_collision_frac"],
                "cap_eff_dim": cap["eff_dim"],
                "cap_n_words": cap["n_words"],
                "partial_cue_retention": pc["retention"],
                "partial_cue_n": pc["n"],
                "active_per_word_mean": stats["active_per_word_mean"],
                "n_l23_total": stats["n_l23_total"],
                "elapsed_s": stats["elapsed_s"],
                "wall_s": time.monotonic() - t0,
            }
        )
        print(
            f"  t1_sparse: simlex={s['spearman']:.3f} (n={s['n_pairs']}) "
            f"analogy={a['top1']:.3f} (n={a['n_entries']}) "
            f"| cap: mean_sim={cap['mean_pairwise_sim']:.3f} "
            f"coll={cap['high_collision_frac']:.2f} ed={cap['eff_dim']:.1f} "
            f"| pc_retention={pc['retention']:.3f} "
            f"l23_active={stats['active_per_word_mean']:.1f}/{stats['n_l23_total']} "
            f"({stats['elapsed_s']:.1f}s)"
        )

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Sparse vs dense word embedding comparison")
    p.add_argument(
        "--max-tokens",
        type=int,
        nargs="+",
        default=[1_000_000],
        help="Token counts to sweep (one per training run).",
    )
    p.add_argument("--vocab-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--w2v-epochs", type=int, default=5)
    p.add_argument("--t1-epochs", type=int, default=1)
    p.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=["word2vec", "t1"],
        help="Skip these models.",
    )
    p.add_argument("--csv", type=str, default=None)
    p.add_argument(
        "--t1-cols",
        type=int,
        default=256,
        help="T1 n_columns. Default 256 — bigger than char-level due to vocab size.",
    )
    p.add_argument("--t1-k", type=int, default=16, help="T1 k_columns.")
    args = p.parse_args()

    t1_kwargs = {"n_columns": args.t1_cols, "k_columns": args.t1_k}

    all_rows: list[dict] = []
    for n in args.max_tokens:
        all_rows.extend(
            run_one(
                n_tokens=n,
                vocab_size=args.vocab_size,
                seed=args.seed,
                skip=args.skip,
                w2v_epochs=args.w2v_epochs,
                t1_epochs=args.t1_epochs,
                t1_kwargs=t1_kwargs,
            )
        )

    print("\n=== Summary ===")
    for r in all_rows:
        print(
            f"  {r['model']:>10s} @ {r['n_tokens']:>8,} tok | "
            f"simlex={r['simlex_spearman']:.3f} (n={r['simlex_n']}) | "
            f"analogy={r['analogy_top1']:.3f} (n={r['analogy_n']}) | "
            f"{r['elapsed_s']:.1f}s"
        )

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Union of all keys across rows.
        keys = sorted({k for r in all_rows for k in r})
        with out.open("w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
