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
import pickle
import time
from pathlib import Path

from examples.text_exploration.sparse_vs_dense.data import (
    build_vocab,
    encode_tokens,
    load_analogy,
    load_simlex,
    load_text8,
)
from examples.text_exploration.sparse_vs_dense.brown_cluster_baseline import (
    train_brown_cluster,
)
from examples.text_exploration.sparse_vs_dense.evaluation import (
    evaluate_analogy,
    evaluate_bundling_capacity,
    evaluate_capacity,
    evaluate_partial_cue,
    evaluate_simlex,
)
from examples.text_exploration.sparse_vs_dense.random_indexing_baseline import (
    train_random_indexing,
)
from examples.text_exploration.sparse_vs_dense.sparse_skipgram_hebbian_baseline import (
    train_sparse_skipgram_hebbian,
)
from examples.text_exploration.sparse_vs_dense.t1_word import train_t1_word
from examples.text_exploration.sparse_vs_dense.word2vec_baseline import train_word2vec


def _eval_and_row(
    emb,
    *,
    model: str,
    n_tokens: int,
    vocab_size: int,
    seed: int,
    simlex: list,
    analogy: list,
    stats: dict,
    wall_s: float,
    extra: dict | None = None,
) -> dict:
    """Run the shared eval battery on `emb` and assemble one row."""
    s = evaluate_simlex(emb, simlex)
    a = evaluate_analogy(emb, analogy)
    cap = evaluate_capacity(emb, seed=seed)
    bundle = evaluate_bundling_capacity(emb, seed=seed)
    pc = evaluate_partial_cue(emb, simlex, seed=seed)
    row = {
        "model": model,
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
        "bundling_capacity": bundle["capacity_estimate"],
        "bundling_margin_at_k8": _margin_at_k(bundle, 8),
        "bundling_margin_at_k32": _margin_at_k(bundle, 32),
        "partial_cue_retention": pc.get("retention", 0.0),
        "partial_cue_n": pc.get("n", 0),
        "elapsed_s": stats["elapsed_s"],
        "wall_s": wall_s,
    }
    if extra:
        row.update(extra)
    print(
        f"  {model}: simlex={s['spearman']:.3f} (n={s['n_pairs']}) "
        f"analogy={a['top1']:.3f} (n={a['n_entries']}) "
        f"| cap: mean_sim={cap['mean_pairwise_sim']:.3f} "
        f"coll={cap['high_collision_frac']:.2f} ed={cap['eff_dim']:.1f} "
        f"| bundle_k*={bundle['capacity_estimate']} "
        f"({stats['elapsed_s']:.1f}s)"
    )
    return row


def _margin_at_k(bundle: dict, k: int) -> float:
    for entry in bundle.get("per_k", []):
        if entry["k"] == k:
            return float(entry["margin"])
    return 0.0


def run_one(
    *,
    n_tokens: int,
    vocab_size: int,
    seed: int,
    skip: list[str],
    w2v_epochs: int,
    t1_epochs: int,
    t1_kwargs: dict | None = None,
    ri_kwargs: dict | None = None,
    ssh_kwargs: dict | None = None,
    dump_dir: Path | None = None,
) -> list[dict]:
    """Train + eval all enabled architectures on a slice of text8."""
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

    token_ids_cache: list[int] | None = None

    def token_ids() -> list[int]:
        nonlocal token_ids_cache
        if token_ids_cache is None:
            token_ids_cache = encode_tokens(tokens, token_to_id)
        return token_ids_cache

    rows: list[dict] = []

    if "word2vec" not in skip:
        t0 = time.monotonic()
        emb, stats = train_word2vec(
            tokens,
            vocab=id_to_token,
            vector_size=100,
            window=5,
            min_count=5,
            epochs=w2v_epochs,
            seed=seed,
        )
        rows.append(
            _eval_and_row(
                emb,
                model="word2vec",
                n_tokens=n_tokens,
                vocab_size=vocab_size,
                seed=seed,
                simlex=simlex,
                analogy=analogy,
                stats=stats,
                wall_s=time.monotonic() - t0,
            )
        )
        if dump_dir is not None:
            _dump_embeddings(dump_dir, emb, "word2vec", n_tokens=n_tokens, seed=seed)

    if "random_indexing" not in skip:
        t0 = time.monotonic()
        emb, stats = train_random_indexing(
            token_ids(),
            id_to_token=id_to_token,
            seed=seed,
            **(ri_kwargs or {}),
        )
        rows.append(
            _eval_and_row(
                emb,
                model="random_indexing",
                n_tokens=n_tokens,
                vocab_size=vocab_size,
                seed=seed,
                simlex=simlex,
                analogy=analogy,
                stats=stats,
                wall_s=time.monotonic() - t0,
                extra={
                    "active_per_word_mean": stats["active_per_word_mean"],
                    "n_dims": stats["n_dims"],
                },
            )
        )
        if dump_dir is not None:
            _dump_embeddings(
                dump_dir, emb, "random_indexing", n_tokens=n_tokens, seed=seed
            )

    if "brown_cluster" not in skip:
        t0 = time.monotonic()
        emb, stats = train_brown_cluster(
            token_ids(),
            id_to_token=id_to_token,
            seed=seed,
        )
        rows.append(
            _eval_and_row(
                emb,
                model="brown_cluster",
                n_tokens=n_tokens,
                vocab_size=vocab_size,
                seed=seed,
                simlex=simlex,
                analogy=analogy,
                stats=stats,
                wall_s=time.monotonic() - t0,
                extra={
                    "active_per_word_mean": stats["active_per_word_mean"],
                    "n_dims": stats["n_dims"],
                    "mean_depth": stats["mean_depth"],
                    "max_depth": stats["max_depth"],
                },
            )
        )
        if dump_dir is not None:
            _dump_embeddings(
                dump_dir, emb, "brown_cluster", n_tokens=n_tokens, seed=seed
            )

    if "sparse_skipgram_hebbian" not in skip:
        t0 = time.monotonic()
        emb, stats = train_sparse_skipgram_hebbian(
            token_ids(),
            id_to_token=id_to_token,
            seed=seed,
            **(ssh_kwargs or {}),
        )
        rows.append(
            _eval_and_row(
                emb,
                model="sparse_skipgram_hebbian",
                n_tokens=n_tokens,
                vocab_size=vocab_size,
                seed=seed,
                simlex=simlex,
                analogy=analogy,
                stats=stats,
                wall_s=time.monotonic() - t0,
                extra={
                    "active_per_word_mean": stats["active_per_word_mean"],
                    "n_dims": stats["n_dims"],
                    "k_active": stats["k_active"],
                },
            )
        )
        if dump_dir is not None:
            _dump_embeddings(
                dump_dir,
                emb,
                "sparse_skipgram_hebbian",
                n_tokens=n_tokens,
                seed=seed,
            )

    if "t1" not in skip:
        t0 = time.monotonic()
        emb, stats = train_t1_word(
            token_ids(),
            id_to_token=id_to_token,
            epochs=t1_epochs,
            seed=seed,
            region_kwargs=t1_kwargs or {},
            log_every=0,
        )
        rows.append(
            _eval_and_row(
                emb,
                model="t1_sparse",
                n_tokens=n_tokens,
                vocab_size=vocab_size,
                seed=seed,
                simlex=simlex,
                analogy=analogy,
                stats=stats,
                wall_s=time.monotonic() - t0,
                extra={
                    "active_per_word_mean": stats["active_per_word_mean"],
                    "n_l23_total": stats["n_l23_total"],
                },
            )
        )
        if dump_dir is not None:
            _dump_embeddings(dump_dir, emb, "t1_sparse", n_tokens=n_tokens, seed=seed)

    return rows


def _dump_embeddings(
    dump_dir: Path,
    emb,
    model_name: str,
    *,
    n_tokens: int,
    seed: int,
) -> None:
    """Pickle a dict[word, vector] for post-hoc viz.

    For sparse: pickle the boolean SDRs dict directly. For dense
    (word2vec): materialize a dict from the KeyedVectors so the
    pickle doesn't carry gensim-specific types.
    """
    dump_dir.mkdir(parents=True, exist_ok=True)
    name = f"{model_name}_n{n_tokens}_s{seed}.pkl"
    path = dump_dir / name
    # All baselines return ndarrays from `get()` — boolean SDRs for
    # sparse, float vectors for dense. Copy so the pickle is decoupled
    # from any native storage (e.g. gensim KeyedVectors).
    payload = {w: emb.get(w).copy() for w in emb.vocab() if emb.get(w) is not None}
    with path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"    dumped → {path}")


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
        choices=[
            "word2vec",
            "random_indexing",
            "brown_cluster",
            "sparse_skipgram_hebbian",
            "t1",
        ],
        help="Skip these models.",
    )
    p.add_argument(
        "--ssh-dims",
        type=int,
        default=1024,
        help="Sparse-skipgram-Hebbian n_dims.",
    )
    p.add_argument(
        "--ssh-k",
        type=int,
        default=40,
        help="Sparse-skipgram-Hebbian k_active.",
    )
    p.add_argument(
        "--ssh-lr-pos",
        type=float,
        default=0.05,
        help="Sparse-skipgram-Hebbian positive-pair learning rate.",
    )
    p.add_argument(
        "--ssh-lr-neg",
        type=float,
        default=0.02,
        help="Sparse-skipgram-Hebbian anti-Hebbian (negative) rate.",
    )
    p.add_argument(
        "--ri-dims",
        type=int,
        default=2048,
        help="Random Indexing n_dims (index/context vector size).",
    )
    p.add_argument(
        "--ri-k",
        type=int,
        default=40,
        help="Random Indexing k_active (top-k bits in the binarized SDR).",
    )
    p.add_argument("--csv", type=str, default=None)
    p.add_argument(
        "--t1-cols",
        type=int,
        default=256,
        help="T1 n_columns. Default 256 — bigger than char-level due to vocab size.",
    )
    p.add_argument("--t1-k", type=int, default=16, help="T1 k_columns.")
    p.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help=(
            "If set, pickle per-(model, n_tokens, seed) embedding dicts "
            "into this directory for downstream viz."
        ),
    )
    args = p.parse_args()

    t1_kwargs = {"n_columns": args.t1_cols, "k_columns": args.t1_k}
    ri_kwargs = {"n_dims": args.ri_dims, "k_active": args.ri_k}
    ssh_kwargs = {
        "n_dims": args.ssh_dims,
        "k_active": args.ssh_k,
        "lr_pos": args.ssh_lr_pos,
        "lr_neg": args.ssh_lr_neg,
    }

    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    csv_path = Path(args.csv) if args.csv else None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for n in args.max_tokens:
        new_rows = run_one(
            n_tokens=n,
            vocab_size=args.vocab_size,
            seed=args.seed,
            skip=args.skip,
            w2v_epochs=args.w2v_epochs,
            t1_epochs=args.t1_epochs,
            t1_kwargs=t1_kwargs,
            ri_kwargs=ri_kwargs,
            ssh_kwargs=ssh_kwargs,
            dump_dir=dump_dir,
        )
        all_rows.extend(new_rows)
        # Persist CSV after every token-count batch so a kill mid-sweep
        # doesn't lose completed rows. Re-emits header each time using
        # the current union of keys so a row added later doesn't drop
        # earlier rows' columns.
        if csv_path is not None and all_rows:
            keys = sorted({k for r in all_rows for k in r})
            with csv_path.open("w") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in all_rows:
                    w.writerow(r)
            print(f"  [partial] wrote {csv_path} ({len(all_rows)} rows)")

    print("\n=== Summary ===")
    for r in all_rows:
        print(
            f"  {r['model']:>10s} @ {r['n_tokens']:>8,} tok | "
            f"simlex={r['simlex_spearman']:.3f} (n={r['simlex_n']}) | "
            f"analogy={r['analogy_top1']:.3f} (n={r['analogy_n']}) | "
            f"{r['elapsed_s']:.1f}s"
        )

    if csv_path is not None:
        print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
