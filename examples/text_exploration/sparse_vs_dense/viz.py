#!/usr/bin/env python3
"""Visualization for ARB-139 conviction artifacts.

Produces four blog-ready figures from the `compare.py` outputs:

1. **Per-word representation grid** (Fig 1). For a curated set of words
   (dog, cat, king, queen, ...), show:
   - T1 sparse SDR as a binary grid (lit cells = active).
   - word2vec dense vector as a sorted color bar.
   Side by side, same words. Shows visually why one is "sparse" and the
   other is "dense" — and how distinct the sparse codes look per word.

2. **Pairwise similarity heatmap** (Fig 2). 50 words x 50 words. T1:
   Jaccard. word2vec: cosine. Rendered side by side. Dense collapses to
   mostly-red; sparse shows structure.

3. **Corruption robustness curve** (Fig 3). X=corruption level, Y=mean
   similarity retention on high-rated SimLex pairs. Sparse vs dense
   plotted together. Sparse stays flat; dense degrades faster.

4. **Sample-efficiency / capacity curves** (Fig 4). X=training tokens,
   Y=eff_dim. T1 vs word2vec. Shows how representation capacity
   evolves with data.

All inputs come from:
- `compare.py --dump-dir D` pickles per-(model, n_tokens, seed) dicts.
- `compare.py --csv X.csv` writes the aggregate metrics.

Usage:
    uv run python -m examples.text_exploration.sparse_vs_dense.viz \\
        --dump-dir /tmp/svd_dumps \\
        --csv /tmp/svd_multiseed_s0.csv /tmp/svd_multiseed_s1.csv ... \\
        --out-dir /tmp/svd_figures
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import pickle
from pathlib import Path

import numpy as np

from examples.text_exploration.sparse_vs_dense.evaluation import (
    _corrupt_dense,
    _corrupt_sparse,
    benchmark_nn_query,
    cosine_similarity,
    evaluate_nn_retrieval,
    jaccard_similarity,
    storage_bytes_per_embedding,
)


class _DictEmbeddings:
    """Lightweight `Embeddings`-protocol adapter over a `{word: vec}` dict.

    Lets the viz code drive the evaluation.* helpers directly on the
    pickle dumps produced by compare.py.
    """

    def __init__(self, d: dict[str, np.ndarray], *, sparse: bool, name: str) -> None:
        self._d = d
        self._sparse = sparse
        self.name = name

    def vocab(self) -> list[str]:
        return list(self._d.keys())

    def get(self, word: str) -> np.ndarray | None:
        return self._d.get(word)

    def is_sparse(self) -> bool:
        return self._sparse


# Words chosen for the per-word grid + similarity heatmap. Mix of
# common-function (the, of), concrete-noun (dog, cat), semantic pairs
# (king/queen, man/woman, war/peace), and topic-words (computer,
# science, music, light).
DEFAULT_CURATED = [
    "the",
    "of",
    "and",
    "in",
    "one",
    "two",
    "dog",
    "cat",
    "horse",
    "bird",
    "king",
    "queen",
    "man",
    "woman",
    "war",
    "peace",
    "love",
    "fear",
    "computer",
    "science",
    "music",
    "light",
    "water",
    "fire",
    "good",
    "bad",
    "big",
    "small",
    "red",
    "blue",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_embedding_dump(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        return pickle.load(f)


def load_csv(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with p.open() as f:
            rows.extend(csv.DictReader(f))
    # Coerce numerics.
    numeric_keys = {
        "seed",
        "n_tokens",
        "vocab_size",
        "simlex_spearman",
        "simlex_pearson",
        "simlex_n",
        "analogy_top1",
        "analogy_n",
        "cap_mean_sim",
        "cap_collision_frac",
        "cap_eff_dim",
        "cap_n_words",
        "partial_cue_retention",
        "partial_cue_n",
        "active_per_word_mean",
        "n_l23_total",
        "elapsed_s",
        "wall_s",
    }
    for r in rows:
        for k in numeric_keys:
            if k in r and r[k] not in ("", None):
                with contextlib.suppress(ValueError):
                    r[k] = float(r[k])
    return rows


# ---------------------------------------------------------------------------
# Figure 1: per-word representation grid
# ---------------------------------------------------------------------------


def fig_per_word_grid(
    t1_dump: dict[str, np.ndarray],
    w2v_dump: dict[str, np.ndarray],
    words: list[str],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    # Filter to words that exist in BOTH embeddings.
    words = [w for w in words if w in t1_dump and w in w2v_dump]
    if not words:
        print("  [fig1] no words in both dumps, skipping")
        return

    n = len(words)
    fig, axes = plt.subplots(n, 2, figsize=(12, 0.8 * n), dpi=120)
    if n == 1:
        axes = axes[None, :]

    # T1: treat each SDR as a 1D strip reshaped to 2D for display.
    t1_dim = next(iter(t1_dump.values())).size
    rows = int(np.ceil(np.sqrt(t1_dim)))
    cols = int(np.ceil(t1_dim / rows))
    t1_grid_shape = (rows, cols)

    # word2vec: render as sorted color bar so the most-activated dims
    # are visible regardless of which dims they are.
    w2v_dim = next(iter(w2v_dump.values())).size

    for i, w in enumerate(words):
        t1_vec = t1_dump[w].astype(int)
        padded = np.zeros(rows * cols, dtype=int)
        padded[:t1_dim] = t1_vec
        axes[i, 0].imshow(
            padded.reshape(t1_grid_shape),
            cmap="Greys",
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_ylabel(w, rotation=0, labelpad=30, va="center", fontsize=10)

        w2v_vec = w2v_dump[w]
        axes[i, 1].imshow(
            w2v_vec.reshape(1, w2v_dim),
            cmap="RdBu_r",
            aspect="auto",
            vmin=-np.abs(w2v_vec).max(),
            vmax=np.abs(w2v_vec).max(),
        )
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

    axes[0, 0].set_title("T1 sparse binary SDR", fontsize=11)
    axes[0, 1].set_title("word2vec dense vector", fontsize=11)
    fig.suptitle("Per-word representations", fontsize=13, y=1.002)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig1] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: pairwise similarity heatmap
# ---------------------------------------------------------------------------


def fig_similarity_heatmap(
    t1_dump: dict[str, np.ndarray],
    w2v_dump: dict[str, np.ndarray],
    words: list[str],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    words = [w for w in words if w in t1_dump and w in w2v_dump]
    n = len(words)
    if n < 2:
        print("  [fig2] not enough shared words, skipping")
        return

    t1_mat = np.zeros((n, n))
    w2v_mat = np.zeros((n, n))
    for i, a in enumerate(words):
        for j, b in enumerate(words):
            t1_mat[i, j] = jaccard_similarity(t1_dump[a], t1_dump[b])
            w2v_mat[i, j] = cosine_similarity(w2v_dump[a], w2v_dump[b])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    im0 = axes[0].imshow(t1_mat, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title(
        f"T1 sparse (Jaccard)\nmean={t1_mat[np.triu_indices(n, k=1)].mean():.3f}"
    )
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(words, rotation=90, fontsize=8)
    axes[0].set_yticklabels(words, fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(w2v_mat, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title(
        f"word2vec dense (cosine)\nmean={w2v_mat[np.triu_indices(n, k=1)].mean():.3f}"
    )
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(words, rotation=90, fontsize=8)
    axes[1].set_yticklabels(words, fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Pairwise similarity: sparse is spread, dense collapses", fontsize=12, y=1.0
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig2] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: corruption robustness curve
# ---------------------------------------------------------------------------


def fig_corruption_curve(
    t1_dump: dict[str, np.ndarray],
    w2v_dump: dict[str, np.ndarray],
    simlex_pairs: list[tuple[str, str, float]],
    out_path: Path,
    *,
    levels: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    min_score: float = 6.0,
    seed: int = 0,
) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    high_pairs = [(a, b) for a, b, s in simlex_pairs if s >= min_score]

    t1_curve: list[float] = []
    w2v_curve: list[float] = []
    for level in levels:
        t1_sims, w2v_sims = [], []
        for a, b in high_pairs:
            va_t1, vb_t1 = t1_dump.get(a), t1_dump.get(b)
            if va_t1 is not None and vb_t1 is not None:
                va_c = _corrupt_sparse(va_t1, level, rng)
                t1_sims.append(jaccard_similarity(va_c, vb_t1))
            va_w, vb_w = w2v_dump.get(a), w2v_dump.get(b)
            if va_w is not None and vb_w is not None:
                va_c = _corrupt_dense(va_w, level, rng)
                w2v_sims.append(cosine_similarity(va_c, vb_w))
        t1_curve.append(float(np.mean(t1_sims)) if t1_sims else 0.0)
        w2v_curve.append(float(np.mean(w2v_sims)) if w2v_sims else 0.0)

    # Normalize to retention (share of clean similarity).
    t1_ret = [s / t1_curve[0] if t1_curve[0] > 0 else 0 for s in t1_curve]
    w2v_ret = [s / w2v_curve[0] if w2v_curve[0] > 0 else 0 for s in w2v_curve]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(levels, t1_ret, "o-", label="T1 sparse (Jaccard)", linewidth=2)
    ax.plot(levels, w2v_ret, "s-", label="word2vec dense (cosine)", linewidth=2)
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("Corruption level (fraction of bits flipped / noise sigma)")
    ax.set_ylabel("Similarity retention (vs. clean)")
    ax.set_title(
        f"Corruption robustness: sparse binary codes preserve similarity "
        f"\n(high-rated SimLex pairs, n={len(high_pairs)})"
    )
    ax.set_ylim(-0.1, 1.2)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig3] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: sample-efficiency / capacity curves
# ---------------------------------------------------------------------------


def fig_eff_dim_curve(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    # Group by (model, n_tokens) → list of eff_dim across seeds.
    grouped: dict[tuple[str, int], list[float]] = {}
    for r in rows:
        key = (r["model"], int(r["n_tokens"]))
        grouped.setdefault(key, []).append(float(r.get("cap_eff_dim", 0.0)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120, sharex=True)

    for model, ax in zip(["t1_sparse", "word2vec"], axes, strict=True):
        xs = sorted({k[1] for k in grouped if k[0] == model})
        means = [np.mean(grouped[(model, x)]) for x in xs]
        stds = [np.std(grouped[(model, x)]) for x in xs]
        ax.errorbar(xs, means, yerr=stds, fmt="o-", linewidth=2, capsize=4)
        ax.set_xscale("log")
        ax.set_xlabel("Training tokens")
        ax.set_ylabel("eff_dim (participation ratio)")
        ax.set_title(f"{model}: representation capacity over data scale")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Capacity over training data — sparse preserves more dimensions", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig4] wrote {out_path}")


def fig_simlex_curve(rows: list[dict], out_path: Path) -> None:
    """Bonus: SimLex learning curve, mean ± std across seeds."""
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, int], list[float]] = {}
    for r in rows:
        key = (r["model"], int(r["n_tokens"]))
        grouped.setdefault(key, []).append(float(r.get("simlex_spearman", 0.0)))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    for model, marker in [("word2vec", "s-"), ("t1_sparse", "o-")]:
        xs = sorted({k[1] for k in grouped if k[0] == model})
        if not xs:
            continue
        means = [np.mean(grouped[(model, x)]) for x in xs]
        stds = [np.std(grouped[(model, x)]) for x in xs]
        ax.errorbar(
            xs, means, yerr=stds, fmt=marker, label=model, linewidth=2, capsize=4
        )
    ax.set_xscale("log")
    ax.axhline(0, linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("SimLex-999 Spearman correlation")
    ax.set_title("SimLex learning curves — dense wins its native benchmark")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig5] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 6: nearest-neighbor retrieval (side-by-side top-k)
# ---------------------------------------------------------------------------


NN_QUERY_WORDS = [
    "king",
    "queen",
    "dog",
    "cat",
    "computer",
    "music",
    "war",
    "light",
    "good",
    "small",
]


def fig_nn_retrieval(
    t1_dump: dict[str, np.ndarray],
    w2v_dump: dict[str, np.ndarray],
    query_words: list[str],
    out_path: Path,
    *,
    k: int = 5,
) -> None:
    """Render top-k nearest neighbors for each query, side by side.

    This is the practical "semantic search" use case: given a word, what
    words is the representation telling us are similar? Shows *what* the
    representations have actually learned, in a form non-technical
    readers can parse.
    """
    import matplotlib.pyplot as plt

    t1_emb = _DictEmbeddings(t1_dump, sparse=True, name="t1_sparse")
    w2v_emb = _DictEmbeddings(w2v_dump, sparse=False, name="word2vec")

    queries = [q for q in query_words if q in t1_dump and q in w2v_dump]
    if not queries:
        print("  [fig6] no query words in both dumps, skipping")
        return

    t1_nn = evaluate_nn_retrieval(t1_emb, queries, k=k)["per_query"]
    w2v_nn = evaluate_nn_retrieval(w2v_emb, queries, k=k)["per_query"]

    n = len(queries)
    fig, ax = plt.subplots(figsize=(12, 0.4 * n + 1.5), dpi=120)
    ax.axis("off")

    header = ["query", f"T1 sparse top-{k}", f"word2vec dense top-{k}"]
    rows = []
    for q in queries:
        t1_list = ", ".join(f"{w} ({s:.2f})" for w, s in t1_nn.get(q, []))
        w2v_list = ", ".join(f"{w} ({s:.2f})" for w, s in w2v_nn.get(q, []))
        rows.append([q, t1_list, w2v_list])

    table = ax.table(
        cellText=rows,
        colLabels=header,
        cellLoc="left",
        colLoc="center",
        loc="center",
        colWidths=[0.08, 0.46, 0.46],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for col in range(3):
        table[(0, col)].set_facecolor("#dddddd")
        table[(0, col)].set_text_props(weight="bold")

    ax.set_title(
        f"Nearest-neighbor retrieval (top-{k} per query)\n"
        "Same task, same vocab — what each representation finds similar",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig6] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 7: storage + retrieval speed summary
# ---------------------------------------------------------------------------


def fig_storage_speed(
    t1_dump: dict[str, np.ndarray],
    w2v_dump: dict[str, np.ndarray],
    query_words: list[str],
    out_path: Path,
) -> dict:
    """Paired bar charts: bytes/embedding and ms/query for sparse vs dense.

    Returns a dict of the raw numbers so compare.py can log them too.
    """
    import matplotlib.pyplot as plt

    t1_emb = _DictEmbeddings(t1_dump, sparse=True, name="t1_sparse")
    w2v_emb = _DictEmbeddings(w2v_dump, sparse=False, name="word2vec")

    t1_store = storage_bytes_per_embedding(t1_emb)
    w2v_store = storage_bytes_per_embedding(w2v_emb)
    # Use every word that exists in both as the query stream — realistic
    # "query the whole vocab" workload. Capped to 500 to keep runtime short.
    shared = [w for w in query_words if w in t1_dump and w in w2v_dump]
    if len(shared) < 10:
        shared = [w for w in t1_dump if w in w2v_dump][:500]
    t1_speed = benchmark_nn_query(t1_emb, shared, trials=3)
    w2v_speed = benchmark_nn_query(w2v_emb, shared, trials=3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)

    # Storage.
    labels = ["T1 sparse\n(active-idx list)", "word2vec dense\n(float32 vector)"]
    bytes_vals = [
        t1_store.get("bytes_per_embedding", 0),
        w2v_store.get("bytes_per_embedding", 0),
    ]
    colors = ["#4c9bd6", "#e36a6a"]
    bars = axes[0].bar(labels, bytes_vals, color=colors)
    axes[0].set_ylabel("Bytes per embedding")
    ratio = bytes_vals[1] / max(bytes_vals[0], 1)
    axes[0].set_title(f"Storage footprint\n(dense uses {ratio:.1f}x more bytes)")
    for bar, v in zip(bars, bytes_vals, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f" {v} B",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[0].grid(axis="y", alpha=0.3)

    # Speed.
    ms_vals = [
        t1_speed.get("mean_ms_per_query", 0.0),
        w2v_speed.get("mean_ms_per_query", 0.0),
    ]
    bars = axes[1].bar(labels, ms_vals, color=colors)
    axes[1].set_ylabel("ms per nearest-neighbor query")
    faster = "sparse" if ms_vals[0] < ms_vals[1] else "dense"
    speed_ratio = max(ms_vals) / max(min(ms_vals), 1e-6)
    axes[1].set_title(
        f"NN lookup speed (vocab={t1_speed.get('vocab_size', '?')})\n"
        f"({faster} {speed_ratio:.1f}x faster)"
    )
    for bar, v in zip(bars, ms_vals, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f" {v:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Practical deployment costs: storage & retrieval speed",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig7] wrote {out_path}")

    return {
        "t1_bytes": bytes_vals[0],
        "w2v_bytes": bytes_vals[1],
        "storage_ratio": ratio,
        "t1_ms": ms_vals[0],
        "w2v_ms": ms_vals[1],
        "speed_ratio": speed_ratio,
        "faster": faster,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Blog-ready viz for ARB-139")
    p.add_argument("--dump-dir", type=Path, required=True)
    p.add_argument("--csv", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/svd_figures"))
    p.add_argument(
        "--n-tokens",
        type=int,
        default=None,
        help="Which training scale to use for per-word figures (dumps at this "
        "token count, seed 0). Defaults to max available.",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Pick the dump files.
    dumps = sorted(args.dump_dir.glob("*.pkl"))
    print(f"Found {len(dumps)} dump files in {args.dump_dir}")
    t1_dumps = {
        int(p.stem.split("_n")[1].split("_")[0]): p
        for p in dumps
        if p.stem.startswith("t1_sparse")
    }
    w2v_dumps = {
        int(p.stem.split("_n")[1].split("_")[0]): p
        for p in dumps
        if p.stem.startswith("word2vec")
    }
    if args.n_tokens:
        target = args.n_tokens
    else:
        common = set(t1_dumps) & set(w2v_dumps)
        target = max(common) if common else None

    rows = load_csv(args.csv)
    print(f"Loaded {len(rows)} CSV rows")

    if target is not None and target in t1_dumps and target in w2v_dumps:
        print(f"Using dumps at n_tokens={target}")
        t1_dump = load_embedding_dump(t1_dumps[target])
        w2v_dump = load_embedding_dump(w2v_dumps[target])
        fig_per_word_grid(
            t1_dump, w2v_dump, DEFAULT_CURATED, args.out_dir / "fig1_per_word.png"
        )
        fig_similarity_heatmap(
            t1_dump, w2v_dump, DEFAULT_CURATED, args.out_dir / "fig2_similarity.png"
        )

        # Need simlex for fig 3.
        from examples.text_exploration.sparse_vs_dense.data import load_simlex

        vocab_set = set(t1_dump) & set(w2v_dump)
        simlex = load_simlex(vocab=vocab_set)
        fig_corruption_curve(
            t1_dump,
            w2v_dump,
            simlex,
            args.out_dir / "fig3_corruption.png",
        )
        fig_nn_retrieval(
            t1_dump,
            w2v_dump,
            NN_QUERY_WORDS,
            args.out_dir / "fig6_nn_retrieval.png",
        )
        speed = fig_storage_speed(
            t1_dump,
            w2v_dump,
            DEFAULT_CURATED,
            args.out_dir / "fig7_storage_speed.png",
        )
        print(
            f"  storage: T1={speed['t1_bytes']}B  w2v={speed['w2v_bytes']}B"
            f"  ({speed['storage_ratio']:.1f}x dense/sparse)"
        )
        print(
            f"  speed:   T1={speed['t1_ms']:.2f}ms  w2v={speed['w2v_ms']:.2f}ms"
            f"  ({speed['faster']} {speed['speed_ratio']:.1f}x faster)"
        )
    else:
        print("Skipping dump-based figures (no matching dumps for target)")

    fig_eff_dim_curve(rows, args.out_dir / "fig4_eff_dim.png")
    fig_simlex_curve(rows, args.out_dir / "fig5_simlex.png")

    print(f"\nWrote figures to {args.out_dir}/")


if __name__ == "__main__":
    main()
