"""Evaluation primitives for ARB-139.

Two benchmarks: SimLex-999 (similarity correlation) and Google analogy
(top-1 vector arithmetic). Operates on a generic `Embeddings` interface
so word2vec (dense) and T1 (sparse) plug in the same way.

Similarity:
- Dense → cosine.
- Sparse → Jaccard. (Both are bounded in [0, 1] for our purposes.)

Analogy:
- Dense → 3CosAdd: argmax over vocab of cos(d, b - a + c).
- Sparse → multiple options; we use a "set arithmetic" version: target
  = (b OR c) \\ a, then argmax over vocab by Jaccard with target. This
  isn't perfectly principled (XOR-style might also work), but it's the
  most common sparse analogy formulation in the SDR literature.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

# ---------------------------------------------------------------------------
# Embedding interface
# ---------------------------------------------------------------------------


class Embeddings(Protocol):
    """Minimal interface both dense (word2vec) and sparse (T1) implement."""

    name: str

    def vocab(self) -> list[str]:
        """All known words."""

    def get(self, word: str) -> np.ndarray | None:
        """Vector for `word` (dense) or boolean SDR (sparse). None if OOV."""

    def is_sparse(self) -> bool:
        """True for boolean SDRs, False for dense float vectors."""


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    if union == 0:
        return 0.0
    return float((a & b).sum()) / union


def similarity(a: np.ndarray, b: np.ndarray, *, sparse: bool) -> float:
    return jaccard_similarity(a, b) if sparse else cosine_similarity(a, b)


# ---------------------------------------------------------------------------
# SimLex-999 evaluation
# ---------------------------------------------------------------------------


def evaluate_simlex(
    emb: Embeddings,
    pairs: list[tuple[str, str, float]],
) -> dict:
    """Spearman correlation between predicted and human similarity.

    Returns {n_pairs, spearman, pearson}. `n_pairs` may be less than
    `len(pairs)` if any words are OOV in `emb` — those pairs are
    silently dropped.
    """
    from scipy.stats import pearsonr, spearmanr

    sparse = emb.is_sparse()
    pred: list[float] = []
    human: list[float] = []
    for a, b, score in pairs:
        va, vb = emb.get(a), emb.get(b)
        if va is None or vb is None:
            continue
        pred.append(similarity(va, vb, sparse=sparse))
        human.append(score)
    if len(pred) < 2:
        return {"n_pairs": len(pred), "spearman": 0.0, "pearson": 0.0}
    rho, _ = spearmanr(pred, human)
    r, _ = pearsonr(pred, human)
    return {
        "n_pairs": len(pred),
        "spearman": float(rho),
        "pearson": float(r),
    }


# ---------------------------------------------------------------------------
# Sparse-native metrics: capacity + pattern separation
# ---------------------------------------------------------------------------


def evaluate_capacity(emb: Embeddings, *, sample: int = 500, seed: int = 0) -> dict:
    """Sparse-native capacity / collision metrics.

    For binary SDRs: how well does the embedding use its representational
    capacity? Three signals:

    - `mean_pairwise_sim` — average similarity across all distinct pairs.
      Should be low (around 0.05-0.20 for well-separated codes). High →
      codes collapsing toward each other.
    - `high_collision_frac` — fraction of pairs with similarity > 0.8.
      Proxy for capacity exhaustion / pattern collapse.
    - `eff_dim` — participation ratio of the vocab-SDR matrix.
      Larger = the embedding population spans more independent
      directions = less collapsed.

    Dense embeddings also get these (with cosine instead of Jaccard),
    so the same metrics report across both architectures. That gives us
    a common readout for whether representations are stable or drifting.
    """
    vocab = emb.vocab()
    rng = np.random.default_rng(seed)
    if len(vocab) > sample:
        idx = rng.choice(len(vocab), size=sample, replace=False)
        vocab = [vocab[i] for i in idx]
    vecs: list[np.ndarray] = []
    for w in vocab:
        v = emb.get(w)
        if v is not None:
            vecs.append(v)
    if len(vecs) < 2:
        return {
            "n_words": len(vecs),
            "mean_pairwise_sim": 0.0,
            "std_pairwise_sim": 0.0,
            "high_collision_frac": 0.0,
            "eff_dim": 0.0,
        }

    sparse = emb.is_sparse()
    if sparse:
        X = np.stack(vecs).astype(bool)
        inter = (X[:, None, :] & X[None, :, :]).sum(axis=-1)
        union = (X[:, None, :] | X[None, :, :]).sum(axis=-1)
        sim = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    else:
        X = np.stack(vecs)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = X / np.where(norms > 0, norms, 1)
        sim = Xn @ Xn.T

    n = sim.shape[0]
    tri = np.triu_indices(n, k=1)
    offdiag = sim[tri]

    X_float = X.astype(np.float64) if sparse else X
    X_centered = X_float - X_float.mean(axis=0)
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    lambdas = s**2 / max(X_centered.shape[0] - 1, 1)
    sum_l = float(lambdas.sum())
    sum_l2 = float((lambdas**2).sum())
    eff_dim = sum_l**2 / sum_l2 if sum_l2 > 1e-12 else 0.0

    return {
        "n_words": n,
        "mean_pairwise_sim": float(offdiag.mean()),
        "std_pairwise_sim": float(offdiag.std()),
        "high_collision_frac": float((offdiag > 0.8).mean()),
        "eff_dim": float(eff_dim),
    }


def _corrupt_sparse(v: np.ndarray, corruption: float, rng) -> np.ndarray:
    """Flip `corruption` fraction of active bits off + same number of
    inactive bits on, preserving sparsity."""
    active = np.flatnonzero(v)
    inactive = np.flatnonzero(~v)
    n_flip = max(1, int(len(active) * corruption))
    if n_flip > len(active) or n_flip > len(inactive):
        return v.copy()
    flipped_off = rng.choice(active, size=n_flip, replace=False)
    flipped_on = rng.choice(inactive, size=n_flip, replace=False)
    out = v.copy()
    out[flipped_off] = False
    out[flipped_on] = True
    return out


def _corrupt_dense(v: np.ndarray, corruption: float, rng) -> np.ndarray:
    """Add Gaussian noise proportional to vector std.

    Parametrized so `corruption=0.3` means noise sigma is 30% of the vector's
    own std. Direct analog of "30% of active bits flipped" for dense
    vectors.
    """
    sigma = corruption * float(v.std()) if v.std() > 0 else corruption
    return v + rng.normal(0.0, sigma, size=v.shape).astype(v.dtype)


def evaluate_corruption_robustness(
    emb: Embeddings,
    pairs: list[tuple[str, str, float]],
    *,
    corruption_levels: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    seed: int = 0,
    min_score: float = 6.0,
) -> dict:
    """Corruption robustness curve on high-similarity SimLex pairs.

    Works for both sparse and dense:
    - Sparse: flip `level` fraction of active bits off and same number
      of inactive bits on (preserves sparsity).
    - Dense: add Gaussian noise with sigma = level * vector.std().

    For each level, computes mean similarity between corrupted `a`
    and clean `b` across high-rated pairs. Reports raw means and
    "retention" (corrupted_mean / clean_mean) — closer to 1 = robust.

    Returns {corruption_levels, mean_sims, retentions, n_pairs}.
    """
    sparse = emb.is_sparse()
    rng = np.random.default_rng(seed)
    high_pairs = [(a, b) for a, b, s in pairs if s >= min_score]

    pair_vecs: list[tuple[np.ndarray, np.ndarray]] = []
    for a, b in high_pairs:
        va, vb = emb.get(a), emb.get(b)
        if va is not None and vb is not None:
            pair_vecs.append((va, vb))

    levels = list(corruption_levels)
    mean_sims: list[float] = []
    for level in levels:
        sims: list[float] = []
        for va, vb in pair_vecs:
            va_corrupt = (
                _corrupt_sparse(va, level, rng)
                if sparse
                else _corrupt_dense(va, level, rng)
            )
            sims.append(similarity(va_corrupt, vb, sparse=sparse))
        mean_sims.append(float(np.mean(sims)) if sims else 0.0)

    clean_mean = mean_sims[0] if mean_sims else 0.0
    retentions = [(m / clean_mean) if clean_mean > 0 else 0.0 for m in mean_sims]
    return {
        "corruption_levels": levels,
        "mean_sims": mean_sims,
        "retentions": retentions,
        "n_pairs": len(pair_vecs),
    }


def evaluate_nn_retrieval(
    emb: Embeddings,
    query_words: list[str],
    *,
    k: int = 5,
) -> dict:
    """Top-k nearest neighbors for each query word.

    Practical word2vec use case: "find items similar to this one"
    (recommendation, semantic search, suggestion). Uses the native
    similarity of each embedding (Jaccard for sparse, cosine for
    dense) — so each architecture is evaluated on its own terms.

    Returns:
      {
        "per_query": {query_word: [(neighbor_word, similarity), ...]},
        "n_queries": int (how many had matches),
      }

    Queries not present in `emb.vocab()` are silently skipped.
    """
    sparse = emb.is_sparse()
    vocab = [w for w in emb.vocab() if emb.get(w) is not None]
    if not vocab:
        return {"per_query": {}, "n_queries": 0}
    vec_stack = np.stack([emb.get(w) for w in vocab])
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    if not sparse:
        stack_norms = np.linalg.norm(vec_stack, axis=1)

    results: dict[str, list[tuple[str, float]]] = {}
    for q in query_words:
        if q not in word_to_idx:
            continue
        q_vec = vec_stack[word_to_idx[q]]
        if sparse:
            inter = (vec_stack & q_vec).sum(axis=1)
            union = (vec_stack | q_vec).sum(axis=1)
            sims = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        else:
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm == 0:
                continue
            sims = np.zeros(vec_stack.shape[0])
            mask = stack_norms > 0
            sims[mask] = vec_stack[mask] @ q_vec / (stack_norms[mask] * q_norm)

        sims[word_to_idx[q]] = -np.inf  # exclude self
        top_idx = sims.argsort()[-k:][::-1]
        results[q] = [(vocab[i], float(sims[i])) for i in top_idx]

    return {"per_query": results, "n_queries": len(results)}


def storage_bytes_per_embedding(emb: Embeddings) -> dict:
    """Practical storage cost per embedding.

    Sparse: two reasonable encodings — packed bits (N bits → N/8 bytes)
    or active-index list (k active x 2 bytes per int16 index). We
    report both. Dense: dimensionality x 4 bytes (float32).

    Also reports compression ratio relative to the dense equivalent.
    """
    vocab = emb.vocab()
    if not vocab:
        return {"bytes_per_embedding": 0}
    sample = emb.get(vocab[0])
    if sample is None:
        return {"bytes_per_embedding": 0}

    if emb.is_sparse():
        n_bits = sample.size
        mean_active = float(
            np.mean(
                [int(emb.get(w).sum()) for w in vocab[:500] if emb.get(w) is not None]
            )
        )
        packed_bytes = (n_bits + 7) // 8
        index_bytes = int(mean_active) * 2  # int16 indices
        return {
            "sparse": True,
            "n_bits": n_bits,
            "mean_active": mean_active,
            "packed_bytes": packed_bytes,
            "index_bytes": index_bytes,
            "bytes_per_embedding": index_bytes,  # the usual storage form
        }
    n_dims = sample.size
    return {
        "sparse": False,
        "n_dims": n_dims,
        "dtype": str(sample.dtype),
        "bytes_per_embedding": n_dims * sample.dtype.itemsize,
    }


def benchmark_nn_query(
    emb: Embeddings,
    query_words: list[str],
    *,
    trials: int = 5,
) -> dict:
    """Wall-clock per-query NN lookup cost.

    Measures the actual lookup time including argsort — numbers scale
    with vocab size. Reports mean + std across trials.
    """
    import time

    sparse = emb.is_sparse()
    vocab = [w for w in emb.vocab() if emb.get(w) is not None]
    if not vocab:
        return {"mean_ms_per_query": 0.0}
    vec_stack = np.stack([emb.get(w) for w in vocab])
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    if not sparse:
        stack_norms = np.linalg.norm(vec_stack, axis=1)

    times: list[float] = []
    for _ in range(trials):
        t0 = time.monotonic()
        for q in query_words:
            if q not in word_to_idx:
                continue
            q_vec = vec_stack[word_to_idx[q]]
            if sparse:
                inter = (vec_stack & q_vec).sum(axis=1)
                union = (vec_stack | q_vec).sum(axis=1)
                sims = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
            else:
                q_norm = float(np.linalg.norm(q_vec))
                if q_norm == 0:
                    continue
                sims = np.zeros(vec_stack.shape[0])
                mask = stack_norms > 0
                sims[mask] = vec_stack[mask] @ q_vec / (stack_norms[mask] * q_norm)
            _ = sims.argsort()[-5:]
        times.append(time.monotonic() - t0)

    n_real_queries = sum(1 for q in query_words if q in word_to_idx)
    mean_s = float(np.mean(times))
    return {
        "vocab_size": len(vocab),
        "n_queries": n_real_queries,
        "total_s_mean": mean_s,
        "mean_ms_per_query": 1000.0 * mean_s / max(n_real_queries, 1),
    }


def evaluate_partial_cue(
    emb: Embeddings,
    pairs: list[tuple[str, str, float]],
    *,
    corruption: float = 0.3,
    seed: int = 0,
) -> dict:
    """Sparse-only: scalar partial-cue retention at a single corruption
    level. Kept for the existing single-number metric in `compare.py`;
    see `evaluate_corruption_robustness` for the full curve.
    """
    if not emb.is_sparse():
        return {"n": 0, "retention": 0.0, "note": "dense emb skipped"}

    rng = np.random.default_rng(seed)
    high_pairs = [(a, b) for a, b, s in pairs if s >= 6.0]
    clean_sims: list[float] = []
    corrupted_sims: list[float] = []
    for a, b in high_pairs:
        va, vb = emb.get(a), emb.get(b)
        if va is None or vb is None:
            continue
        clean_sims.append(jaccard_similarity(va, vb))
        va_corrupt = _corrupt_sparse(va, corruption, rng)
        corrupted_sims.append(jaccard_similarity(va_corrupt, vb))

    if len(clean_sims) < 2:
        return {"n": len(clean_sims), "retention": 0.0}
    clean_arr = np.array(clean_sims)
    corrupt_arr = np.array(corrupted_sims)
    if clean_arr.std() == 0:
        retention = 0.0
    else:
        retention = float(np.corrcoef(clean_arr, corrupt_arr)[0, 1])
    return {
        "n": len(clean_sims),
        "retention": retention,
        "mean_clean_sim": float(clean_arr.mean()),
        "mean_corrupt_sim": float(corrupt_arr.mean()),
    }


# ---------------------------------------------------------------------------
# Google analogy evaluation
# ---------------------------------------------------------------------------


def _stack_vocab(emb: Embeddings) -> tuple[list[str], np.ndarray]:
    """Stack all vocab vectors into a 2D array for batched argmax."""
    vocab = [w for w in emb.vocab() if emb.get(w) is not None]
    vecs = [emb.get(w) for w in vocab]
    return vocab, np.stack(vecs)


def _analogy_target_dense(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """3CosAdd-style target: b - a + c."""
    return b - a + c


def _analogy_target_sparse(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Sparse analogy: (b OR c) \\ a.

    Captures "the parts of b and c that aren't in a." For a clean
    analogy like king:man :: queen:woman, this should produce the
    woman-typical bits of queen plus the queen-typical bits of woman.
    Falls back gracefully when any input is empty.
    """
    return (b | c) & ~a


def evaluate_analogy(
    emb: Embeddings,
    entries: list[tuple[str, tuple[str, str, str, str]]],
    *,
    top_k: int = 1,
) -> dict:
    """Top-1 analogy accuracy. Excludes a, b, c from the candidate
    pool (standard 3CosAdd convention)."""
    sparse = emb.is_sparse()
    vocab_words, vocab_vecs = _stack_vocab(emb)
    word_to_idx = {w: i for i, w in enumerate(vocab_words)}

    if not vocab_words:
        return {"n_entries": 0, "top1": 0.0, "per_category": {}}

    n_correct = 0
    n_total = 0
    per_category: dict[str, dict[str, int]] = {}

    for category, (a, b, c, d) in entries:
        # Need all four in our vocab.
        if not all(w in word_to_idx for w in (a, b, c, d)):
            continue
        va = vocab_vecs[word_to_idx[a]]
        vb = vocab_vecs[word_to_idx[b]]
        vc = vocab_vecs[word_to_idx[c]]

        target = (
            _analogy_target_sparse(va, vb, vc)
            if sparse
            else _analogy_target_dense(va, vb, vc)
        )

        if sparse:
            target_bool = target.astype(bool)
            # Vectorized Jaccard between target and all vocab vectors.
            inter = (vocab_vecs & target_bool).sum(axis=1)
            union = (vocab_vecs | target_bool).sum(axis=1)
            sims = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        else:
            # Cosine: normalize then dot.
            tn = np.linalg.norm(target)
            if tn == 0:
                continue
            target_norm = target / tn
            vocab_norms = np.linalg.norm(vocab_vecs, axis=1)
            mask = vocab_norms > 0
            sims = np.zeros(vocab_vecs.shape[0])
            sims[mask] = vocab_vecs[mask] @ target_norm / vocab_norms[mask]

        # Mask out a, b, c from candidates (3CosAdd convention).
        for excl in (a, b, c):
            sims[word_to_idx[excl]] = -np.inf

        top_idx = sims.argsort()[-top_k:][::-1]
        if word_to_idx[d] in top_idx:
            n_correct += 1
        n_total += 1
        cat_stats = per_category.setdefault(category, {"correct": 0, "total": 0})
        cat_stats["total"] += 1
        if word_to_idx[d] in top_idx:
            cat_stats["correct"] += 1

    return {
        "n_entries": n_total,
        f"top{top_k}": n_correct / max(n_total, 1),
        "per_category": {
            cat: {
                "n": s["total"],
                "acc": s["correct"] / max(s["total"], 1),
            }
            for cat, s in per_category.items()
        },
    }
