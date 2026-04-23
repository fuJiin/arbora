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
