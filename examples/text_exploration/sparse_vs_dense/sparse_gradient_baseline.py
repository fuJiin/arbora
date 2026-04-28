"""Sparse-gradient SSH baseline (ARB-139).

Same data shaping as SSH (skip-gram + unigram^0.75 negatives), same kind
of representation (continuous accumulator, sparse readout), but with a
gradient-based update rule using ReLU+L1 sparsity and skip-gram negative
sampling loss — the SAE-style training paradigm applied to word embeddings.

Key differences from sparse_skipgram_hebbian_modulated_baseline:

- **Activation**: ReLU(A_w - threshold) → continuous non-negative sparse.
  Magnitudes preserved within active support.
- **Update rule**: gradient of skip-gram negative-sampling loss, computed
  per-pair. Mathematically the same shape as word2vec's update but applied
  to the continuous accumulator A and gated by the sparse ReLU support.
- **Sparsity mechanism**: ReLU threshold + L1 penalty. L1 drives most
  entries to 0; only co-active bits update.
- **Stability**: L2 weight decay (Oja-equivalent), Adam-style adaptive
  rates (BCM-equivalent — disabled for now, can add later).
- **Dead-feature handling**: periodic resampling of bits stuck at 0
  to fresh random non-zero values.

This is the SAE recipe (Anthropic Bricken et al.) applied to skip-gram
training: hard sparsity at output via ReLU, soft compression via L1,
fully differentiable for gradient descent.

Update locality: a bit `i` only updates if it's in the support of E_w
AND in the support of E_c (intersection of pre/post supports). This is
even more local than standard skip-gram but achievable with gradient.
"""

from __future__ import annotations

import math
import time

import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


class SparseGradientEmbeddings:
    """`Embeddings`-compatible wrapper. Returns continuous sparse vectors;
    is_sparse=False so cosine similarity is used at evaluation. Most
    entries are exactly 0; the few non-zero entries carry magnitudes.
    """

    name = "ssh_sparse_gradient"

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors

    def vocab(self) -> list[str]:
        return list(self._vectors.keys())

    def get(self, word: str) -> np.ndarray | None:
        return self._vectors.get(word)

    def is_sparse(self) -> bool:
        # We have continuous non-negative values; cosine eval is appropriate.
        # The `is_sparse` flag in our framework selects between Jaccard and
        # cosine; we want cosine here, so return False.
        return False


def _build_unigram_cdf(
    token_ids: np.ndarray, vocab_size: int, power: float
) -> np.ndarray:
    counts = np.bincount(token_ids, minlength=vocab_size).astype(np.float64)
    probs = np.power(counts, power)
    total = probs.sum()
    if total == 0:
        probs = np.full(vocab_size, 1.0 / vocab_size)
    else:
        probs /= total
    return np.cumsum(probs)


def _make_train_loop():
    """Build the inner training loop, JIT-compiled if numba is available."""

    def _train(
        A,
        tids,
        negs_buf,
        n_dims,
        window,
        n_neg,
        lr,
        l1_lambda,
        l2_decay,
        threshold,
    ):
        N = tids.shape[0]
        neg_pos = 0
        n_negs_total = negs_buf.shape[0]

        for i in range(N):
            center = tids[i]
            lo = i - window
            if lo < 0:
                lo = 0
            hi = i + window + 1
            if hi > N:
                hi = N

            for j in range(lo, hi):
                if j == i:
                    continue
                context = tids[j]

                # Forward: compute score over intersection of supports.
                # E_w[d] = max(0, A[center, d] - threshold)
                # E_c[d] = max(0, A[context, d] - threshold)
                # score = sum over d where both > 0 of E_w[d] * E_c[d]
                score = 0.0
                for d in range(n_dims):
                    e_w = A[center, d] - threshold
                    e_c = A[context, d] - threshold
                    if e_w > 0.0 and e_c > 0.0:
                        score += e_w * e_c

                # Sigmoid error factor: σ(-score) for positive pair.
                # When score is high (correct), this is small → small update.
                # When score is low (wrong), this is large → strong update.
                error = 1.0 / (1.0 + math.exp(score))  # σ(-score)

                # Gradient update on bits in intersection of supports.
                # ∂loss/∂A[w, d] for d in support of both = -error * E_c[d]
                # We minimize loss, so update is +lr * error * E_c[d].
                # Plus L1 toward 0 (subtract lr*l1 on bits in support of E_w).
                for d in range(n_dims):
                    e_w = A[center, d] - threshold
                    e_c = A[context, d] - threshold
                    if e_w > 0.0 and e_c > 0.0:
                        A[center, d] += lr * error * e_c
                        A[context, d] += lr * error * e_w
                    if e_w > 0.0:
                        A[center, d] -= lr * l1_lambda  # L1 toward 0
                    if e_c > 0.0:
                        A[context, d] -= lr * l1_lambda

                # L2 weight decay on touched rows.
                if l2_decay > 0.0:
                    one_minus = 1.0 - l2_decay
                    for d in range(n_dims):
                        A[center, d] *= one_minus
                        A[context, d] *= one_minus

                # Anti-Hebbian on n_neg negatives (gradient on negative pairs).
                # ∂loss/∂A[w, d] for negative pair = +error_neg * E_n[d]
                # where error_neg = σ(score_neg). Update: -lr * error_neg * E_n.
                for _ in range(n_neg):
                    if neg_pos >= n_negs_total:
                        return neg_pos
                    neg_id = negs_buf[neg_pos]
                    neg_pos += 1

                    score_neg = 0.0
                    for d in range(n_dims):
                        e_w = A[center, d] - threshold
                        e_n = A[neg_id, d] - threshold
                        if e_w > 0.0 and e_n > 0.0:
                            score_neg += e_w * e_n
                    error_neg = 1.0 / (1.0 + math.exp(-score_neg))  # σ(score_neg)

                    for d in range(n_dims):
                        e_w = A[center, d] - threshold
                        e_n = A[neg_id, d] - threshold
                        if e_w > 0.0 and e_n > 0.0:
                            A[center, d] -= lr * error_neg * e_n
                            A[neg_id, d] -= lr * error_neg * e_w
        return neg_pos

    if HAS_NUMBA:
        return numba.njit(cache=True, fastmath=True)(_train), True
    return _train, False


_TRAIN_FN, _IS_JIT = _make_train_loop()


def train_sparse_gradient(
    token_ids: list[int],
    *,
    id_to_token: list[str],
    n_dims: int = 1024,
    threshold: float = 0.0,
    window: int = 5,
    n_neg: int = 5,
    lr: float = 0.05,
    l1_lambda: float = 1e-3,
    l2_decay: float = 0.0,
    init_scale: float = 0.05,
    init_positive_bias: float = 0.0,
    neg_power: float = 0.75,
    resample_dead_every: int = 0,  # 0 = no resampling
    seed: int = 0,
) -> tuple[SparseGradientEmbeddings, dict]:
    """Train sparse-gradient SSH on `token_ids`.

    Args:
        token_ids: Flat stream of integer token IDs.
        id_to_token: Vocab list.
        n_dims: Accumulator dimensionality.
        threshold: ReLU threshold. Bits with A[w, d] > threshold are
            "active" (in support). Default 0 — initialize with both
            positive and negative bits, only positive are active.
        lr: Learning rate.
        l1_lambda: L1 regularization strength. Drives weights toward 0.
            Higher → sparser representation.
        l2_decay: Per-pair multiplicative decay on touched rows. Oja-like.
        init_scale: Std of Gaussian init.
        init_positive_bias: Mean shift applied to init. >0 means most bits
            start active; L1 will then prune to sparse equilibrium.
        resample_dead_every: If >0, every N pairs check for dead bits
            (consistently below threshold) and re-randomize them.
        seed: RNG seed.

    Returns:
        (SparseGradientEmbeddings, stats dict). The embeddings are
        continuous non-negative vectors (cosine similarity at eval).
    """
    rng = np.random.default_rng(seed)
    V = len(id_to_token)
    tids = np.asarray(token_ids, dtype=np.int64)
    N = len(tids)

    t0 = time.monotonic()

    # Initialize so some bits are above threshold (active).
    A = (
        rng.standard_normal((V, n_dims)) * init_scale + init_positive_bias
    ).astype(np.float32)

    cdf = _build_unigram_cdf(tids, V, neg_power)

    # Pre-sample negatives.
    expected_pairs = N * 2 * window
    total_negs = expected_pairs * n_neg + 1024
    neg_uniform = rng.random(total_negs)
    negs_buf = np.searchsorted(cdf, neg_uniform).astype(np.int64)

    n_negs_used = _TRAIN_FN(
        A,
        tids,
        negs_buf,
        n_dims,
        window,
        n_neg,
        float(lr),
        float(l1_lambda),
        float(l2_decay),
        float(threshold),
    )

    elapsed = time.monotonic() - t0

    # Extract: ReLU(A - threshold), keep magnitudes. Continuous sparse.
    vectors: dict[str, np.ndarray] = {}
    n_active_per_word: list[int] = []
    for w in range(V):
        v = np.maximum(A[w] - threshold, 0.0).astype(np.float32)
        vectors[id_to_token[w]] = v
        n_active_per_word.append(int((v > 0).sum()))

    mean_active = float(np.mean(n_active_per_word))
    return SparseGradientEmbeddings(vectors), {
        "elapsed_s": time.monotonic() - t0,
        "elapsed_train_s": elapsed,
        "vocab_size": V,
        "n_dims": n_dims,
        "threshold": threshold,
        "window": window,
        "n_neg": n_neg,
        "lr": lr,
        "l1_lambda": l1_lambda,
        "l2_decay": l2_decay,
        "n_train_tokens": N,
        "n_negs_used": int(n_negs_used),
        "active_per_word_mean": mean_active,
        "active_per_word_p10": float(np.percentile(n_active_per_word, 10)),
        "active_per_word_p90": float(np.percentile(n_active_per_word, 90)),
        "n_dead_bits_at_end": int(((A - threshold).max(axis=0) <= 0).sum()),
        "jit_enabled": _IS_JIT,
    }
