"""Entmax-1.5 sparse-gradient SSH baseline (ARB-139).

Like sparsemax_baseline but uses **entmax with α=1.5** for the activation.
α-entmax (Peters et al. 2019) is a generalization: α=1 is softmax (dense),
α=2 is sparsemax (very sparse). α=1.5 sits between — sparse but less
aggressive narrowing than sparsemax.

Forward: entmax_α(z)_i = max(0, (α−1)·z_i − τ)^(1/(α−1))
For α=1.5: entmax_1.5(z)_i = max(0, 0.5·z_i − τ)²
τ chosen so output sums to 1 (simplex constraint).

We hypothesized that pure sparsemax over-prunes at scale (collapses to ~2
active bits per word). Entmax-1.5's less-aggressive sparsity should
preserve more representational capacity.

Gradient: we use sparsemax-style approximation (mask through support +
centering by support mean). Exact entmax-1.5 gradient is more complex
(Peters paper Eq. 9) but sparsemax form is a reasonable approximation.

Usage matches sparsemax_baseline; just substitute the activation.
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


class Entmax15Embeddings:
    name = "ssh_entmax15"

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors

    def vocab(self) -> list[str]:
        return list(self._vectors.keys())

    def get(self, word: str) -> np.ndarray | None:
        return self._vectors.get(word)

    def is_sparse(self) -> bool:
        return False  # continuous values; cosine eval


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


def _entmax15_into_py(z, out, sort_buf, max_iter=30):
    """Compute entmax-1.5(z), writing result into `out`. Uses bisection
    on τ to satisfy the simplex constraint sum(out) = 1.

    Forward formula: out_i = max(0, 0.5·z_i − τ)²
    Bisection bounds:
      - τ_low = 0.5·max(z) − 1.0 (gives single-element support)
      - τ_high = 0.5·max(z) (gives empty support)

    Returns the support size for diagnostics.
    """
    n = z.shape[0]

    # Find max(z) for bisection bounds.
    max_z = z[0]
    for i in range(1, n):
        if z[i] > max_z:
            max_z = z[i]

    tau_low = 0.5 * max_z - 1.0
    tau_high = 0.5 * max_z

    # Bisection on τ.
    for _ in range(max_iter):
        tau = (tau_low + tau_high) * 0.5
        total = 0.0
        for i in range(n):
            x = 0.5 * z[i] - tau
            if x > 0.0:
                total += x * x
        if total > 1.0:
            tau_low = tau
        else:
            tau_high = tau

    # Final apply.
    tau = (tau_low + tau_high) * 0.5
    support_size = 0
    for i in range(n):
        x = 0.5 * z[i] - tau
        if x > 0.0:
            out[i] = x * x
            support_size += 1
        else:
            out[i] = 0.0
    return support_size


if HAS_NUMBA:
    _entmax15_into = numba.njit(cache=True, fastmath=True)(_entmax15_into_py)
else:
    _entmax15_into = _entmax15_into_py


def _make_train_loop():
    """Build the inner training loop, JIT-compiled if numba is available.

    Gradient approximation: same form as sparsemax (mask through support
    + centering by support mean). Exact entmax-1.5 gradient is more
    complex but this approximation should still drive learning.
    """

    def _train(
        A,
        tids,
        negs_buf,
        n_dims,
        window,
        n_neg,
        lr,
        l2_decay,
        e_w_buf,
        e_c_buf,
        e_n_buf,
        sort_buf,
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

                _entmax15_into(A[center], e_w_buf, sort_buf)
                _entmax15_into(A[context], e_c_buf, sort_buf)

                # Score = E_w · E_c
                score = 0.0
                for d in range(n_dims):
                    score += e_w_buf[d] * e_c_buf[d]

                error = 1.0 / (1.0 + math.exp(score))  # σ(-score)

                # Gradient on A_w (sparsemax-style approximation).
                support_size_w = 0
                sum_c_in_support_w = 0.0
                for d in range(n_dims):
                    if e_w_buf[d] > 0.0:
                        support_size_w += 1
                        sum_c_in_support_w += e_c_buf[d]
                if support_size_w > 0:
                    mean_c = sum_c_in_support_w / support_size_w
                    for d in range(n_dims):
                        if e_w_buf[d] > 0.0:
                            A[center, d] += lr * error * (e_c_buf[d] - mean_c)

                # Gradient on A_c.
                support_size_c = 0
                sum_w_in_support_c = 0.0
                for d in range(n_dims):
                    if e_c_buf[d] > 0.0:
                        support_size_c += 1
                        sum_w_in_support_c += e_w_buf[d]
                if support_size_c > 0:
                    mean_w = sum_w_in_support_c / support_size_c
                    for d in range(n_dims):
                        if e_c_buf[d] > 0.0:
                            A[context, d] += lr * error * (e_w_buf[d] - mean_w)

                if l2_decay > 0.0:
                    one_minus = 1.0 - l2_decay
                    for d in range(n_dims):
                        A[center, d] *= one_minus
                        A[context, d] *= one_minus

                # Anti-Hebbian on negatives.
                for _ in range(n_neg):
                    if neg_pos >= n_negs_total:
                        return neg_pos
                    neg_id = negs_buf[neg_pos]
                    neg_pos += 1

                    _entmax15_into(A[neg_id], e_n_buf, sort_buf)

                    score_neg = 0.0
                    for d in range(n_dims):
                        score_neg += e_w_buf[d] * e_n_buf[d]

                    error_neg = 1.0 / (1.0 + math.exp(-score_neg))

                    support_size = 0
                    sum_n = 0.0
                    for d in range(n_dims):
                        if e_w_buf[d] > 0.0:
                            support_size += 1
                            sum_n += e_n_buf[d]
                    if support_size > 0:
                        mean_n = sum_n / support_size
                        for d in range(n_dims):
                            if e_w_buf[d] > 0.0:
                                A[center, d] -= lr * error_neg * (e_n_buf[d] - mean_n)

                    support_size_n = 0
                    sum_w_n = 0.0
                    for d in range(n_dims):
                        if e_n_buf[d] > 0.0:
                            support_size_n += 1
                            sum_w_n += e_w_buf[d]
                    if support_size_n > 0:
                        mean_w_at_n = sum_w_n / support_size_n
                        for d in range(n_dims):
                            if e_n_buf[d] > 0.0:
                                A[neg_id, d] -= lr * error_neg * (e_w_buf[d] - mean_w_at_n)
        return neg_pos

    if HAS_NUMBA:
        return numba.njit(cache=True, fastmath=True)(_train), True
    return _train, False


_TRAIN_FN, _IS_JIT = _make_train_loop()


def train_entmax15_ssh(
    token_ids: list[int],
    *,
    id_to_token: list[str],
    n_dims: int = 1024,
    window: int = 5,
    n_neg: int = 5,
    lr: float = 1.0,
    l2_decay: float = 0.0,
    init_scale: float = 0.001,
    neg_power: float = 0.75,
    seed: int = 0,
) -> tuple[Entmax15Embeddings, dict]:
    """Train entmax-1.5 sparse-gradient SSH on `token_ids`."""
    rng = np.random.default_rng(seed)
    V = len(id_to_token)
    tids = np.asarray(token_ids, dtype=np.int64)
    N = len(tids)

    t0 = time.monotonic()

    A = (rng.standard_normal((V, n_dims)) * init_scale).astype(np.float32)

    cdf = _build_unigram_cdf(tids, V, neg_power)

    expected_pairs = N * 2 * window
    total_negs = expected_pairs * n_neg + 1024
    neg_uniform = rng.random(total_negs)
    negs_buf = np.searchsorted(cdf, neg_uniform).astype(np.int64)

    e_w_buf = np.empty(n_dims, dtype=np.float32)
    e_c_buf = np.empty(n_dims, dtype=np.float32)
    e_n_buf = np.empty(n_dims, dtype=np.float32)
    sort_buf = np.empty(n_dims, dtype=np.float32)

    n_negs_used = _TRAIN_FN(
        A, tids, negs_buf,
        n_dims, window, n_neg,
        float(lr), float(l2_decay),
        e_w_buf, e_c_buf, e_n_buf, sort_buf,
    )

    elapsed = time.monotonic() - t0

    # Extract entmax-1.5 of each row.
    vectors: dict[str, np.ndarray] = {}
    n_active_per_word: list[int] = []
    for w in range(V):
        out = np.empty(n_dims, dtype=np.float32)
        sb = np.empty(n_dims, dtype=np.float32)
        _entmax15_into_py(A[w].astype(np.float32), out, sb)
        vectors[id_to_token[w]] = out
        n_active_per_word.append(int((out > 0).sum()))

    mean_active = float(np.mean(n_active_per_word))
    return Entmax15Embeddings(vectors), {
        "elapsed_s": time.monotonic() - t0,
        "elapsed_train_s": elapsed,
        "vocab_size": V,
        "n_dims": n_dims,
        "window": window,
        "n_neg": n_neg,
        "lr": lr,
        "l2_decay": l2_decay,
        "n_train_tokens": N,
        "n_negs_used": int(n_negs_used),
        "active_per_word_mean": mean_active,
        "active_per_word_p10": float(np.percentile(n_active_per_word, 10)),
        "active_per_word_p90": float(np.percentile(n_active_per_word, 90)),
        "jit_enabled": _IS_JIT,
    }
