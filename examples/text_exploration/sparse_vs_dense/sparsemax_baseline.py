"""Sparsemax sparse-gradient SSH baseline (ARB-139).

Like sparse_gradient_baseline.py but uses **sparsemax** instead of
ReLU+L1 for the sparse activation. Sparsemax (Martins & Astudillo 2016):

  sparsemax(z)_i = max(0, z_i - τ(z))

where τ(z) is chosen so that the output sums to 1 (projection onto
the probability simplex).

Key advantages over ReLU+L1:

- **Auto-sparsity**: number of active bits is determined by the input,
  no hyperparameter to tune.
- **No dead features**: a bit can re-enter the support when its value
  rises above τ (or when other values fall). Unlike ReLU+L1, the
  threshold τ is dynamic — bits aren't permanently lost.
- **Native gradient**: ∂sparsemax(z)_i/∂z_j = δ_ij - 1/|S| if both
  i,j in support S, 0 otherwise. The "centering by support mean" gives
  natural simplex-respecting gradient flow.

Algorithm (per row):

  1. Sort z_(1) ≥ z_(2) ≥ ... ≥ z_(D) descending
  2. Find ρ = max{i : 1 + i·z_(i) > Σ_{j=1}^i z_(j)}
  3. τ = (Σ_{j=1}^ρ z_(j) − 1) / ρ
  4. p_i = max(z_i − τ, 0)

Cost: O(D log D) per row vs O(D) for ReLU. Slower per pair, but the
auto-sparsity and no-dead-features properties may be worth it.

Update: gradient on A[w, i] for i in support of E_w:
    grad_i = error_factor · (E_c[i] - mean(E_c[support_w]))

The "−mean" term comes from the simplex constraint and naturally
keeps updates zero-sum across support.
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


class SparsemaxEmbeddings:
    """Embeddings wrapper for sparsemax-based sparse-gradient SSH.

    Returns continuous non-negative vectors that sum to 1 per row;
    use cosine similarity at evaluation (is_sparse=False).
    """

    name = "ssh_sparsemax"

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors

    def vocab(self) -> list[str]:
        return list(self._vectors.keys())

    def get(self, word: str) -> np.ndarray | None:
        return self._vectors.get(word)

    def is_sparse(self) -> bool:
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


def _sparsemax_into_py(z, out, sort_buf):
    """Compute sparsemax(z), writing result into `out`. Uses `sort_buf`
    (length n) as scratch.

    Module-level so numba can JIT it independently and call it from the
    training loop. Returns the support size ρ for diagnostics.
    """
    n = z.shape[0]
    for i in range(n):
        sort_buf[i] = z[i]
    sort_buf_sorted = np.sort(sort_buf)[::-1]

    cumsum = 0.0
    rho = 1
    for i in range(n):
        cumsum += sort_buf_sorted[i]
        if 1.0 + (i + 1) * sort_buf_sorted[i] > cumsum:
            rho = i + 1

    cumsum_rho = 0.0
    for i in range(rho):
        cumsum_rho += sort_buf_sorted[i]
    tau = (cumsum_rho - 1.0) / rho

    for i in range(n):
        v = z[i] - tau
        if v > 0.0:
            out[i] = v
        else:
            out[i] = 0.0
    return rho


if HAS_NUMBA:
    _sparsemax_into = numba.njit(cache=True, fastmath=True)(_sparsemax_into_py)
else:
    _sparsemax_into = _sparsemax_into_py


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

                # Compute sparsemax for both rows.
                _sparsemax_into(A[center], e_w_buf, sort_buf)
                _sparsemax_into(A[context], e_c_buf, sort_buf)

                # Forward: score = E_w · E_c
                score = 0.0
                for d in range(n_dims):
                    score += e_w_buf[d] * e_c_buf[d]

                # Positive-pair loss gradient:
                # L = -log σ(score)
                # ∂L/∂score = -σ(-score)
                # error = σ(-score)  (positive scalar)
                # Update minimizes loss, so direction is +score
                error = 1.0 / (1.0 + math.exp(score))

                # Gradient on A_w[i] for i in support of E_w:
                #   grad_i = error * (E_c[i] - mean_E_c_over_support_w)
                # Compute mean of E_c over support of w.
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

                # Symmetric: gradient on A_c[i] for i in support of E_c
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

                # L2 weight decay on touched rows.
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

                    _sparsemax_into(A[neg_id], e_n_buf, sort_buf)

                    score_neg = 0.0
                    for d in range(n_dims):
                        score_neg += e_w_buf[d] * e_n_buf[d]

                    # error_neg = σ(score_neg) — large when score is wrongly high
                    error_neg = 1.0 / (1.0 + math.exp(-score_neg))

                    # Gradient on A_w (push away from E_n)
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

                    # And symmetric: gradient on A_n (push away from E_w)
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


def train_sparsemax_ssh(
    token_ids: list[int],
    *,
    id_to_token: list[str],
    n_dims: int = 1024,
    window: int = 5,
    n_neg: int = 5,
    lr: float = 0.05,
    l2_decay: float = 0.0,
    init_scale: float = 0.01,
    neg_power: float = 0.75,
    seed: int = 0,
) -> tuple[SparsemaxEmbeddings, dict]:
    """Train sparsemax sparse-gradient SSH on `token_ids`.

    Args:
        token_ids: Flat stream of integer token IDs.
        id_to_token: Vocab list.
        n_dims: Accumulator dimensionality.
        lr: Gradient learning rate.
        l2_decay: Per-pair multiplicative weight decay (Oja-equivalent).
        init_scale: Std of Gaussian init.
        seed: RNG seed.

    Returns:
        (SparsemaxEmbeddings, stats dict).
    """
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

    # Scratch buffers (reused across pairs).
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

    # Extract: sparsemax of each row.
    vectors: dict[str, np.ndarray] = {}
    n_active_per_word: list[int] = []
    for w in range(V):
        out = np.empty(n_dims, dtype=np.float32)
        sb = np.empty(n_dims, dtype=np.float32)
        if HAS_NUMBA:
            # Use the JIT'd inner sparsemax via the inner-loop function's
            # helper. Since _sparsemax_into is closed over, we re-implement
            # the projection at python level here for extraction.
            pass
        # Pure-python sparsemax for extraction:
        z = A[w].copy()
        z_sorted = np.sort(z)[::-1]
        cumsum = np.cumsum(z_sorted)
        idx = np.arange(1, n_dims + 1)
        cond = 1.0 + idx * z_sorted > cumsum
        valid = np.where(cond)[0]
        rho = int(valid.max() + 1) if len(valid) > 0 else 1
        tau = (cumsum[rho - 1] - 1.0) / rho
        out = np.maximum(z - tau, 0.0).astype(np.float32)
        vectors[id_to_token[w]] = out
        n_active_per_word.append(int((out > 0).sum()))

    mean_active = float(np.mean(n_active_per_word))
    return SparsemaxEmbeddings(vectors), {
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
