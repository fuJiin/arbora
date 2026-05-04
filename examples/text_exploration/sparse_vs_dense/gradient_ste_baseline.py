"""Gradient-via-STE on top-k binary SSH baseline (ARB-139).

Same representation as existing SSH (continuous accumulator → top-k binary
via top-k operation), same eval (Jaccard on binary). DIFFERENT update rule:
gradient of skip-gram negative-sampling loss, computed using a
straight-through estimator on the non-differentiable top-k operation.

Forward:
  E_w = top_k(A_w)  binary, k active bits
  score = E_w · E_c (bit overlap)
  L_pos = -log σ(score), L_neg = -log σ(-score)

Backward (STE):
  ∂E_w/∂A_w = 1 (identity, STE)
  ∂L_pos/∂A_w[i] = -σ(-score) · E_c[i]  (non-zero only where E_c is active)
  Update: A_w[i] += lr · σ(-score) · E_c[i]  (i.e., gradient descent)

Compare to existing modulated SSH:
  A_w[i] += lr · (1 - overlap/k) · 1  (Hebbian, linear modulator)

Both update only on bits in support of E_c. Difference: the modulator
shape (sigmoid σ(-score) vs linear (1-overlap/k)).

For shallow models, gradient descent and Hebbian-with-right-modulator
are equivalent. This experiment isolates the effect of the modulator
shape — if results differ from existing SSH, the modulator shape matters.

Single table (no separate context vector). Sigmoid bounding NOT applied
to keep this clean — pure STE gradient.
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


class GradientSTEEmbeddings:
    name = "ssh_gradient_ste"

    def __init__(self, sdrs: dict[str, np.ndarray]) -> None:
        self._sdrs = sdrs

    def vocab(self) -> list[str]:
        return list(self._sdrs.keys())

    def get(self, word: str) -> np.ndarray | None:
        return self._sdrs.get(word)

    def is_sparse(self) -> bool:
        return True


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


def _top_k_into_py(row, k, out):
    """Top-k indices by value (returns indices, not boolean mask).

    Same algorithm as modulated baseline — manual scan for numba friendliness.
    """
    n = row.shape[0]
    for i in range(k):
        out[i] = i
    min_pos = 0
    min_val = row[out[0]]
    for i in range(1, k):
        v = row[out[i]]
        if v < min_val:
            min_val = v
            min_pos = i
    for i in range(k, n):
        v = row[i]
        if v > min_val:
            out[min_pos] = i
            min_val = row[out[0]]
            min_pos = 0
            for j in range(1, k):
                vj = row[out[j]]
                if vj < min_val:
                    min_val = vj
                    min_pos = j


if HAS_NUMBA:
    _top_k_into = numba.njit(cache=True, fastmath=True)(_top_k_into_py)
else:
    _top_k_into = _top_k_into_py


def _make_train_loop():
    def _train(
        A,
        tids,
        negs_buf,
        e_w_buf,
        e_c_buf,
        e_n_buf,
        n_dims,
        k_active,
        window,
        n_neg,
        lr,
        l2_decay,
    ):
        N = tids.shape[0]
        neg_pos = 0
        n_negs_total = negs_buf.shape[0]

        for i in range(N):
            center = tids[i]
            _top_k_into(A[center], k_active, e_w_buf)

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
                _top_k_into(A[context], k_active, e_c_buf)

                # Compute score = bit overlap = |E_w ∩ E_c|
                # e_w_buf and e_c_buf are arrays of indices (length k_active)
                overlap = 0
                for a in range(k_active):
                    bit_a = e_w_buf[a]
                    for b in range(k_active):
                        if e_c_buf[b] == bit_a:
                            overlap += 1
                            break
                score = float(overlap)

                # Sigmoid error: σ(-score). Large for low overlap, small for
                # high overlap — the gradient strength.
                error = 1.0 / (1.0 + math.exp(score))

                # Update A_w on bits where E_c is active (E_c[i] = 1).
                # ∂score/∂A_w[i] via STE = E_c[i] = 1 if i in support of E_c.
                # So update A[center, e_c_buf[bi]] += lr * error.
                step = lr * error
                for bi in range(k_active):
                    A[center, e_c_buf[bi]] += step

                # Symmetric: update A_c on bits where E_w is active.
                for bi in range(k_active):
                    A[context, e_w_buf[bi]] += step

                if l2_decay > 0.0:
                    one_minus = 1.0 - l2_decay
                    for d in range(n_dims):
                        A[center, d] *= one_minus
                        A[context, d] *= one_minus

                # Negatives: gradient descent on L_neg = -log σ(-score_neg)
                # ∂L_neg/∂score_neg = σ(score_neg)
                # Update on A_w[i] for i in support of E_n: -lr · σ(score_neg)
                for _ in range(n_neg):
                    if neg_pos >= n_negs_total:
                        return neg_pos
                    neg_id = negs_buf[neg_pos]
                    neg_pos += 1
                    _top_k_into(A[neg_id], k_active, e_n_buf)

                    overlap_n = 0
                    for a in range(k_active):
                        bit_a = e_w_buf[a]
                        for b in range(k_active):
                            if e_n_buf[b] == bit_a:
                                overlap_n += 1
                                break
                    score_n = float(overlap_n)

                    # σ(score_n) — large when score is wrongly high
                    error_n = 1.0 / (1.0 + math.exp(-score_n))
                    step_n = lr * error_n

                    for bi in range(k_active):
                        A[center, e_n_buf[bi]] -= step_n
                    for bi in range(k_active):
                        A[neg_id, e_w_buf[bi]] -= step_n
        return neg_pos

    if HAS_NUMBA:
        return numba.njit(cache=True, fastmath=True)(_train), True
    return _train, False


_TRAIN_FN, _IS_JIT = _make_train_loop()


def train_gradient_ste(
    token_ids: list[int],
    *,
    id_to_token: list[str],
    n_dims: int = 1024,
    k_active: int = 40,
    window: int = 5,
    n_neg: int = 5,
    lr: float = 0.05,
    l2_decay: float = 0.0,
    init_scale: float = 0.01,
    neg_power: float = 0.75,
    seed: int = 0,
) -> tuple[GradientSTEEmbeddings, dict]:
    """Train gradient-via-STE SSH on `token_ids`."""
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

    e_w_buf = np.empty(k_active, dtype=np.int64)
    e_c_buf = np.empty(k_active, dtype=np.int64)
    e_n_buf = np.empty(k_active, dtype=np.int64)

    n_negs_used = _TRAIN_FN(
        A,
        tids,
        negs_buf,
        e_w_buf,
        e_c_buf,
        e_n_buf,
        n_dims,
        k_active,
        window,
        n_neg,
        float(lr),
        float(l2_decay),
    )

    elapsed = time.monotonic() - t0

    sdrs: dict[str, np.ndarray] = {}
    for w in range(V):
        top_k_idx = np.argpartition(-A[w], k_active)[:k_active]
        code = np.zeros(n_dims, dtype=np.bool_)
        code[top_k_idx] = True
        sdrs[id_to_token[w]] = code

    mean_active = float(np.mean([int(v.sum()) for v in sdrs.values()]))
    return GradientSTEEmbeddings(sdrs), {
        "elapsed_s": time.monotonic() - t0,
        "elapsed_train_s": elapsed,
        "vocab_size": V,
        "n_dims": n_dims,
        "k_active": k_active,
        "window": window,
        "n_neg": n_neg,
        "lr": lr,
        "l2_decay": l2_decay,
        "n_train_tokens": N,
        "n_negs_used": int(n_negs_used),
        "active_per_word_mean": mean_active,
        "jit_enabled": _IS_JIT,
    }
