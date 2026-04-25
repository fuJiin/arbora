"""Modulated Sparse Skip-gram Hebbian baseline (ARB-139).

The "primitive-equipped" version of `sparse_skipgram_hebbian_baseline.py`.

Same data shaping (skip-gram + unigram^0.75 negatives), same representation
(real accumulator → top-k binary code), same locality (only k bits touched
per update). The change is in the update *magnitude*:

- **Surprise modulation** (default on): scale the positive Hebbian update by
  (1 - overlap/k) — small when the pair is already aligned, large when it's
  novel. Anti-Hebbian on negatives scales by (overlap/k) — small when there's
  no spurious alignment to fix, large when the model has wrongly grouped a
  negative with the center. This is the structural analog of word2vec's
  σ(-score) factor.

- **Optional Oja-like decay**: per-update uniform shrinkage of the touched
  rows. Off by default — modulation alone should be sufficient stabilizer
  per the analysis in conversation Part 1. Knob is here for ablations.

- **Numba-JITed inner loop** for ~10x speedup over pure-Python — necessary
  because we can't iterate fast enough through 1M-10M tokens otherwise.

The training-time scoring function is bit overlap (intersection size). At
extraction time we read top-k(A_center) per word, matching word2vec's
"return center vectors" convention.
"""

from __future__ import annotations

import time

import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


class ModulatedSSHEmbeddings:
    """`Embeddings`-compatible wrapper for modulated SSH SDRs."""

    name = "ssh_modulated"

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


# Inner loop. Written so the entire body is numba-friendly: only ndarray ops,
# explicit loops, no Python objects. The argpartition dance is replaced with
# a manual top-k scan that numba can compile efficiently.

def _make_train_loop():
    """Return a JIT-compiled inner-loop function (or a Python fallback)."""

    def _top_k_into(row, k, out):
        """Fill `out[:k]` with indices of the k largest entries of `row`.

        Manual top-k that numba can compile cleanly (numba doesn't support
        np.argpartition). For our scale (D=1024, k=40) this is comparable
        to argpartition — heap-style is overkill, the scan-and-replace
        approach is fine and branch-predictor-friendly.
        """
        n = row.shape[0]
        # Initialize out with first k indices.
        for i in range(k):
            out[i] = i
        # Find the position of the smallest among them.
        min_pos = 0
        min_val = row[out[0]]
        for i in range(1, k):
            v = row[out[i]]
            if v < min_val:
                min_val = v
                min_pos = i
        # Walk the rest; replace min when we find something larger.
        for i in range(k, n):
            v = row[i]
            if v > min_val:
                out[min_pos] = i
                # Recompute new min — O(k), called rarely once row is hot.
                min_val = row[out[0]]
                min_pos = 0
                for j in range(1, k):
                    vj = row[out[j]]
                    if vj < min_val:
                        min_val = vj
                        min_pos = j

    def _overlap(a, b, k):
        """|a ∩ b| where a and b are sorted-or-unsorted arrays of k indices each."""
        # Use a small bool mask. For k=40 this allocates k bools — cheap.
        # Mark a's indices, count how many of b's are marked.
        # We're called with k in {32, 40, 64} — boolean mask of size n_dims is
        # too big for stack allocation, so we do an O(k^2) inner-loop check.
        c = 0
        for i in range(k):
            ai = a[i]
            for j in range(k):
                if b[j] == ai:
                    c += 1
                    break
        return c

    def _train(
        A_center,
        A_context,
        tids,
        cdf,
        negs_buf,
        e_center_buf,
        e_context_buf,
        e_neg_buf,
        n_dims,
        k_active,
        window,
        n_neg,
        lr_pos,
        lr_neg,
        decay,
        modulate,
    ):
        N = tids.shape[0]
        neg_pos = 0  # pointer into negs_buf
        n_negs_total = negs_buf.shape[0]
        for i in range(N):
            center = tids[i]
            # Cache E_center for this token position.
            _top_k_into(A_center[center], k_active, e_center_buf)

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
                _top_k_into(A_context[context], k_active, e_context_buf)

                # Surprise modulator for positive pair.
                if modulate:
                    overlap_pos = _overlap(e_center_buf, e_context_buf, k_active)
                    mod_pos = 1.0 - overlap_pos / k_active
                else:
                    mod_pos = 1.0

                # Hebbian (positive): both directions.
                step_pos = lr_pos * mod_pos
                for bi in range(k_active):
                    A_center[center, e_context_buf[bi]] += step_pos
                    A_context[context, e_center_buf[bi]] += step_pos

                # Optional uniform decay applied only to the rows we just touched.
                if decay > 0.0:
                    one_minus = 1.0 - decay
                    for d in range(n_dims):
                        A_center[center, d] *= one_minus
                        A_context[context, d] *= one_minus
                    # E_center is now stale in absolute magnitude but the top-k
                    # set is preserved by uniform scaling, so no recompute needed.

                # Anti-Hebbian on n_neg negatives.
                for _ in range(n_neg):
                    neg_id = negs_buf[neg_pos]
                    neg_pos += 1
                    if neg_pos >= n_negs_total:
                        # Out of pre-sampled negatives; signal to caller by stopping.
                        return neg_pos
                    _top_k_into(A_context[neg_id], k_active, e_neg_buf)
                    if modulate:
                        overlap_neg = _overlap(e_center_buf, e_neg_buf, k_active)
                        mod_neg = overlap_neg / k_active  # bigger when wrongly aligned
                    else:
                        mod_neg = 1.0
                    step_neg = lr_neg * mod_neg
                    for bi in range(k_active):
                        A_center[center, e_neg_buf[bi]] -= step_neg
        return neg_pos

    if HAS_NUMBA:
        # Compile both helpers and the main loop. nopython=True forces a fast
        # native build; we accept a one-time JIT cost on first call.
        _top_k_jit = numba.njit(cache=True, fastmath=True)(_top_k_into)
        _overlap_jit = numba.njit(cache=True, fastmath=True)(_overlap)

        # Re-bind the helpers inside _train so numba inlines them.
        train_src = _train

        @numba.njit(cache=True, fastmath=True)
        def _train_jit(
            A_center, A_context, tids, cdf,
            negs_buf, e_center_buf, e_context_buf, e_neg_buf,
            n_dims, k_active, window, n_neg,
            lr_pos, lr_neg, decay, modulate,
        ):
            N = tids.shape[0]
            neg_pos = 0
            n_negs_total = negs_buf.shape[0]
            for i in range(N):
                center = tids[i]
                _top_k_jit(A_center[center], k_active, e_center_buf)

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
                    _top_k_jit(A_context[context], k_active, e_context_buf)

                    if modulate:
                        overlap_pos = _overlap_jit(e_center_buf, e_context_buf, k_active)
                        mod_pos = 1.0 - overlap_pos / k_active
                    else:
                        mod_pos = 1.0

                    step_pos = lr_pos * mod_pos
                    for bi in range(k_active):
                        A_center[center, e_context_buf[bi]] += step_pos
                        A_context[context, e_center_buf[bi]] += step_pos

                    if decay > 0.0:
                        one_minus = 1.0 - decay
                        for d in range(n_dims):
                            A_center[center, d] *= one_minus
                            A_context[context, d] *= one_minus

                    for _ in range(n_neg):
                        if neg_pos >= n_negs_total:
                            return neg_pos
                        neg_id = negs_buf[neg_pos]
                        neg_pos += 1
                        _top_k_jit(A_context[neg_id], k_active, e_neg_buf)
                        if modulate:
                            overlap_neg = _overlap_jit(e_center_buf, e_neg_buf, k_active)
                            mod_neg = overlap_neg / k_active
                        else:
                            mod_neg = 1.0
                        step_neg = lr_neg * mod_neg
                        for bi in range(k_active):
                            A_center[center, e_neg_buf[bi]] -= step_neg
            return neg_pos
        return _train_jit, True
    else:
        return _train, False


_TRAIN_FN, _IS_JIT = _make_train_loop()


def train_sparse_skipgram_hebbian_modulated(
    token_ids: list[int],
    *,
    id_to_token: list[str],
    n_dims: int = 1024,
    k_active: int = 40,
    window: int = 5,
    n_neg: int = 5,
    lr_pos: float = 0.05,
    lr_neg: float = 0.05,
    modulate: bool = True,
    decay: float = 0.0,
    single_table: bool = False,
    init_scale: float = 0.01,
    neg_power: float = 0.75,
    seed: int = 0,
) -> tuple[ModulatedSSHEmbeddings, dict]:
    """Train modulated SSH on `token_ids`.

    Args:
        token_ids: Flat stream of integer token IDs.
        id_to_token: Vocab list.
        n_dims: Accumulator / SDR dimensionality.
        k_active: Active bits per word's final SDR.
        window: Context window half-width.
        n_neg: Negative samples per positive pair.
        lr_pos: Hebbian rate on positive pairs.
        lr_neg: Anti-Hebbian rate on negatives. With modulation enabled, the
            symmetry between LTP and LTD is restored automatically by the
            modulator scaling, so default lr_pos == lr_neg here (vanilla SSH
            had to use lr_neg < lr_pos to compensate for unmodulated rates).
        modulate: If True, scale positive update by (1 - overlap/k) and
            negative update by (overlap/k). If False, recovers vanilla SSH.
        decay: If > 0, apply uniform Oja-like decay to the touched rows after
            each pair: `A *= (1 - decay)`. 0 means no decay.
        single_table: If True, use a single accumulator table for both center
            and context roles (A_center and A_context aliased to same array).
            Halves memory and tests whether word2vec's two-table asymmetry
            buys SSH anything — likely not, since bit-overlap scoring is
            symmetric. If False, separate tables (default, matches word2vec).
        init_scale: Stdev of Gaussian init for accumulators.
        neg_power: Exponent on unigram counts for negative sampling.
        seed: RNG seed.

    Returns:
        (ModulatedSSHEmbeddings, stats dict).
    """
    rng = np.random.default_rng(seed)
    V = len(id_to_token)
    tids = np.asarray(token_ids, dtype=np.int64)
    N = len(tids)

    t0 = time.monotonic()

    A_center = (rng.standard_normal((V, n_dims)) * init_scale).astype(np.float32)
    if single_table:
        # Alias: every update path writes through this view; effectively
        # there is one table indexed by word, regardless of role.
        A_context = A_center
    else:
        A_context = (rng.standard_normal((V, n_dims)) * init_scale).astype(np.float32)

    cdf = _build_unigram_cdf(tids, V, neg_power)

    # Pre-sample all negatives upfront so the inner loop can be numba-friendly.
    # Total negatives needed: N * (2 * window) * n_neg, with some slack.
    expected_pairs = N * 2 * window
    total_negs = expected_pairs * n_neg + 1024
    neg_uniform = rng.random(total_negs)
    negs_buf = np.searchsorted(cdf, neg_uniform).astype(np.int64)

    # Scratch buffers reused inside the loop.
    e_center_buf = np.empty(k_active, dtype=np.int64)
    e_context_buf = np.empty(k_active, dtype=np.int64)
    e_neg_buf = np.empty(k_active, dtype=np.int64)

    n_negs_used = _TRAIN_FN(
        A_center, A_context, tids, cdf,
        negs_buf, e_center_buf, e_context_buf, e_neg_buf,
        n_dims, k_active, window, n_neg,
        float(lr_pos), float(lr_neg), float(decay), bool(modulate),
    )

    elapsed_train = time.monotonic() - t0

    # Extract: top-k of A_center for each word.
    sdrs: dict[str, np.ndarray] = {}
    out_buf = np.empty(k_active, dtype=np.int64)
    for w in range(V):
        # Reuse the same top-k routine for consistency.
        top_k_idx = np.argpartition(-A_center[w], k_active)[:k_active]
        code = np.zeros(n_dims, dtype=np.bool_)
        code[top_k_idx] = True
        sdrs[id_to_token[w]] = code

    mean_active = float(np.mean([int(v.sum()) for v in sdrs.values()]))
    return ModulatedSSHEmbeddings(sdrs), {
        "elapsed_s": time.monotonic() - t0,
        "elapsed_train_s": elapsed_train,
        "vocab_size": V,
        "n_dims": n_dims,
        "k_active": k_active,
        "window": window,
        "n_neg": n_neg,
        "lr_pos": lr_pos,
        "lr_neg": lr_neg,
        "modulate": modulate,
        "decay": decay,
        "single_table": single_table,
        "n_train_tokens": N,
        "n_negs_used": int(n_negs_used),
        "active_per_word_mean": mean_active,
        "jit_enabled": _IS_JIT,
    }
