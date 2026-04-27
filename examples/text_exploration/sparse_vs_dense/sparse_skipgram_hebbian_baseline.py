"""Sparse Skip-gram with local Hebbian + anti-Hebbian updates (ARB-139).

Does the same thing word2vec does — slide a window, pair center/context
tokens, contrast against unigram-sampled negatives — but with:

- **Sparse binary codes** instead of dense vectors. Each word has a
  real-valued accumulator `A_w` in R^D that is never exposed; the
  "embedding" is `E_w = top_k(A_w)` in {0,1}^D. k-WTA is the only
  non-Hebbian component.
- **Local Hebbian / anti-Hebbian updates** instead of gradient descent.
  Positive pair → strengthen connection; negative pair → weaken.
  Each step touches exactly (1 + 1 + n_neg) accumulator rows.
- **Two tables** (A_center, A_context), mirroring word2vec's
  in-vs-out vectors — empirically important for quality.

Philosophy: keeps word2vec's sampling procedure (the part that does
the real work) and swaps only the representation substrate + the
update rule. Isolates "sparse binary + Hebbian" from "cortical
architecture" — so its result tells us whether arbora's T1 anti-
correlation is due to the representation + learning rule, or due to
the cortical circuit around them.

Update rule, per positive (center, context):

    E_context_idx = argpartition_top_k(A_context[context])
    E_center_idx  = argpartition_top_k(A_center[center])
    A_center[center,   E_context_idx] += lr_pos    # Hebbian
    A_context[context, E_center_idx]  += lr_pos    # Hebbian (symmetric)

    for each sampled negative n:
        E_n_idx = argpartition_top_k(A_context[n])
        A_center[center, E_n_idx] -= lr_neg        # anti-Hebbian

At extraction time: E_w = top_k(A_center[w]) for every w — matching
word2vec's convention of using center vectors as the final embedding.
"""

from __future__ import annotations

import time

import numpy as np


class SparseSkipgramHebbianEmbeddings:
    """`Embeddings`-compatible wrapper for locally-learned sparse skip-gram."""

    name = "sparse_skipgram_hebbian"

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
    """Unigram^power distribution CDF for negative sampling.

    Power=0.75 matches word2vec's default. Flattens the distribution so
    rare words get sampled more often than their frequency would suggest.
    Returns a (V,) CDF that np.searchsorted can use for O(log V) sampling.
    """
    counts = np.bincount(token_ids, minlength=vocab_size).astype(np.float64)
    probs = np.power(counts, power)
    total = probs.sum()
    if total == 0:
        probs = np.full(vocab_size, 1.0 / vocab_size)
    else:
        probs /= total
    return np.cumsum(probs)


def train_sparse_skipgram_hebbian(
    token_ids: list[int],
    *,
    id_to_token: list[str],
    n_dims: int = 1024,
    k_active: int = 40,
    window: int = 5,
    n_neg: int = 5,
    lr_pos: float = 0.05,
    lr_neg: float = 0.02,
    init_scale: float = 0.01,
    neg_power: float = 0.75,
    seed: int = 0,
) -> tuple[SparseSkipgramHebbianEmbeddings, dict]:
    """Train local Hebbian + anti-Hebbian skip-gram on `token_ids`.

    Args:
        token_ids: Flat stream of integer token IDs.
        id_to_token: Vocab list.
        n_dims: Accumulator / SDR dimensionality. 1024 matches T1's
            `n_columns * n_l23 = 256 * 4` so per-bit capacity is
            comparable at the representation level.
        k_active: Number of active bits per word's final SDR. 40/1024
            ≈ 4% density — close to T1's observed active-per-word mean.
        window: Context window half-width (symmetric). Matches word2vec.
        n_neg: Negative samples per positive pair. Matches word2vec.
        lr_pos: Hebbian strengthening rate on positive pairs.
        lr_neg: Anti-Hebbian weakening rate on negatives.
            Default lr_neg < lr_pos reflects that each positive pair
            contributes n_neg anti-Hebbian updates; total anti-Hebbian
            mass per step ≈ n_neg * lr_neg, which we want comparable to
            lr_pos to avoid collapse.
        init_scale: Stdev of Gaussian init for accumulators. Small and
            non-zero so the very first top_k has a well-defined (random)
            selection rather than ties.
        neg_power: Exponent on unigram counts for negative sampling
            (word2vec uses 0.75).
        seed: RNG seed.

    Returns:
        (SparseSkipgramHebbianEmbeddings, stats dict).
    """
    rng = np.random.default_rng(seed)
    V = len(id_to_token)
    tids = np.asarray(token_ids, dtype=np.int32)
    N = len(tids)

    t0 = time.monotonic()

    # Two accumulator tables — mirror word2vec's center/context split.
    A_center = rng.standard_normal((V, n_dims)).astype(np.float32) * init_scale
    A_context = rng.standard_normal((V, n_dims)).astype(np.float32) * init_scale

    cdf = _build_unigram_cdf(tids, V, neg_power)

    def top_k_indices(row: np.ndarray) -> np.ndarray:
        # argpartition: top-k by value, unordered. Negate because we want
        # the k largest.
        return np.argpartition(-row, k_active)[:k_active]

    # Main training loop: iterate positions; for each, iterate window;
    # for each positive pair do the three updates.
    #
    # Pre-sample negatives in a flat batch to amortize RNG overhead —
    # for large corpora this matters more than the loop structure. We
    # sample lazily in chunks so memory stays bounded.
    CHUNK = 100_000
    pairs_processed = 0
    neg_cache: np.ndarray | None = None
    neg_cache_offset = 0

    def get_negatives() -> np.ndarray:
        nonlocal neg_cache, neg_cache_offset
        if neg_cache is None or neg_cache_offset >= len(neg_cache):
            neg_cache = np.searchsorted(cdf, rng.random(CHUNK * n_neg))
            neg_cache_offset = 0
        out = neg_cache[neg_cache_offset : neg_cache_offset + n_neg]
        neg_cache_offset += n_neg
        return out

    for i in range(N):
        center = int(tids[i])
        lo = max(0, i - window)
        hi = min(N, i + window + 1)
        for j in range(lo, hi):
            if j == i:
                continue
            context = int(tids[j])

            e_context = top_k_indices(A_context[context])
            e_center = top_k_indices(A_center[center])

            # Positive Hebbian (both directions)
            A_center[center, e_context] += lr_pos
            A_context[context, e_center] += lr_pos

            # Anti-Hebbian against n_neg random negatives
            negs = get_negatives()
            for n_id in negs:
                e_n = top_k_indices(A_context[int(n_id)])
                A_center[center, e_n] -= lr_neg

            pairs_processed += 1

    # Extract final embeddings: top-k of A_center for each word.
    # Matches word2vec's convention of using center vectors as the
    # exported representation.
    sdrs: dict[str, np.ndarray] = {}
    for w in range(V):
        idx = top_k_indices(A_center[w])
        code = np.zeros(n_dims, dtype=np.bool_)
        code[idx] = True
        sdrs[id_to_token[w]] = code

    mean_active = float(np.mean([int(v.sum()) for v in sdrs.values()]))
    return SparseSkipgramHebbianEmbeddings(sdrs), {
        "elapsed_s": time.monotonic() - t0,
        "vocab_size": V,
        "n_dims": n_dims,
        "k_active": k_active,
        "window": window,
        "n_neg": n_neg,
        "lr_pos": lr_pos,
        "lr_neg": lr_neg,
        "pairs_processed": pairs_processed,
        "n_train_tokens": N,
        "active_per_word_mean": mean_active,
    }
