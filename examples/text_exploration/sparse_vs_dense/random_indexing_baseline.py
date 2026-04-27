"""Random Indexing baseline for ARB-139.

Sahlgren / Kanerva-style Random Indexing. Each word is assigned a fixed
sparse ternary "index vector" at init (mostly zeros, a few +/-1s). A
word's context vector is the accumulated sum of the index vectors of
its context-window neighbors over the whole corpus. The final sparse-
binary embedding keeps the top-k positions by magnitude (sign dropped)
so the output is a boolean SDR directly comparable to T1 via Jaccard.

Why this belongs here:

- **No learning rule.** Just random projection + accumulation. Isolates
  "sparse-binary representation" from "local Hebbian learning rule" in
  the four-way comparison (word2vec / RI / Brown / T1).
- **One pass, incremental.** A new word gets a new index vector, its
  context vector grows as more corpus flows through. No retraining.
- **Same input format as word2vec and T1.** Token IDs in, sparse bool
  SDRs out. Plugs into the existing `Embeddings` protocol.

Design choices:

- Ternary index vectors (+1, -1, 0) per Sahlgren — parity with the
  standard RI formulation. We do NOT binarize the index vector itself.
- Context aggregation: windowed sum, no distance weighting (keeps it
  minimal; distance weighting is a knob to sweep later).
- Binarization: top-k by absolute value → True. This drops sign, which
  loses the bipolar anti-correlation signal, but is necessary to match
  the Jaccard-based sparse eval path. Keeping signed ints is a plausible
  variant worth a separate baseline.
"""

from __future__ import annotations

import time

import numpy as np


class RandomIndexingEmbeddings:
    """`Embeddings`-compatible wrapper around RI-derived sparse SDRs."""

    name = "random_indexing"

    def __init__(self, sdrs: dict[str, np.ndarray]) -> None:
        self._sdrs = sdrs

    def vocab(self) -> list[str]:
        return list(self._sdrs.keys())

    def get(self, word: str) -> np.ndarray | None:
        return self._sdrs.get(word)

    def is_sparse(self) -> bool:
        return True


def _build_index_vectors(
    vocab_size: int,
    n_dims: int,
    n_nonzero: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sparse ternary index matrix, shape (vocab_size, n_dims).

    Each row has exactly `n_nonzero` nonzero entries, half +1 and half
    -1 (approximately — if `n_nonzero` is odd, one side gets the extra).
    Stored as int8 since the values are in {-1, 0, +1}.
    """
    idx = np.zeros((vocab_size, n_dims), dtype=np.int8)
    n_pos = n_nonzero // 2
    n_neg = n_nonzero - n_pos
    for w in range(vocab_size):
        pos = rng.choice(n_dims, size=n_nonzero, replace=False)
        idx[w, pos[:n_pos]] = 1
        idx[w, pos[n_pos : n_pos + n_neg]] = -1
    return idx


def train_random_indexing(
    token_ids: list[int],
    *,
    id_to_token: list[str],
    n_dims: int = 2048,
    n_nonzero: int = 20,
    window: int = 5,
    k_active: int = 40,
    seed: int = 0,
) -> tuple[RandomIndexingEmbeddings, dict]:
    """Train Random Indexing on `token_ids`, return (embeddings, stats).

    Args:
        token_ids: Flat stream of integer token IDs.
        id_to_token: Vocabulary list; index -> token string.
        n_dims: Dimensionality of index vectors (and context vectors).
            Higher = more capacity, lower collision. 2048 is the standard
            Kanerva-style "high dimensional" default for this scale.
        n_nonzero: Number of nonzero entries per index vector. Sahlgren's
            original paper suggests ~2-20 — we use 20 so the accumulated
            context vector gets meaningful signal quickly.
        window: Context window half-width (same convention as word2vec).
            Each token contributes its index vector to the context of the
            `2*window` surrounding tokens.
        k_active: Number of active bits in the final binary SDR (top-k
            by |context_vector|). 40/2048 ≈ 2% density matches the T1
            region default (16 active columns × ~2 l23 cells = ~32/1024 ≈
            3%) roughly.
        seed: RNG seed for index vectors.

    Returns:
        (RandomIndexingEmbeddings, stats dict).
    """
    from scipy.sparse import coo_matrix

    rng = np.random.default_rng(seed)
    V = len(id_to_token)

    t0 = time.monotonic()
    index_vecs = _build_index_vectors(V, n_dims, n_nonzero, rng)

    # Build a sparse V x V co-occurrence matrix `co[d, s]` = count of
    # (src-token s seen in window of dst-token d) across the corpus.
    # Then context = co @ index_vecs in one BLAS-backed sparse matmul —
    # much faster than a per-token Python loop for N >= 10^5.
    N = len(token_ids)
    tids = np.asarray(token_ids, dtype=np.int32)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for d in range(-window, window + 1):
        if d == 0:
            continue
        if d > 0:
            dst = tids[: N - d]
            src = tids[d:]
        else:
            dst = tids[-d:]
            src = tids[: N + d]
        rows.append(dst)
        cols.append(src)
    rows_arr = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    cols_arr = np.concatenate(cols) if cols else np.empty(0, dtype=np.int32)
    data = np.ones(rows_arr.size, dtype=np.int32)
    co = coo_matrix((data, (rows_arr, cols_arr)), shape=(V, V)).tocsr()
    context = (co @ index_vecs.astype(np.int32)).astype(np.int32)

    # Binarize: top-k_active positions by absolute value.
    sdrs: dict[str, np.ndarray] = {}
    abs_ctx = np.abs(context)
    # argpartition is O(V*n_dims) which is fine for vocab=5000, dims=2048.
    top_idx = np.argpartition(-abs_ctx, kth=k_active, axis=1)[:, :k_active]
    for w in range(V):
        v = np.zeros(n_dims, dtype=np.bool_)
        v[top_idx[w]] = True
        sdrs[id_to_token[w]] = v

    mean_active = float(np.mean([int(v.sum()) for v in sdrs.values()]))
    return RandomIndexingEmbeddings(sdrs), {
        "elapsed_s": time.monotonic() - t0,
        "vocab_size": V,
        "n_dims": n_dims,
        "n_nonzero": n_nonzero,
        "window": window,
        "k_active": k_active,
        "n_train_tokens": N,
        "active_per_word_mean": mean_active,
    }
