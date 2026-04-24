"""Brown-cluster-style baseline for ARB-139.

Classical Brown clustering (Brown et al. 1992) greedily merges word
classes to maximize class-based bigram likelihood, producing a binary
tree over words. Each word's "embedding" is its path from the root of
that tree.

Implementing the true exchange algorithm is heavy and a stand-alone
research project. What we want here is the *representation shape* Brown
clustering produces — a hierarchical, discrete, sparse-binary code — so
we can ask "does hierarchical structure in a sparse binary code buy you
anything over flat random projection (RI)?"

Approximation used here:

1. Build a vocab x vocab co-occurrence matrix on a fixed window.
2. Apply PPMI weighting (positive pointwise mutual information) — the
   same signal Brown's exchange algorithm is implicitly optimizing.
3. Reduce to a ~200-dim dense representation via truncated SVD so
   hierarchical clustering is tractable.
4. Agglomerative clustering (Ward linkage) produces a binary tree.
5. **Embedding of a word = boolean vector of all its ancestor cluster
   IDs** (multi-resolution prefix code). For a vocab of V, the tree has
   V-1 internal nodes; a word lights up exactly depth-to-root ≈ log2(V)
   ancestors. That's a genuinely sparse binary code with log-depth
   density.

This is "Brown-inspired", not Brown proper — the clustering objective
is cosine over PPMI rows instead of class-bigram likelihood, and the
linkage is Ward instead of likelihood-optimal merges. The tree shape
is the part that matters for the representation comparison.

Why not just use path-bits directly? A path-bit embedding has only
log2(V) ≈ 13 dimensions for V=5000, which has almost no capacity for
similarity judgments. The ancestor-set encoding is the standard way to
turn a Brown tree into a useful feature vector (Turian et al. 2010,
"Word Representations: A Simple and General Method for Semi-Supervised
Learning" — prefix features).
"""

from __future__ import annotations

import time

import numpy as np


class BrownClusterEmbeddings:
    """`Embeddings`-compatible wrapper around ancestor-set sparse SDRs."""

    name = "brown_cluster"

    def __init__(
        self,
        sdrs: dict[str, np.ndarray],
        *,
        path_bits: dict[str, np.ndarray] | None = None,
    ) -> None:
        self._sdrs = sdrs
        # Optional: the pure binary-path form (length = tree depth).
        # Kept around for analyses that want the "true Brown bit-path";
        # `get()` returns the ancestor-set encoding as the default since
        # it has useful capacity.
        self._path_bits = path_bits or {}

    def vocab(self) -> list[str]:
        return list(self._sdrs.keys())

    def get(self, word: str) -> np.ndarray | None:
        return self._sdrs.get(word)

    def path_bits(self, word: str) -> np.ndarray | None:
        return self._path_bits.get(word)

    def is_sparse(self) -> bool:
        return True


def _build_cooccurrence(
    token_ids: list[int],
    vocab_size: int,
    window: int,
) -> np.ndarray:
    """V x V symmetric co-occurrence counts from a windowed pass."""
    from scipy.sparse import coo_matrix

    tids = np.asarray(token_ids, dtype=np.int32)
    N = len(tids)
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
    co = coo_matrix((data, (rows_arr, cols_arr)), shape=(vocab_size, vocab_size))
    return co.toarray()


def _ppmi(counts: np.ndarray, smoothing: float = 0.75) -> np.ndarray:
    """Positive Pointwise Mutual Information with context smoothing.

    PPMI(w, c) = max(0, log(p(w,c) / (p(w) * p_smooth(c)))). The
    smoothing exponent raises context probabilities to `smoothing`
    power before normalizing — standard trick (Levy & Goldberg 2015)
    that prevents rare-context artifacts from dominating.
    """
    total = counts.sum()
    if total == 0:
        return counts.astype(np.float32)
    row_sum = counts.sum(axis=1, keepdims=True)
    col_sum = counts.sum(axis=0, keepdims=True)
    col_sum_smooth = np.power(col_sum, smoothing)
    col_sum_smooth = col_sum_smooth / col_sum_smooth.sum() * total

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(
            (counts * total) / np.maximum(row_sum * col_sum_smooth, 1e-12) + 1e-12
        )
    pmi[~np.isfinite(pmi)] = 0.0
    return np.maximum(pmi, 0.0).astype(np.float32)


def _ancestors_from_linkage(linkage: np.ndarray, n_leaves: int) -> list[list[int]]:
    """Given a scipy/sklearn linkage matrix, return each leaf's ancestor
    chain (list of internal-node IDs from immediate parent to root).

    Internal nodes in the linkage are indexed n_leaves, n_leaves+1, ...
    (scipy convention): row `k` in the linkage creates internal node
    `n_leaves + k`.
    """
    n_internal = linkage.shape[0]
    parent = np.full(n_leaves + n_internal, -1, dtype=np.int64)
    for k in range(n_internal):
        left, right = int(linkage[k, 0]), int(linkage[k, 1])
        parent[left] = n_leaves + k
        parent[right] = n_leaves + k

    ancestors: list[list[int]] = [[] for _ in range(n_leaves)]
    for leaf in range(n_leaves):
        chain: list[int] = []
        node = parent[leaf]
        while node != -1:
            chain.append(int(node) - n_leaves)  # map to 0..n_internal-1
            node = parent[node]
        ancestors[leaf] = chain
    return ancestors


def _path_bits_from_linkage(linkage: np.ndarray, n_leaves: int) -> np.ndarray:
    """Binary path: for each leaf, a bool vector of length tree-depth.

    Bit `k` = did the ancestor at depth k come from the "left" child of
    its parent? The length per leaf varies (unbalanced tree), so we pad
    to max_depth with False. This is the strict Brown-style bit-path
    representation; it's kept as a secondary output because its capacity
    is small (log2(V) bits).
    """
    n_internal = linkage.shape[0]
    parent = np.full(n_leaves + n_internal, -1, dtype=np.int64)
    is_left = np.zeros(n_leaves + n_internal, dtype=np.bool_)
    for k in range(n_internal):
        left, right = int(linkage[k, 0]), int(linkage[k, 1])
        parent[left] = n_leaves + k
        parent[right] = n_leaves + k
        is_left[left] = True
        is_left[right] = False

    paths: list[list[bool]] = []
    max_depth = 0
    for leaf in range(n_leaves):
        path: list[bool] = []
        node = leaf
        while parent[node] != -1:
            path.append(bool(is_left[node]))
            node = int(parent[node])
        path.reverse()
        paths.append(path)
        if len(path) > max_depth:
            max_depth = len(path)

    out = np.zeros((n_leaves, max_depth), dtype=np.bool_)
    for leaf, path in enumerate(paths):
        out[leaf, : len(path)] = path
    return out


def train_brown_cluster(
    token_ids: list[int],
    *,
    id_to_token: list[str],
    window: int = 5,
    svd_dim: int = 200,
    ppmi_smoothing: float = 0.75,
    linkage: str = "ward",
    seed: int = 0,
) -> tuple[BrownClusterEmbeddings, dict]:
    """Train a Brown-inspired hierarchical clustering of the vocab.

    Args:
        token_ids: Flat stream of token IDs.
        id_to_token: Vocab list.
        window: Co-occurrence window (each side).
        svd_dim: Truncated-SVD target dimension for clustering input.
            Agglomerative clustering scales O(V^2 svd_dim) in the distance
            computation, so this matters. 200 is the standard PMI+SVD
            sweet spot (Levy & Goldberg 2015).
        ppmi_smoothing: Context-probability smoothing exponent for PPMI.
        linkage: Agglomerative linkage ('ward', 'average', 'complete').
            Ward is default because it produces more balanced trees.
        seed: RNG seed for truncated SVD.

    Returns:
        (BrownClusterEmbeddings, stats dict).
    """
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from sklearn.decomposition import TruncatedSVD

    V = len(id_to_token)
    t0 = time.monotonic()

    counts = _build_cooccurrence(token_ids, V, window)
    pmi = _ppmi(counts, smoothing=ppmi_smoothing)

    # Reduce PPMI rows to a compact dense representation for clustering.
    svd = TruncatedSVD(n_components=min(svd_dim, V - 1), random_state=seed)
    reduced = svd.fit_transform(pmi)

    # scipy's linkage returns an (n-1, 4) matrix. Use the reduced rows
    # directly — Ward on raw PPMI would be noisier and much slower.
    Z = scipy_linkage(reduced, method=linkage)

    ancestors = _ancestors_from_linkage(Z, n_leaves=V)
    n_internal = V - 1

    # Ancestor-set SDRs: one bit per internal node, True for each of the
    # word's ancestors on the path to root. Exactly `depth_of_leaf`
    # active bits per word.
    sdrs: dict[str, np.ndarray] = {}
    depths: list[int] = []
    for leaf in range(V):
        v = np.zeros(n_internal, dtype=np.bool_)
        for node_id in ancestors[leaf]:
            v[node_id] = True
        sdrs[id_to_token[leaf]] = v
        depths.append(len(ancestors[leaf]))

    path_bits_mat = _path_bits_from_linkage(Z, n_leaves=V)
    path_bits: dict[str, np.ndarray] = {
        id_to_token[leaf]: path_bits_mat[leaf].copy() for leaf in range(V)
    }

    emb = BrownClusterEmbeddings(sdrs, path_bits=path_bits)
    return emb, {
        "elapsed_s": time.monotonic() - t0,
        "vocab_size": V,
        "n_dims": n_internal,
        "svd_dim": svd_dim,
        "window": window,
        "linkage": linkage,
        "mean_depth": float(np.mean(depths)),
        "max_depth": int(np.max(depths)),
        "n_train_tokens": len(token_ids),
        "active_per_word_mean": float(np.mean(depths)),
    }
