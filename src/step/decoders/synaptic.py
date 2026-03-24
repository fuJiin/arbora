"""Synaptic decoder: nearest-neighbor decode via ff_weight reconstruction.

Instead of argmax -> string -> compare (lossy), this decoder:
1. Reconstructs predicted encoding from ff_weights, weighted by
   prediction confidence per column.
2. Compares against all observed token encodings via dot product.
3. Returns the token with highest affinity.

Also provides column-level decode: maps column activation patterns
to tokens via inverted index, avoiding burst-inflated neuron sets.
"""

import numpy as np

from step.cortex.region import CorticalRegion


class SynapticDecoder:
    """Decodes predicted neuron activations back to tokens via ff_weights."""

    def __init__(self):
        self._token_ids: list[int] = []
        self._token_strs: list[str] = []
        self._encodings: list[np.ndarray] = []  # flat encoding vectors
        self._token_id_set: set[int] = set()
        self._encoding_matrix: np.ndarray | None = None  # (n_tokens, input_dim)

        # Column-level inverted index: col -> list of token indices
        self._col_index: dict[int, list[int]] = {}

    def observe(
        self,
        token_id: int,
        token_str: str,
        encoding: np.ndarray,
        active_columns: np.ndarray | None = None,
    ) -> None:
        """Record a token's encoding and column activation for future decode."""
        if token_id in self._token_id_set:
            return
        idx = len(self._token_ids)
        self._token_id_set.add(token_id)
        self._token_ids.append(token_id)
        self._token_strs.append(token_str)
        self._encodings.append(encoding.flatten().astype(np.float64))
        self._encoding_matrix = None  # invalidate cache

        # Column-level index: record which columns activated for this token
        if active_columns is not None:
            for col in np.nonzero(active_columns)[0]:
                self._col_index.setdefault(int(col), []).append(idx)

    def decode_synaptic(
        self,
        predicted_neurons: np.ndarray,
        region: CorticalRegion,
    ) -> tuple[int, str]:
        """Decode via ff_weight reconstruction + nearest-neighbor.

        Reconstructs the expected encoding from predicted columns'
        ff_weights (uniform weighting), then finds the observed token
        whose encoding has highest dot product similarity.
        """
        if not self._token_ids:
            return -1, ""

        if self._encoding_matrix is None:
            self._encoding_matrix = np.array(self._encodings)

        cols = np.unique(predicted_neurons // region.n_l4)
        if len(cols) == 0:
            return -1, ""

        # Uniform reconstruction from ff_weights of predicted columns
        reconstruction = region.ff_weights[:, cols].sum(axis=1)

        # Dot product against all observed encodings
        scores = self._encoding_matrix @ reconstruction
        best_idx = int(np.argmax(scores))
        return self._token_ids[best_idx], self._token_strs[best_idx]

    def decode_columns(
        self,
        predicted_neurons: np.ndarray,
        n_l4: int,
    ) -> tuple[int, str]:
        """Decode via column-level inverted index.

        Maps predicted columns to tokens that previously activated
        those same columns. Scores by overlap count (how many predicted
        columns match observed columns for each token).

        More stable than neuron-level decode because columns are the
        functional unit — not inflated by burst.
        """
        if not self._token_ids or not self._col_index:
            return -1, ""

        cols = np.unique(predicted_neurons // n_l4)
        scores: dict[int, float] = {}
        for col in cols:
            for idx in self._col_index.get(int(col), ()):
                scores[idx] = scores.get(idx, 0.0) + 1.0

        if not scores:
            return -1, ""

        best_idx = max(scores, key=scores.__getitem__)
        return self._token_ids[best_idx], self._token_strs[best_idx]
