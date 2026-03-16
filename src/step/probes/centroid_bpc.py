"""Centroid-based BPC probe: non-learned prediction quality measurement.

Uses a running centroid (mean L2/3 pattern) per token. Predicts by
comparing the current L2/3 state to all centroids via dot product,
then softmax-normalizes to get probabilities for BPC computation.

No learned parameters — just exponential moving averages. Cannot
"break" or fail to track representational drift, because the centroids
update every step with no capacity limit.

Random baseline over V chars: log2(V) ~ 4.9 for 30 chars.
"""

import math

import numpy as np


class CentroidBPCProbe:
    """BPC measurement using centroid-based nearest-neighbor decoding."""

    def __init__(
        self,
        source_dim: int,
        *,
        ema_alpha: float = 0.01,
        temperature: float = 1.0,
        boundary_window: int = 10,
    ):
        self._source_dim = source_dim
        self._alpha = ema_alpha
        self._temperature = temperature

        # Per-token centroid: token_id -> ndarray(source_dim,) float
        self._centroids: dict[int, np.ndarray] = {}
        self._counts: dict[int, int] = {}

        # BPC accumulators
        self._total_bits: float = 0.0
        self._n_chars: int = 0
        self._recent_bits: list[float] = []
        self._window_size: int = 500

        # Per-dialogue tracking
        self._boundary_window = boundary_window
        self._dialogue_bits: float = 0.0
        self._dialogue_chars: int = 0
        self._boundary_bits: float = 0.0
        self._boundary_chars: int = 0
        self._steady_bits: float = 0.0
        self._steady_chars: int = 0
        self._dialogue_bpcs: list[float] = []
        self._boundary_bpcs: list[float] = []
        self._steady_bpcs: list[float] = []

    @property
    def bpc(self) -> float:
        if self._n_chars == 0:
            return float("inf")
        return self._total_bits / self._n_chars

    @property
    def recent_bpc(self) -> float:
        if not self._recent_bits:
            return float("inf")
        return sum(self._recent_bits) / len(self._recent_bits)

    @property
    def n_tokens(self) -> int:
        return len(self._centroids)

    @property
    def dialogue_bpcs(self) -> list[float]:
        return self._dialogue_bpcs

    @property
    def boundary_bpcs(self) -> list[float]:
        return self._boundary_bpcs

    @property
    def steady_bpcs(self) -> list[float]:
        return self._steady_bpcs

    def observe(self, token_id: int, l23_state: np.ndarray) -> None:
        """Update centroid for token_id with the current L2/3 pattern."""
        pattern = l23_state.astype(np.float64)
        if token_id not in self._centroids:
            self._centroids[token_id] = pattern.copy()
            self._counts[token_id] = 1
        else:
            self._counts[token_id] += 1
            self._centroids[token_id] += self._alpha * (
                pattern - self._centroids[token_id]
            )

    def step(self, token_id: int, l23_state: np.ndarray) -> float:
        """Compute bits for one character prediction.

        Args:
            token_id: The actual character (ord(char)).
            l23_state: L2/3 activations from the PREVIOUS timestep.

        Returns:
            Bits for this character (lower = better).
        """
        n_tokens = len(self._centroids)
        if n_tokens < 2:
            return 0.0

        # Compute similarity to all centroids
        pattern = l23_state.astype(np.float64)
        token_ids = list(self._centroids.keys())
        centroids = np.stack([self._centroids[t] for t in token_ids])

        # Dot product similarity (pattern and centroids are sparse binary-ish)
        scores = centroids @ pattern  # (n_tokens,)
        scores = scores / self._temperature

        # Softmax
        scores -= scores.max()
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()

        if token_id in self._centroids:
            idx = token_ids.index(token_id)
            prob = max(probs[idx], 1e-10)
        else:
            prob = 1.0 / max(n_tokens, 2)

        bits = -math.log2(prob)

        self._total_bits += bits
        self._n_chars += 1
        self._recent_bits.append(bits)
        if len(self._recent_bits) > self._window_size:
            self._recent_bits.pop(0)

        # Per-dialogue tracking
        self._dialogue_bits += bits
        self._dialogue_chars += 1
        if self._dialogue_chars <= self._boundary_window:
            self._boundary_bits += bits
            self._boundary_chars += 1
        else:
            self._steady_bits += bits
            self._steady_chars += 1

        return bits

    def dialogue_boundary(self) -> None:
        """Call at STORY_BOUNDARY to snapshot per-dialogue BPC."""
        if self._dialogue_chars > 0:
            self._dialogue_bpcs.append(
                self._dialogue_bits / self._dialogue_chars
            )
        if self._boundary_chars > 0:
            self._boundary_bpcs.append(
                self._boundary_bits / self._boundary_chars
            )
        if self._steady_chars > 0:
            self._steady_bpcs.append(
                self._steady_bits / self._steady_chars
            )
        self._dialogue_bits = 0.0
        self._dialogue_chars = 0
        self._boundary_bits = 0.0
        self._boundary_chars = 0
        self._steady_bits = 0.0
        self._steady_chars = 0

    def reset(self) -> None:
        """Reset BPC accumulators (keeps centroids)."""
        self._total_bits = 0.0
        self._n_chars = 0
        self._recent_bits.clear()
        self._dialogue_bits = 0.0
        self._dialogue_chars = 0
        self._boundary_bits = 0.0
        self._boundary_chars = 0
        self._steady_bits = 0.0
        self._steady_chars = 0
        self._dialogue_bpcs.clear()
        self._boundary_bpcs.clear()
        self._steady_bpcs.clear()
