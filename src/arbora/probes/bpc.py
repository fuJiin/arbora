"""Bits-per-character (BPC) probe for measuring prediction quality.

BPC = -1/N * Σ log₂(P(char_i | context))

Uses the dendritic decoder's segment overlap scores as unnormalized
log-likelihoods, softmax-normalized to get probabilities. This measures
how well S1's L2/3 representations support downstream token prediction —
exactly what a downstream cortical region would compute.

Random baseline over V chars: log₂(V) ≈ 4.9 for 30 chars.
Good char-level models: 1.0-1.5 BPC.
"""

import math

import numpy as np

from arbora.decoders.dendritic import DendriticDecoder


class BPCProbe:
    """Accumulates bits-per-character from dendritic decoder scores."""

    def __init__(self, temperature: float = 1.0, boundary_window: int = 10):
        self._temperature = temperature
        self._total_bits: float = 0.0
        self._n_chars: int = 0
        # Rolling window for trend tracking
        self._recent_bits: list[float] = []
        self._window_size: int = 500
        # Per-dialogue tracking
        self._dialogue_bits: float = 0.0
        self._dialogue_chars: int = 0
        self._boundary_window = boundary_window
        self._boundary_bits: float = 0.0
        self._boundary_chars: int = 0
        self._steady_bits: float = 0.0
        self._steady_chars: int = 0
        self._dialogue_bpcs: list[float] = []
        self._boundary_bpcs: list[float] = []  # BPC of first N chars after reset
        self._steady_bpcs: list[float] = []  # BPC after boundary window

    @property
    def bpc(self) -> float:
        """Overall BPC across all observed characters."""
        if self._n_chars == 0:
            return float("inf")
        return self._total_bits / self._n_chars

    @property
    def recent_bpc(self) -> float:
        """BPC over the last window_size characters."""
        if not self._recent_bits:
            return float("inf")
        return sum(self._recent_bits) / len(self._recent_bits)

    def step(
        self,
        token_id: int,
        l23_state: np.ndarray,
        decoder: DendriticDecoder,
    ) -> float:
        """Compute bits for one character prediction.

        Args:
            token_id: The actual character (ord(char)).
            l23_state: L2/3 binary activations from previous timestep.
            decoder: The dendritic decoder with learned token→segment mappings.

        Returns:
            Bits for this character (lower = better). Returns log₂(n_tokens)
            if the decoder has no score for this token.
        """
        scores = decoder.decode_scores(l23_state)
        n_tokens = decoder.n_tokens

        if n_tokens == 0:
            return 0.0  # Skip before decoder has any observations

        if not scores:
            # No segments active → uniform over known tokens
            bits = math.log2(max(n_tokens, 2))
        else:
            # Softmax over overlap scores to get probability distribution
            # Only tokens with active segments get non-uniform probability;
            # unseen tokens share remaining mass uniformly.
            raw = np.array(list(scores.values()), dtype=np.float64)
            keys = list(scores.keys())
            raw = raw / self._temperature

            # Numerical stability
            raw -= raw.max()
            exp_scores = np.exp(raw)

            # Reserve some mass for tokens not in scores
            n_unseen = max(n_tokens - len(scores), 0)
            # Each unseen token gets exp(0 - max) = exp(-max) ≈ 0
            # but we give them a floor of 0.01 * avg to avoid -inf
            floor = 0.01 * exp_scores.mean() if n_unseen > 0 else 0.0
            total = exp_scores.sum() + n_unseen * floor

            if token_id in scores:
                idx = keys.index(token_id)
                prob = exp_scores[idx] / total
            else:
                # Token exists in decoder but no active segments matched
                prob = floor / total if floor > 0 else 1.0 / max(n_tokens, 2)

            prob = max(prob, 1e-10)  # Clamp to avoid log(0)
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
        """Call at STORY_BOUNDARY to snapshot per-dialogue BPC.

        Separates within-dialogue BPC into boundary spike (first N chars)
        and steady-state. Does NOT reset the overall accumulators.
        """
        if self._dialogue_chars > 0:
            self._dialogue_bpcs.append(self._dialogue_bits / self._dialogue_chars)
        if self._boundary_chars > 0:
            self._boundary_bpcs.append(self._boundary_bits / self._boundary_chars)
        if self._steady_chars > 0:
            self._steady_bpcs.append(self._steady_bits / self._steady_chars)
        self._dialogue_bits = 0.0
        self._dialogue_chars = 0
        self._boundary_bits = 0.0
        self._boundary_chars = 0
        self._steady_bits = 0.0
        self._steady_chars = 0

    @property
    def dialogue_bpcs(self) -> list[float]:
        """BPC for each completed dialogue."""
        return self._dialogue_bpcs

    @property
    def boundary_bpcs(self) -> list[float]:
        """Mean BPC during first N chars of each dialogue (boundary spike)."""
        return self._boundary_bpcs

    @property
    def steady_bpcs(self) -> list[float]:
        """Mean BPC after boundary window settles (within-dialogue)."""
        return self._steady_bpcs

    def reset(self) -> None:
        """Reset all accumulators (e.g., for held-out evaluation)."""
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
