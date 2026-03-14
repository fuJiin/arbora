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

from step.decoders.dendritic import DendriticDecoder


class BPCProbe:
    """Accumulates bits-per-character from dendritic decoder scores."""

    def __init__(self, temperature: float = 1.0):
        self._temperature = temperature
        self._total_bits: float = 0.0
        self._n_chars: int = 0
        # Rolling window for trend tracking
        self._recent_bits: list[float] = []
        self._window_size: int = 500

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

        return bits

    def reset(self) -> None:
        """Reset all accumulators (e.g., for held-out evaluation)."""
        self._total_bits = 0.0
        self._n_chars = 0
        self._recent_bits.clear()
