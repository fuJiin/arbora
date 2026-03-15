"""Motor cortical region: maps sensory context to output predictions.

Subclasses SensoryRegion (encoding_width=0, full connectivity) and adds
an L5-like output layer with confidence gating. Columns self-organize
to represent output tokens via Hebbian learning — same cortical algorithm
as sensory regions, different wiring.

The column→token mapping is learned: each column tracks which token_id
most frequently activates it, forming a self-organizing motor map.
"""

import numpy as np

from step.cortex.sensory import SensoryRegion


class MotorRegion(SensoryRegion):
    """Cortical region with L5 output gating for token prediction.

    Like S2, receives S1's L2/3 firing rate as input (encoding_width=0).
    Unlike S2, adds:
    - Per-column L5 readout score (mean L2/3 firing rate in column)
    - Confidence threshold: below → silent (no prediction)
    - Self-organizing column→token mapping via activation frequency
    """

    def __init__(
        self,
        input_dim: int,
        *,
        n_columns: int = 32,
        output_threshold: float = 0.3,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            encoding_width=0,
            n_columns=n_columns,
            seed=seed,
            **kwargs,
        )
        self.output_threshold = output_threshold

        # L5 output state (updated each step in process())
        self.output_scores = np.zeros(n_columns)

        # Self-organizing column→token mapping
        # _col_token_counts[col][token_id] = activation count
        self._col_token_counts: list[dict[int, int]] = [
            {} for _ in range(n_columns)
        ]
        # Cached best mapping: col → token_id (-1 = unassigned)
        self._col_token_map = np.full(n_columns, -1, dtype=np.int64)

        # Last step output (set by topology run loop after BG gating)
        self.last_output: tuple[int, float] = (-1, 0.0)
        self.last_gate: float = 0.5
        self.last_reward: float = 0.0

    def process(self, encoding: np.ndarray) -> np.ndarray:
        """Feedforward + compute L5 output scores."""
        active = super().process(encoding)

        # L5 readout: per-column mean L2/3 firing rate
        rates = self.firing_rate_l23.reshape(self.n_columns, self.n_l23)
        self.output_scores = rates.mean(axis=1)

        return active

    def observe_token(self, token_id: int) -> None:
        """Update column→token mapping based on current activation.

        Called each step with the actual token_id. Active columns
        accumulate counts for this token, building the motor map.
        """
        for col in np.nonzero(self.active_columns)[0]:
            counts = self._col_token_counts[col]
            counts[token_id] = counts.get(token_id, 0) + 1
            self._col_token_map[col] = max(counts, key=counts.get)

    def get_output(self) -> tuple[int, float]:
        """Return (predicted_token_id, confidence).

        Returns (-1, 0.0) if no column is above threshold or if the
        best column has no token assignment yet.
        """
        above = self.output_scores >= self.output_threshold
        if not above.any():
            return (-1, 0.0)

        masked = self.output_scores.copy()
        masked[~above] = -1.0
        best_col = int(masked.argmax())
        confidence = float(self.output_scores[best_col])

        token_id = int(self._col_token_map[best_col])
        if token_id < 0:
            return (-1, 0.0)

        return (token_id, confidence)

    def get_population_output(self) -> tuple[int, float]:
        """Return (predicted_token_id, confidence) via population vote.

        All active columns above threshold vote for their most-frequent
        token. Votes weighted by column output score (L5 readout).
        Models L5 population coding: the motor command is encoded by
        the *pattern* of active columns, not any single column.

        Returns (-1, 0.0) if no column is above threshold or no votes.
        """
        above = self.output_scores >= self.output_threshold
        if not above.any():
            return (-1, 0.0)

        votes: dict[int, float] = {}
        for col in np.nonzero(above)[0]:
            token_id = int(self._col_token_map[col])
            if token_id < 0:
                continue
            votes[token_id] = (
                votes.get(token_id, 0.0)
                + float(self.output_scores[col])
            )

        if not votes:
            return (-1, 0.0)

        best_token = max(votes, key=votes.get)
        confidence = float(self.output_scores[above].max())
        return (best_token, confidence)

    def get_decoded_output(self, decoder) -> tuple[int, float]:
        """Return (predicted_token_id, confidence) using a dendritic decoder.

        L5 readout + BG gating decides IF M1 speaks (output_scores >= threshold).
        The decoder decides WHAT M1 says (replaces column→token frequency map).
        """
        above = self.output_scores >= self.output_threshold
        if not above.any():
            return (-1, 0.0)

        predictions = decoder.decode(self.active_l23, k=1)
        if not predictions:
            return (-1, 0.0)

        confidence = float(self.output_scores[above].max())
        return (predictions[0], confidence)

    def get_output_distribution(self) -> np.ndarray:
        """Return per-column output scores (for visualization)."""
        return self.output_scores.copy()

    def reset_working_memory(self):
        """Reset transient state, preserving learned mappings."""
        super().reset_working_memory()
        self.output_scores[:] = 0.0
