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
    - Babbling mode: noise injection for motor exploration
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

        # Babbling mode: mix random sparse drive with normal feedforward.
        # 0.0 = normal (no noise), 1.0 = pure random babbling.
        # Models brainstem pattern generators in early infant vocalization.
        self.babbling_noise: float = 0.0

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
        """Feedforward + compute L5 output scores.

        When babbling_noise > 0, forces random column activations directly,
        bypassing ff_weights and k-WTA competition. This ensures M1 explores
        its full output space uniformly rather than collapsing to dominant
        patterns. Models brainstem pattern generators driving motor cortex.
        """
        if self.babbling_noise >= 1.0:
            active = self._babble_direct(encoding)
        elif self.babbling_noise > 0.0:
            # Partial noise: mix random columns with normal processing
            if self._rng.random() < self.babbling_noise:
                active = self._babble_direct(encoding)
            else:
                active = super().process(encoding)
        else:
            active = super().process(encoding)

        # L5 readout: per-column mean L2/3 firing rate
        rates = self.firing_rate_l23.reshape(self.n_columns, self.n_l23)
        self.output_scores = rates.mean(axis=1)

        return active

    def _babble_direct(self, encoding: np.ndarray | None = None) -> np.ndarray:
        """Force random column activations while training ff_weights.

        Randomly selects k columns (ensuring diverse exploration), then
        runs normal L2/3 activation and learning. Crucially, also trains
        ff_weights via _learn_ff so the sensorimotor mapping develops —
        M1 learns "when S1 pattern is X, I should activate columns Y."

        Args:
            encoding: The real feedforward input (S1 L2/3). If provided,
                      ff_weights are trained to map this input to the
                      forced columns. Builds the forward model.
        """
        # Pick random k columns
        cols = self._rng.choice(self.n_columns, self.k_columns, replace=False)

        # Run the standard step pipeline but inject our columns
        # 1. Decay voltages
        self.voltage_l4 *= self.voltage_decay
        self.voltage_l23 *= self.voltage_decay

        # 2. Compute predictions (for segment learning)
        self._compute_predictions()

        # 3. Save prediction context
        self._pred_context_l23[:] = self.active_l23
        self._pred_context_l4[:] = self.active_l4

        # 4-6. Force our chosen columns active
        for col in cols:
            start = col * self.n_l4
            end = start + self.n_l4
            self.voltage_l4[start:end] = 1.0

        scores_l4 = self.voltage_l4 + self.excitability_l4
        self._activate_l4_burst(cols, scores_l4)

        # 7. Activate L2/3
        self._activate_l23(cols)

        # 8. Learn — segments AND ff_weights
        if self.learning_enabled:
            self._learn()
            # Train ff_weights: learn to map S1 input → forced columns.
            # This builds the sensorimotor forward model.
            if encoding is not None:
                flat = encoding.flatten().astype(np.float64)
                self._learn_ff(flat)

        # 9-13. Standard housekeeping
        self._update_traces()
        self._update_excitability()
        self.voltage_l4[self.active_l4] = 0.0
        self.voltage_l23[self.active_l23] = 0.0
        np.clip(self.voltage_l4, 0.0, 1.0, out=self.voltage_l4)
        np.clip(self.voltage_l23, 0.0, 1.0, out=self.voltage_l23)
        self.firing_rate_l23 *= self.voltage_decay
        self.firing_rate_l23[self.active_l23] += 1.0 - self.voltage_decay

        return np.nonzero(self.active_l4)[0]

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
