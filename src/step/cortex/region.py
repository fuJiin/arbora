"""Cortical region with L4 (input) and L2/3 (associative) layers.

Two-layer architecture modeled on neocortical minicolumns:
- L4 (input): receives feedforward drive, modulated by feedback context
- L2/3 (associative): receives L4 feedforward + lateral context from
  other L2/3 neurons, enabling associative binding and pattern completion

Activation uses two-level competition:
1. Top-k columns selected by strongest L4 neuron score
2. One winning L4 neuron per active column (voltage + excitability)
3. One winning L2/3 neuron per active column (L4 bias + lateral + excitability)
"""

import numpy as np


class CorticalRegion:
    def __init__(
        self,
        n_columns: int,
        n_l4: int,
        n_l23: int,
        k_columns: int,
        *,
        fb_threshold: float = 0.5,
        voltage_decay: float = 0.9,
        eligibility_decay: float = 0.95,
        synapse_decay: float = 0.999,
        learning_rate: float = 0.01,
        seed: int = 0,
    ):
        self.n_columns = n_columns
        self.n_l4 = n_l4
        self.n_l23 = n_l23
        self.k_columns = k_columns
        self.fb_threshold = fb_threshold
        self.voltage_decay = voltage_decay
        self.eligibility_decay = eligibility_decay
        self.synapse_decay = synapse_decay
        self.learning_rate = learning_rate
        self._rng = np.random.default_rng(seed)

        self.n_l4_total = n_columns * n_l4
        self.n_l23_total = n_columns * n_l23

        # Per-neuron state
        self.voltage_l4 = np.zeros(self.n_l4_total)
        self.voltage_l23 = np.zeros(self.n_l23_total)
        self.excitability_l4 = np.zeros(self.n_l4_total)
        self.excitability_l23 = np.zeros(self.n_l23_total)

        # Active masks (updated each step)
        self.active_l4 = np.zeros(self.n_l4_total, dtype=np.bool_)
        self.active_l23 = np.zeros(self.n_l23_total, dtype=np.bool_)
        self.active_columns = np.zeros(n_columns, dtype=np.bool_)

        # Feedback weights (neuron-to-neuron)
        self.fb_weights = np.zeros((self.n_l23_total, self.n_l4_total))
        self.lateral_weights = np.zeros(
            (self.n_l4_total, self.n_l4_total)
        )
        # L2/3 lateral weights (associative binding across columns)
        self.l23_lateral_weights = np.zeros(
            (self.n_l23_total, self.n_l23_total)
        )

        # Per-neuron eligibility traces
        self.trace_l4 = np.zeros(self.n_l4_total)
        self.trace_l23 = np.zeros(self.n_l23_total)

    def reset_working_memory(self):
        """Reset transient state, preserving learned synaptic weights."""
        self.voltage_l4[:] = 0.0
        self.voltage_l23[:] = 0.0
        self.trace_l4[:] = 0.0
        self.trace_l23[:] = 0.0
        self.excitability_l4[:] = 0.0
        self.excitability_l23[:] = 0.0
        self.active_l4[:] = False
        self.active_l23[:] = False
        self.active_columns[:] = False

    def get_prediction(self, k: int) -> np.ndarray:
        """Predict which L4 neurons will fire next (read-only).

        Simulates voltage decay + feedback/lateral without feedforward
        input, returning the top-k neuron indices by predicted score.
        Excludes excitability so this measures learned prediction only.
        """
        v = self.voltage_l4 * self.voltage_decay

        if self.active_l23.any():
            fb = self.active_l23.astype(np.float64) @ self.fb_weights
            v += fb * (fb > self.fb_threshold)

        if self.active_l4.any():
            lat = self.active_l4.astype(np.float64) @ self.lateral_weights
            v += lat * (lat > self.fb_threshold)

        if k >= len(v):
            return np.arange(len(v))
        return np.argpartition(v, -k)[-k:]

    def step(self, column_drive: np.ndarray) -> np.ndarray:
        """Run one timestep given column-level feedforward drive.

        Args:
            column_drive: (n_columns,) feedforward activation per column.

        Returns:
            Array of global indices of active L4 neurons.
        """
        # 1. Decay voltages
        self.voltage_l4 *= self.voltage_decay
        self.voltage_l23 *= self.voltage_decay

        # 2. Feedforward: column drive -> all L4 neurons in each column
        self.voltage_l4 += np.repeat(column_drive, self.n_l4)

        # 3. Feedback from previous activity -> L4 voltage bonus
        self._apply_feedback()

        # 4. Activate L4: top-k columns, winner per column
        scores_l4 = self.voltage_l4 + self.excitability_l4
        top_cols, l4_winners_in_col = self._activate_l4(scores_l4)

        # 5. Activate L2/3: L4 feedforward + lateral context
        self._activate_l23(top_cols, l4_winners_in_col)

        # 6. Learn (previous traces x current activation)
        self._learn()

        # 7. Update eligibility traces for newly active neurons
        self._update_traces()

        # 8. Homeostatic excitability
        self._update_excitability()

        # 9. Refractory: reset voltage for active neurons
        self.voltage_l4[self.active_l4] = 0.0
        self.voltage_l23[self.active_l23] = 0.0

        return top_cols * self.n_l4 + l4_winners_in_col

    def _apply_feedback(self):
        """Add threshold-gated feedback voltage to L4 neurons."""
        if self.active_l23.any():
            fb = self.active_l23.astype(np.float64) @ self.fb_weights
            self.voltage_l4 += fb * (fb > self.fb_threshold)
        if self.active_l4.any():
            lat = self.active_l4.astype(np.float64) @ self.lateral_weights
            self.voltage_l4 += lat * (lat > self.fb_threshold)

    def _activate_l4(
        self, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select top-k columns and winning L4 neuron per column.

        Returns (top_cols, winners_in_col) index arrays.
        """
        by_col = scores.reshape(self.n_columns, self.n_l4)
        col_scores = by_col.max(axis=1)

        if self.k_columns >= self.n_columns:
            top_cols = np.arange(self.n_columns)
        else:
            top_cols = np.argpartition(
                col_scores, -self.k_columns
            )[-self.k_columns :]

        winners_in_col = by_col[top_cols].argmax(axis=1)
        global_winners = top_cols * self.n_l4 + winners_in_col

        self.active_columns[:] = False
        self.active_columns[top_cols] = True

        self.active_l4[:] = False
        self.active_l4[global_winners] = True

        return top_cols, winners_in_col

    def _activate_l23(
        self,
        top_cols: np.ndarray,
        l4_winners_in_col: np.ndarray,
    ):
        """Activate L2/3 neurons in active columns.

        All L2/3 neurons in active columns receive base feedforward
        drive from L4. The L2/3 neuron matching the L4 winner gets a
        bonus (biologically: denser intra-column connectivity at the
        same position). Lateral context from previous L2/3 activity
        can override this bias.
        """
        # L4 -> L2/3 feedforward: base drive to all neurons in column
        for col in top_cols:
            start = col * self.n_l23
            self.voltage_l23[start : start + self.n_l23] += 0.5

        # Bonus for L2/3 neuron matching the L4 winner
        valid = l4_winners_in_col < self.n_l23
        matching = (
            top_cols[valid] * self.n_l23 + l4_winners_in_col[valid]
        )
        self.voltage_l23[matching] += 0.5

        # L2/3 lateral: previous L2/3 activity biases current selection
        if self.active_l23.any():
            lat = (
                self.active_l23.astype(np.float64)
                @ self.l23_lateral_weights
            )
            self.voltage_l23 += lat

        # Competitive selection: one winner per active column
        l23_scores = self.voltage_l23 + self.excitability_l23
        by_col = l23_scores.reshape(self.n_columns, self.n_l23)
        winners_in_col = by_col[top_cols].argmax(axis=1)
        l23_winners = top_cols * self.n_l23 + winners_in_col

        self.active_l23[:] = False
        self.active_l23[l23_winners] = True

    def _learn(self):
        """Strengthen synapses where dest is active and source has trace."""
        active_l4_f = self.active_l4.astype(np.float64)
        active_l23_f = self.active_l23.astype(np.float64)

        # L2/3 -> L4 feedback synapses
        self.fb_weights += (
            self.learning_rate
            * self.trace_l23[:, np.newaxis]
            * active_l4_f[np.newaxis, :]
        )

        # L4 -> L4 lateral synapses
        self.lateral_weights += (
            self.learning_rate
            * self.trace_l4[:, np.newaxis]
            * active_l4_f[np.newaxis, :]
        )

        # L2/3 -> L2/3 lateral synapses
        self.l23_lateral_weights += (
            self.learning_rate
            * self.trace_l23[:, np.newaxis]
            * active_l23_f[np.newaxis, :]
        )

        # Decay all synapses
        self.fb_weights *= self.synapse_decay
        self.lateral_weights *= self.synapse_decay
        self.l23_lateral_weights *= self.synapse_decay

        np.clip(self.fb_weights, 0, 1, out=self.fb_weights)
        np.clip(self.lateral_weights, 0, 1, out=self.lateral_weights)
        np.clip(
            self.l23_lateral_weights,
            0,
            1,
            out=self.l23_lateral_weights,
        )

    def _update_traces(self):
        """Set active neuron traces to 1, decay the rest."""
        self.trace_l4 *= self.eligibility_decay
        self.trace_l23 *= self.eligibility_decay
        self.trace_l4[self.active_l4] = 1.0
        self.trace_l23[self.active_l23] = 1.0

    def _update_excitability(self):
        """Boost inactive neurons, reset active ones."""
        self.excitability_l4 += ~self.active_l4
        self.excitability_l23 += ~self.active_l23
        self.excitability_l4[self.active_l4] = 0.0
        self.excitability_l23[self.active_l23] = 0.0
