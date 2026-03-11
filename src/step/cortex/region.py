"""Cortical region with L4 (input) and L2/3 (associative) layers.

Two-layer architecture modeled on neocortical minicolumns:
- L4 (input): receives feedforward drive, modulated by feedback context
- L2/3 (associative): activated via fixed intra-column wiring from L4

Activation uses two-level competition:
1. Top-k columns selected by strongest neuron score (feedforward + feedback)
2. One winning neuron per active column (highest voltage + excitability)
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

        # Per-neuron eligibility traces
        self.trace_l4 = np.zeros(self.n_l4_total)
        self.trace_l23 = np.zeros(self.n_l23_total)

    def step(self, column_drive: np.ndarray) -> np.ndarray:
        """Run one timestep given column-level feedforward drive.

        Args:
            column_drive: (n_columns,) feedforward activation per column.

        Returns:
            Array of global indices of active L4 neurons.
        """
        # 1. Decay voltages from previous step
        self.voltage_l4 *= self.voltage_decay

        # 2. Feedforward: column drive → all neurons in each column
        self.voltage_l4 += np.repeat(column_drive, self.n_l4)

        # 3. Feedback from previous activity → L4 voltage bonus
        self._apply_feedback()

        # 4. Score = voltage + excitability, then activate
        scores = self.voltage_l4 + self.excitability_l4
        active_l4_indices = self._activate(scores)

        # 5. Learn (previous traces x current activation)
        self._learn()

        # 6. Update eligibility traces for newly active neurons
        self._update_traces()

        # 7. Homeostatic excitability
        self._update_excitability()

        # 8. Refractory: reset voltage for active neurons
        self.voltage_l4[self.active_l4] = 0.0

        return active_l4_indices

    def _apply_feedback(self):
        """Add threshold-gated feedback voltage to L4 neurons."""
        if self.active_l23.any():
            fb = self.active_l23.astype(np.float64) @ self.fb_weights
            self.voltage_l4 += fb * (fb > self.fb_threshold)
        if self.active_l4.any():
            lat = self.active_l4.astype(np.float64) @ self.lateral_weights
            self.voltage_l4 += lat * (lat > self.fb_threshold)

    def _activate(self, scores: np.ndarray) -> np.ndarray:
        """Select top-k columns and winning neuron per column."""
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

        # L2/3: fixed 1-to-1 wiring from L4 (same index within column)
        self.active_l23[:] = False
        valid = winners_in_col < self.n_l23
        l23_indices = (
            top_cols[valid] * self.n_l23 + winners_in_col[valid]
        )
        self.active_l23[l23_indices] = True

        return global_winners

    def _learn(self):
        """Strengthen synapses where dest is active and source has trace."""
        active_l4_f = self.active_l4.astype(np.float64)

        # L2/3 → L4 feedback synapses
        self.fb_weights += (
            self.learning_rate
            * self.trace_l23[:, np.newaxis]
            * active_l4_f[np.newaxis, :]
        )

        # L4 → L4 lateral synapses
        self.lateral_weights += (
            self.learning_rate
            * self.trace_l4[:, np.newaxis]
            * active_l4_f[np.newaxis, :]
        )

        # Decay all synapses
        self.fb_weights *= self.synapse_decay
        self.lateral_weights *= self.synapse_decay

        np.clip(self.fb_weights, 0, 1, out=self.fb_weights)
        np.clip(self.lateral_weights, 0, 1, out=self.lateral_weights)

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
