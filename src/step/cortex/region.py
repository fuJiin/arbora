"""Cortical region with L4 (input) and L2/3 (associative) layers.

Two-layer architecture modeled on neocortical minicolumns:
- L4 (input): receives feedforward drive, modulated by feedback context
- L2/3 (associative): receives L4 feedforward + lateral context from
  other L2/3 neurons, enabling associative binding and pattern completion

Activation uses burst/precise distinction:
1. Before feedforward input, compute which neurons are "predicted"
   (feedback/lateral exceeds dendritic spike threshold)
2. Top-k columns selected by strongest feedforward + predicted score
3. Per active column: if a neuron was predicted → precise (only it fires).
   If none predicted → burst (all neurons fire, best-match gets trace).
4. Burst = surprise signal → stronger learning to rewire predictions.
   Precise = expected → reinforcing learning to strengthen predictions.
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
        voltage_decay: float = 0.5,
        eligibility_decay: float = 0.95,
        synapse_decay: float = 0.999,
        learning_rate: float = 0.05,
        max_excitability: float = 0.2,
        fb_boost_threshold: float = 0.3,
        fb_boost: float = 0.4,
        burst_learning_scale: float = 3.0,
        seed: int = 0,
    ):
        self.n_columns = n_columns
        self.n_l4 = n_l4
        self.n_l23 = n_l23
        self.k_columns = k_columns
        self.voltage_decay = voltage_decay
        self.eligibility_decay = eligibility_decay
        self.synapse_decay = synapse_decay
        self.learning_rate = learning_rate
        self.max_excitability = max_excitability
        self.fb_boost_threshold = fb_boost_threshold
        self.fb_boost = fb_boost
        self.burst_learning_scale = burst_learning_scale
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

        # Burst state (updated each step)
        self.bursting_columns = np.zeros(n_columns, dtype=np.bool_)
        self.predicted_l4 = np.zeros(self.n_l4_total, dtype=np.bool_)

        # Feedback weights (neuron-to-neuron)
        self.fb_weights = np.zeros((self.n_l23_total, self.n_l4_total))
        self.lateral_weights = np.zeros((self.n_l4_total, self.n_l4_total))
        # L2/3 lateral weights (associative binding across columns)
        self.l23_lateral_weights = np.zeros((self.n_l23_total, self.n_l23_total))

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
        self.bursting_columns[:] = False
        self.predicted_l4[:] = False

    def get_prediction(self, k: int) -> np.ndarray:
        """Predict which L4 neurons will fire next (read-only).

        Simulates voltage decay + feedback/lateral without feedforward
        input, returning the top-k neuron indices by predicted score.
        Excludes excitability so this measures learned prediction only.
        """
        v = self.voltage_l4 * self.voltage_decay

        if self.active_l23.any():
            fb = self.active_l23.astype(np.float64) @ self.fb_weights
            v += self.fb_boost * (fb > self.fb_boost_threshold)

        if self.active_l4.any():
            lat = self.active_l4.astype(np.float64) @ self.lateral_weights
            v += self.fb_boost * (lat > self.fb_boost_threshold)

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

        # 2. Compute predictive state BEFORE feedforward input.
        #    A neuron is "predicted" if feedback or lateral drive
        #    exceeds the dendritic spike threshold.
        self._compute_predictions()

        # 3. Feedforward: column drive -> all L4 neurons in each column
        self.voltage_l4 += np.repeat(column_drive, self.n_l4)

        # 4. Predicted neurons get a voltage boost (they're primed)
        self.voltage_l4[self.predicted_l4] += self.fb_boost

        # 5. Activate L4: top-k columns, then burst/precise per column
        scores_l4 = self.voltage_l4 + self.excitability_l4
        top_cols = self._select_columns(scores_l4)
        self._activate_l4_burst(top_cols, scores_l4)

        # 6. Activate L2/3: L4 feedforward + lateral context
        self._activate_l23(top_cols)

        # 7. Learn (previous traces x current activation, scaled by burst)
        self._learn()

        # 8. Update eligibility traces for newly active neurons
        self._update_traces()

        # 9. Homeostatic excitability (capped)
        self._update_excitability()

        # 10. Refractory: reset voltage for active neurons
        self.voltage_l4[self.active_l4] = 0.0
        self.voltage_l23[self.active_l23] = 0.0

        # 11. Clamp voltage (bounded membrane potential)
        np.clip(self.voltage_l4, 0.0, 1.0, out=self.voltage_l4)
        np.clip(self.voltage_l23, 0.0, 1.0, out=self.voltage_l23)

        return np.nonzero(self.active_l4)[0]

    def _compute_predictions(self):
        """Determine which L4 neurons are in predictive state.

        Uses feedback from L2/3 and lateral from L4 (previous step).
        A neuron is predicted if either signal exceeds the dendritic
        spike threshold. Computed before feedforward input arrives.
        """
        self.predicted_l4[:] = False

        if self.active_l23.any():
            fb = self.active_l23.astype(np.float64) @ self.fb_weights
            self.predicted_l4 |= fb > self.fb_boost_threshold

        if self.active_l4.any():
            lat = self.active_l4.astype(np.float64) @ self.lateral_weights
            self.predicted_l4 |= lat > self.fb_boost_threshold

    def _select_columns(self, scores: np.ndarray) -> np.ndarray:
        """Select top-k columns by max neuron score."""
        by_col = scores.reshape(self.n_columns, self.n_l4)
        col_scores = by_col.max(axis=1)

        if self.k_columns >= self.n_columns:
            return np.arange(self.n_columns)
        return np.argpartition(col_scores, -self.k_columns)[-self.k_columns :]

    def _activate_l4_burst(self, top_cols: np.ndarray, scores: np.ndarray):
        """Activate L4 neurons with burst/precise distinction.

        For each active column:
        - If any neuron was predicted: PRECISE — only the predicted
          neuron with highest score fires.
        - If no neuron predicted: BURST — all neurons in column fire,
          but only the highest-score neuron gets the eligibility trace.
        """
        self.active_columns[:] = False
        self.active_columns[top_cols] = True
        self.active_l4[:] = False
        self.bursting_columns[:] = False

        predicted_by_col = self.predicted_l4.reshape(self.n_columns, self.n_l4)
        scores_by_col = scores.reshape(self.n_columns, self.n_l4)

        for col in top_cols:
            start = col * self.n_l4
            end = start + self.n_l4
            col_predicted = predicted_by_col[col]

            if col_predicted.any():
                # PRECISE: only the best predicted neuron fires
                # Among predicted neurons, pick highest score
                col_scores = scores_by_col[col].copy()
                col_scores[~col_predicted] = -np.inf
                winner = col_scores.argmax()
                self.active_l4[start + winner] = True
            else:
                # BURST: all neurons in column fire
                self.active_l4[start:end] = True
                self.bursting_columns[col] = True

    def _activate_l23(self, top_cols: np.ndarray):
        """Activate L2/3 neurons in active columns.

        Precise columns: one L2/3 winner (feedforward from L4 + lateral).
        Bursting columns: all L2/3 neurons fire (propagate surprise).
        """
        # L4 -> L2/3 feedforward: base drive to all neurons in column
        for col in top_cols:
            start = col * self.n_l23
            self.voltage_l23[start : start + self.n_l23] += 0.5

        # Bonus for L2/3 neuron matching the L4 winner (precise only)
        for col in top_cols:
            if not self.bursting_columns[col]:
                l4_start = col * self.n_l4
                l4_winner = np.argmax(self.active_l4[l4_start : l4_start + self.n_l4])
                if l4_winner < self.n_l23:
                    self.voltage_l23[col * self.n_l23 + l4_winner] += 0.5

        # L2/3 lateral: previous L2/3 activity biases current selection
        if self.active_l23.any():
            lat = self.active_l23.astype(np.float64) @ self.l23_lateral_weights
            self.voltage_l23 += lat

        # Competitive selection per column
        self.active_l23[:] = False
        l23_scores = self.voltage_l23 + self.excitability_l23
        by_col = l23_scores.reshape(self.n_columns, self.n_l23)

        for col in top_cols:
            start = col * self.n_l23
            end = start + self.n_l23
            if self.bursting_columns[col]:
                # Burst: all L2/3 neurons fire
                self.active_l23[start:end] = True
            else:
                # Precise: one winner
                winner = by_col[col].argmax()
                self.active_l23[start + winner] = True

    def _learn(self):
        """Strengthen synapses where dest is active and source has trace.

        Learning rate is scaled by burst_learning_scale for neurons
        in bursting columns — stronger updates to rewire predictions
        for surprising inputs.
        """
        # Build per-neuron learning rate based on burst state
        lr_l4 = np.full(self.n_l4_total, self.learning_rate)
        lr_l23 = np.full(self.n_l23_total, self.learning_rate)
        for col in np.nonzero(self.bursting_columns)[0]:
            l4_start = col * self.n_l4
            l23_start = col * self.n_l23
            lr_l4[l4_start : l4_start + self.n_l4] *= self.burst_learning_scale
            lr_l23[l23_start : l23_start + self.n_l23] *= self.burst_learning_scale

        active_l4_f = self.active_l4.astype(np.float64)
        active_l23_f = self.active_l23.astype(np.float64)

        # L2/3 -> L4 feedback synapses
        # Scale by destination (L4) learning rate
        self.fb_weights += (
            self.trace_l23[:, np.newaxis] * (lr_l4 * active_l4_f)[np.newaxis, :]
        )

        # L4 -> L4 lateral synapses
        self.lateral_weights += (
            self.trace_l4[:, np.newaxis] * (lr_l4 * active_l4_f)[np.newaxis, :]
        )

        # L2/3 -> L2/3 lateral synapses
        self.l23_lateral_weights += (
            self.trace_l23[:, np.newaxis] * (lr_l23 * active_l23_f)[np.newaxis, :]
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
        """Set active neuron traces to 1, decay the rest.

        For bursting columns, only the highest-score neuron gets the
        trace — prevents smearing learning across all burst neurons.
        """
        self.trace_l4 *= self.eligibility_decay
        self.trace_l23 *= self.eligibility_decay

        # Precise columns: all active neurons get trace (just 1 per col)
        # Burst columns: only best-match neuron gets trace
        for col in np.nonzero(self.active_columns)[0]:
            if self.bursting_columns[col]:
                # Pick the neuron with highest voltage as trace winner
                l4_start = col * self.n_l4
                l4_end = l4_start + self.n_l4
                best = l4_start + np.argmax(self.voltage_l4[l4_start:l4_end])
                self.trace_l4[best] = 1.0

                l23_start = col * self.n_l23
                l23_end = l23_start + self.n_l23
                best_l23 = l23_start + np.argmax(self.voltage_l23[l23_start:l23_end])
                self.trace_l23[best_l23] = 1.0
            else:
                # Precise: the single active neuron gets the trace
                l4_start = col * self.n_l4
                l4_end = l4_start + self.n_l4
                active_in_col = self.active_l4[l4_start:l4_end]
                if active_in_col.any():
                    self.trace_l4[l4_start + active_in_col.argmax()] = 1.0

                l23_start = col * self.n_l23
                l23_end = l23_start + self.n_l23
                active_l23 = self.active_l23[l23_start:l23_end]
                if active_l23.any():
                    self.trace_l23[l23_start + active_l23.argmax()] = 1.0

    def _update_excitability(self):
        """Boost inactive neurons, reset active ones (capped).

        Increment is max_excitability / n_l4, so it takes n_l4 steps
        for an inactive neuron to reach full excitability — enough for
        one complete rotation through all neurons in a column.
        """
        inc_l4 = self.max_excitability / self.n_l4
        inc_l23 = self.max_excitability / self.n_l23
        self.excitability_l4[~self.active_l4] += inc_l4
        self.excitability_l23[~self.active_l23] += inc_l23
        self.excitability_l4[self.active_l4] = 0.0
        self.excitability_l23[self.active_l23] = 0.0
        np.clip(
            self.excitability_l4,
            0,
            self.max_excitability,
            out=self.excitability_l4,
        )
        np.clip(
            self.excitability_l23,
            0,
            self.max_excitability,
            out=self.excitability_l23,
        )
