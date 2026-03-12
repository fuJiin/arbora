"""Cortical region with L4 (input) and L2/3 (associative) layers.

Two-layer architecture modeled on neocortical minicolumns:
- L4 (input): receives feedforward drive, modulated by feedback context
- L2/3 (associative): receives L4 feedforward + lateral context from
  other L2/3 neurons, enabling associative binding and pattern completion

Prediction uses dendritic segments — each L4 neuron has multiple
short dendritic branches that each recognize a specific pattern of
source neuron activity. A segment fires when enough of its connected
synapses have active sources, predicting the neuron. Two segment types:
- Feedback segments (L2/3 → L4): context from associative layer
- Lateral segments (L4 → L4): context from same-layer temporal patterns

Activation uses burst/precise distinction:
1. Before feedforward input, check dendritic segments for predicted neurons
2. Top-k columns selected by strongest feedforward + predicted score
3. Per active column: if a neuron was predicted → precise (only it fires).
   If none predicted → burst (all neurons fire, best-match gets trace).
4. Burst = surprise signal → grow new segment connections.
   Precise = expected → reinforce segment connections.
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
        fb_boost: float = 0.4,
        burst_learning_scale: float = 3.0,
        # Dendritic segment parameters
        n_fb_segments: int = 4,
        n_lat_segments: int = 4,
        n_synapses_per_segment: int = 24,
        perm_threshold: float = 0.5,
        perm_init: float = 0.6,
        perm_increment: float = 0.2,
        perm_decrement: float = 0.05,
        seg_activation_threshold: int = 2,
        prediction_gain: float = 1.0,
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
        self.fb_boost = fb_boost
        self.burst_learning_scale = burst_learning_scale
        self.n_fb_segments = n_fb_segments
        self.n_lat_segments = n_lat_segments
        self.n_synapses_per_segment = n_synapses_per_segment
        self.perm_threshold = perm_threshold
        self.perm_init = perm_init
        self.perm_increment = perm_increment
        self.perm_decrement = perm_decrement
        self.seg_activation_threshold = seg_activation_threshold
        self.prediction_gain = prediction_gain
        self._rng = np.random.default_rng(seed)

        # Third-factor neuromodulatory signal (set externally each step).
        # Scales learning rates: 1.0 = normal, >1 = surprise boosts learning.
        self.surprise_modulator: float = 1.0

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

        # L2/3 firing rate estimate (EMA of boolean activations).
        # Models postsynaptic temporal integration of spike trains.
        self.firing_rate_l23 = np.zeros(self.n_l23_total)

        # L2/3 lateral weights (associative binding across columns)
        self.l23_lateral_weights = np.zeros((self.n_l23_total, self.n_l23_total))

        # Per-neuron eligibility traces
        self.trace_l4 = np.zeros(self.n_l4_total)
        self.trace_l23 = np.zeros(self.n_l23_total)

        # Prediction-time context (saved for segment learning)
        self._pred_context_l23 = np.zeros(self.n_l23_total, dtype=np.bool_)
        self._pred_context_l4 = np.zeros(self.n_l4_total, dtype=np.bool_)

        # Initialize dendritic segments
        self._init_segments()

    def _init_segments(self):
        """Initialize dendritic segment arrays with random connectivity."""
        n = self.n_l4_total
        n_syn = self.n_synapses_per_segment

        # Feedback segments: L2/3 → L4
        self.fb_seg_indices = np.zeros(
            (n, self.n_fb_segments, n_syn), dtype=np.int32
        )
        self.fb_seg_perm = np.zeros((n, self.n_fb_segments, n_syn))

        # Lateral segments: L4 → L4
        self.lat_seg_indices = np.zeros(
            (n, self.n_lat_segments, n_syn), dtype=np.int32
        )
        self.lat_seg_perm = np.zeros((n, self.n_lat_segments, n_syn))

        fb_pool = np.arange(self.n_l23_total)
        lat_pool = np.arange(self.n_l4_total)

        for i in range(n):
            for s in range(self.n_fb_segments):
                self.fb_seg_indices[i, s] = self._rng.choice(
                    fb_pool, n_syn, replace=len(fb_pool) < n_syn
                )
            for s in range(self.n_lat_segments):
                self.lat_seg_indices[i, s] = self._rng.choice(
                    lat_pool, n_syn, replace=len(lat_pool) < n_syn
                )

    def predict_neuron(
        self, l4_idx: int, source_idx: int, segment_type: str = "fb"
    ):
        """Set up a dendritic segment that fires when source_idx is active.

        For testing. Fills all synapses in segment 0 with the given source
        index and sets permanences to 1.0, guaranteeing the segment fires
        whenever the source neuron is active.
        """
        if segment_type == "fb":
            self.fb_seg_indices[l4_idx, 0, :] = source_idx
            self.fb_seg_perm[l4_idx, 0, :] = 1.0
        else:
            self.lat_seg_indices[l4_idx, 0, :] = source_idx
            self.lat_seg_perm[l4_idx, 0, :] = 1.0

    def _get_source_pool(self, neuron: int, seg_type: str) -> np.ndarray:
        """Get valid source neuron indices for growing synapses."""
        if seg_type == "fb":
            return np.arange(self.n_l23_total)
        return np.arange(self.n_l4_total)

    def reset_working_memory(self):
        """Reset transient state, preserving learned synaptic weights and segments."""
        self.voltage_l4[:] = 0.0
        self.voltage_l23[:] = 0.0
        self.firing_rate_l23[:] = 0.0
        self.trace_l4[:] = 0.0
        self.trace_l23[:] = 0.0
        self.excitability_l4[:] = 0.0
        self.excitability_l23[:] = 0.0
        self.active_l4[:] = False
        self.active_l23[:] = False
        self.active_columns[:] = False
        self.bursting_columns[:] = False
        self.predicted_l4[:] = False
        self._pred_context_l23[:] = False
        self._pred_context_l4[:] = False

    def _predict_from_segments(self) -> np.ndarray:
        """Check which L4 neurons have active dendritic segments.

        A segment is active when >= seg_activation_threshold of its
        connected synapses (permanence > perm_threshold) have active
        source neurons. A neuron is predicted if any segment is active.

        Returns boolean mask of shape (n_l4_total,).
        """
        predicted = np.zeros(self.n_l4_total, dtype=np.bool_)

        if self.active_l23.any():
            active_at_syn = self.active_l23[self.fb_seg_indices]
            connected = self.fb_seg_perm > self.perm_threshold
            counts = (active_at_syn & connected).sum(axis=2)
            predicted |= (counts >= self.seg_activation_threshold).any(axis=1)

        if self.active_l4.any():
            active_at_syn = self.active_l4[self.lat_seg_indices]
            connected = self.lat_seg_perm > self.perm_threshold
            counts = (active_at_syn & connected).sum(axis=2)
            predicted |= (counts >= self.seg_activation_threshold).any(axis=1)

        return predicted

    def get_prediction(self, k: int) -> np.ndarray:
        """Return predicted L4 neuron indices via dendritic segments.

        The k parameter is accepted for API compatibility but ignored —
        segment prediction is binary (predicted or not), not top-k.
        """
        return np.nonzero(self._predict_from_segments())[0]

    def step(self, drive: np.ndarray) -> np.ndarray:
        """Run one timestep given per-neuron feedforward drive.

        Args:
            drive: (n_l4_total,) per-neuron feedforward drive.

        Returns:
            Array of global indices of active L4 neurons.
        """
        # 1. Decay voltages
        self.voltage_l4 *= self.voltage_decay
        self.voltage_l23 *= self.voltage_decay

        # 2. Compute predictive state BEFORE feedforward input.
        self._compute_predictions()

        # 3. Save prediction-time context for segment learning
        #    (current active state is from the previous step)
        self._pred_context_l23[:] = self.active_l23
        self._pred_context_l4[:] = self.active_l4

        # 4. Feedforward drive to L4 neurons
        self.voltage_l4 += drive

        # 5a. Thalamic gating: amplify drive for predicted columns.
        #     Models thalamocortical feedback (L6→thalamus) that modulates
        #     relay gain, biasing column competition toward predicted columns.
        if self.prediction_gain > 1.0 and self.predicted_l4.any():
            pred_by_col = self.predicted_l4.reshape(self.n_columns, self.n_l4)
            pred_cols = pred_by_col.any(axis=1)
            # Broadcast column mask to per-neuron gain
            gain_mask = np.repeat(pred_cols, self.n_l4)
            self.voltage_l4[gain_mask] *= self.prediction_gain

        # 5b. Predicted neurons get a voltage boost (they're primed)
        self.voltage_l4[self.predicted_l4] += self.fb_boost

        # 6. Activate L4: top-k columns, then burst/precise per column
        scores_l4 = self.voltage_l4 + self.excitability_l4
        top_cols = self._select_columns(scores_l4)
        self._activate_l4_burst(top_cols, scores_l4)

        # 7. Activate L2/3: L4 feedforward + lateral context
        self._activate_l23(top_cols)

        # 8. Learn (L2/3 lateral Hebbian + segment permanence updates)
        self._learn()

        # 9. Update eligibility traces for newly active neurons
        self._update_traces()

        # 10. Homeostatic excitability (capped)
        self._update_excitability()

        # 11. Refractory: reset voltage for active neurons
        self.voltage_l4[self.active_l4] = 0.0
        self.voltage_l23[self.active_l23] = 0.0

        # 12. Clamp voltage (bounded membrane potential)
        np.clip(self.voltage_l4, 0.0, 1.0, out=self.voltage_l4)
        np.clip(self.voltage_l23, 0.0, 1.0, out=self.voltage_l23)

        # 13. Update L2/3 firing rate estimate (EMA of spike train)
        self.firing_rate_l23 *= self.voltage_decay
        self.firing_rate_l23[self.active_l23] += 1.0 - self.voltage_decay

        return np.nonzero(self.active_l4)[0]

    def _compute_predictions(self):
        """Determine which L4 neurons are in predictive state via segments."""
        self.predicted_l4 = self._predict_from_segments()

    def _select_columns(self, scores: np.ndarray) -> np.ndarray:
        """Select top-k columns by max neuron score."""
        by_col = scores.reshape(self.n_columns, self.n_l4)
        col_scores = by_col.max(axis=1)

        if self.k_columns >= self.n_columns:
            return np.arange(self.n_columns)
        return np.argpartition(col_scores, -self.k_columns)[-self.k_columns :]

    def _activate_l4_burst(self, top_cols: np.ndarray, scores: np.ndarray):
        """Activate L4 neurons with burst/precise distinction.

        For each winning column:
        - If any neuron was predicted by dendritic segments → precise activation:
          only the best-scoring predicted neuron fires.
        - If no neuron was predicted → burst: all neurons fire (surprise signal).
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
                col_scores = scores_by_col[col].copy()
                col_scores[~col_predicted] = -np.inf
                winner = col_scores.argmax()
                self.active_l4[start + winner] = True
            else:
                # BURST: all neurons in column fire
                self.active_l4[start:end] = True
                self.bursting_columns[col] = True

    def _activate_l23(self, top_cols: np.ndarray):
        """Activate L2/3 associative neurons in active columns.

        L2/3 receives three sources of drive:
        1. L4 feedforward: base drive to all neurons in the column
        2. L4 winner bonus: precise columns boost the matching L2/3 neuron
        3. L2/3 lateral: previous L2/3 activity biases current selection

        Then competitive selection: burst columns → all fire,
        precise columns → single winner.
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
        """L2/3 lateral Hebbian learning + dendritic segment updates.

        L2/3 lateral weights use dense Hebbian learning (no segments yet).
        L4 prediction is handled entirely by dendritic segments.
        """
        active_l23_f = self.active_l23.astype(np.float64)
        lr_l23 = np.full(self.n_l23_total, self.learning_rate * self.surprise_modulator)
        for col in np.nonzero(self.bursting_columns)[0]:
            l23_start = col * self.n_l23
            lr_l23[l23_start : l23_start + self.n_l23] *= self.burst_learning_scale

        # L2/3 -> L2/3 lateral: Hebbian with trace
        self.l23_lateral_weights += (
            self.trace_l23[:, np.newaxis] * (lr_l23 * active_l23_f)[np.newaxis, :]
        )
        self.l23_lateral_weights *= self.synapse_decay
        np.clip(self.l23_lateral_weights, 0, 1, out=self.l23_lateral_weights)

        # Dendritic segment learning (prediction)
        self._learn_segments()

    def _learn_segments(self):
        """Update dendritic segment permanences based on prediction outcomes.

        - Burst (unpredicted): grow best-matching segment on trace winner
        - Precise + predicted: reinforce the active segments
        - Predicted but didn't fire: punish the active segments
        """
        for col in np.nonzero(self.active_columns)[0]:
            l4_start = col * self.n_l4
            l4_end = l4_start + self.n_l4

            if self.bursting_columns[col]:
                # Grow segment on trace winner (highest voltage neuron)
                best = l4_start + np.argmax(self.voltage_l4[l4_start:l4_end])
                self._grow_best_segment(best)
            else:
                # Reinforce segments on precisely-predicted neurons
                for neuron in range(l4_start, l4_end):
                    if self.active_l4[neuron] and self.predicted_l4[neuron]:
                        self._adapt_segments(neuron, reinforce=True)

        # Punish false predictions (predicted but didn't fire)
        false_predicted = self.predicted_l4 & ~self.active_l4
        for neuron in np.nonzero(false_predicted)[0]:
            self._adapt_segments(neuron, reinforce=False)

    def _find_best_segment(
        self, neuron: int
    ) -> tuple[str | None, int, int]:
        """Find the segment with most overlap with prediction-time context.

        Returns (seg_type, seg_index, overlap_count) or (None, 0, 0).
        """
        best_overlap = -1
        best_type = None
        best_seg_idx = 0

        for seg_type, seg_indices, ctx in [
            ("fb", self.fb_seg_indices, self._pred_context_l23),
            ("lat", self.lat_seg_indices, self._pred_context_l4),
        ]:
            if not ctx.any():
                continue
            for s in range(seg_indices.shape[1]):
                overlap = int(ctx[seg_indices[neuron, s]].sum())
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_type = seg_type
                    best_seg_idx = s

        return best_type, best_seg_idx, max(best_overlap, 0)

    def _grow_best_segment(self, neuron: int):
        """Grow the best-matching segment for a bursting neuron.

        Finds the segment with most overlap with prediction-time context,
        then strengthens matching synapses and replaces weakest non-matching
        ones with new connections to active source neurons.
        """
        best_type, best_seg_idx, best_overlap = self._find_best_segment(neuron)
        if best_type is None or best_overlap == 0:
            return

        if best_type == "fb":
            seg_indices = self.fb_seg_indices
            seg_perm = self.fb_seg_perm
            ctx = self._pred_context_l23
        else:
            seg_indices = self.lat_seg_indices
            seg_perm = self.lat_seg_perm
            ctx = self._pred_context_l4

        idx = seg_indices[neuron, best_seg_idx].copy()
        perm = seg_perm[neuron, best_seg_idx].copy()
        syn_active = ctx[idx]

        # Strengthen active synapses, weaken inactive (modulated by surprise)
        inc = self.perm_increment * self.surprise_modulator
        dec = self.perm_decrement * self.surprise_modulator
        perm[syn_active] = np.minimum(perm[syn_active] + inc, 1.0)
        perm[~syn_active] = np.maximum(perm[~syn_active] - dec, 0.0)

        # Grow: replace weakest inactive synapses with active sources
        pool = self._get_source_pool(neuron, best_type)
        active_in_pool = np.intersect1d(np.nonzero(ctx)[0], pool)
        existing_set = set(idx.tolist())
        new_sources = [s for s in active_in_pool if s not in existing_set]

        if new_sources:
            inactive_slots = np.where(~syn_active)[0]
            if len(inactive_slots) > 0:
                order = np.argsort(perm[inactive_slots])
                n_grow = min(len(new_sources), len(inactive_slots))
                for i in range(n_grow):
                    slot = inactive_slots[order[i]]
                    idx[slot] = new_sources[i]
                    perm[slot] = self.perm_init

        seg_indices[neuron, best_seg_idx] = idx
        seg_perm[neuron, best_seg_idx] = perm

    def _adapt_segments(self, neuron: int, reinforce: bool):
        """Reinforce or punish active segments on a neuron.

        reinforce=True: strengthen synapses on correctly-predicting segments
        reinforce=False: weaken synapses on falsely-predicting segments
        """
        for seg_indices, seg_perm, ctx in [
            (self.fb_seg_indices, self.fb_seg_perm, self._pred_context_l23),
            (self.lat_seg_indices, self.lat_seg_perm, self._pred_context_l4),
        ]:
            if not ctx.any():
                continue
            for s in range(seg_indices.shape[1]):
                idx = seg_indices[neuron, s]
                perm = seg_perm[neuron, s]
                syn_active = ctx[idx]
                connected = perm > self.perm_threshold
                count = (syn_active & connected).sum()

                if count >= self.seg_activation_threshold:
                    inc = self.perm_increment * self.surprise_modulator
                    dec = self.perm_decrement * self.surprise_modulator
                    if reinforce:
                        # Strengthen active, weaken inactive
                        perm[syn_active] = np.minimum(
                            perm[syn_active] + inc, 1.0
                        )
                        perm[~syn_active] = np.maximum(
                            perm[~syn_active] - dec, 0.0
                        )
                    else:
                        # Punish: weaken active connected synapses
                        mask = syn_active & connected
                        perm[mask] = np.maximum(
                            perm[mask] - dec, 0.0
                        )

                    seg_perm[neuron, s] = perm

    def _update_traces(self):
        """Set active neuron traces to 1, decay the rest."""
        self.trace_l4 *= self.eligibility_decay
        self.trace_l23 *= self.eligibility_decay

        for col in np.nonzero(self.active_columns)[0]:
            if self.bursting_columns[col]:
                l4_start = col * self.n_l4
                l4_end = l4_start + self.n_l4
                best = l4_start + np.argmax(self.voltage_l4[l4_start:l4_end])
                self.trace_l4[best] = 1.0

                l23_start = col * self.n_l23
                l23_end = l23_start + self.n_l23
                best_l23 = l23_start + np.argmax(self.voltage_l23[l23_start:l23_end])
                self.trace_l23[best_l23] = 1.0
            else:
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
        """Boost inactive neurons, reset active ones (capped)."""
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
