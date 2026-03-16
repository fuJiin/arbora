"""Cortical region with L4 (input) and L2/3 (associative) layers.

Two-layer architecture modeled on neocortical minicolumns:
- L4 (input): receives feedforward drive, modulated by feedback context
- L2/3 (associative): receives L4 feedforward + lateral context from
  other L2/3 neurons, enabling associative binding and pattern completion

Prediction uses dendritic segments — each neuron has multiple short
dendritic branches that recognize specific patterns of source activity.
A segment fires when enough connected synapses have active sources.

L4 segment types:
- Feedback segments (L2/3 → L4): context from associative layer
- Lateral segments (L4 → L4): context from same-layer temporal patterns

L2/3 segment types:
- L2/3 lateral segments (L2/3 → L2/3): selective pattern-specific
  lateral predictions, replacing dense Hebbian with sparse connectivity.
  Each L2/3 neuron has dendritic branches recognizing specific L2/3
  patterns, biasing competitive selection via voltage boost.

Activation uses burst/precise distinction:
1. Before feedforward input, check dendritic segments for predicted neurons
2. Top-k columns selected by strongest feedforward + predicted score
3. Per active column: if a neuron was predicted → precise (only it fires).
   If none predicted → burst (all neurons fire, best-match gets trace).
4. Burst = surprise signal → grow new segment connections.
   Precise = expected → reinforce segment connections.

Feedforward weights map input dimensions to L4 neuron drive. Each column
has a structural receptive field mask; neurons in the same column share
the mask but learn different weight patterns. Hebbian LTP/LTD on the
feedforward synapses.
"""

import numpy as np

try:
    from step.cortex._numba_kernels import (
        predict_segments as _nb_predict,
        grow_segment as _nb_grow,
        adapt_segments_batch as _nb_adapt,
    )
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


class CorticalRegion:
    def __init__(
        self,
        input_dim: int,
        n_columns: int,
        n_l4: int,
        n_l23: int,
        k_columns: int,
        *,
        ltd_rate: float = 0.01,
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
        n_l23_segments: int = 4,
        n_synapses_per_segment: int = 24,
        perm_threshold: float = 0.5,
        perm_init: float = 0.6,
        perm_increment: float = 0.2,
        perm_decrement: float = 0.05,
        seg_activation_threshold: int = 2,
        prediction_gain: float = 2.5,
        n_apical_segments: int = 4,
        l23_prediction_boost: float = 0.0,
        seed: int = 0,
    ):
        self.input_dim = input_dim
        self.n_columns = n_columns
        self.n_l4 = n_l4
        self.n_l23 = n_l23
        self.k_columns = k_columns
        self.ltd_rate = ltd_rate
        self.voltage_decay = voltage_decay
        self.eligibility_decay = eligibility_decay
        self.synapse_decay = synapse_decay
        self.learning_rate = learning_rate
        self.max_excitability = max_excitability
        self.fb_boost = fb_boost
        self.burst_learning_scale = burst_learning_scale
        self.n_fb_segments = n_fb_segments
        self.n_lat_segments = n_lat_segments
        self.n_l23_segments = n_l23_segments
        self.n_synapses_per_segment = n_synapses_per_segment
        self.perm_threshold = perm_threshold
        self.perm_init = perm_init
        self.perm_increment = perm_increment
        self.perm_decrement = perm_decrement
        self.seg_activation_threshold = seg_activation_threshold
        self.prediction_gain = prediction_gain
        self.n_apical_segments = n_apical_segments
        # L2/3 segment prediction boost (0 = use fb_boost for both layers)
        self.l23_prediction_boost = l23_prediction_boost
        self._rng = np.random.default_rng(seed)

        # Third-factor neuromodulatory signals (set externally each step).
        # Scales learning rates: 1.0 = normal, >1 = boosts learning.
        self.surprise_modulator: float = 1.0
        # Dopaminergic reward signal: gates eligibility consolidation.
        self.reward_modulator: float = 1.0

        # Learning gate: set False to freeze all plasticity (ff, segments, apical).
        # Forward pass still runs — region processes input but doesn't learn.
        self.learning_enabled: bool = True

        self.n_l4_total: int = n_columns * n_l4
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
        self.predicted_l23 = np.zeros(self.n_l23_total, dtype=np.bool_)

        # L2/3 firing rate estimate (EMA of boolean activations).
        # Models postsynaptic temporal integration of spike trains.
        self.firing_rate_l23 = np.zeros(self.n_l23_total)

        # L2/3 lateral connections use dendritic segments only (no dense matrix).
        # Segments provide sparse pattern-specific predictions, matching biology.

        # Per-neuron eligibility traces
        self.trace_l4 = np.zeros(self.n_l4_total)
        self.trace_l23 = np.zeros(self.n_l23_total)

        # Prediction-time context (saved for segment learning)
        self._pred_context_l23 = np.zeros(self.n_l23_total, dtype=np.bool_)
        self._pred_context_l4 = np.zeros(self.n_l4_total, dtype=np.bool_)

        # Apical feedback (S2 L2/3 → S1 L4): initialized lazily via
        # init_apical_context() once the higher region exists.
        # Uses per-neuron learned gain weights (BAC firing model):
        # gain = 1 + context @ apical_gain_weights (multiplicative on voltage)
        self._apical_source_dim: int = 0
        self._apical_context = np.zeros(0, dtype=np.float64)
        self._apical_gain_weights: np.ndarray | None = None

        # --- Feedforward weights ---
        col_mask = self._build_ff_mask(input_dim)
        self.ff_mask = np.repeat(col_mask, self.n_l4, axis=1)
        self.ff_weights = np.zeros((input_dim, self.n_l4_total))
        self.ff_weights[self.ff_mask] = self._rng.uniform(
            0.1, 0.5, int(self.ff_mask.sum())
        )
        self._col_mask = col_mask

        # Efference copy: predicted sensory consequence of motor output.
        # Set by set_efference_copy(), consumed (cleared) in process().
        # Gain controls suppression strength: 1.0 = full cancellation,
        # <1 = partial, >1 = overcompensation (amplifies mismatch).
        self._efference_copy: np.ndarray | None = None
        self.efference_gain: float = 1.0

        # Column drive from last process() call (for diagnostics)
        self.last_column_drive = np.zeros(n_columns)

        # Initialize dendritic segments
        self._init_segments()

    def _build_ff_mask(self, input_dim: int) -> np.ndarray:
        """Build structural connectivity mask: (input_dim, n_columns).

        Default: full connectivity (every column sees every input dim).
        Subclasses override for encoding-specific receptive fields.
        """
        return np.ones((input_dim, self.n_columns), dtype=np.bool_)

    # ------------------------------------------------------------------
    # Feedforward processing and learning
    # ------------------------------------------------------------------

    def set_efference_copy(self, encoding: np.ndarray | None) -> None:
        """Set efference copy signal from motor output.

        When set, the next process() call subtracts the predicted sensory
        consequence from L4 drive — converting the response from "what is
        this input?" to "what's unexpected about it?". This breaks the
        autoregressive fixed point: the expected motor-generated input is
        suppressed, letting residual activity drive downstream regions
        to a new state.

        Models corollary discharge in biological sensorimotor loops.
        """
        self._efference_copy = encoding

    def process(self, encoding: np.ndarray) -> np.ndarray:
        """Feedforward an encoding through L4 → L2/3 pipeline."""
        flat = encoding.flatten().astype(np.float64)

        neuron_drive = flat @ self.ff_weights

        # Efference copy: suppress expected sensory consequence
        if self._efference_copy is not None:
            ef_flat = self._efference_copy.flatten().astype(np.float64)
            predicted_drive = ef_flat @ self.ff_weights
            neuron_drive -= self.efference_gain * predicted_drive
            self._efference_copy = None

        self.last_column_drive = neuron_drive.reshape(self.n_columns, self.n_l4).max(
            axis=1
        )
        active = self.step(neuron_drive)

        if self.learning_enabled:
            self._learn_ff(flat)
        return active

    def reconstruct(
        self,
        columns: np.ndarray | None = None,
        neurons: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reconstruct encoding from active columns/neurons via ff_weights."""
        if neurons is not None and len(neurons) > 0:
            return self.ff_weights[:, neurons].sum(axis=1)

        if columns is None:
            columns = np.nonzero(self.active_columns)[0]
        if len(columns) == 0:
            return np.zeros(self.input_dim)

        neuron_indices = []
        for col in columns:
            neuron_indices.extend(range(col * self.n_l4, (col + 1) * self.n_l4))
        return self.ff_weights[:, neuron_indices].sum(axis=1)

    def _learn_ff(self, flat_input: np.ndarray):
        """Per-neuron Hebbian ff learning with LTD.

        LTP on active neurons (the winning neuron in each active column).
        LTD on active neurons' inactive input connections.
        Subthreshold LTP on all neurons in inactive columns.
        """
        # Find winning neurons (one per active column) — vectorized
        active_cols = np.nonzero(self.active_columns)[0]
        winner_indices = np.empty(len(active_cols), dtype=np.intp)

        if len(active_cols) > 0:
            voltage_by_col = self.voltage_l4.reshape(self.n_columns, self.n_l4)
            active_by_col = self.active_l4.reshape(self.n_columns, self.n_l4)
            is_burst = self.bursting_columns[active_cols]
            if is_burst.any():
                winner_indices[is_burst] = active_cols[
                    is_burst
                ] * self.n_l4 + voltage_by_col[active_cols[is_burst]].argmax(axis=1)
            precise = ~is_burst
            if precise.any():
                winner_indices[precise] = active_cols[
                    precise
                ] * self.n_l4 + active_by_col[active_cols[precise]].argmax(axis=1)

        # Build per-neuron masks as compact vectors (only winners need them)
        neuromod = self.surprise_modulator * self.reward_modulator
        ltp_rate = self.learning_rate * neuromod

        # LTP: sparse update on winner neurons only
        if len(winner_indices) > 0:
            self.ff_weights[:, winner_indices] += ltp_rate * flat_input[:, np.newaxis]

        # LTD: sparse update on winner neurons only
        if len(winner_indices) > 0:
            ltd_rate = self.ltd_rate * neuromod
            inactive_input = 1.0 - flat_input
            winner_cols = winner_indices // self.n_l4
            col_masks = self._col_mask[:, winner_cols]
            local_on = np.maximum(
                (flat_input[:, np.newaxis] * col_masks).sum(axis=0), 1.0
            )
            local_off = np.maximum(
                (inactive_input[:, np.newaxis] * col_masks).sum(axis=0), 1.0
            )
            local_scales = local_on / local_off
            neuron_masks = self.ff_mask[:, winner_indices]
            self.ff_weights[:, winner_indices] -= (
                ltd_rate
                * local_scales[np.newaxis, :]
                * inactive_input[:, np.newaxis]
                * neuron_masks
            )
            # Clip winners in-place
            w = self.ff_weights[:, winner_indices]
            w[~neuron_masks] = 0.0
            np.clip(w, 0, 1, out=w)
            self.ff_weights[:, winner_indices] = w

        # Subthreshold: weak LTP on ALL neurons. Exploit input sparsity:
        # only rows where flat_input > 0 need updating.
        active_dims = np.flatnonzero(flat_input)
        if len(active_dims) > 0:
            sub_ltp = ltp_rate * 0.1 * flat_input[active_dims, np.newaxis]
            self.ff_weights[active_dims] += sub_ltp
            # Enforce structural mask on modified rows — vectorized
            self.ff_weights[active_dims] *= self.ff_mask[active_dims]
            # Upper clamp only on modified rows
            np.minimum(
                self.ff_weights[active_dims], 1, out=self.ff_weights[active_dims]
            )

    # ------------------------------------------------------------------
    # Dendritic segments
    # ------------------------------------------------------------------

    def _init_segments(self):
        """Initialize dendritic segment arrays with random connectivity."""
        n = self.n_l4_total
        n_syn = self.n_synapses_per_segment

        # Feedback segments: L2/3 → L4
        self.fb_seg_indices = np.zeros((n, self.n_fb_segments, n_syn), dtype=np.int32)
        self.fb_seg_perm = np.zeros((n, self.n_fb_segments, n_syn))

        # Lateral segments: L4 → L4
        self.lat_seg_indices = np.zeros((n, self.n_lat_segments, n_syn), dtype=np.int32)
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

        # L2/3 lateral segments: L2/3 → L2/3
        n23 = self.n_l23_total
        self.l23_seg_indices = np.zeros(
            (n23, self.n_l23_segments, n_syn), dtype=np.int32
        )
        self.l23_seg_perm = np.zeros((n23, self.n_l23_segments, n_syn))

        l23_pool = np.arange(n23)
        for i in range(n23):
            for s in range(self.n_l23_segments):
                self.l23_seg_indices[i, s] = self._rng.choice(
                    l23_pool, n_syn, replace=len(l23_pool) < n_syn
                )

    def init_apical_context(self, source_dim: int):
        """Initialize apical gain modulation from a higher region.

        Creates a learned weight matrix mapping the higher region's L2/3
        firing rates to per-neuron gain factors on this region's L4.
        Models biological BAC (backpropagation-activated calcium) firing:
        apical input lowers firing threshold without directly causing spikes.

        The gain is multiplicative on voltage: neurons with no feedforward
        drive are unaffected regardless of apical signal. This prevents
        the instructive feedback loop where top-down signals override
        bottom-up representations.

        Learning is slow (10x slower than feedforward) to prevent
        sender-receiver coupling that disrupts the sender's representations.
        """
        self._apical_source_dim = source_dim
        self._apical_context = np.zeros(source_dim, dtype=np.float64)
        # Per-neuron gain weights: (source_dim, n_l4_total)
        # Initialized small positive — slight uniform gain initially
        self._apical_gain_weights = self._rng.uniform(
            0.0, 0.1, (source_dim, self.n_l4_total),
        )

    # Keep old name as alias for backward compat (topology.connect uses it)
    def init_apical_segments(self, source_dim: int):
        """Backward-compatible alias for init_apical_context."""
        self.init_apical_context(source_dim)

    @property
    def has_apical(self) -> bool:
        """Whether apical context has been initialized."""
        return self._apical_source_dim > 0

    def predict_neuron(self, l4_idx: int, source_idx: int, segment_type: str = "fb"):
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
        self.predicted_l23[:] = False
        self._pred_context_l23[:] = False
        self._pred_context_l4[:] = False
        self._efference_copy = None
        if self.has_apical:
            self._apical_context[:] = 0.0
            # Preserve _apical_gain_weights (learned, not transient)

    def _predict_from_segments(self) -> np.ndarray:
        """Check which L4 neurons have active dendritic segments.

        A segment is active when >= seg_activation_threshold of its
        connected synapses (permanence > perm_threshold) have active
        source neurons. A neuron is predicted if any segment is active.

        Returns boolean mask of shape (n_l4_total,).
        """
        if _HAS_NUMBA:
            predicted = np.zeros(self.n_l4_total, dtype=np.bool_)
            if self.active_l23.any():
                predicted |= _nb_predict(
                    self.active_l23, self.fb_seg_indices, self.fb_seg_perm,
                    self.perm_threshold, self.seg_activation_threshold,
                )
            if self.active_l4.any():
                predicted |= _nb_predict(
                    self.active_l4, self.lat_seg_indices, self.lat_seg_perm,
                    self.perm_threshold, self.seg_activation_threshold,
                )
            return predicted

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
        # Apical context used directly as gain in step 5a — no pred snapshot needed.

        # 4. Feedforward drive to L4 neurons
        self.voltage_l4 += drive

        # 5a. Apical gating: per-neuron gain from top-down context.
        #     Models BAC firing: apical calcium plateau lowers threshold.
        #     gain_per_neuron = 1 + context @ weights (always >= 1).
        #     Multiplicative on voltage — can't fire without basal drive.
        if self.has_apical and self._apical_context.any():
            raw_gain = self._apical_context @ self._apical_gain_weights
            # Clamp to [0, prediction_gain-1] then shift to [1, prediction_gain]
            np.clip(raw_gain, 0.0, self.prediction_gain - 1.0, out=raw_gain)
            raw_gain += 1.0
            self.voltage_l4 *= raw_gain

        # 5b. Predicted neurons get a voltage boost (they're primed)
        self.voltage_l4[self.predicted_l4] += self.fb_boost

        # 6. Activate L4: top-k columns, then burst/precise per column
        scores_l4 = self.voltage_l4 + self.excitability_l4
        top_cols = self._select_columns(scores_l4)
        self._activate_l4_burst(top_cols, scores_l4)

        # 7. Activate L2/3: L4 feedforward + lateral context
        self._activate_l23(top_cols)

        # 8. Learn (dendritic segment permanence updates)
        if self.learning_enabled:
            self._learn()

            # 8b. Apical gain learning: slow Hebbian on gain weights.
            #     Strengthen connections from active context → active neurons.
            #     10x slower than feedforward to prevent sender disruption.
            if self.has_apical and self._apical_context.any():
                self._learn_apical_gain()

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

    def _predict_l23_from_segments(self) -> np.ndarray:
        """Check which L2/3 neurons have active lateral dendritic segments.

        Uses previous active_l23 as context (saved in _pred_context_l23).
        Returns boolean mask of shape (n_l23_total,).
        """
        if _HAS_NUMBA and self.active_l23.any():
            return _nb_predict(
                self.active_l23, self.l23_seg_indices, self.l23_seg_perm,
                self.perm_threshold, self.seg_activation_threshold,
            )

        predicted = np.zeros(self.n_l23_total, dtype=np.bool_)
        if self.active_l23.any():
            active_at_syn = self.active_l23[self.l23_seg_indices]
            connected = self.l23_seg_perm > self.perm_threshold
            counts = (active_at_syn & connected).sum(axis=2)
            predicted |= (counts >= self.seg_activation_threshold).any(axis=1)
        return predicted

    def set_apical_context(self, context: np.ndarray):
        """Set the apical feedback signal from a higher region.

        Called each step by the runner before this region's step().
        context: continuous firing rate signal from the higher region's L2/3.
        """
        self._apical_context[:] = context

    def _predict_apical_columns(self):
        """Check which columns have apical dendritic segment activity.

        Apical segments source from a higher region's L2/3. A segment is
        active when enough connected synapses have active sources (thresholded
        from continuous firing rate). Apical prediction is column-level:
        if any neuron in a column has an active apical segment, the column
        is apically predicted.

        Sets self.apical_predicted_cols (boolean, per-column).
        """
        self.apical_predicted_cols[:] = False
        if not self.has_apical:
            return

        # Threshold continuous firing rate to boolean for segment matching
        ctx_bool = self._apical_context > 0

        if not ctx_bool.any():
            return

        active_at_syn = ctx_bool[self.apical_seg_indices]
        connected = self.apical_seg_perm > self.perm_threshold
        counts = (active_at_syn & connected).sum(axis=2)
        neuron_predicted = (counts >= self.seg_activation_threshold).any(axis=1)

        # Aggregate to column level
        pred_by_col = neuron_predicted.reshape(self.n_columns, self.n_l4)
        self.apical_predicted_cols = pred_by_col.any(axis=1)

    def _compute_predictions(self):
        """Determine which neurons are in predictive state via segments."""
        self.predicted_l4 = self._predict_from_segments()
        self.predicted_l23 = self._predict_l23_from_segments()
        # Apical context is now used as continuous gain modulation in step(),
        # not through segment-based column prediction.

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

        # Work only with the winning columns
        tc_predicted = predicted_by_col[top_cols]  # (k, n_l4)
        tc_scores = scores_by_col[top_cols].copy()  # (k, n_l4)
        has_prediction = tc_predicted.any(axis=1)  # (k,)

        # Burst columns: no prediction — all neurons fire
        burst_cols = top_cols[~has_prediction]
        self.bursting_columns[burst_cols] = True
        if len(burst_cols) > 0:
            active_l4_by_col = self.active_l4.reshape(self.n_columns, self.n_l4)
            active_l4_by_col[burst_cols] = True

        # Precise columns: mask unpredicted neurons, pick best scorer
        precise_mask = has_prediction
        if precise_mask.any():
            precise_cols = top_cols[precise_mask]
            p_scores = tc_scores[precise_mask]  # (n_precise, n_l4)
            p_predicted = tc_predicted[precise_mask]  # (n_precise, n_l4)
            p_scores[~p_predicted] = -np.inf
            winners = p_scores.argmax(axis=1)  # (n_precise,)
            global_indices = precise_cols * self.n_l4 + winners
            self.active_l4[global_indices] = True

    def _activate_l23(self, top_cols: np.ndarray):
        """Activate L2/3 associative neurons in active columns.

        L2/3 receives three sources of drive:
        1. L4 feedforward: base drive to all neurons in the column
        2. L4 winner bonus: precise columns boost the matching L2/3 neuron
        3. L2/3 segment prediction: predicted neurons get voltage boost

        Then competitive selection: burst columns → all fire,
        precise columns → single winner.
        """
        # L4 -> L2/3 feedforward: base drive to all neurons in active columns
        v_l23 = self.voltage_l23.reshape(self.n_columns, self.n_l23)
        v_l23[top_cols] += 0.5

        # Bonus for L2/3 neuron matching the L4 winner (precise only)
        precise_cols = top_cols[~self.bursting_columns[top_cols]]
        if len(precise_cols) > 0:
            active_l4_by_col = self.active_l4.reshape(self.n_columns, self.n_l4)
            l4_winners = active_l4_by_col[precise_cols].argmax(axis=1)
            # Only apply bonus where L4 winner index fits in L2/3
            valid = l4_winners < self.n_l23
            if valid.any():
                valid_cols = precise_cols[valid]
                valid_winners = l4_winners[valid]
                self.voltage_l23[valid_cols * self.n_l23 + valid_winners] += 0.5

        # L2/3 segment prediction boost: predicted neurons are primed
        l23_boost = self.l23_prediction_boost or self.fb_boost
        self.voltage_l23[self.predicted_l23] += l23_boost

        # Competitive selection per column
        self.active_l23[:] = False
        l23_scores = self.voltage_l23 + self.excitability_l23
        by_col = l23_scores.reshape(self.n_columns, self.n_l23)

        # Burst columns: all L2/3 neurons fire
        burst_cols = top_cols[self.bursting_columns[top_cols]]
        if len(burst_cols) > 0:
            active_l23_by_col = self.active_l23.reshape(self.n_columns, self.n_l23)
            active_l23_by_col[burst_cols] = True

        # Precise columns: single winner per column
        if len(precise_cols) > 0:
            winners = by_col[precise_cols].argmax(axis=1)
            self.active_l23[precise_cols * self.n_l23 + winners] = True

    def _learn_apical_gain(self):
        """Slow Hebbian learning on apical gain weights.

        Models plasticity at apical synapses — which higher-region neurons
        can modulate which lower-region neurons. Learning rate is 10x
        slower than feedforward to prevent sender-receiver coupling.

        Rule: if context was active AND neuron fired → strengthen (LTP).
        Passive decay keeps weights bounded.
        """
        # Slow learning rate: 10x slower than feedforward
        apical_lr = self.learning_rate * 0.1

        # Context as column vector, active neurons as row vector
        ctx = self._apical_context  # (source_dim,)
        active = self.active_l4.astype(np.float64)  # (n_l4_total,)

        # Sparse update: only where context is nonzero
        ctx_nz = np.flatnonzero(ctx)
        active_nz = np.flatnonzero(active)
        if len(ctx_nz) > 0 and len(active_nz) > 0:
            # LTP: outer product of active context × active neurons
            delta = apical_lr * ctx[ctx_nz, np.newaxis] * active[np.newaxis, active_nz]
            self._apical_gain_weights[np.ix_(ctx_nz, active_nz)] += delta

        # Passive decay to prevent unbounded growth
        self._apical_gain_weights *= 0.9999
        np.clip(self._apical_gain_weights, 0.0, 1.0, out=self._apical_gain_weights)

    def _learn(self):
        """Dendritic segment updates for L4 and L2/3 prediction.

        L4 segments: feedback (L2/3→L4) and lateral (L4→L4) prediction.
        L2/3 segments: lateral (L2/3→L2/3) pattern-specific prediction.
        All use sparse dendritic segment learning (grow/reinforce/punish).
        """
        self._learn_segments()
        self._learn_l23_segments()

    def _learn_segments(self):
        """Update dendritic segment permanences based on prediction outcomes.

        - Burst (unpredicted): grow best-matching segment on trace winner
        - Precise + predicted: reinforce the active segments
        - Predicted but didn't fire: punish the active segments
        """
        active_cols = np.nonzero(self.active_columns)[0]
        voltage_by_col = self.voltage_l4.reshape(self.n_columns, self.n_l4)

        # Burst columns: grow segment on trace winner
        burst_cols = active_cols[self.bursting_columns[active_cols]]
        if len(burst_cols) > 0:
            best_in_col = voltage_by_col[burst_cols].argmax(axis=1)
            for i, col in enumerate(burst_cols):
                self._grow_best_segment(col * self.n_l4 + best_in_col[i])

        # Precise + predicted: batch reinforce
        reinforce_neurons = np.nonzero(
            self.active_l4
            & self.predicted_l4
            & np.repeat(self.active_columns & ~self.bursting_columns, self.n_l4)
        )[0]
        if len(reinforce_neurons) > 0:
            self._adapt_segments_batch(
                reinforce_neurons,
                self.fb_seg_indices,
                self.fb_seg_perm,
                self._pred_context_l23,
                reinforce=True,
            )
            self._adapt_segments_batch(
                reinforce_neurons,
                self.lat_seg_indices,
                self.lat_seg_perm,
                self._pred_context_l4,
                reinforce=True,
            )

        # Punish false predictions (predicted but didn't fire)
        false_predicted = np.nonzero(self.predicted_l4 & ~self.active_l4)[0]
        if len(false_predicted) > 0:
            self._adapt_segments_batch(
                false_predicted,
                self.fb_seg_indices,
                self.fb_seg_perm,
                self._pred_context_l23,
                reinforce=False,
            )
            self._adapt_segments_batch(
                false_predicted,
                self.lat_seg_indices,
                self.lat_seg_perm,
                self._pred_context_l4,
                reinforce=False,
            )

        # Apical feedback is now pure gain modulation — no segment learning.

    # ------------------------------------------------------------------
    # Generic segment operations (shared by L4 and L2/3 segments)
    # ------------------------------------------------------------------

    def _grow_segment(
        self,
        neuron: int,
        seg_indices: np.ndarray,
        seg_perm: np.ndarray,
        ctx: np.ndarray,
        pool: np.ndarray,
    ):
        """Grow the best-matching segment for a bursting neuron.

        Finds the segment with most overlap with context, strengthens
        matching synapses, and replaces weakest non-matching ones with
        new connections to active source neurons.
        """
        if not ctx.any():
            return

        neuromod = self.surprise_modulator * self.reward_modulator
        inc = self.perm_increment * neuromod
        dec = self.perm_decrement * neuromod

        if _HAS_NUMBA:
            _nb_grow(
                neuron, seg_indices, seg_perm, ctx,
                pool.astype(np.int32) if pool.dtype != np.int32 else pool,
                inc, dec, self.perm_init,
            )
            return

        # NumPy fallback
        overlaps = ctx[seg_indices[neuron]].sum(axis=1)
        best_seg_idx = int(overlaps.argmax())
        if overlaps[best_seg_idx] <= 0:
            return

        idx = seg_indices[neuron, best_seg_idx].copy()
        perm = seg_perm[neuron, best_seg_idx].copy()
        syn_active = ctx[idx]

        perm[syn_active] = np.minimum(perm[syn_active] + inc, 1.0)
        perm[~syn_active] = np.maximum(perm[~syn_active] - dec, 0.0)

        active_in_pool = pool[ctx[pool]]
        if len(active_in_pool) > 0:
            existing_set = set(idx.tolist())
            new_sources = np.array(
                [s for s in active_in_pool if s not in existing_set],
                dtype=idx.dtype,
            )
        else:
            new_sources = active_in_pool

        if len(new_sources) > 0:
            inactive_slots = np.where(~syn_active)[0]
            if len(inactive_slots) > 0:
                order = np.argsort(perm[inactive_slots])
                n_grow = min(len(new_sources), len(inactive_slots))
                slots = inactive_slots[order[:n_grow]]
                idx[slots] = new_sources[:n_grow]
                perm[slots] = self.perm_init

        seg_indices[neuron, best_seg_idx] = idx
        seg_perm[neuron, best_seg_idx] = perm

    def _adapt_segment_array(
        self,
        neuron: int,
        seg_indices: np.ndarray,
        seg_perm: np.ndarray,
        ctx: np.ndarray,
        reinforce: bool,
    ):
        """Reinforce or punish active segments for a single neuron."""
        self._adapt_segments_batch(
            np.array([neuron]), seg_indices, seg_perm, ctx, reinforce
        )

    def _adapt_segments_batch(
        self,
        neurons: np.ndarray,
        seg_indices: np.ndarray,
        seg_perm: np.ndarray,
        ctx: np.ndarray,
        reinforce: bool,
    ):
        """Reinforce or punish active segments for a batch of neurons."""
        if len(neurons) == 0 or not ctx.any():
            return

        neuromod = self.surprise_modulator * self.reward_modulator
        inc = self.perm_increment * neuromod
        dec = self.perm_decrement * neuromod

        if _HAS_NUMBA:
            _nb_adapt(
                neurons.astype(np.intp),
                seg_indices, seg_perm, ctx,
                self.perm_threshold, self.seg_activation_threshold,
                inc, dec, reinforce,
            )
            return

        # NumPy fallback
        batch_idx = seg_indices[neurons]
        batch_perm = seg_perm[neurons]
        syn_active = ctx[batch_idx]
        connected = batch_perm > self.perm_threshold
        counts = (syn_active & connected).sum(axis=2)
        active_mask = counts >= self.seg_activation_threshold

        if not active_mask.any():
            return

        active_f = active_mask[:, :, np.newaxis].astype(np.float64)
        syn_f = syn_active.astype(np.float64)

        if reinforce:
            delta = active_f * (syn_f * inc - (1.0 - syn_f) * dec)
            batch_perm += delta
            np.clip(batch_perm, 0.0, 1.0, out=batch_perm)
        else:
            punish_f = active_f * syn_f * connected.astype(np.float64)
            batch_perm -= punish_f * dec
            np.maximum(batch_perm, 0.0, out=batch_perm)

        seg_perm[neurons] = batch_perm

    # ------------------------------------------------------------------
    # L4 segment learning (feedback + lateral)
    # ------------------------------------------------------------------

    def _grow_best_segment(self, neuron: int):
        """Grow the best-matching L4 segment for a bursting neuron.

        Checks both fb (L2/3→L4) and lat (L4→L4) segment types,
        picks the one with most context overlap.
        """
        best_overlap = -1
        best_type = None

        # Vectorized overlap computation per segment type
        for seg_type, seg_indices, ctx in [
            ("fb", self.fb_seg_indices, self._pred_context_l23),
            ("lat", self.lat_seg_indices, self._pred_context_l4),
        ]:
            if not ctx.any():
                continue
            # (n_seg, n_syn) -> sum over synapses
            overlaps = ctx[seg_indices[neuron]].sum(axis=1)
            max_overlap = int(overlaps.max())
            if max_overlap > best_overlap:
                best_overlap = max_overlap
                best_type = seg_type

        if best_type is None or best_overlap <= 0:
            return

        if best_type == "fb":
            seg_indices = self.fb_seg_indices
            seg_perm = self.fb_seg_perm
            ctx = self._pred_context_l23
        else:
            seg_indices = self.lat_seg_indices
            seg_perm = self.lat_seg_perm
            ctx = self._pred_context_l4

        pool = self._get_source_pool(neuron, best_type)
        self._grow_segment(neuron, seg_indices, seg_perm, ctx, pool)

    def _adapt_segments(self, neuron: int, reinforce: bool):
        """Reinforce or punish active L4 segments (both fb and lat)."""
        self._adapt_segment_array(
            neuron,
            self.fb_seg_indices,
            self.fb_seg_perm,
            self._pred_context_l23,
            reinforce,
        )
        self._adapt_segment_array(
            neuron,
            self.lat_seg_indices,
            self.lat_seg_perm,
            self._pred_context_l4,
            reinforce,
        )

    # ------------------------------------------------------------------
    # Apical segment learning (S2 L2/3 → S1 L4)
    # ------------------------------------------------------------------

    def _learn_apical_segments(self):
        """Update apical segment permanences based on prediction outcomes.

        Same grow/reinforce/punish pattern as fb/lat segments, but using
        apical context (higher region's L2/3 firing rate) as the source.
        """
        if not self.has_apical:
            return

        ctx = self._pred_apical_context > 0  # threshold to boolean
        if not ctx.any():
            return

        pool = np.arange(self._apical_source_dim)
        active_cols = np.nonzero(self.active_columns)[0]
        voltage_by_col = self.voltage_l4.reshape(self.n_columns, self.n_l4)

        # Burst columns: grow segment on trace winner
        burst_cols = active_cols[self.bursting_columns[active_cols]]
        if len(burst_cols) > 0:
            best_in_col = voltage_by_col[burst_cols].argmax(axis=1)
            for i, col in enumerate(burst_cols):
                self._grow_segment(
                    col * self.n_l4 + best_in_col[i],
                    self.apical_seg_indices,
                    self.apical_seg_perm,
                    ctx,
                    pool,
                )

        # Precise columns: batch reinforce active neurons
        reinforce_neurons = np.nonzero(
            self.active_l4
            & np.repeat(self.active_columns & ~self.bursting_columns, self.n_l4)
        )[0]
        if len(reinforce_neurons) > 0:
            self._adapt_segments_batch(
                reinforce_neurons,
                self.apical_seg_indices,
                self.apical_seg_perm,
                ctx,
                reinforce=True,
            )

        # Punish: all neurons in columns with apical prediction that didn't activate
        punish_cols = np.nonzero(self.apical_predicted_cols & ~self.active_columns)[0]
        if len(punish_cols) > 0:
            punish_neurons = np.repeat(punish_cols, self.n_l4) * self.n_l4 + np.tile(
                np.arange(self.n_l4), len(punish_cols)
            )
            self._adapt_segments_batch(
                punish_neurons,
                self.apical_seg_indices,
                self.apical_seg_perm,
                ctx,
                reinforce=False,
            )

    # ------------------------------------------------------------------
    # L2/3 segment learning (lateral)
    # ------------------------------------------------------------------

    def _learn_l23_segments(self):
        """Update L2/3 lateral segment permanences based on prediction outcomes.

        - Burst column: grow best-matching L2/3 segment on trace winner
        - Precise + predicted L2/3: reinforce active segments
        - Predicted L2/3 but didn't fire: punish active segments
        """
        active_cols = np.nonzero(self.active_columns)[0]
        voltage_l23_by_col = self.voltage_l23.reshape(self.n_columns, self.n_l23)

        # Burst columns: grow segment on trace winner
        burst_cols = active_cols[self.bursting_columns[active_cols]]
        if len(burst_cols) > 0:
            best_in_col = voltage_l23_by_col[burst_cols].argmax(axis=1)
            for i, col in enumerate(burst_cols):
                best = col * self.n_l23 + best_in_col[i]
                pool = self._get_l23_source_pool(best)
                self._grow_segment(
                    best,
                    self.l23_seg_indices,
                    self.l23_seg_perm,
                    self._pred_context_l23,
                    pool,
                )

        # Precise + predicted: batch reinforce
        reinforce_neurons = np.nonzero(
            self.active_l23
            & self.predicted_l23
            & np.repeat(self.active_columns & ~self.bursting_columns, self.n_l23)
        )[0]
        if len(reinforce_neurons) > 0:
            self._adapt_segments_batch(
                reinforce_neurons,
                self.l23_seg_indices,
                self.l23_seg_perm,
                self._pred_context_l23,
                reinforce=True,
            )

        # Punish false predictions
        false_predicted = np.nonzero(self.predicted_l23 & ~self.active_l23)[0]
        if len(false_predicted) > 0:
            self._adapt_segments_batch(
                false_predicted,
                self.l23_seg_indices,
                self.l23_seg_perm,
                self._pred_context_l23,
                reinforce=False,
            )

    def _get_l23_source_pool(self, neuron: int) -> np.ndarray:
        """Get valid L2/3 source neuron indices for growing synapses."""
        return np.arange(self.n_l23_total)

    def _update_traces(self):
        """Set active neuron traces to 1, decay the rest."""
        self.trace_l4 *= self.eligibility_decay
        self.trace_l23 *= self.eligibility_decay

        active_cols = np.nonzero(self.active_columns)[0]
        if len(active_cols) == 0:
            return

        voltage_l4_by_col = self.voltage_l4.reshape(self.n_columns, self.n_l4)
        voltage_l23_by_col = self.voltage_l23.reshape(self.n_columns, self.n_l23)
        active_l4_by_col = self.active_l4.reshape(self.n_columns, self.n_l4)
        active_l23_by_col = self.active_l23.reshape(self.n_columns, self.n_l23)

        # Burst columns: trace to highest-voltage neuron
        burst_cols = active_cols[self.bursting_columns[active_cols]]
        if len(burst_cols) > 0:
            best_l4 = voltage_l4_by_col[burst_cols].argmax(axis=1)
            self.trace_l4[burst_cols * self.n_l4 + best_l4] = 1.0
            best_l23 = voltage_l23_by_col[burst_cols].argmax(axis=1)
            self.trace_l23[burst_cols * self.n_l23 + best_l23] = 1.0

        # Precise columns: trace to the active neuron
        precise_cols = active_cols[~self.bursting_columns[active_cols]]
        if len(precise_cols) > 0:
            # argmax on boolean active mask gives the first True index
            best_l4 = active_l4_by_col[precise_cols].argmax(axis=1)
            self.trace_l4[precise_cols * self.n_l4 + best_l4] = 1.0
            best_l23 = active_l23_by_col[precise_cols].argmax(axis=1)
            self.trace_l23[precise_cols * self.n_l23 + best_l23] = 1.0

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
