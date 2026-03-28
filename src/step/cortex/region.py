"""Cortical region with L4 (input), L2/3 (associative), and L5 (output) layers.

Three-layer architecture modeled on neocortical minicolumns:
- L4 (input): receives feedforward drive, modulated by lateral context
- L2/3 (associative): receives L4 feedforward + lateral context from
  other L2/3 neurons, enabling associative binding and pattern completion
- L5 (output): receives intra-columnar L2/3 drive, projects subcortically
  to BG (corticostriatal), thalamus, brainstem, and cerebellum

Prediction uses dendritic segments — each neuron has multiple short
dendritic branches that recognize specific patterns of source activity.
A segment fires when enough connected synapses have active sources.

L4 segment types:
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

from step.config import PlasticityRule
from step.cortex.lamina import Lamina, LaminaID

try:
    from step.cortex._numba_kernels import (
        adapt_segments_batch as _nb_adapt,
    )
    from step.cortex._numba_kernels import (
        grow_segment as _nb_grow,
    )
    from step.cortex._numba_kernels import (
        predict_segments as _nb_predict,
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
        n_l5: int | None = None,
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
        n_l4_lat_segments: int = 4,
        n_l23_segments: int = 4,
        n_synapses_per_segment: int = 24,
        perm_threshold: float = 0.5,
        perm_init: float = 0.6,
        perm_increment: float = 0.2,
        perm_decrement: float = 0.05,
        seg_activation_threshold: int = 2,
        prediction_gain: float = 2.5,
        n_apical_segments: int = 4,
        n_l5_segments: int = 4,
        l23_prediction_boost: float = 0.0,
        source_dims: list[int] | None = None,
        ff_sparsity: float = 0.0,
        pre_trace_decay: float = 0.0,
        plasticity_rule: PlasticityRule = PlasticityRule.HEBBIAN,
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
        self.n_l4_lat_segments = n_l4_lat_segments
        self.n_l23_segments = n_l23_segments
        self.n_synapses_per_segment = n_synapses_per_segment
        self.perm_threshold = perm_threshold
        self.perm_init = perm_init
        self.perm_increment = perm_increment
        self.perm_decrement = perm_decrement
        self.seg_activation_threshold = seg_activation_threshold
        self.prediction_gain = prediction_gain
        self.n_apical_segments = n_apical_segments
        self.n_l5_segments = n_l5_segments
        # L2/3 segment prediction boost (0 = use fb_boost for both layers)
        self.l23_prediction_boost = l23_prediction_boost
        self._rng = np.random.default_rng(seed)
        # Source-aware structural sparsity for multi-ff inputs.
        self._source_dims = source_dims
        self._ff_sparsity = ff_sparsity

        # STDP-like presynaptic trace: decaying record of recent input
        # activity. When a postsynaptic neuron fires (k-WTA), synapses
        # from recently-active inputs get strengthened — giving temporal
        # credit to inputs that preceded activation, not just coincided.
        # 0.0 = disabled (pure coincidence Hebbian, original behavior).
        self._pre_trace_decay = pre_trace_decay
        self._pre_trace_threshold = 0.0  # min value to use (sparsity)

        # Third-factor neuromodulatory signals (set externally each step).
        # Scales learning rates: 1.0 = normal, >1 = boosts learning.
        self.surprise_modulator: float = 1.0
        # Dopaminergic reward signal: gates eligibility consolidation.
        self.reward_modulator: float = 1.0

        # Learning gate: set False to freeze all plasticity (ff, segments, apical).
        # Forward pass still runs — region processes input but doesn't learn.
        self.learning_enabled: bool = True

        self.n_l5 = n_l5 if n_l5 is not None else n_l23

        # --- Lamina registry ---
        self.laminae: dict[LaminaID, Lamina] = {}
        self.register_lamina(Lamina(n_columns, n_l4, lamina_id=LaminaID.L4))
        self.register_lamina(Lamina(n_columns, n_l23, lamina_id=LaminaID.L23))
        self.register_lamina(Lamina(n_columns, self.n_l5, lamina_id=LaminaID.L5))

        # Convenience totals (widely used in step logic and subclasses)
        self.n_l4_total: int = self.l4.n_total
        self.n_l23_total = self.l23.n_total
        self.n_l5_total = self.l5.n_total

        # Pre-allocated source pools for segment growth (avoids per-call arange)
        self._l4_lat_source_pool = np.arange(self.n_l4_total)
        self._l23_source_pool = np.arange(self.n_l23_total)

        # Column-level state (not per-lamina)
        self.active_columns = np.zeros(n_columns, dtype=np.bool_)
        self.bursting_columns = np.zeros(n_columns, dtype=np.bool_)

        # L5 output scores: per-column mean L5 firing rate.
        # Primary output signal for subcortical targets (BG, cerebellum).
        self.output_scores = np.zeros(n_columns)

        # L2/3 lateral connections use dendritic segments only (no dense matrix).
        # Segments provide sparse pattern-specific predictions, matching biology.

        # Prediction-time context (saved for segment learning).
        # Boolean snapshots used for prediction (threshold-based).
        self._pred_context_l23 = np.zeros(self.n_l23_total, dtype=np.bool_)
        self._pred_context_l4 = np.zeros(self.n_l4_total, dtype=np.bool_)
        # Continuous traces for segment learning (STDP-like temporal credit).
        # When pre_trace_decay > 0, segment learning uses these traces
        # (which include recent history) instead of boolean snapshots.
        if self._pre_trace_decay > 0:
            self._seg_trace_l23 = np.zeros(self.n_l23_total)
            self._seg_trace_l4 = np.zeros(self.n_l4_total)
            self._seg_trace_l5 = np.zeros(self.n_l5_total)
        else:
            self._seg_trace_l23 = None
            self._seg_trace_l4 = None
            self._seg_trace_l5 = None

        # Apical feedback: per-source L5 dendritic segments (BAC firing model).
        # Multiple regions can send apical to the same target — each has
        # its own learned segments. Predictions from all sources are OR'd.
        # Initialized lazily via init_apical_context() per source.
        self._apical_sources: dict[str, dict] = {}
        # Backward-compat aliases (set to first source on init)
        self._apical_source_dim: int = 0
        self._apical_context = np.zeros(0, dtype=np.float64)

        # Optional goal drive (initialized lazily via init_goal_drive())
        self._goal_drive: np.ndarray | None = None
        self._goal_weights: np.ndarray | None = None

        # --- Feedforward weights ---
        # Target the input lamina: L4 in granular regions, L2/3 in agranular.
        _ff_n_per_col = self.input_lamina.n_per_col
        _ff_n_total = self.input_lamina.n_total
        col_mask = self._build_ff_mask(input_dim)
        self.ff_mask = np.repeat(col_mask, _ff_n_per_col, axis=1)
        self.ff_weights = np.zeros((input_dim, _ff_n_total))
        self.ff_weights[self.ff_mask] = self._rng.uniform(
            0.1, 0.5, int(self.ff_mask.sum())
        )
        self._col_mask = col_mask

        # --- Intra-column feedforward weights ---
        # Per-column weight matrices for within-column pathways.
        # L4→L2/3: which L4 patterns drive which L2/3 neurons
        # L2/3→L5: which L2/3 patterns drive which L5 outputs
        self.l4_to_l23_weights = self._rng.uniform(
            0.1, 0.5, (self.n_columns, self.n_l4, self.n_l23)
        )
        self.l23_to_l5_weights = self._rng.uniform(
            0.1, 0.5, (self.n_columns, self.n_l23, self.n_l5)
        )

        # Presynaptic trace buffer (only allocated if enabled)
        if self._pre_trace_decay > 0:
            self._pre_trace = np.zeros(input_dim)
        else:
            self._pre_trace = None

        # Plasticity rule: HEBBIAN (immediate LTP/LTD) vs THREE_FACTOR
        # (eligibility trace, consolidated by apply_reward).
        self.plasticity_rule = plasticity_rule

        # Three-factor learning: eligibility clip threshold and trace.
        # Allocated when plasticity_rule is THREE_FACTOR.
        self._eligibility_clip: float = 0.0
        if plasticity_rule == PlasticityRule.THREE_FACTOR:
            self._ff_eligibility: np.ndarray | None = np.zeros((input_dim, _ff_n_total))
        else:
            self._ff_eligibility = None

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

        If source_dims is set (multi-ff input), applies per-source
        random sparsity so each column connects to a subset of each
        source. This prevents neurons from being overloaded by the
        full concatenated drive and encourages source specialization.

        Subclasses override for encoding-specific receptive fields.
        """
        source_dims = getattr(self, "_source_dims", None)
        ff_sparsity = getattr(self, "_ff_sparsity", 0.0)
        if source_dims and ff_sparsity > 0:
            # Source-aware sparsity: random mask per source per column
            mask = np.zeros((input_dim, self.n_columns), dtype=np.bool_)
            pos = 0
            for sdim in source_dims:
                # Each column connects to (1 - ff_sparsity) of this source
                src_mask = self._rng.random((sdim, self.n_columns)) < (
                    1.0 - ff_sparsity
                )
                mask[pos : pos + sdim] = src_mask
                pos += sdim
            # Fill any remaining dims (rounding) with full connectivity
            if pos < input_dim:
                mask[pos:] = True
            return mask
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

    def process(
        self,
        encoding: np.ndarray,
        *,
        forced_columns: np.ndarray | None = None,
    ) -> np.ndarray:
        """Feedforward an encoding through the input→association→output pipeline.

        Args:
            encoding: Input encoding vector.
            forced_columns: If provided, skip k-WTA and use these columns.
                Used for motor exploration (random column forcing).
        """
        flat = encoding.flatten().astype(np.float64)

        neuron_drive = flat @ self.ff_weights

        # Efference copy: suppress expected sensory consequence
        if self._efference_copy is not None:
            ef_flat = self._efference_copy.flatten().astype(np.float64)
            predicted_drive = ef_flat @ self.ff_weights
            neuron_drive -= self.efference_gain * predicted_drive
            self._efference_copy = None

        # Optional goal drive (PFC→motor additive signal)
        if self._goal_drive is not None and self._goal_weights is not None:
            goal_drive = self._goal_drive @ self._goal_weights
            neuron_drive += goal_drive
            if hasattr(self, "_goal_eligibility"):
                self._goal_eligibility *= self.eligibility_decay
                self._pending_goal_signal = self._goal_drive
            self._goal_drive = None

        _n_per = self.input_lamina.n_per_col
        self.last_column_drive = neuron_drive.reshape(self.n_columns, _n_per).max(
            axis=1
        )
        active = self.step(neuron_drive, forced_columns=forced_columns)

        # Goal eligibility for winner neurons
        if hasattr(self, "_goal_eligibility") and hasattr(self, "_pending_goal_signal"):
            winner_cols = np.nonzero(self.active_columns)[0]
            if len(winner_cols) > 0:
                winner_neurons = []
                for col in winner_cols:
                    winner_neurons.extend(range(col * _n_per, (col + 1) * _n_per))
                goal_lr = self.learning_rate * 0.1
                self._goal_eligibility[:, winner_neurons] += (
                    goal_lr * self._pending_goal_signal[:, np.newaxis]
                )
            del self._pending_goal_signal

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

        n_per = self.input_lamina.n_per_col
        neuron_indices = []
        for col in columns:
            neuron_indices.extend(range(col * n_per, (col + 1) * n_per))
        return self.ff_weights[:, neuron_indices].sum(axis=1)

    def _find_winners(self) -> np.ndarray:
        """Find winning neurons (one per active column) on the input lamina.

        For bursting columns: winner is the neuron with highest voltage.
        For precise columns: winner is the neuron with highest activation.

        Returns flat indices into the input lamina neuron array.
        Returns empty array if no columns are active.
        """
        active_cols = np.nonzero(self.active_columns)[0]
        if len(active_cols) == 0:
            return np.empty(0, dtype=np.intp)

        lamina = self.input_lamina
        n_per_col = lamina.n_per_col
        voltage_by_col = lamina.voltage.reshape(self.n_columns, n_per_col)
        active_by_col = lamina.active.reshape(self.n_columns, n_per_col)
        is_burst = self.bursting_columns[active_cols]

        winner_indices = np.empty(len(active_cols), dtype=np.intp)
        if is_burst.any():
            winner_indices[is_burst] = active_cols[
                is_burst
            ] * n_per_col + voltage_by_col[active_cols[is_burst]].argmax(axis=1)
        precise = ~is_burst
        if precise.any():
            winner_indices[precise] = active_cols[precise] * n_per_col + active_by_col[
                active_cols[precise]
            ].argmax(axis=1)
        return winner_indices

    def _learn_ff(self, flat_input: np.ndarray):
        """Per-neuron feedforward learning, dispatched by plasticity_rule.

        Always updates presynaptic traces (STDP, orthogonal to rule).

        HEBBIAN path: immediate LTP/LTD/subthreshold on ff_weights.
        THREE_FACTOR path: decay eligibility, accumulate Hebbian
        coincidences into eligibility trace. Weights updated only
        when apply_reward() is called.
        """
        # Update presynaptic trace: decay + accumulate current input
        if self._pre_trace is not None:
            self._pre_trace *= self._pre_trace_decay
            self._pre_trace += flat_input
            # Threshold for sparsity: ignore faint echoes
            if self._pre_trace_threshold > 0:
                ltp_signal = np.where(
                    self._pre_trace > self._pre_trace_threshold,
                    self._pre_trace,
                    0.0,
                )
            else:
                ltp_signal = self._pre_trace
        else:
            # No trace: pure coincidence (original behavior)
            ltp_signal = flat_input

        # Find winning neurons (one per active column) — vectorized
        winner_indices = self._find_winners()

        if self.plasticity_rule == PlasticityRule.THREE_FACTOR:
            self._learn_ff_three_factor(ltp_signal, winner_indices)
        else:
            self._learn_ff_hebbian(flat_input, ltp_signal, winner_indices)

    def _learn_ff_three_factor(
        self, ltp_signal: np.ndarray, winner_indices: np.ndarray
    ):
        """Three-factor: accumulate Hebbian coincidences in eligibility trace.

        Eligibility trace decays each step. Consolidated into ff_weights
        only when apply_reward() is called with nonzero reward.
        """
        # Decay eligibility trace
        assert self._ff_eligibility is not None
        self._ff_eligibility *= self.eligibility_decay

        if len(winner_indices) == 0:
            return

        # Record in eligibility trace using temporal signal
        self._ff_eligibility[:, winner_indices] += (
            self.learning_rate * ltp_signal[:, np.newaxis]
        )

    def _learn_ff_hebbian(
        self,
        flat_input: np.ndarray,
        ltp_signal: np.ndarray,
        winner_indices: np.ndarray,
    ):
        """Hebbian: immediate LTP/LTD/subthreshold on ff_weights.

        When pre_trace_decay > 0, uses presynaptic traces for temporal
        credit assignment: inputs that fired recently (not just now) get
        credit when a postsynaptic neuron activates. This is the core
        STDP mechanism — pre before post → strengthen.

        When pre_trace_decay == 0, uses standard coincidence Hebbian
        (original behavior: only current input gets credit).

        LTP on active neurons (the winning neuron in each active column).
        LTD on active neurons' inactive input connections.
        Subthreshold LTP on all neurons in inactive columns.
        """
        # Cache attribute lookups for hot loop
        ff_weights = self.ff_weights
        neuromod = self.surprise_modulator * self.reward_modulator
        ltp_rate = self.learning_rate * neuromod

        # LTP + LTD on winner neurons (one per active column)
        if len(winner_indices) > 0:
            # Read winner weight slice once
            w = ff_weights[:, winner_indices]

            # LTP: strengthen synapses from recently-active inputs → winners
            w += ltp_rate * ltp_signal[:, np.newaxis]

            # LTD: weaken synapses from NOT-recently-active inputs → winners
            # Uses flat_input (not trace) for LTD — only current step's
            # inactive inputs get weakened. Trace-based LTD would be too
            # aggressive (would weaken inputs that were active recently).
            ltd_rate = self.ltd_rate * neuromod
            inactive_input = 1.0 - flat_input
            _n_per = self.input_lamina.n_per_col
            winner_cols = winner_indices // _n_per
            # col_masks == neuron_masks since ff_mask = repeat(col_mask, n_l4)
            col_masks = self._col_mask[:, winner_cols]
            # Matrix-vector products instead of element-wise broadcast+sum
            local_on = np.maximum(flat_input @ col_masks, 1.0)
            local_off = np.maximum(inactive_input @ col_masks, 1.0)
            local_scales = local_on / local_off
            w -= (
                ltd_rate
                * local_scales[np.newaxis, :]
                * inactive_input[:, np.newaxis]
                * col_masks
            )

            # Mask and clamp in-place, write back once
            w[~col_masks] = 0.0
            np.clip(w, 0, 1, out=w)
            ff_weights[:, winner_indices] = w

        # Subthreshold: weak LTP on ALL neurons (uses trace too)
        active_dims = np.flatnonzero(ltp_signal > 0.01)
        if len(active_dims) > 0:
            # Single read-modify-write: add sub-LTP, mask, clamp
            ff_mask = self.ff_mask
            w_sub = ff_weights[active_dims]
            w_sub += ltp_rate * 0.1 * ltp_signal[active_dims, np.newaxis]
            w_sub *= ff_mask[active_dims]
            np.minimum(w_sub, 1, out=w_sub)
            ff_weights[active_dims] = w_sub

    # ------------------------------------------------------------------
    # Three-factor reward consolidation
    # ------------------------------------------------------------------

    # Minimum reward magnitude to trigger consolidation.
    # Rewards below this are treated as zero (avoids noise accumulation).
    REWARD_DEAD_ZONE: float = 1e-6

    def apply_reward(self, reward: float) -> None:
        """Consolidate eligibility traces into ff_weights using reward.

        Three-factor rule: dw = reward * eligibility_trace
        Positive reward strengthens recent pathways, negative weakens.

        Subclasses with additional eligibility traces (output_weights,
        goal_weights) should override and call super() first.

        No-op for regions without _ff_eligibility (sensory regions use
        two-factor Hebbian, not reward-gated learning).
        """
        if not self.learning_enabled:
            return
        if abs(reward) < self.REWARD_DEAD_ZONE:
            return
        if self._ff_eligibility is None:
            return

        # Clamp eligibility traces before consolidation
        if self._eligibility_clip > 0:
            clip = self._eligibility_clip
            np.clip(self._ff_eligibility, -clip, clip, out=self._ff_eligibility)

        self.ff_weights += reward * self._ff_eligibility
        self.ff_weights *= self.ff_mask
        np.clip(self.ff_weights, 0, 1, out=self.ff_weights)

    # ------------------------------------------------------------------
    # Dendritic segments
    # ------------------------------------------------------------------

    def _init_segments(self):
        """Initialize dendritic segment arrays with random connectivity."""
        n = self.n_l4_total
        n_syn = self.n_synapses_per_segment

        # Lateral segments: L4 → L4
        n_lat = self.n_l4_lat_segments
        self.l4_lat_seg_indices = np.zeros((n, n_lat, n_syn), dtype=np.int32)
        self.l4_lat_seg_perm = np.zeros((n, n_lat, n_syn))

        lat_pool = np.arange(self.n_l4_total)

        for i in range(n):
            for s in range(self.n_l4_lat_segments):
                self.l4_lat_seg_indices[i, s] = self._rng.choice(
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

        # L5 lateral segments: L5 → L5 (output-layer sequence prediction)
        # Only allocated when n_l5_segments > 0 (disabled by default).
        n5 = self.n_l5_total
        self._l5_source_pool = np.arange(n5)
        if self.n_l5_segments > 0:
            self.l5_seg_indices = np.zeros(
                (n5, self.n_l5_segments, n_syn), dtype=np.int32
            )
            self.l5_seg_perm = np.zeros((n5, self.n_l5_segments, n_syn))
            for i in range(n5):
                for s in range(self.n_l5_segments):
                    self.l5_seg_indices[i, s] = self._rng.choice(
                        self._l5_source_pool,
                        n_syn,
                        replace=len(self._l5_source_pool) < n_syn,
                    )
        else:
            self.l5_seg_indices = np.zeros((n5, 0, n_syn), dtype=np.int32)
            self.l5_seg_perm = np.zeros((n5, 0, n_syn))

    def init_apical_context(
        self,
        source_dim: int,
        source_name: str = "",
        *,
        target_lamina: LaminaID = LaminaID.L5,
    ):
        """Initialize apical feedback from a higher region.

        Creates dendritic segments on the target lamina for
        context-specific gating. Models biological apical dendrites:
        top-down input modulates firing threshold without directly
        causing spikes.

        Args:
            source_dim: Dimensionality of the source signal.
            source_name: Identifier for this source (e.g., "S2").
            target_lamina: Which lamina receives the apical input.
                L5 (default) and L2/3 are biologically motivated.
        """
        name = source_name or f"src_{len(self._apical_sources)}"
        tgt_lam = self.get_lamina(target_lamina)
        n_neurons = tgt_lam.n_total

        n_seg = self.n_apical_segments
        n_syn = self.n_synapses_per_segment
        seg_indices = np.zeros((n_neurons, n_seg, n_syn), dtype=np.int32)
        seg_perm = np.zeros((n_neurons, n_seg, n_syn))
        source_pool = np.arange(source_dim)
        for i in range(n_neurons):
            for s in range(n_seg):
                seg_indices[i, s] = self._rng.choice(
                    source_pool, n_syn, replace=len(source_pool) < n_syn
                )
        self._apical_sources[name] = {
            "dim": source_dim,
            "target_lamina": target_lamina,
            "context": np.zeros(source_dim, dtype=np.float64),
            "seg_indices": seg_indices,
            "seg_perm": seg_perm,
            "source_pool": source_pool,
        }

        # Backward-compat: set single-source aliases to first source
        if self._apical_source_dim == 0:
            self._apical_source_dim = source_dim
            src = self._apical_sources[name]
            self._apical_context = src["context"]

    # Alias for circuit.connect()
    def init_apical_segments(
        self,
        source_dim: int,
        source_name: str = "",
        *,
        target_lamina: LaminaID = LaminaID.L5,
    ):
        """Alias for init_apical_context."""
        self.init_apical_context(source_dim, source_name, target_lamina=target_lamina)

    def register_lamina(self, lam: Lamina) -> None:
        """Register a lamina with this region."""
        lam.region = self
        self.laminae[lam.id] = lam

    def get_lamina(self, lid: LaminaID) -> Lamina:
        """Look up a lamina by ID."""
        return self.laminae[lid]

    @property
    def l4(self) -> Lamina:
        return self.laminae[LaminaID.L4]

    @property
    def l23(self) -> Lamina:
        return self.laminae[LaminaID.L23]

    @property
    def l5(self) -> Lamina:
        return self.laminae[LaminaID.L5]

    @property
    def has_l4(self) -> bool:
        """Whether this region has an L4 (input) lamina."""
        return self.n_l4 > 0

    @property
    def has_l5(self) -> bool:
        """Whether this region has an L5 (output) lamina."""
        return self.n_l5 > 0

    @property
    def input_lamina(self) -> Lamina:
        """The lamina that receives feedforward drive (L4 or L2/3)."""
        return self.l4 if self.has_l4 else self.l23

    @property
    def output_lamina(self) -> Lamina:
        """The lamina that provides output (L5 or L2/3)."""
        return self.l5 if self.has_l5 else self.l23

    @property
    def has_apical(self) -> bool:
        """Whether any apical source has been initialized."""
        return len(self._apical_sources) > 0

    def predict_neuron(self, l4_idx: int, source_idx: int, segment_type: str = "lat"):
        """Set up a dendritic segment that fires when source_idx is active.

        For testing. Fills all synapses in segment 0 with the given source
        index and sets permanences to 1.0, guaranteeing the segment fires
        whenever the source neuron is active.
        """
        self.l4_lat_seg_indices[l4_idx, 0, :] = source_idx
        self.l4_lat_seg_perm[l4_idx, 0, :] = 1.0

    def _get_source_pool(self, neuron: int) -> np.ndarray:
        """Get valid source neuron indices for growing lateral synapses."""
        return self._l4_lat_source_pool

    def reset_working_memory(self):
        """Reset transient state, preserving learned synaptic weights and segments."""
        self.l4.voltage[:] = 0.0
        self.l23.voltage[:] = 0.0
        self.l23.firing_rate[:] = 0.0
        self.l5.firing_rate[:] = 0.0
        self.l5.active[:] = False
        self.output_scores[:] = 0.0
        self.l4.trace[:] = 0.0
        self.l23.trace[:] = 0.0
        self.l4.excitability[:] = 0.0
        self.l23.excitability[:] = 0.0
        self.l4.active[:] = False
        self.l23.active[:] = False
        self.active_columns[:] = False
        self.bursting_columns[:] = False
        self.l4.predicted[:] = False
        self.l23.predicted[:] = False
        self.l5.predicted[:] = False
        self._pred_context_l23[:] = False
        self._pred_context_l4[:] = False
        self._efference_copy = None
        if self._pre_trace is not None:
            self._pre_trace[:] = 0.0
        if self._seg_trace_l23 is not None:
            self._seg_trace_l23[:] = 0.0
        if self._seg_trace_l4 is not None:
            self._seg_trace_l4[:] = 0.0
        if self._seg_trace_l5 is not None:
            self._seg_trace_l5[:] = 0.0
        if self._ff_eligibility is not None:
            self._ff_eligibility[:] = 0.0
        # Clear apical contexts (preserve learned weights)
        for src in self._apical_sources.values():
            src["context"][:] = 0.0

    def _check_segments(
        self,
        ctx: np.ndarray,
        seg_indices: np.ndarray,
        seg_perm: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """Check which neurons have active segments given context.

        Shared prediction logic for L4 (fb/lat), L2/3, and L5 segments.
        ORs results into `out` (caller initializes to False).
        """
        if not ctx.any():
            return
        if _HAS_NUMBA:
            out |= _nb_predict(
                ctx,
                seg_indices,
                seg_perm,
                self.perm_threshold,
                self.seg_activation_threshold,
            )
        else:
            active_at_syn = ctx[seg_indices]
            connected = seg_perm > self.perm_threshold
            counts = (active_at_syn & connected).sum(axis=2)
            out |= (counts >= self.seg_activation_threshold).any(axis=1)

    def _predict_from_segments(self) -> np.ndarray:
        """Check which L4 neurons have active dendritic segments.

        Returns boolean mask of shape (n_l4_total,).
        """
        predicted = np.zeros(self.n_l4_total, dtype=np.bool_)
        self._check_segments(
            self.l4.active, self.l4_lat_seg_indices, self.l4_lat_seg_perm, predicted
        )
        return predicted

    def get_prediction(self, k: int) -> np.ndarray:
        """Return predicted L4 neuron indices via dendritic segments.

        The k parameter is accepted for API compatibility but ignored —
        segment prediction is binary (predicted or not), not top-k.
        """
        return np.nonzero(self._predict_from_segments())[0]

    def step(
        self,
        drive: np.ndarray,
        *,
        forced_columns: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run one timestep given per-neuron feedforward drive.

        Args:
            drive: Per-neuron feedforward drive targeting the input lamina.
                   Shape (n_l4_total,) for granular, (n_l23_total,) for agranular.
            forced_columns: If provided, skip k-WTA and use these column
                indices instead. Used for motor exploration (random column forcing).

        Returns:
            Array of global indices of active input-lamina neurons.
        """
        if not self.has_l4:
            return self._step_no_l4(drive, forced_columns=forced_columns)

        # 1. Decay voltages
        self.l4.voltage *= self.voltage_decay
        self.l23.voltage *= self.voltage_decay

        # 2. Compute predictive state BEFORE feedforward input.
        self._compute_predictions()

        # 3. Save prediction-time context for segment learning
        #    (current active state is from the previous step)
        self._pred_context_l23[:] = self.l23.active
        self._pred_context_l4[:] = self.l4.active
        # Update continuous segment traces (STDP-like)
        if self._seg_trace_l23 is not None:
            self._seg_trace_l23 *= self._pre_trace_decay
            self._seg_trace_l23[self.l23.active] += 1.0
        if self._seg_trace_l4 is not None:
            self._seg_trace_l4 *= self._pre_trace_decay
            self._seg_trace_l4[self.l4.active] += 1.0
        if self._seg_trace_l5 is not None:
            self._seg_trace_l5 *= self._pre_trace_decay
            self._seg_trace_l5[self.l5.active] += 1.0

        # 4. Feedforward drive to L4 neurons
        self.l4.voltage += drive

        # 5. Predicted neurons get a voltage boost (they're primed)
        # Apical context acts via L5 segments (predicted_l5 boost in _activate_l5).
        self.l4.voltage[self.l4.predicted] += self.fb_boost

        # 6. Activate L4: top-k columns, then burst/precise per column
        scores_l4 = self.l4.voltage + self.l4.excitability
        top_cols = (
            forced_columns
            if forced_columns is not None
            else self._select_columns(scores_l4)
        )
        self._activate_l4_burst(top_cols, scores_l4)

        # 6b. Update L4 firing rate (EMA) so L4→L2/3 weights see non-zero drive
        self.l4.firing_rate *= self.voltage_decay
        self.l4.firing_rate[self.l4.active] += 1.0 - self.voltage_decay

        # 7. Activate L2/3: L4 feedforward + lateral context
        self._activate_l23(top_cols)

        # 7b. Activate L5: intra-columnar drive from L2/3
        self._activate_l5(top_cols)

        # 8. Learn (dendritic segment permanence updates)
        if self.learning_enabled:
            self._learn()

            # 8b. Apical segment learning (L5 and L2/3 targets)
            if self.has_apical:
                self._learn_apical()

        # 9. Update eligibility traces for newly active neurons
        self._update_traces()

        # 10. Homeostatic excitability (capped)
        self._update_excitability()

        # 11. Refractory: reset voltage for active neurons
        self.l4.voltage[self.l4.active] = 0.0
        self.l23.voltage[self.l23.active] = 0.0

        # 12. Clamp voltage (bounded membrane potential)
        np.clip(self.l4.voltage, 0.0, 1.0, out=self.l4.voltage)
        np.clip(self.l23.voltage, 0.0, 1.0, out=self.l23.voltage)

        # 13. Update firing rate estimates (EMA of spike train)
        self.l23.firing_rate *= self.voltage_decay
        self.l23.firing_rate[self.l23.active] += 1.0 - self.voltage_decay
        self.l5.firing_rate *= self.voltage_decay
        self.l5.firing_rate[self.l5.active] += 1.0 - self.voltage_decay

        # 14. Update L5 output scores (per-column mean L5 firing rate)
        if self.n_l5 > 0:
            self.output_scores[:] = self.l5.firing_rate.reshape(
                self.n_columns, self.n_l5
            ).mean(axis=1)
        else:
            self.output_scores[:] = 0.0

        return np.nonzero(self.l4.active)[0]

    def _step_no_l4(
        self,
        drive: np.ndarray,
        *,
        forced_columns: np.ndarray | None = None,
    ) -> np.ndarray:
        """Agranular processing: feedforward drive targets L2/3 directly.

        Used when n_l4=0 (motor cortex, PFC). Input reception, column
        selection, and burst/precise determination all happen on L2/3.
        L2/3 then drives L5 via standard intra-column pathway.
        """
        # 1. Decay L2/3 voltage (no L4 to decay)
        self.l23.voltage *= self.voltage_decay

        # 2. Predictions on L2/3 (L4 predictions are no-ops on size-0 arrays)
        self._compute_predictions()

        # 3. Save prediction context (L2/3 only)
        self._pred_context_l23[:] = self.l23.active
        if self._seg_trace_l23 is not None:
            self._seg_trace_l23 *= self._pre_trace_decay
            self._seg_trace_l23[self.l23.active] += 1.0
        if self._seg_trace_l5 is not None:
            self._seg_trace_l5 *= self._pre_trace_decay
            self._seg_trace_l5[self.l5.active] += 1.0

        # 4. Feedforward drive to L2/3 (not L4)
        self.l23.voltage += drive

        # 5. Prediction boost on L2/3
        self.l23.voltage[self.l23.predicted] += self.fb_boost

        # 6. Column selection on L2/3
        scores = self.l23.voltage + self.l23.excitability
        top_cols = (
            forced_columns
            if forced_columns is not None
            else self._select_columns_generic(scores, self.n_l23)
        )

        # 7. Burst/precise on L2/3
        self._activate_input_burst(top_cols, scores, self.l23, self.n_l23)

        # 8. L2/3 firing rate (EMA)
        self.l23.firing_rate *= self.voltage_decay
        self.l23.firing_rate[self.l23.active] += 1.0 - self.voltage_decay

        # 9. L5 activation via L2/3→L5 pathway
        self._activate_l5(top_cols)

        # 10. Learn (L2/3 segments only — no L4 segments)
        if self.learning_enabled:
            self._learn()
            if self.has_apical:
                self._learn_apical()

        # 11. Traces (L2/3 only)
        self.l23.trace *= self.eligibility_decay
        active_cols = np.nonzero(self.active_columns)[0]
        if len(active_cols) > 0:
            v_by_col = self.l23.voltage.reshape(self.n_columns, self.n_l23)
            a_by_col = self.l23.active.reshape(self.n_columns, self.n_l23)
            burst_cols = active_cols[self.bursting_columns[active_cols]]
            if len(burst_cols) > 0:
                best = v_by_col[burst_cols].argmax(axis=1)
                self.l23.trace[burst_cols * self.n_l23 + best] = 1.0
            precise_cols = active_cols[~self.bursting_columns[active_cols]]
            if len(precise_cols) > 0:
                best = a_by_col[precise_cols].argmax(axis=1)
                self.l23.trace[precise_cols * self.n_l23 + best] = 1.0

        # 12. Excitability (L2/3 only)
        inc = self.max_excitability / self.n_l23
        self.l23.excitability[~self.l23.active] += inc
        self.l23.excitability[self.l23.active] = 0.0
        np.clip(
            self.l23.excitability,
            0,
            self.max_excitability,
            out=self.l23.excitability,
        )

        # 13. Refractory reset
        self.l23.voltage[self.l23.active] = 0.0
        np.clip(self.l23.voltage, 0.0, 1.0, out=self.l23.voltage)

        # 14. L5 firing rate + output scores
        if self.has_l5:
            self.l5.firing_rate *= self.voltage_decay
            self.l5.firing_rate[self.l5.active] += 1.0 - self.voltage_decay
            self.output_scores[:] = self.l5.firing_rate.reshape(
                self.n_columns, self.n_l5
            ).mean(axis=1)
        else:
            self.output_scores[:] = 0.0

        # Store column drive for diagnostics
        self.last_column_drive[:] = self.l23.voltage.reshape(
            self.n_columns, self.n_l23
        ).max(axis=1)

        return np.nonzero(self.l23.active)[0]

    def _predict_l23_from_segments(self) -> np.ndarray:
        """Check which L2/3 neurons have active lateral dendritic segments.

        Returns boolean mask of shape (n_l23_total,).
        """
        predicted = np.zeros(self.n_l23_total, dtype=np.bool_)
        self._check_segments(
            self.l23.active, self.l23_seg_indices, self.l23_seg_perm, predicted
        )
        return predicted

    def set_apical_context(self, context: np.ndarray, source_name: str = ""):
        """Set apical feedback signal from a specific source.

        Called each step by the runner before this region's step().
        Multiple sources accumulate — each has its own context buffer.

        Args:
            context: Firing rate signal from the source region's L2/3.
            source_name: Which source (must match init_apical_context).
        """
        if source_name and source_name in self._apical_sources:
            self._apical_sources[source_name]["context"][:] = context
        elif self._apical_sources:
            # Backward compat: if no name given, match by dimension
            for src in self._apical_sources.values():
                if src["dim"] == len(context):
                    src["context"][:] = context
                    break
            else:
                # Fallback: set first source
                first = next(iter(self._apical_sources.values()))
                n = min(len(context), first["dim"])
                first["context"][:n] = context[:n]
        # Update backward-compat alias to whichever source has signal
        for src in self._apical_sources.values():
            if src["context"].any():
                self._apical_context = src["context"]
                break

    def _compute_predictions(self):
        """Determine which neurons are in predictive state via segments."""
        self.l4.predicted[:] = self._predict_from_segments()
        self.l23.predicted[:] = self._predict_l23_from_segments()
        # L5 prediction from lateral segments (if enabled)
        if self.n_l5_segments > 0:
            self.l5.predicted[:] = self._predict_l5_lateral_from_segments()
        else:
            self.l5.predicted[:] = False
        # Apical predictions (additive with lateral)
        if self._apical_sources:
            self.l5.predicted |= self._predict_from_apical_segments(LaminaID.L5)
            self.l23.predicted |= self._predict_from_apical_segments(LaminaID.L23)

    def _predict_l5_lateral_from_segments(self) -> np.ndarray:
        """Check which L5 neurons have active lateral dendritic segments.

        L5→L5 lateral prediction: which L5 neurons are expected given
        the current L5 activation pattern? Enables output-layer sequence
        prediction.

        Returns boolean mask of shape (n_l5_total,).
        """
        predicted = np.zeros(self.n_l5_total, dtype=np.bool_)
        self._check_segments(
            self.l5.active, self.l5_seg_indices, self.l5_seg_perm, predicted
        )
        return predicted

    def _predict_from_apical_segments(self, target_lamina: LaminaID) -> np.ndarray:
        """Check which neurons have active apical dendritic segments.

        Filters apical sources by target lamina. Each matching source
        contributes independently. A neuron is predicted if any of
        its segments has enough connected synapses with active context.

        Returns boolean mask of shape (n_total,) for the target lamina.
        """
        tgt_lam = self.get_lamina(target_lamina)
        predicted = np.zeros(tgt_lam.n_total, dtype=np.bool_)
        for src in self._apical_sources.values():
            if src.get("target_lamina") != target_lamina:
                continue
            seg_idx = src.get("seg_indices")
            seg_p = src.get("seg_perm")
            if seg_idx is None:
                continue
            ctx = src["context"]
            if not ctx.any():
                continue
            self._check_segments(ctx > 0.01, seg_idx, seg_p, predicted)
        return predicted

    def _select_columns(self, scores: np.ndarray) -> np.ndarray:
        """Select top-k columns by max neuron score (L4)."""
        return self._select_columns_generic(scores, self.n_l4)

    def _select_columns_generic(self, scores: np.ndarray, n_per_col: int) -> np.ndarray:
        """Select top-k columns by max neuron score on any lamina."""
        by_col = scores.reshape(self.n_columns, n_per_col)
        col_scores = by_col.max(axis=1)

        if self.k_columns >= self.n_columns:
            return np.arange(self.n_columns)
        return np.argpartition(col_scores, -self.k_columns)[-self.k_columns :]

    def _activate_l4_burst(self, top_cols: np.ndarray, scores: np.ndarray):
        """Activate L4 neurons with burst/precise distinction."""
        self._activate_input_burst(top_cols, scores, self.l4, self.n_l4)

    def _activate_input_burst(
        self,
        top_cols: np.ndarray,
        scores: np.ndarray,
        lamina: Lamina,
        n_per_col: int,
    ):
        """Activate neurons on any lamina with burst/precise distinction.

        For each winning column:
        - If any neuron was predicted by dendritic segments → precise activation:
          only the best-scoring predicted neuron fires.
        - If no neuron was predicted → burst: all neurons fire (surprise signal).
        """
        self.active_columns[:] = False
        self.active_columns[top_cols] = True
        lamina.active[:] = False
        self.bursting_columns[:] = False

        predicted_by_col = lamina.predicted.reshape(self.n_columns, n_per_col)
        scores_by_col = scores.reshape(self.n_columns, n_per_col)

        # Work only with the winning columns
        tc_predicted = predicted_by_col[top_cols]  # (k, n_per_col)
        tc_scores = scores_by_col[top_cols].copy()
        has_prediction = tc_predicted.any(axis=1)  # (k,)

        # Burst columns: no prediction — all neurons fire
        burst_cols = top_cols[~has_prediction]
        self.bursting_columns[burst_cols] = True
        if len(burst_cols) > 0:
            active_by_col = lamina.active.reshape(self.n_columns, n_per_col)
            active_by_col[burst_cols] = True

        # Precise columns: mask unpredicted neurons, pick best scorer
        precise_mask = has_prediction
        if precise_mask.any():
            precise_cols = top_cols[precise_mask]
            p_scores = tc_scores[precise_mask]
            p_predicted = tc_predicted[precise_mask]
            p_scores[~p_predicted] = -np.inf
            winners = p_scores.argmax(axis=1)
            global_indices = precise_cols * n_per_col + winners
            lamina.active[global_indices] = True

    def _activate_downstream(
        self,
        top_cols: np.ndarray,
        src_lamina: Lamina,
        tgt_lamina: Lamina,
        weights: np.ndarray,
    ) -> None:
        """Activate a downstream lamina via per-column ff weights.

        Shared logic for L4→L2/3 and L2/3→L5 intra-column pathways.
        Drive = source firing_rate @ weights + prediction boost.
        Burst columns: all target neurons fire.
        Precise columns: single winner by score.
        """
        n_src = src_lamina.n_per_col
        n_tgt = tgt_lamina.n_per_col

        # Feedforward drive via per-column weights
        tgt_lamina.active[:] = False
        if len(top_cols) == 0:
            return

        src_fr = src_lamina.firing_rate.reshape(self.n_columns, n_src)
        if len(top_cols) > 0:
            drive = np.einsum(
                "ci,cij->cj",
                src_fr[top_cols],
                weights[top_cols],
            )
            tgt_v = tgt_lamina.voltage.reshape(self.n_columns, n_tgt)
            tgt_v[top_cols] += drive

        # Prediction boost
        boost = self.l23_prediction_boost or self.fb_boost
        tgt_lamina.voltage[tgt_lamina.predicted] += boost

        # Competitive selection
        scores = tgt_lamina.voltage + tgt_lamina.excitability
        by_col = scores.reshape(self.n_columns, n_tgt)

        # Burst: all fire
        burst_cols = top_cols[self.bursting_columns[top_cols]]
        if len(burst_cols) > 0:
            active_by_col = tgt_lamina.active.reshape(self.n_columns, n_tgt)
            active_by_col[burst_cols] = True

        # Precise: single winner
        precise_cols = top_cols[~self.bursting_columns[top_cols]]
        if len(precise_cols) > 0:
            winners = by_col[precise_cols].argmax(axis=1)
            tgt_lamina.active[precise_cols * n_tgt + winners] = True

    def _activate_l23(self, top_cols: np.ndarray):
        """Activate L2/3 via L4 → L2/3 per-column weights."""
        self._activate_downstream(top_cols, self.l4, self.l23, self.l4_to_l23_weights)

    def _activate_l5(self, top_cols: np.ndarray):
        """Activate L5 via L2/3 → L5 per-column weights."""
        if self.n_l5 == 0:
            return
        self._activate_downstream(top_cols, self.l23, self.l5, self.l23_to_l5_weights)

    def _learn_apical(self):
        """Update apical segment permanences on all target laminae.

        Per-source, per-target-lamina grow/reinforce/punish:
        - Active column, neuron predicted → reinforce matching segments
        - Active column, neuron NOT predicted → grow on winner
        - Predicted but not active → punish false positives
        """
        active_cols = np.nonzero(self.active_columns)[0]

        for src in self._apical_sources.values():
            seg_idx = src.get("seg_indices")
            seg_perm = src.get("seg_perm")
            pool = src.get("source_pool")
            tgt_lid = src.get("target_lamina", LaminaID.L5)
            if seg_idx is None:
                continue
            ctx = src["context"]
            if not ctx.any():
                continue
            ctx_bool = ctx > 0.01

            tgt_lam = self.get_lamina(tgt_lid)
            n_per = tgt_lam.n_per_col
            active_by_col = tgt_lam.active.reshape(self.n_columns, n_per)
            pred_by_col = tgt_lam.predicted.reshape(self.n_columns, n_per)

            reinforce_neurons = []
            grow_neurons = []
            for col in active_cols:
                active_in_col = np.nonzero(active_by_col[col])[0]
                if len(active_in_col) == 0:
                    continue
                predicted_in_col = active_by_col[col] & pred_by_col[col]
                if predicted_in_col.any():
                    for local_idx in np.nonzero(predicted_in_col)[0]:
                        reinforce_neurons.append(col * n_per + local_idx)
                else:
                    winner = active_in_col[0]
                    grow_neurons.append(col * n_per + winner)

            if reinforce_neurons:
                self._adapt_segments_batch(
                    np.array(reinforce_neurons, dtype=np.intp),
                    seg_idx,
                    seg_perm,
                    ctx_bool,
                    reinforce=True,
                )
            for neuron in grow_neurons:
                self._grow_segment(neuron, seg_idx, seg_perm, ctx_bool, pool)

            # Punish false positives
            false_pred = np.nonzero(tgt_lam.predicted & ~tgt_lam.active)[0]
            if len(false_pred) > 0:
                self._adapt_segments_batch(
                    false_pred,
                    seg_idx,
                    seg_perm,
                    ctx_bool,
                    reinforce=False,
                )

    def _learn(self):
        """Dendritic segment updates for all layers.

        L4 segments: feedback (L2/3→L4) and lateral (L4→L4).
        L2/3 segments: lateral (L2/3→L2/3).
        L5 segments: lateral (L5→L5) output-layer sequence prediction.
        All use sparse dendritic segment learning (grow/reinforce/punish).
        """
        self._learn_l4_segments()
        self._learn_l23_segments()
        self._learn_intra_column_ff()
        if self.n_l5_segments > 0:
            self._learn_l5_lateral_segments()

    def _learn_intra_column_ff(self):
        """Hebbian learning on intra-column feedforward weights.

        L4→L2/3 and L2/3→L5: same rule applied to both pathways.
        """
        active_cols = np.nonzero(self.active_columns)[0]
        if len(active_cols) == 0:
            return

        ltp_rate = self.learning_rate * self.surprise_modulator
        ltd_rate = self.ltd_rate * 0.5  # gentler LTD for within-column

        if self.n_l4 > 0:
            self._learn_column_weights(
                active_cols,
                self.l4_to_l23_weights,
                self.l4.firing_rate,
                self.l23.active,
                self.n_l4,
                self.n_l23,
                ltp_rate,
                ltd_rate,
            )
        if self.n_l5 > 0:
            self._learn_column_weights(
                active_cols,
                self.l23_to_l5_weights,
                self.l23.firing_rate,
                self.l5.active,
                self.n_l23,
                self.n_l5,
                ltp_rate,
                ltd_rate,
            )

    def _learn_column_weights(
        self,
        active_cols: np.ndarray,
        weights: np.ndarray,
        src_fr: np.ndarray,
        tgt_active: np.ndarray,
        n_src: int,
        n_tgt: int,
        ltp_rate: float,
        ltd_rate: float,
    ) -> None:
        """Hebbian update on per-column weight matrix.

        For each active column, strengthen connections from active
        source neurons to active target neurons (LTP), weaken from
        inactive source neurons (LTD). Clamps to [0, 1].

        Vectorized: computes outer-product updates for all active
        columns simultaneously using broadcasting.
        """
        if len(active_cols) == 0:
            return
        src_by_col = src_fr.reshape(self.n_columns, n_src)
        tgt_by_col = tgt_active.reshape(self.n_columns, n_tgt)

        # Gather source firing rates and target activity for active columns
        # src_active: (n_active, n_src), tgt_f: (n_active, n_tgt)
        src_active = src_by_col[active_cols]
        tgt_f = tgt_by_col[active_cols]

        # Per-winner update: w[:, j] += ltp_rate * src - ltd_rate * (1 - src)
        #                   = (ltp_rate + ltd_rate) * src * tgt - ltd_rate * tgt
        # Vectorised as outer product over all active columns at once:
        # delta[c, i, j] = ((ltp_rate + ltd_rate) * src[c, i] - ltd_rate) * tgt[c, j]
        combined_rate = ltp_rate + ltd_rate
        # (n_active, n_src, 1) * (n_active, 1, n_tgt) -> (n_active, n_src, n_tgt)
        scaled_src = combined_rate * src_active - ltd_rate
        delta = scaled_src[:, :, np.newaxis] * tgt_f[:, np.newaxis, :]

        weights[active_cols] += delta
        np.clip(weights, 0.0, 1.0, out=weights)

    def _learn_l4_segments(self):
        """Update L4 dendritic segment permanences based on prediction outcomes.

        No-op when n_l4=0 (agranular regions have no L4 segments).

        - Burst (unpredicted): grow best-matching segment on trace winner
        - Precise + predicted: reinforce the active segments
        - Predicted but didn't fire: punish the active segments

        When segment traces are enabled, uses continuous traces (with
        temporal depth) for segment growth. Adapt still uses boolean
        context (segments need active/inactive distinction for
        permanence increment/decrement).
        """
        if self.n_l4 == 0:
            return
        active_cols = np.nonzero(self.active_columns)[0]
        voltage_by_col = self.l4.voltage.reshape(self.n_columns, self.n_l4)

        # Burst columns: grow segment on trace winner
        burst_cols = active_cols[self.bursting_columns[active_cols]]
        if len(burst_cols) > 0:
            best_in_col = voltage_by_col[burst_cols].argmax(axis=1)
            for i, col in enumerate(burst_cols):
                self._grow_best_segment(col * self.n_l4 + best_in_col[i])

        # Context for adapt: use continuous traces (if enabled) for
        # weighted scoring, or boolean context for binary scoring
        if self._seg_trace_l4 is not None:
            lat_ctx = self._seg_trace_l4
        else:
            lat_ctx = self._pred_context_l4

        # Precise + predicted: batch reinforce
        reinforce_neurons = np.nonzero(
            self.l4.active
            & self.l4.predicted
            & np.repeat(self.active_columns & ~self.bursting_columns, self.n_l4)
        )[0]
        if len(reinforce_neurons) > 0:
            self._adapt_segments_batch(
                reinforce_neurons,
                self.l4_lat_seg_indices,
                self.l4_lat_seg_perm,
                lat_ctx,
                reinforce=True,
            )

        # Punish false predictions (predicted but didn't fire)
        false_predicted = np.nonzero(self.l4.predicted & ~self.l4.active)[0]
        if len(false_predicted) > 0:
            self._adapt_segments_batch(
                false_predicted,
                self.l4_lat_seg_indices,
                self.l4_lat_seg_perm,
                lat_ctx,
                reinforce=False,
            )

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
                neuron,
                seg_indices,
                seg_perm,
                ctx,
                pool.astype(np.int32) if pool.dtype != np.int32 else pool,
                inc,
                dec,
                self.perm_init,
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
        """Reinforce or punish active segments for a batch of neurons.

        Supports both boolean ctx (original) and continuous trace ctx.
        With traces, segment activation scoring uses weighted sums
        instead of binary counts.
        """
        if len(neurons) == 0 or not ctx.any():
            return

        neuromod = self.surprise_modulator * self.reward_modulator
        inc = self.perm_increment * neuromod
        dec = self.perm_decrement * neuromod

        is_continuous = ctx.dtype != np.bool_

        # Numba path only for boolean ctx
        if not is_continuous and _HAS_NUMBA:
            _nb_adapt(
                neurons.astype(np.intp),
                seg_indices,
                seg_perm,
                ctx,
                self.perm_threshold,
                self.seg_activation_threshold,
                inc,
                dec,
                reinforce,
            )
            return

        batch_idx = seg_indices[neurons]
        batch_perm = seg_perm[neurons]
        ctx_at_syn = ctx[batch_idx]
        connected = batch_perm > self.perm_threshold

        # Score segments: continuous weighted sum or binary count
        if is_continuous:
            scores = (ctx_at_syn * connected).sum(axis=2)
        else:
            scores = (ctx_at_syn & connected).sum(axis=2)
        threshold = float(self.seg_activation_threshold)
        active_mask = scores >= threshold

        if not active_mask.any():
            return

        active_f = active_mask[:, :, np.newaxis].astype(np.float64)
        syn_f = ctx_at_syn.astype(np.float64)

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
        """Grow the best-matching L4 lateral segment for a bursting neuron.

        When segment traces are enabled, uses continuous traces (recent
        history) for overlap scoring and growth — giving segments
        temporal credit.
        """
        # Use traces if available, else boolean context
        lat_ctx = self._pred_context_l4
        if self._seg_trace_l4 is not None:
            lat_ctx = self._seg_trace_l4

        if not lat_ctx.any():
            return

        # Convert continuous trace to boolean for _grow_segment
        # (growth uses active/inactive distinction)
        ctx = lat_ctx
        if ctx.dtype != np.bool_:
            threshold = self._pre_trace_threshold or 0.01
            ctx = ctx > threshold

        pool = self._get_source_pool(neuron)
        self._grow_segment(
            neuron,
            self.l4_lat_seg_indices,
            self.l4_lat_seg_perm,
            ctx,
            pool,
        )

    def _adapt_segments(self, neuron: int, reinforce: bool):
        """Reinforce or punish active L4 lateral segments."""
        self._adapt_segment_array(
            neuron,
            self.l4_lat_seg_indices,
            self.l4_lat_seg_perm,
            self._pred_context_l4,
            reinforce,
        )

    # ------------------------------------------------------------------
    # L2/3 segment learning (lateral)
    # ------------------------------------------------------------------

    def _learn_l23_segments(self):
        """Update L2/3 lateral segment permanences.

        Uses segment traces (if enabled) for temporal credit in growth.
        Adapt uses thresholded traces or boolean context.
        """
        active_cols = np.nonzero(self.active_columns)[0]
        voltage_l23_by_col = self.l23.voltage.reshape(self.n_columns, self.n_l23)

        # Context for growth: use trace if available
        grow_ctx = self._pred_context_l23
        if self._seg_trace_l23 is not None:
            grow_ctx_continuous = self._seg_trace_l23
        else:
            grow_ctx_continuous = None

        # Context for adapt: continuous trace or boolean
        if self._seg_trace_l23 is not None:
            adapt_ctx = self._seg_trace_l23
        else:
            adapt_ctx = self._pred_context_l23

        # Burst columns: grow segment on trace winner
        burst_cols = active_cols[self.bursting_columns[active_cols]]
        if len(burst_cols) > 0:
            best_in_col = voltage_l23_by_col[burst_cols].argmax(axis=1)
            # Growth context: threshold trace to boolean for _grow_segment
            if grow_ctx_continuous is not None:
                threshold = self._pre_trace_threshold or 0.01
                grow_bool = grow_ctx_continuous > threshold
            else:
                grow_bool = grow_ctx
            for i, col in enumerate(burst_cols):
                best = col * self.n_l23 + best_in_col[i]
                pool = self._get_l23_source_pool(best)
                self._grow_segment(
                    best,
                    self.l23_seg_indices,
                    self.l23_seg_perm,
                    grow_bool,
                    pool,
                )

        # Precise + predicted: batch reinforce
        reinforce_neurons = np.nonzero(
            self.l23.active
            & self.l23.predicted
            & np.repeat(
                self.active_columns & ~self.bursting_columns,
                self.n_l23,
            )
        )[0]
        if len(reinforce_neurons) > 0:
            self._adapt_segments_batch(
                reinforce_neurons,
                self.l23_seg_indices,
                self.l23_seg_perm,
                adapt_ctx,
                reinforce=True,
            )

        # Punish false predictions
        false_predicted = np.nonzero(self.l23.predicted & ~self.l23.active)[0]
        if len(false_predicted) > 0:
            self._adapt_segments_batch(
                false_predicted,
                self.l23_seg_indices,
                self.l23_seg_perm,
                adapt_ctx,
                reinforce=False,
            )

    def _learn_l5_lateral_segments(self):
        """Update L5 lateral segment permanences.

        Same grow/reinforce/punish pattern as L2/3 lateral segments.
        Uses continuous traces (if enabled) for temporal credit,
        matching L2/3 segment learning.
        """
        active_cols = np.nonzero(self.active_columns)[0]
        if len(active_cols) == 0:
            return

        # Context: continuous trace or boolean L5 activation
        if self._seg_trace_l5 is not None:
            grow_ctx_continuous = self._seg_trace_l5
            adapt_ctx = self._seg_trace_l5
        else:
            grow_ctx_continuous = None
            adapt_ctx = self.l5.active

        # Growth context: threshold trace to boolean for _grow_segment
        if grow_ctx_continuous is not None:
            threshold = self._pre_trace_threshold or 0.01
            grow_bool = grow_ctx_continuous > threshold
        else:
            grow_bool = self.l5.active

        # Burst columns: grow L5 lateral segment on best L5 neuron
        burst_cols = active_cols[self.bursting_columns[active_cols]]
        if len(burst_cols) > 0:
            fr_by_col = self.l23.firing_rate.reshape(self.n_columns, self.n_l23)
            l23_winners = fr_by_col[burst_cols].argmax(axis=1)
            l5_best = np.minimum(l23_winners, self.n_l5 - 1)
            for i, col in enumerate(burst_cols):
                neuron = col * self.n_l5 + l5_best[i]
                self._grow_segment(
                    neuron,
                    self.l5_seg_indices,
                    self.l5_seg_perm,
                    grow_bool,
                    self._l5_source_pool,
                )

        # Precise + predicted: reinforce
        reinforce_neurons = np.nonzero(
            self.l5.active
            & self.l5.predicted
            & np.repeat(
                self.active_columns & ~self.bursting_columns,
                self.n_l5,
            )
        )[0]
        if len(reinforce_neurons) > 0:
            self._adapt_segments_batch(
                reinforce_neurons,
                self.l5_seg_indices,
                self.l5_seg_perm,
                adapt_ctx,
                reinforce=True,
            )

        # Punish false predictions
        false_predicted = np.nonzero(self.l5.predicted & ~self.l5.active)[0]
        if len(false_predicted) > 0:
            self._adapt_segments_batch(
                false_predicted,
                self.l5_seg_indices,
                self.l5_seg_perm,
                adapt_ctx,
                reinforce=False,
            )

    def _get_l23_source_pool(self, neuron: int) -> np.ndarray:
        """Get valid L2/3 source neuron indices for growing synapses."""
        return self._l23_source_pool

    def _update_traces(self):
        """Set active neuron traces to 1, decay the rest."""
        self.l4.trace *= self.eligibility_decay
        self.l23.trace *= self.eligibility_decay

        active_cols = np.nonzero(self.active_columns)[0]
        if len(active_cols) == 0:
            return

        voltage_l4_by_col = self.l4.voltage.reshape(self.n_columns, self.n_l4)
        voltage_l23_by_col = self.l23.voltage.reshape(self.n_columns, self.n_l23)
        active_l4_by_col = self.l4.active.reshape(self.n_columns, self.n_l4)
        active_l23_by_col = self.l23.active.reshape(self.n_columns, self.n_l23)

        # Burst columns: trace to highest-voltage neuron
        burst_cols = active_cols[self.bursting_columns[active_cols]]
        if len(burst_cols) > 0:
            best_l4 = voltage_l4_by_col[burst_cols].argmax(axis=1)
            self.l4.trace[burst_cols * self.n_l4 + best_l4] = 1.0
            best_l23 = voltage_l23_by_col[burst_cols].argmax(axis=1)
            self.l23.trace[burst_cols * self.n_l23 + best_l23] = 1.0

        # Precise columns: trace to the active neuron
        precise_cols = active_cols[~self.bursting_columns[active_cols]]
        if len(precise_cols) > 0:
            # argmax on boolean active mask gives the first True index
            best_l4 = active_l4_by_col[precise_cols].argmax(axis=1)
            self.l4.trace[precise_cols * self.n_l4 + best_l4] = 1.0
            best_l23 = active_l23_by_col[precise_cols].argmax(axis=1)
            self.l23.trace[precise_cols * self.n_l23 + best_l23] = 1.0

    def _update_excitability(self):
        """Boost inactive neurons, reset active ones (capped)."""
        inc_l4 = self.max_excitability / self.n_l4
        inc_l23 = self.max_excitability / self.n_l23
        self.l4.excitability[~self.l4.active] += inc_l4
        self.l23.excitability[~self.l23.active] += inc_l23
        self.l4.excitability[self.l4.active] = 0.0
        self.l23.excitability[self.l23.active] = 0.0
        np.clip(
            self.l4.excitability,
            0,
            self.max_excitability,
            out=self.l4.excitability,
        )
        np.clip(
            self.l23.excitability,
            0,
            self.max_excitability,
            out=self.l23.excitability,
        )
