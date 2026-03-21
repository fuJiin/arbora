"""Sensory cortical region: encoding-specific receptive fields.

Optional three-factor learning: hybrid Hebbian + eligibility traces.
When enabled, a fraction of the Hebbian update goes into traces that
consolidate based on downstream surprise (synaptic tagging model).
This enables longer-range causal learning: S1 ff_weights improve
when the representations they produce help S2 predict better.
"""

import numpy as np

from step.cortex.region import CorticalRegion


class SensoryRegion(CorticalRegion):
    """Cortical region with encoding-aware structural connectivity.

    Extends CorticalRegion with:
    - Encoding-width-based receptive field masking for ff_weights
    - Local connectivity for dendritic segments (topographic constraint)
    - L2/3 lateral connectivity mask (local neighborhood)
    - Optional three-factor learning (trace_fraction > 0)
    """

    def __init__(
        self,
        input_dim: int,
        *,
        encoding_width: int = 0,
        trace_fraction: float = 0.0,
        eligibility_clip: float = 0.05,
        seed: int = 0,
        **kwargs,
    ):
        # Store before super().__init__ so _build_ff_mask can use it
        self.encoding_width = encoding_width
        super().__init__(input_dim=input_dim, seed=seed, **kwargs)

        # Three-factor hybrid learning.
        # trace_fraction: what fraction of Hebbian update goes to traces
        # (remainder applied directly as two-factor). 0.0 = pure two-factor.
        self.trace_fraction = trace_fraction
        self._eligibility_clip = eligibility_clip
        if trace_fraction > 0:
            self._ff_eligibility = np.zeros_like(self.ff_weights)

    def _learn_ff(self, flat_input: np.ndarray):
        """Hybrid Hebbian + eligibility trace learning.

        When trace_fraction > 0, splits the Hebbian update:
        - (1 - trace_fraction) applied directly (two-factor baseline)
        - trace_fraction recorded in eligibility traces, consolidated
          when apply_surprise() is called by downstream surprise signal

        This lets sensory regions learn from input statistics (two-factor)
        while also learning from downstream consequences (three-factor).
        Models synaptic tagging: all synapses form short-term tags, but
        only tags present when a modulatory signal arrives become permanent.
        """
        if self.trace_fraction <= 0:
            # Pure two-factor: delegate to base class
            super()._learn_ff(flat_input)
            return

        # Decay eligibility traces
        self._ff_eligibility *= self.eligibility_decay

        active_cols = np.nonzero(self.active_columns)[0]
        if len(active_cols) == 0:
            return

        # Find winner neurons (same logic as base class)
        voltage_by_col = self.voltage_l4.reshape(
            self.n_columns, self.n_l4
        )
        active_by_col = self.active_l4.reshape(
            self.n_columns, self.n_l4
        )
        is_burst = self.bursting_columns[active_cols]
        winner_indices = np.empty(len(active_cols), dtype=np.intp)
        if is_burst.any():
            winner_indices[is_burst] = (
                active_cols[is_burst] * self.n_l4
                + voltage_by_col[active_cols[is_burst]].argmax(axis=1)
            )
        precise = ~is_burst
        if precise.any():
            winner_indices[precise] = (
                active_cols[precise] * self.n_l4
                + active_by_col[active_cols[precise]].argmax(axis=1)
            )

        neuromod = self.surprise_modulator * self.reward_modulator
        ltp_rate = self.learning_rate * neuromod

        # Split: direct fraction goes to weights, trace fraction to traces
        direct_rate = ltp_rate * (1.0 - self.trace_fraction)
        trace_rate = ltp_rate * self.trace_fraction

        # Direct LTP (two-factor portion)
        if len(winner_indices) > 0 and direct_rate > 0:
            self.ff_weights[:, winner_indices] += (
                direct_rate * flat_input[:, np.newaxis]
            )

        # Trace LTP (three-factor portion)
        if len(winner_indices) > 0 and trace_rate > 0:
            self._ff_eligibility[:, winner_indices] += (
                trace_rate * flat_input[:, np.newaxis]
            )

        # LTD: always applied directly (structural refinement)
        if len(winner_indices) > 0:
            ltd_rate = self.ltd_rate * neuromod
            inactive_input = 1.0 - flat_input
            winner_cols = winner_indices // self.n_l4
            col_masks = self._col_mask[:, winner_cols]
            local_on = np.maximum(
                (flat_input[:, np.newaxis] * col_masks).sum(axis=0),
                1.0,
            )
            local_off = np.maximum(
                (inactive_input[:, np.newaxis] * col_masks).sum(axis=0),
                1.0,
            )
            local_scales = local_on / local_off
            neuron_masks = self.ff_mask[:, winner_indices]
            self.ff_weights[:, winner_indices] -= (
                ltd_rate
                * local_scales[np.newaxis, :]
                * inactive_input[:, np.newaxis]
                * neuron_masks
            )
            w = self.ff_weights[:, winner_indices]
            w[~neuron_masks] = 0.0
            np.clip(w, 0, 1, out=w)
            self.ff_weights[:, winner_indices] = w

        # Subthreshold LTP: direct only (background learning)
        active_dims = np.flatnonzero(flat_input)
        if len(active_dims) > 0:
            sub_ltp = direct_rate * 0.1 * flat_input[
                active_dims, np.newaxis
            ]
            self.ff_weights[active_dims] += sub_ltp
            self.ff_weights[active_dims] *= self.ff_mask[active_dims]
            np.minimum(
                self.ff_weights[active_dims],
                1,
                out=self.ff_weights[active_dims],
            )

    def apply_surprise(self, surprise_improvement: float) -> None:
        """Consolidate eligibility traces using downstream surprise.

        Called when the downstream region's surprise decreases (= this
        region's representations helped). Positive = good (consolidate
        traces), negative = bad (reverse traces).

        Models synaptic tagging: norepinephrine from locus coeruleus
        signals "what just happened was important" and converts
        short-term synaptic tags into long-term changes.
        """
        if self.trace_fraction <= 0 or not self.learning_enabled:
            return
        if abs(surprise_improvement) < 1e-6:
            return

        if self._eligibility_clip > 0:
            np.clip(
                self._ff_eligibility,
                -self._eligibility_clip,
                self._eligibility_clip,
                out=self._ff_eligibility,
            )

        self.ff_weights += surprise_improvement * self._ff_eligibility
        self.ff_weights *= self.ff_mask
        np.clip(self.ff_weights, 0, 1, out=self.ff_weights)

    def reset_working_memory(self):
        """Reset transient state, preserving learned weights."""
        super().reset_working_memory()
        if self.trace_fraction > 0:
            self._ff_eligibility[:] = 0.0

    def _build_ff_mask(self, input_dim: int) -> np.ndarray:
        """Build encoding-width-aware receptive field mask."""
        col_mask = np.zeros((input_dim, self.n_columns), dtype=np.bool_)
        encoding_width = getattr(self, "encoding_width", 0)

        if encoding_width > 0:
            n_positions = input_dim // encoding_width
            stride = max(1, encoding_width // self.n_columns)
            window = stride * 3

            for col in range(self.n_columns):
                w_start = col * stride
                for pos in range(n_positions):
                    for w in range(window):
                        idx = pos * encoding_width + ((w_start + w) % encoding_width)
                        col_mask[idx, col] = True
        elif input_dim <= self.n_columns:
            col_mask[:, :] = True
        else:
            window_size = (2 * input_dim) // max(self.n_columns, 1)
            stride = max(1, (input_dim - window_size) // max(self.n_columns - 1, 1))
            for col in range(self.n_columns):
                start = min(col * stride, input_dim - window_size)
                end = min(start + window_size, input_dim)
                col_mask[start:end, col] = True

        return col_mask

    def _init_segments(self):
        """Override: constrain segment indices to local connectivity."""
        radius = max(1, self.n_columns // 4)
        n = self.n_l4_total
        n_syn = self.n_synapses_per_segment

        # Build per-column source pools (local neighborhood)
        self._fb_col_pools = {}
        self._lat_col_pools = {}
        self._l23_col_pools = {}
        for col in range(self.n_columns):
            neighbors = [c for c in range(self.n_columns) if abs(c - col) <= radius]
            self._fb_col_pools[col] = np.concatenate(
                [np.arange(c * self.n_l23, (c + 1) * self.n_l23) for c in neighbors]
            )
            self._lat_col_pools[col] = np.concatenate(
                [np.arange(c * self.n_l4, (c + 1) * self.n_l4) for c in neighbors]
            )
            self._l23_col_pools[col] = self._fb_col_pools[col]  # same L2/3 neighborhood

        # L4 segments (feedback + lateral)
        self.fb_seg_indices = np.zeros((n, self.n_fb_segments, n_syn), dtype=np.int32)
        self.fb_seg_perm = np.zeros((n, self.n_fb_segments, n_syn))
        self.lat_seg_indices = np.zeros((n, self.n_lat_segments, n_syn), dtype=np.int32)
        self.lat_seg_perm = np.zeros((n, self.n_lat_segments, n_syn))

        for i in range(n):
            col = i // self.n_l4
            fb_pool = self._fb_col_pools[col]
            lat_pool = self._lat_col_pools[col]
            for s in range(self.n_fb_segments):
                self.fb_seg_indices[i, s] = self._rng.choice(
                    fb_pool, n_syn, replace=len(fb_pool) < n_syn
                )
            for s in range(self.n_lat_segments):
                self.lat_seg_indices[i, s] = self._rng.choice(
                    lat_pool, n_syn, replace=len(lat_pool) < n_syn
                )

        # L2/3 lateral segments
        n23 = self.n_l23_total
        self.l23_seg_indices = np.zeros(
            (n23, self.n_l23_segments, n_syn), dtype=np.int32
        )
        self.l23_seg_perm = np.zeros((n23, self.n_l23_segments, n_syn))

        for i in range(n23):
            col = i // self.n_l23
            l23_pool = self._l23_col_pools[col]
            for s in range(self.n_l23_segments):
                self.l23_seg_indices[i, s] = self._rng.choice(
                    l23_pool, n_syn, replace=len(l23_pool) < n_syn
                )

    def _get_source_pool(self, neuron: int, seg_type: str) -> np.ndarray:
        """Override: return local connectivity pool for this neuron's column."""
        col = neuron // self.n_l4
        if seg_type == "fb":
            return self._fb_col_pools[col]
        return self._lat_col_pools[col]

    def _get_l23_source_pool(self, neuron: int) -> np.ndarray:
        """Override: return local L2/3 pool for this neuron's column."""
        col = neuron // self.n_l23
        return self._l23_col_pools[col]

    # L2/3 lateral connectivity is handled by dendritic segments with
    # local connectivity pools (_l23_col_pools), not a dense weight matrix.
