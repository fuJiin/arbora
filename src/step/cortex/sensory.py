"""Sensory cortical region: encoding-specific receptive fields and local connectivity."""

import numpy as np

from step.cortex.region import CorticalRegion


class SensoryRegion(CorticalRegion):
    """Cortical region with encoding-aware structural connectivity.

    Extends CorticalRegion with:
    - Encoding-width-based receptive field masking for ff_weights
    - Local connectivity for dendritic segments (topographic constraint)
    - L2/3 lateral connectivity mask (local neighborhood)
    """

    def __init__(
        self,
        input_dim: int,
        *,
        encoding_width: int = 0,
        seed: int = 0,
        **kwargs,
    ):
        # Store before super().__init__ so _build_ff_mask can use it
        self.encoding_width = encoding_width
        super().__init__(input_dim=input_dim, seed=seed, **kwargs)

        # --- L2/3 lateral structural mask: local connectivity ---
        radius = max(1, self.n_columns // 4)
        self._init_l23_lateral_mask(radius)

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
            neighbors = [
                c for c in range(self.n_columns) if abs(c - col) <= radius
            ]
            self._fb_col_pools[col] = np.concatenate(
                [np.arange(c * self.n_l23, (c + 1) * self.n_l23) for c in neighbors]
            )
            self._lat_col_pools[col] = np.concatenate(
                [np.arange(c * self.n_l4, (c + 1) * self.n_l4) for c in neighbors]
            )
            self._l23_col_pools[col] = self._fb_col_pools[col]  # same L2/3 neighborhood

        # L4 segments (feedback + lateral)
        self.fb_seg_indices = np.zeros(
            (n, self.n_fb_segments, n_syn), dtype=np.int32
        )
        self.fb_seg_perm = np.zeros((n, self.n_fb_segments, n_syn))
        self.lat_seg_indices = np.zeros(
            (n, self.n_lat_segments, n_syn), dtype=np.int32
        )
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

    def _init_l23_lateral_mask(self, radius: int):
        """Build local connectivity mask for L2/3 lateral weights."""
        self.l23_lat_mask = np.zeros_like(self.l23_lateral_weights, dtype=np.bool_)
        for src_col in range(self.n_columns):
            for dst_col in range(self.n_columns):
                if abs(src_col - dst_col) <= radius:
                    src_start = src_col * self.n_l23
                    src_end = src_start + self.n_l23
                    dst_start = dst_col * self.n_l23
                    dst_end = dst_start + self.n_l23
                    self.l23_lat_mask[src_start:src_end, dst_start:dst_end] = True

        self.l23_lateral_weights[~self.l23_lat_mask] = 0.0

    def _learn(self):
        """Override to enforce L2/3 lateral mask after learning."""
        super()._learn()
        self.l23_lateral_weights[~self.l23_lat_mask] = 0.0
