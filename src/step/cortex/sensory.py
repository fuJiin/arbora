"""Sensory cortical region: encoding-specific receptive fields."""

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

        # L5 lateral segments (local connectivity matching L2/3)
        n5 = self.n_l5_total
        self.l5_seg_indices = np.zeros((n5, self.n_l5_segments, n_syn), dtype=np.int32)
        self.l5_seg_perm = np.zeros((n5, self.n_l5_segments, n_syn))

        # Build L5 local pools (same neighborhood as L2/3)
        self._l5_col_pools = {}
        for col in range(self.n_columns):
            neighbors = [c for c in range(self.n_columns) if abs(c - col) <= radius]
            self._l5_col_pools[col] = np.concatenate(
                [np.arange(c * self.n_l5, (c + 1) * self.n_l5) for c in neighbors]
            )

        l5_pool_all = np.arange(n5)
        self._l5_source_pool = l5_pool_all
        for i in range(n5):
            col = i // self.n_l5
            l5_pool = self._l5_col_pools[col]
            for s in range(self.n_l5_segments):
                self.l5_seg_indices[i, s] = self._rng.choice(
                    l5_pool, n_syn, replace=len(l5_pool) < n_syn
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
