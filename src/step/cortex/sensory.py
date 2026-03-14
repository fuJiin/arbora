"""Sensory cortical region: interfaces with encodings."""

import numpy as np

from step.cortex.region import CorticalRegion


class SensoryRegion(CorticalRegion):
    """Cortical region with feedforward input from an encoding.

    Adds per-neuron feedforward weights mapping encoding dimensions to
    neuron drive, with Hebbian learning on the feedforward synapses.

    Structural sparsity throughout:
    - ff_weights: (input_dim, n_l4_total). Neurons in the same column
      share a receptive field mask but learn different weight patterns —
      like biological neurons sampling different synapses from the same
      thalamic bundle. Column drive = max neuron drive in each column.
    - dendritic segments: local connectivity — neurons only connect to
      neurons in nearby columns, like biological topography.
    - L2/3 lateral weights: local connectivity mask.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        encoding_width: int = 0,
        ltd_rate: float = 0.01,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.input_dim = input_dim
        self.ltd_rate = ltd_rate

        # --- FF structural mask: tile by character width ---
        col_mask = np.zeros((input_dim, self.n_columns), dtype=np.bool_)

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
            # Small input (e.g., one-hot characters): full connectivity.
            # Every column sees every input dimension. Differentiation
            # comes from learned weights, not structural masking.
            col_mask[:, :] = True
        else:
            window_size = (2 * input_dim) // max(self.n_columns, 1)
            stride = max(1, (input_dim - window_size) // max(self.n_columns - 1, 1))
            for col in range(self.n_columns):
                start = min(col * stride, input_dim - window_size)
                end = min(start + window_size, input_dim)
                col_mask[start:end, col] = True

        # Per-neuron: (input_dim, n_l4_total), neurons in same column share mask
        self.ff_mask = np.repeat(col_mask, self.n_l4, axis=1)
        self.ff_weights = np.zeros((input_dim, self.n_l4_total))
        self.ff_weights[self.ff_mask] = self._rng.uniform(
            0.1, 0.5, int(self.ff_mask.sum())
        )
        self._col_mask = col_mask

        # --- L2/3 lateral structural mask: local connectivity ---
        radius = max(1, self.n_columns // 4)
        self._init_l23_lateral_mask(radius)

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

    def process(self, encoding: np.ndarray) -> np.ndarray:
        """Feedforward an encoding through L4 → L2/3 pipeline."""
        flat = encoding.flatten().astype(np.float64)

        neuron_drive = flat @ self.ff_weights
        self.last_column_drive = neuron_drive.reshape(
            self.n_columns, self.n_l4
        ).max(axis=1)
        active = self.step(neuron_drive)

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
        # Build active neuron mask (only winners, not all burst neurons)
        # For learning, we want the neuron that gets the trace
        active_neurons_f = np.zeros(self.n_l4_total)
        for col in np.nonzero(self.active_columns)[0]:
            l4_start = col * self.n_l4
            l4_end = l4_start + self.n_l4
            if self.bursting_columns[col]:
                # Burst: trace goes to highest-voltage neuron
                best = np.argmax(self.voltage_l4[l4_start:l4_end])
                active_neurons_f[l4_start + best] = 1.0
            else:
                # Precise: the single active neuron
                active_in_col = self.active_l4[l4_start:l4_end]
                if active_in_col.any():
                    active_neurons_f[l4_start + active_in_col.argmax()] = 1.0

        # LTP: active input x winning neurons (modulated by neuromodulators)
        neuromod = self.surprise_modulator * self.reward_modulator
        ltp_rate = self.learning_rate * neuromod
        self.ff_weights += (
            ltp_rate
            * flat_input[:, np.newaxis]
            * active_neurons_f[np.newaxis, :]
        )

        # LTD: inactive input x winning neurons, local sparsity scaling
        ltd_rate = self.ltd_rate * neuromod
        inactive_input = 1.0 - flat_input
        for neuron_idx in np.nonzero(active_neurons_f)[0]:
            col = neuron_idx // self.n_l4
            neuron_mask = self.ff_mask[:, neuron_idx]
            # Use column mask for sparsity calculation
            col_mask_vec = self._col_mask[:, col]
            local_on = max(flat_input[col_mask_vec].sum(), 1.0)
            local_off = max(inactive_input[col_mask_vec].sum(), 1.0)
            local_scale = local_on / local_off
            self.ff_weights[:, neuron_idx] -= (
                ltd_rate * local_scale * inactive_input * neuron_mask
            )

        # Subthreshold: weak LTP on neurons in inactive columns
        inactive_neurons_f = np.zeros(self.n_l4_total)
        for col in np.nonzero(~self.active_columns)[0]:
            l4_start = col * self.n_l4
            inactive_neurons_f[l4_start: l4_start + self.n_l4] = 1.0

        self.ff_weights += (
            ltp_rate * 0.1
            * flat_input[:, np.newaxis]
            * inactive_neurons_f[np.newaxis, :]
        )

        self.ff_weights[~self.ff_mask] = 0.0
        np.clip(self.ff_weights, 0, 1, out=self.ff_weights)

    def _learn(self):
        """Override to enforce L2/3 lateral mask after learning."""
        super()._learn()
        self.l23_lateral_weights[~self.l23_lat_mask] = 0.0
