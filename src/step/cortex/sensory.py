"""Sensory cortical region: interfaces with encodings."""

import numpy as np

from step.cortex.region import CorticalRegion


class SensoryRegion(CorticalRegion):
    """Cortical region with feedforward input from an encoding.

    Adds feedforward weights mapping encoding dimensions to column drive,
    with Hebbian learning on the feedforward synapses.

    Structural sparsity throughout:
    - ff_weights: columns tile the character dimension (width) of the
      encoding, so each column is a character-range detector across
      all positions. No column wastes capacity on padding.
    - lateral/feedback weights: local connectivity — neurons only
      connect to neurons in nearby columns, like biological topography.

    Per-neuron ff_weights mode (per_neuron_ff=True):
    - Each L4 neuron has its own ff_weights within its column's mask.
    - Neurons in the same column share the same receptive field (mask)
      but learn different weight patterns — like biological neurons
      sampling different synapses from the same thalamic bundle.
    - Column drive for selection = max neuron drive in each column.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        encoding_width: int = 0,
        ltd_rate: float = 0.01,
        per_neuron_ff: bool = False,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.input_dim = input_dim
        self.ltd_rate = ltd_rate
        self.per_neuron_ff = per_neuron_ff

        # --- FF structural mask: tile by character width ---
        # Build column-level mask first (always needed)
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
        else:
            window_size = (2 * input_dim) // max(self.n_columns, 1)
            stride = max(1, (input_dim - window_size) // max(self.n_columns - 1, 1))
            for col in range(self.n_columns):
                start = min(col * stride, input_dim - window_size)
                end = min(start + window_size, input_dim)
                col_mask[start:end, col] = True

        if per_neuron_ff:
            # Per-neuron: (input_dim, n_l4_total), neurons in same column share mask
            self.ff_mask = np.repeat(col_mask, self.n_l4, axis=1)
            self.ff_weights = np.zeros((input_dim, self.n_l4_total))
            self.ff_weights[self.ff_mask] = self._rng.uniform(
                0.1, 0.5, int(self.ff_mask.sum())
            )
            # Keep column mask for LTD scaling
            self._col_mask = col_mask
        else:
            # Column-level: (input_dim, n_columns)
            self.ff_mask = col_mask
            self.ff_weights = np.zeros((input_dim, self.n_columns))
            self.ff_weights[self.ff_mask] = self._rng.uniform(
                0.1, 0.5, int(self.ff_mask.sum())
            )
            self._col_mask = col_mask

        # --- Lateral/feedback structural masks: local connectivity ---
        radius = max(1, self.n_columns // 4)
        self._init_internal_masks(radius)

    def _init_internal_masks(self, radius: int):
        """Build connectivity masks for lateral and feedback weights."""
        col_mask = np.zeros((self.n_columns, self.n_columns), dtype=np.bool_)
        for i in range(self.n_columns):
            for j in range(self.n_columns):
                if abs(i - j) <= radius:
                    col_mask[i, j] = True

        self.fb_mask = np.zeros_like(self.fb_weights, dtype=np.bool_)
        for src_col in range(self.n_columns):
            for dst_col in range(self.n_columns):
                if col_mask[src_col, dst_col]:
                    src_start = src_col * self.n_l23
                    src_end = src_start + self.n_l23
                    dst_start = dst_col * self.n_l4
                    dst_end = dst_start + self.n_l4
                    self.fb_mask[src_start:src_end, dst_start:dst_end] = True

        self.lat_mask = np.zeros_like(self.lateral_weights, dtype=np.bool_)
        for src_col in range(self.n_columns):
            for dst_col in range(self.n_columns):
                if col_mask[src_col, dst_col]:
                    src_start = src_col * self.n_l4
                    src_end = src_start + self.n_l4
                    dst_start = dst_col * self.n_l4
                    dst_end = dst_start + self.n_l4
                    self.lat_mask[src_start:src_end, dst_start:dst_end] = True

        self.l23_lat_mask = np.zeros_like(self.l23_lateral_weights, dtype=np.bool_)
        for src_col in range(self.n_columns):
            for dst_col in range(self.n_columns):
                if col_mask[src_col, dst_col]:
                    src_start = src_col * self.n_l23
                    src_end = src_start + self.n_l23
                    dst_start = dst_col * self.n_l23
                    dst_end = dst_start + self.n_l23
                    self.l23_lat_mask[src_start:src_end, dst_start:dst_end] = True

        self.fb_weights[~self.fb_mask] = 0.0
        self.lateral_weights[~self.lat_mask] = 0.0
        self.l23_lateral_weights[~self.l23_lat_mask] = 0.0

    def process(self, encoding: np.ndarray) -> np.ndarray:
        """Feedforward an encoding through L4 → L2/3 pipeline."""
        flat = encoding.flatten().astype(np.float64)

        if self.per_neuron_ff:
            # Per-neuron drive: (n_l4_total,)
            neuron_drive = flat @ self.ff_weights
            # Column drive for diagnostics: max per column
            self.last_column_drive = neuron_drive.reshape(
                self.n_columns, self.n_l4
            ).max(axis=1)
            active = self.step(neuron_drive)
        else:
            # Column drive: (n_columns,)
            self.last_column_drive = flat @ self.ff_weights
            active = self.step(self.last_column_drive)

        self._learn_ff(flat)
        return active

    def reconstruct(
        self,
        columns: np.ndarray | None = None,
        neurons: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reconstruct encoding from active columns/neurons via ff_weights.

        In per-neuron mode, can reconstruct from specific neurons for
        sharper reconstruction. Falls back to column-level if neurons
        not provided.
        """
        if self.per_neuron_ff and neurons is not None and len(neurons) > 0:
            return self.ff_weights[:, neurons].sum(axis=1)

        if columns is None:
            columns = np.nonzero(self.active_columns)[0]
        if len(columns) == 0:
            return np.zeros(self.input_dim)

        if self.per_neuron_ff:
            # Sum all neuron weights in the specified columns
            neuron_indices = []
            for col in columns:
                neuron_indices.extend(range(col * self.n_l4, (col + 1) * self.n_l4))
            return self.ff_weights[:, neuron_indices].sum(axis=1)
        else:
            return self.ff_weights[:, columns].sum(axis=1)

    def _learn_ff(self, flat_input: np.ndarray):
        """Hebbian update with LTD, respecting structural connectivity."""
        if self.per_neuron_ff:
            self._learn_ff_per_neuron(flat_input)
        else:
            self._learn_ff_column(flat_input)

    def _learn_ff_column(self, flat_input: np.ndarray):
        """Column-level ff learning (original)."""
        active_cols_f = self.active_columns.astype(np.float64)
        inactive_cols_f = 1.0 - active_cols_f

        # LTP: active input × active columns
        self.ff_weights += (
            self.learning_rate
            * flat_input[:, np.newaxis]
            * active_cols_f[np.newaxis, :]
        )

        # LTD: inactive input × active columns, scaled by local sparsity
        inactive_input = 1.0 - flat_input
        for col in np.nonzero(self.active_columns)[0]:
            col_mask = self.ff_mask[:, col]
            local_on = max(flat_input[col_mask].sum(), 1.0)
            local_off = max(inactive_input[col_mask].sum(), 1.0)
            local_scale = local_on / local_off
            self.ff_weights[:, col] -= (
                self.ltd_rate * local_scale * inactive_input * col_mask
            )

        # Subthreshold: weak LTP on inactive columns
        self.ff_weights += (
            self.learning_rate
            * 0.1
            * flat_input[:, np.newaxis]
            * inactive_cols_f[np.newaxis, :]
        )

        self.ff_weights[~self.ff_mask] = 0.0
        np.clip(self.ff_weights, 0, 1, out=self.ff_weights)

    def _learn_ff_per_neuron(self, flat_input: np.ndarray):
        """Per-neuron ff learning.

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

        # LTP: active input × winning neurons
        self.ff_weights += (
            self.learning_rate
            * flat_input[:, np.newaxis]
            * active_neurons_f[np.newaxis, :]
        )

        # LTD: inactive input × winning neurons, local sparsity scaling
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
                self.ltd_rate * local_scale * inactive_input * neuron_mask
            )

        # Subthreshold: weak LTP on neurons in inactive columns
        inactive_neurons_f = np.zeros(self.n_l4_total)
        for col in np.nonzero(~self.active_columns)[0]:
            l4_start = col * self.n_l4
            inactive_neurons_f[l4_start: l4_start + self.n_l4] = 1.0

        self.ff_weights += (
            self.learning_rate * 0.1
            * flat_input[:, np.newaxis]
            * inactive_neurons_f[np.newaxis, :]
        )

        self.ff_weights[~self.ff_mask] = 0.0
        np.clip(self.ff_weights, 0, 1, out=self.ff_weights)

    def _learn(self):
        """Override to enforce structural masks after Hebbian update."""
        super()._learn()
        self.fb_weights[~self.fb_mask] = 0.0
        self.lateral_weights[~self.lat_mask] = 0.0
        self.l23_lateral_weights[~self.l23_lat_mask] = 0.0
