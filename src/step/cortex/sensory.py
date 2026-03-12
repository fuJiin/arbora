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
        self.ff_mask = np.zeros((input_dim, self.n_columns), dtype=np.bool_)

        if encoding_width > 0:
            # 2D-aware: input is (n_positions, encoding_width) flattened.
            # Each column covers a slice of the width dimension, replicated
            # across all positions.
            n_positions = input_dim // encoding_width
            stride = max(1, encoding_width // self.n_columns)
            window = stride * 3  # ~67% overlap between adjacent columns

            for col in range(self.n_columns):
                w_start = col * stride
                for pos in range(n_positions):
                    for w in range(window):
                        idx = pos * encoding_width + ((w_start + w) % encoding_width)
                        self.ff_mask[idx, col] = True
        else:
            # Fallback: sliding window over flat dims
            window_size = (2 * input_dim) // max(self.n_columns, 1)
            stride = max(1, (input_dim - window_size) // max(self.n_columns - 1, 1))
            for col in range(self.n_columns):
                start = min(col * stride, input_dim - window_size)
                end = min(start + window_size, input_dim)
                self.ff_mask[start:end, col] = True

        # Initialize weights only within the mask
        self.ff_weights = np.zeros((input_dim, self.n_columns))
        self.ff_weights[self.ff_mask] = self._rng.uniform(
            0.1, 0.5, int(self.ff_mask.sum())
        )

        # --- Lateral/feedback structural masks: local connectivity ---
        # Neurons connect to neurons in nearby columns (within radius).
        # Radius = n_columns // 4 gives ~50% connectivity.
        radius = max(1, self.n_columns // 4)
        self._init_internal_masks(radius)

    def _init_internal_masks(self, radius: int):
        """Build connectivity masks for lateral and feedback weights.

        Neurons in column i can connect to neurons in columns
        [i-radius, i+radius]. This creates topographic locality.
        """
        # Column-level adjacency
        col_mask = np.zeros((self.n_columns, self.n_columns), dtype=np.bool_)
        for i in range(self.n_columns):
            for j in range(self.n_columns):
                if abs(i - j) <= radius:
                    col_mask[i, j] = True

        # Expand to neuron-level masks
        # fb_weights: (n_l23_total, n_l4_total) — L2/3 source, L4 dest
        self.fb_mask = np.zeros_like(self.fb_weights, dtype=np.bool_)
        for src_col in range(self.n_columns):
            for dst_col in range(self.n_columns):
                if col_mask[src_col, dst_col]:
                    src_start = src_col * self.n_l23
                    src_end = src_start + self.n_l23
                    dst_start = dst_col * self.n_l4
                    dst_end = dst_start + self.n_l4
                    self.fb_mask[src_start:src_end, dst_start:dst_end] = True

        # lateral_weights: (n_l4_total, n_l4_total)
        self.lat_mask = np.zeros_like(self.lateral_weights, dtype=np.bool_)
        for src_col in range(self.n_columns):
            for dst_col in range(self.n_columns):
                if col_mask[src_col, dst_col]:
                    src_start = src_col * self.n_l4
                    src_end = src_start + self.n_l4
                    dst_start = dst_col * self.n_l4
                    dst_end = dst_start + self.n_l4
                    self.lat_mask[src_start:src_end, dst_start:dst_end] = True

        # l23_lateral_weights: (n_l23_total, n_l23_total)
        self.l23_lat_mask = np.zeros_like(self.l23_lateral_weights, dtype=np.bool_)
        for src_col in range(self.n_columns):
            for dst_col in range(self.n_columns):
                if col_mask[src_col, dst_col]:
                    src_start = src_col * self.n_l23
                    src_end = src_start + self.n_l23
                    dst_start = dst_col * self.n_l23
                    dst_end = dst_start + self.n_l23
                    self.l23_lat_mask[src_start:src_end, dst_start:dst_end] = True

        # Enforce masks on initial (zero) weights
        self.fb_weights[~self.fb_mask] = 0.0
        self.lateral_weights[~self.lat_mask] = 0.0
        self.l23_lateral_weights[~self.l23_lat_mask] = 0.0

    def process(self, encoding: np.ndarray) -> np.ndarray:
        """Feedforward an encoding through L4 → L2/3 pipeline.

        Args:
            encoding: Array of any shape (will be flattened).

        Returns:
            Array of global indices of active L4 neurons.
        """
        flat = encoding.flatten().astype(np.float64)
        self.last_column_drive = flat @ self.ff_weights
        active = self.step(self.last_column_drive)

        # Hebbian learning on feedforward synapses
        self._learn_ff(flat)

        return active

    def reconstruct(self, columns: np.ndarray | None = None) -> np.ndarray:
        """Reconstruct encoding from active columns via ff_weights.

        Walks backward through feedforward synapses: sums the ff_weight
        columns for each active column to produce a reconstructed
        encoding vector (same shape as flattened input).

        Args:
            columns: Column indices to reconstruct from.
                     Defaults to currently active columns.

        Returns:
            Reconstructed encoding vector (input_dim,).
        """
        if columns is None:
            columns = np.nonzero(self.active_columns)[0]
        if len(columns) == 0:
            return np.zeros(self.input_dim)
        return self.ff_weights[:, columns].sum(axis=1)

    def _learn_ff(self, flat_input: np.ndarray):
        """Hebbian update with LTD, respecting structural connectivity.

        LTP: active input bits × active columns → strengthen
        LTD: inactive input bits × active columns → weaken
        Subthreshold: weak LTP on inactive columns

        All updates masked to the structural connectivity — columns
        can never grow connections outside their receptive field.
        """
        active_cols_f = self.active_columns.astype(np.float64)
        inactive_cols_f = 1.0 - active_cols_f

        # --- Winners ---
        # LTP: strengthen connections to active inputs
        self.ff_weights += (
            self.learning_rate
            * flat_input[:, np.newaxis]
            * active_cols_f[np.newaxis, :]
        )

        # LTD: weaken connections to inactive inputs
        # Scale per-column by local sparsity within that column's
        # receptive field, not global input sparsity.
        inactive_input = 1.0 - flat_input
        for col in np.nonzero(self.active_columns)[0]:
            col_mask = self.ff_mask[:, col]
            local_on = max(flat_input[col_mask].sum(), 1.0)
            local_off = max(inactive_input[col_mask].sum(), 1.0)
            local_scale = local_on / local_off
            self.ff_weights[:, col] -= (
                self.ltd_rate
                * local_scale
                * inactive_input
                * col_mask  # only within receptive field
            )

        # --- Losers: subthreshold plasticity ---
        self.ff_weights += (
            self.learning_rate * 0.1
            * flat_input[:, np.newaxis]
            * inactive_cols_f[np.newaxis, :]
        )

        # Enforce structural sparsity — no synapse_decay on ff_weights
        # since LTD already provides the forgetting pressure.
        self.ff_weights[~self.ff_mask] = 0.0
        np.clip(self.ff_weights, 0, 1, out=self.ff_weights)

    def _learn(self):
        """Override to enforce structural masks after Hebbian update."""
        super()._learn()
        self.fb_weights[~self.fb_mask] = 0.0
        self.lateral_weights[~self.lat_mask] = 0.0
        self.l23_lateral_weights[~self.l23_lat_mask] = 0.0
