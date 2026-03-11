"""Sensory cortical region: interfaces with encodings."""

import numpy as np

from step.cortex.region import CorticalRegion


class SensoryRegion(CorticalRegion):
    """Cortical region with feedforward input from an encoding.

    Adds feedforward weights mapping encoding dimensions to column drive,
    with Hebbian learning on the feedforward synapses.
    """

    def __init__(self, input_dim: int, *, seed: int = 0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.input_dim = input_dim

        # Feedforward weights: encoding dimension → column drive
        self.ff_weights = self._rng.uniform(
            0, 0.1, (input_dim, self.n_columns)
        )

    def process(self, encoding: np.ndarray) -> np.ndarray:
        """Feedforward an encoding through L4 → L2/3 pipeline.

        Args:
            encoding: Array of any shape (will be flattened).

        Returns:
            Array of global indices of active L4 neurons.
        """
        flat = encoding.flatten().astype(np.float64)
        column_drive = flat @ self.ff_weights
        active = self.step(column_drive)

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
        """Hebbian update: strengthen input→column where both active."""
        active_cols_f = self.active_columns.astype(np.float64)
        self.ff_weights += (
            self.learning_rate
            * flat_input[:, np.newaxis]
            * active_cols_f[np.newaxis, :]
        )
        self.ff_weights *= self.synapse_decay
        np.clip(self.ff_weights, 0, 1, out=self.ff_weights)
