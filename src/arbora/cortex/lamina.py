"""Lamina: cortex-specific neuron group with columnar structure.

Extends NeuronGroup with columns, voltage, predictions, excitability,
and burst/precise dynamics. Each cortical region has L4, L2/3, L5 laminae.
"""

from __future__ import annotations

import enum

import numpy as np

from arbora.neuron_group import NeuronGroup


class LaminaID(enum.Enum):
    """Cortical layer identifiers. Values are the NeuronGroup string IDs."""

    L4 = "L4"  # Input layer — receives feedforward drive
    L23 = "L2/3"  # Associative layer — lateral context, corticocortical output
    L5 = "L5"  # Output layer — subcortical projections


class Lamina(NeuronGroup):
    """Per-layer state for a cortical region.

    Extends NeuronGroup with cortex-specific dynamics:
    - Column structure (n_columns x n_per_col)
    - Voltage (membrane potential accumulation)
    - Active (binary spike state from k-WTA competition)
    - Predicted (dendritic segment predictions)
    - Excitability (homeostatic boosting of inactive neurons)
    - Trace (eligibility for three-factor learning)
    """

    def __init__(
        self,
        n_columns: int,
        n_per_col: int,
        *,
        lamina_id: LaminaID,
        region: object | None = None,
    ):
        super().__init__(
            n_neurons=n_columns * n_per_col,
            group_id=lamina_id.value,
            region=region,
        )
        # Override: column structure
        self.n_per_col = n_per_col
        self.n_columns = n_columns

        # Cortex-specific per-neuron state
        self.active = np.zeros(self.n_total, dtype=np.bool_)
        self.voltage = np.zeros(self.n_total)
        self.predicted = np.zeros(self.n_total, dtype=np.bool_)
        self.excitability = np.zeros(self.n_total)
        self.trace = np.zeros(self.n_total)

    def reset(self):
        """Zero all transient state, preserving configuration."""
        super().reset()
        self.active[:] = False
        self.voltage[:] = 0.0
        self.predicted[:] = False
        self.excitability[:] = 0.0
        self.trace[:] = 0.0
