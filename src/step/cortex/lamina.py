"""NeuronGroup and Lamina: neuron group containers for circuit wiring.

NeuronGroup — base class for any group of neurons with firing rate.
  Used by circuit.connect() as the connectable surface.
  Subcortical regions (BG, cerebellum) use NeuronGroup directly.

Lamina(NeuronGroup) — cortex-specific: adds columns, voltage,
  predictions, excitability, burst/precise dynamics. Each cortical
  region has L4, L2/3, L5 laminae.
"""

from __future__ import annotations

import enum

import numpy as np


class LaminaID(enum.Enum):
    """Identifies a neural population for connection routing."""

    L4 = "L4"  # Input layer — receives feedforward drive
    L23 = "L2/3"  # Associative layer — lateral context, corticocortical output
    L5 = "L5"  # Output layer — subcortical projections


class NeuronGroup:
    """A group of neurons — the minimal connectable surface.

    circuit.connect() takes NeuronGroup objects to wire regions together.
    Both cortical laminae and subcortical nuclei satisfy this interface.

    Universal properties:
        n_total: Number of neurons.
        firing_rate: Output signal (read by downstream connections).
        _modulation: Pending modulatory input (written by circuit).
        id: Routing identifier for connection resolution.
        region: Back-reference to the owning region.
    """

    def __init__(
        self,
        n_neurons: int,
        *,
        group_id: LaminaID,
        region: object | None = None,
    ):
        self.n_total = n_neurons
        self.n_per_col = n_neurons  # No column structure
        self.n_columns = 1
        self.id = group_id
        self.region = region

        # Universal: every neuron group has an output signal
        self.firing_rate = np.zeros(n_neurons)

        # Universal: any region can receive modulatory input
        self._modulation: np.ndarray | None = None

    def reset(self):
        """Zero transient state."""
        self.firing_rate[:] = 0.0
        self._modulation = None


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
            group_id=lamina_id,
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
