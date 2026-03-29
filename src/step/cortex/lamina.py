"""NeuronGroup and Lamina: neuron group containers for circuit wiring.

NeuronGroup — base class for any group of neurons with firing rate.
  Used by circuit.connect() as the connectable surface.
  Subcortical regions (BG, cerebellum) use NeuronGroup directly.

Lamina(NeuronGroup) — cortex-specific: adds columns, predictions,
  excitability, burst/precise dynamics. Each cortical region has
  L4, L2/3, L5 laminae.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class LaminaID(enum.Enum):
    """Identifies a neural population for connection routing."""

    L4 = "L4"  # Input layer — receives feedforward drive
    L23 = "L2/3"  # Associative layer — lateral context, corticocortical output
    L5 = "L5"  # Output layer — subcortical projections


class NeuronGroup:
    """A group of neurons with firing rate — the minimal connectable surface.

    circuit.connect() takes NeuronGroup objects to wire regions together.
    Both cortical laminae and subcortical nuclei satisfy this interface.

    Attributes:
        n_total: Total number of neurons.
        firing_rate: Per-neuron firing rate (read by downstream connections).
        voltage: Per-neuron voltage (written to by modulatory connections).
        active: Per-neuron binary activation state.
        id: Routing identifier (LaminaID for cortex, arbitrary for subcortical).
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
        self.n_per_col = n_neurons  # No column structure; treat as single group
        self.n_columns = 1
        self.id = group_id
        self.region = region

        self.active = np.zeros(n_neurons, dtype=np.bool_)
        self.voltage = np.zeros(n_neurons)
        self.firing_rate = np.zeros(n_neurons)

        # Pending modulatory signal (set by circuit, consumed in step).
        self._modulation: np.ndarray | None = None

    def reset(self):
        """Zero all transient state."""
        self.active[:] = False
        self.voltage[:] = 0.0
        self.firing_rate[:] = 0.0
        self._modulation = None


class Lamina(NeuronGroup):
    """Per-layer state for a cortical region.

    Extends NeuronGroup with column structure (n_columns x n_per_col),
    prediction tracking, excitability, and eligibility traces — all
    cortex-specific features for burst/precise dynamics.

    The orchestration logic (burst/precise selection, prediction, learning)
    lives in CorticalRegion.step() because it crosses lamina boundaries.
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
        self.predicted = np.zeros(self.n_total, dtype=np.bool_)
        self.excitability = np.zeros(self.n_total)
        self.trace = np.zeros(self.n_total)

    def reset(self):
        """Zero all transient state, preserving configuration."""
        super().reset()
        self.predicted[:] = False
        self.excitability[:] = 0.0
        self.trace[:] = 0.0
