"""NeuronGroup: base class for any group of neurons in a circuit.

The minimal connectable surface for circuit wiring. Both cortical
laminae and subcortical nuclei inherit from this. The circuit only
needs firing_rate (to read output) and modulation (to write input).

Subclasses:
  Lamina(NeuronGroup) — cortex: adds columns, voltage, predictions
  BG uses NeuronGroup directly with domain-specific IDs ("striatum", "gpi")
"""

from __future__ import annotations

import numpy as np


class NeuronGroup:
    """A group of neurons — the minimal connectable surface.

    circuit.connect() takes NeuronGroup objects to wire regions together.
    Both cortical laminae and subcortical nuclei satisfy this interface.

    Attributes:
        n_total: Number of neurons.
        firing_rate: Output signal (read by downstream connections).
        modulation: Pending modulatory input (accumulated via add_modulation).
        id: String identifier for connection routing (e.g., "L4", "striatum").
        region: Back-reference to the owning region.
    """

    def __init__(
        self,
        n_neurons: int,
        *,
        group_id: str,
        region: object | None = None,
    ):
        self.n_total = n_neurons
        self.n_per_col = n_neurons  # No column structure
        self.n_columns = 1
        self.id = group_id
        self.region = region

        self.firing_rate = np.zeros(n_neurons)
        self.modulation: np.ndarray | None = None

    def add_modulation(self, signal: np.ndarray) -> None:
        """Accumulate a modulatory signal. Additive if multiple sources."""
        if self.modulation is None:
            self.modulation = signal.copy()
        else:
            self.modulation += signal

    def clear_modulation(self) -> None:
        """Consume pending modulation (called by region after applying)."""
        self.modulation = None

    def reset(self):
        """Zero transient state."""
        self.firing_rate[:] = 0.0
        self.modulation = None
