"""Lamina: per-layer state container for cortical regions.

Each cortical region has multiple laminae (L4, L2/3, L5) with similar
state: neurons, voltage, activation, firing rate, etc. This class
encapsulates the per-layer arrays to eliminate the _l4/_l23/_l5 suffix
duplication in CorticalRegion.

Named 'Lamina' (neuroscience term for cortical layer) to avoid collision
with ML 'Layer'.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from step.cortex.region import CorticalRegion


class LaminaID(enum.Enum):
    """Identifies a cortical layer for connection routing."""

    L4 = "L4"  # Input layer — receives feedforward drive
    L23 = "L2/3"  # Associative layer — lateral context, corticocortical output
    L5 = "L5"  # Output layer — subcortical projections (BG, cerebellum, thalamus)


class Lamina:
    """Per-layer state for a cortical region.

    A lightweight state container, not a self-running unit. The
    orchestration logic (burst/precise selection, prediction, learning)
    lives in CorticalRegion.step() because it crosses lamina boundaries.

    All features enabled by default. Can be selectively disabled
    if a lamina genuinely doesn't need a particular state (e.g., a
    lamina that only provides a readout with no competitive dynamics).
    """

    def __init__(
        self,
        n_columns: int,
        n_per_col: int,
        *,
        lamina_id: LaminaID,
        region: CorticalRegion | None = None,
    ):
        self.n_per_col = n_per_col
        self.n_columns = n_columns
        self.n_total = n_columns * n_per_col
        self.id = lamina_id
        self.region = region  # Set by CorticalRegion.register_lamina()

        # Per-neuron state arrays (all laminae have all features)
        self.active = np.zeros(self.n_total, dtype=np.bool_)
        self.predicted = np.zeros(self.n_total, dtype=np.bool_)
        self.voltage = np.zeros(self.n_total)
        self.excitability = np.zeros(self.n_total)
        self.trace = np.zeros(self.n_total)
        self.firing_rate = np.zeros(self.n_total)

        # Pending modulatory signal (set by circuit, consumed in step).
        # Applied after feedforward drive, before column selection.
        self._modulation: np.ndarray | None = None

    def reset(self):
        """Zero all transient state, preserving configuration."""
        self.active[:] = False
        self.predicted[:] = False
        self.voltage[:] = 0.0
        self.excitability[:] = 0.0
        self.trace[:] = 0.0
        self.firing_rate[:] = 0.0
