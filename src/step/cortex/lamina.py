"""Lamina: per-layer state container for cortical regions.

Each cortical region has multiple laminae (L4, L2/3, L5) with similar
state: neurons, voltage, activation, firing rate, etc. This class
encapsulates the per-layer arrays to eliminate the _l4/_l23/_l5 suffix
duplication in CorticalRegion.

Named 'Lamina' (neuroscience term for cortical layer) to avoid collision
with ML 'Layer'.
"""

import enum

import numpy as np


class LaminaID(enum.Enum):
    """Identifies a cortical layer for connection routing."""

    L4 = "L4"
    L23 = "L2/3"
    L5 = "L5"


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
        has_voltage: bool = True,
        has_excitability: bool = True,
        has_trace: bool = True,
        has_firing_rate: bool = True,
    ):
        self.n_per_col = n_per_col
        self.n_columns = n_columns
        self.n_total = n_columns * n_per_col

        # All laminae have active + predicted masks
        self.active = np.zeros(self.n_total, dtype=np.bool_)
        self.predicted = np.zeros(self.n_total, dtype=np.bool_)

        # Optional state (configured per lamina)
        self.voltage = np.zeros(self.n_total) if has_voltage else None
        self.excitability = np.zeros(self.n_total) if has_excitability else None
        self.trace = np.zeros(self.n_total) if has_trace else None
        self.firing_rate = np.zeros(self.n_total) if has_firing_rate else None

    def reset(self):
        """Zero all transient state, preserving configuration."""
        self.active[:] = False
        self.predicted[:] = False
        if self.voltage is not None:
            self.voltage[:] = 0.0
        if self.excitability is not None:
            self.excitability[:] = 0.0
        if self.trace is not None:
            self.trace[:] = 0.0
        if self.firing_rate is not None:
            self.firing_rate[:] = 0.0
