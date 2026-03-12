"""Per-step timeline capture for cortex visualization.

Lightweight recorder that captures column/neuron activation, voltage,
and ff_weight norms at each step for post-hoc analysis.
"""

from dataclasses import dataclass, field

import numpy as np

from step.cortex.sensory import SensoryRegion


@dataclass
class TimelineFrame:
    t: int
    active_columns: list[int]
    active_l4: list[int]
    active_l23: list[int]
    column_drive: np.ndarray  # (n_columns,) ff drive per column
    ff_weight_norms: np.ndarray  # (n_columns,) L2 norm per column
    voltage_l4_by_col: np.ndarray  # (n_columns,) max voltage per column
    excitability_l4_by_col: np.ndarray  # (n_columns,) max excitability per column


@dataclass
class Timeline:
    """Records per-step state for visualization."""

    frames: list[TimelineFrame] = field(default_factory=list)

    def capture(self, t: int, region: SensoryRegion, column_drive: np.ndarray):
        """Call after region.step() to capture current state."""
        active_cols = np.nonzero(region.active_columns)[0].tolist()
        active_l4 = np.nonzero(region.active_l4)[0].tolist()
        active_l23 = np.nonzero(region.active_l23)[0].tolist()

        ff_norms = np.linalg.norm(region.ff_weights, axis=0)

        voltage_by_col = region.voltage_l4.reshape(region.n_columns, region.n_l4).max(
            axis=1
        )
        excitability_by_col = region.excitability_l4.reshape(
            region.n_columns, region.n_l4
        ).max(axis=1)

        self.frames.append(
            TimelineFrame(
                t=t,
                active_columns=active_cols,
                active_l4=active_l4,
                active_l23=active_l23,
                column_drive=column_drive.copy(),
                ff_weight_norms=ff_norms.copy(),
                voltage_l4_by_col=voltage_by_col.copy(),
                excitability_l4_by_col=excitability_by_col.copy(),
            )
        )
