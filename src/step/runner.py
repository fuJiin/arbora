"""Cortex training convenience functions."""

from __future__ import annotations

from collections.abc import Sequence

from step.agent import ChatAgent
from step.cortex.circuit import Circuit, Encoder
from step.cortex.sensory import SensoryRegion
from step.data import STORY_BOUNDARY  # noqa: F401 — re-exported for tests
from step.environment import ChatEnv
from step.probes.core import Probe
from step.probes.diagnostics import CortexDiagnostics
from step.train import TrainResult, train

__all__ = [
    "Encoder",
    "run_cortex",
]

_DIAG_FIELDS = [
    "snapshots",
    "_column_counts",
    "_l4_neuron_window",
    "_l23_neuron_window",
    "_l4_l23_matches",
    "_l4_l23_total",
    "_unique_col_sets",
    "_burst_count",
    "_precise_count",
    "_unique_prediction_sets",
    "_prediction_correct_neuron",
    "_prediction_correct_column",
    "_prediction_total",
]


def _copy_diag(
    cortex: Circuit,
    name: str,
    caller_diag: CortexDiagnostics | None,
) -> None:
    """Copy diagnostics state from Circuit back to caller's object."""
    if caller_diag is None:
        return
    topo_diag = cortex.diagnostics.get(name)
    if topo_diag is None:
        return
    for attr in _DIAG_FIELDS:
        setattr(caller_diag, attr, getattr(topo_diag, attr))


def run_cortex(
    region: SensoryRegion,
    encoder: Encoder,
    tokens: list[tuple[int, str]],
    log_interval: int = 100,
    rolling_window: int = 100,
    diagnostics: CortexDiagnostics | None = None,
    probes: Sequence[Probe] = (),
) -> TrainResult:
    """Run single-region cortex model on a token sequence.

    Returns TrainResult with probe snapshots and modulator time series.
    """
    diag_interval = diagnostics.snapshot_interval if diagnostics else 100
    cortex = Circuit(encoder, diagnostics_interval=diag_interval)
    cortex.add_region("S1", region, entry=True, diagnostics=diagnostics is not None)

    env = ChatEnv(tokens)
    agent = ChatAgent(encoder=encoder, circuit=cortex)
    result = train(
        env,
        agent,
        log_interval=log_interval,
        rolling_window=rolling_window,
        probes=probes,
    )
    _copy_diag(cortex, "S1", diagnostics)
    return result
