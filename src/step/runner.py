"""Cortex training loop with natural prediction measurement."""

from dataclasses import dataclass, field

from step.cortex.circuit import Circuit, ConnectionRole, Encoder, RunMetrics
from step.cortex.modulators import SurpriseTracker, ThalamicGate
from step.cortex.sensory import SensoryRegion
from step.data import STORY_BOUNDARY  # noqa: F401 — re-exported for tests
from step.probes.diagnostics import CortexDiagnostics

__all__ = [
    "Encoder",
    "HierarchyMetrics",
    "RunMetrics",
    "run_cortex",
    "run_hierarchy",
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
    show_predictions: int = 0,
) -> RunMetrics:
    """Run cortex model on a token sequence, measuring prediction quality.

    tokens: list of (token_id, token_string) pairs.
            token_id == STORY_BOUNDARY signals a story boundary.
    show_predictions: if > 0, print this many prediction samples at each
                      log interval (actual vs predicted for each decoder).
    """
    diag_interval = diagnostics.snapshot_interval if diagnostics else 100
    cortex = Circuit(encoder, diagnostics_interval=diag_interval)
    cortex.add_region("S1", region, entry=True, diagnostics=diagnostics is not None)
    result = cortex.run(
        tokens,
        log_interval=log_interval,
        rolling_window=rolling_window,
        show_predictions=show_predictions,
    )
    _copy_diag(cortex, "S1", diagnostics)
    return result.per_region["S1"]


@dataclass
class HierarchyMetrics:
    region1: RunMetrics = field(default_factory=RunMetrics)
    region2: RunMetrics = field(default_factory=RunMetrics)
    surprise_modulators: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def run_hierarchy(
    region1: SensoryRegion,
    region2: SensoryRegion,
    encoder: Encoder,
    tokens: list[tuple[int, str]],
    *,
    surprise_tracker: SurpriseTracker | None = None,
    enable_apical_feedback: bool = False,
    buffer_depth: int = 1,
    burst_gate: bool = False,
    gate_feedback: bool = False,
    log_interval: int = 100,
    rolling_window: int = 100,
    diagnostics1: CortexDiagnostics | None = None,
    diagnostics2: CortexDiagnostics | None = None,
) -> HierarchyMetrics:
    """Run two-region hierarchy: S1 → S2.

    S2 receives S1's L2/3 firing rate as its encoding.
    Surprise (S1 burst rate) modulates S2 learning rate.
    """
    # Default: always create a surprise tracker (matches pre-refactor behavior)
    if surprise_tracker is None:
        surprise_tracker = SurpriseTracker()
    diag_interval = 100
    if diagnostics1 is not None:
        diag_interval = diagnostics1.snapshot_interval
    elif diagnostics2 is not None:
        diag_interval = diagnostics2.snapshot_interval

    cortex = Circuit(encoder, diagnostics_interval=diag_interval)
    cortex.add_region(
        "S1",
        region1,
        entry=True,
        diagnostics=diagnostics1 is not None,
    )
    cortex.add_region(
        "S2",
        region2,
        diagnostics=diagnostics2 is not None,
    )
    cortex.connect(
        "S1",
        "S2",
        ConnectionRole.FEEDFORWARD,
        buffer_depth=buffer_depth,
        burst_gate=burst_gate,
        surprise_tracker=surprise_tracker,
    )
    if enable_apical_feedback:
        gate = ThalamicGate() if gate_feedback else None
        cortex.connect("S2", "S1", ConnectionRole.APICAL, thalamic_gate=gate)

    result = cortex.run(
        tokens,
        log_interval=log_interval,
        rolling_window=rolling_window,
    )

    _copy_diag(cortex, "S1", diagnostics1)
    _copy_diag(cortex, "S2", diagnostics2)

    metrics = HierarchyMetrics()
    metrics.region1 = result.per_region["S1"]
    metrics.region2 = result.per_region["S2"]
    metrics.surprise_modulators = result.surprise_modulators.get("S2", [])
    metrics.elapsed_seconds = result.elapsed_seconds
    return metrics
