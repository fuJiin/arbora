"""Cortex training loop with natural prediction measurement."""

from dataclasses import dataclass, field

from step.cortex.sensory import SensoryRegion
from step.cortex.surprise import SurpriseTracker
from step.cortex.topology import Encoder, RunMetrics, Topology
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
    cortex: Topology,
    name: str,
    caller_diag: CortexDiagnostics | None,
) -> None:
    """Copy diagnostics state from Topology back to caller's object."""
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
    diag_interval = (
        diagnostics.snapshot_interval if diagnostics else 100
    )
    cortex = Topology(encoder, diagnostics_interval=diag_interval)
    cortex.add_region(
        "R1", region, entry=True, diagnostics=diagnostics is not None
    )
    result = cortex.run(
        tokens,
        log_interval=log_interval,
        rolling_window=rolling_window,
        show_predictions=show_predictions,
    )
    _copy_diag(cortex, "R1", diagnostics)
    return result.per_region["R1"]


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
    log_interval: int = 100,
    rolling_window: int = 100,
    diagnostics1: CortexDiagnostics | None = None,
    diagnostics2: CortexDiagnostics | None = None,
) -> HierarchyMetrics:
    """Run two-region hierarchy: Region 1 → Region 2.

    Region 2 receives Region 1's L2/3 firing rate as its encoding.
    Surprise (Region 1 burst rate) modulates Region 2 learning rate.
    """
    diag_interval = 100
    if diagnostics1 is not None:
        diag_interval = diagnostics1.snapshot_interval
    elif diagnostics2 is not None:
        diag_interval = diagnostics2.snapshot_interval

    cortex = Topology(encoder, diagnostics_interval=diag_interval)
    cortex.add_region(
        "R1", region1, entry=True,
        diagnostics=diagnostics1 is not None,
    )
    cortex.add_region(
        "R2", region2,
        diagnostics=diagnostics2 is not None,
    )
    cortex.connect("R1", "R2", "feedforward")
    cortex.connect(
        "R1", "R2", "surprise",
        surprise_tracker=surprise_tracker,
    )
    if enable_apical_feedback:
        cortex.connect("R2", "R1", "apical")

    result = cortex.run(
        tokens,
        log_interval=log_interval,
        rolling_window=rolling_window,
    )

    _copy_diag(cortex, "R1", diagnostics1)
    _copy_diag(cortex, "R2", diagnostics2)

    metrics = HierarchyMetrics()
    metrics.region1 = result.per_region["R1"]
    metrics.region2 = result.per_region["R2"]
    metrics.surprise_modulators = result.surprise_modulators.get("R2", [])
    metrics.elapsed_seconds = result.elapsed_seconds
    return metrics
