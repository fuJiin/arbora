"""Circuit factories for MiniGrid benchmarks.

Two minimal topologies, sharing the same S1/BG/M1 configuration so the
only difference between them is whether the hippocampal module mediates
S1 → M1. This symmetry is what lets the ARB-118 ablation attribute
performance differences to HC itself rather than to the surrounding
architecture.

Baseline
--------
::

    S1 → M1
    S1 → BG → M1 (modulatory)

Hippocampal
-----------
::

    S1 → HC → M1
    S1 → BG → M1 (modulatory)

HC is inserted on the sensory → motor feedforward path only. BG still
reads S1 directly, so the reward-driven action-selection signal is
identical between arms — the only thing that changes is whether M1's
feedforward drive is HC-augmented (memory-informed) or raw sensory.

HC is symmetric (input_dim == output_dim == S1.n_l23_total), so M1's
input dimension is the same in both arms and can be configured
identically.
"""

from __future__ import annotations

from dataclasses import dataclass

from arbora.basal_ganglia import BasalGangliaRegion
from arbora.cortex import SensoryRegion
from arbora.cortex.circuit import Circuit, ConnectionRole
from arbora.cortex.motor import MotorRegion
from arbora.hippocampus import HippocampalRegion
from examples.minigrid.encoder import MiniGridEncoder

# Shared default dimensions. Both arms use the same S1/BG/M1 so the
# ablation holds everything but HC presence constant.
_S1_DEFAULTS = dict(
    n_columns=64,
    n_l4=4,
    n_l23=4,
    n_l5=0,
    k_columns=4,
    seed=42,
)
_BG_DEFAULTS = dict(
    n_actions=7,
    seed=789,
)
_M1_DEFAULTS = dict(
    n_columns=16,
    n_l4=0,
    n_l23=4,
    k_columns=2,
    n_output_tokens=7,
    seed=456,
)

# HC defaults tuned for MiniGrid scale: inputs at S1.n_l23_total (256),
# ec/ca3 kept at a similar order, DG overprovisioned ~4x. Keeps
# benchmark runs tractable while leaving room for tens of episodic
# memories in CA3.
_HC_DEFAULTS = dict(
    ec_dim=512,
    dg_dim=2000,
    ca3_dim=512,
    ec_sparsity=0.02,
    ca3_k_active=0.02,
    ca3_mossy_sparsity=0.02,
    ca3_learning_rate=0.5,
    retrieval_iterations=3,
    seed=1234,
)


@dataclass
class _Regions:
    s1: SensoryRegion
    bg: BasalGangliaRegion
    m1: MotorRegion
    hc: HippocampalRegion | None = None


def _build_shared_regions(
    encoder: MiniGridEncoder,
    s1_overrides: dict | None,
    bg_overrides: dict | None,
    m1_overrides: dict | None,
) -> _Regions:
    """Construct S1, BG, M1 used by both arms of the ablation."""
    s1_cfg = {**_S1_DEFAULTS, **(s1_overrides or {})}
    s1 = SensoryRegion(
        input_dim=encoder.input_dim,
        encoding_width=encoder.encoding_width,
        **s1_cfg,
    )

    bg_cfg = {**_BG_DEFAULTS, **(bg_overrides or {})}
    bg = BasalGangliaRegion(
        input_dim=s1.n_l23_total,
        **bg_cfg,
    )

    m1_cfg = {**_M1_DEFAULTS, **(m1_overrides or {})}
    m1 = MotorRegion(
        input_dim=s1.n_l23_total,
        **m1_cfg,
    )

    return _Regions(s1=s1, bg=bg, m1=m1)


def build_baseline_circuit(
    encoder: MiniGridEncoder,
    *,
    s1_overrides: dict | None = None,
    bg_overrides: dict | None = None,
    m1_overrides: dict | None = None,
    finalize: bool = True,
) -> Circuit:
    """Build `S1 → M1 + BG` — the no-HC baseline for the ARB-118 ablation.

    Matches the topology already used by `examples/minigrid/train.py`;
    extracted here so the baseline and hippocampal arms share identical
    shared-region configuration.
    """
    regions = _build_shared_regions(encoder, s1_overrides, bg_overrides, m1_overrides)
    circuit = Circuit(encoder)
    circuit.add_region("S1", regions.s1, entry=True, input_region=True)
    circuit.add_region("BG", regions.bg)
    circuit.add_region("M1", regions.m1, output_region=True)

    circuit.connect(
        regions.s1.output_port, regions.bg.input_port, ConnectionRole.FEEDFORWARD
    )
    circuit.connect(
        regions.s1.output_port, regions.m1.input_port, ConnectionRole.FEEDFORWARD
    )
    circuit.connect(
        regions.bg.output_port, regions.m1.input_port, ConnectionRole.MODULATORY
    )

    if finalize:
        circuit.finalize()
    return circuit


def build_hippocampal_circuit(
    encoder: MiniGridEncoder,
    *,
    s1_overrides: dict | None = None,
    hc_overrides: dict | None = None,
    bg_overrides: dict | None = None,
    m1_overrides: dict | None = None,
    finalize: bool = True,
) -> Circuit:
    """Build `S1 → HC → M1 + BG` — the with-HC arm of the ARB-118 ablation.

    HC is inserted on the sensory→motor feedforward path. BG wiring is
    unchanged from the baseline so the reward-driven action-selection
    signal is identical between arms.
    """
    regions = _build_shared_regions(encoder, s1_overrides, bg_overrides, m1_overrides)
    hc_cfg = {**_HC_DEFAULTS, **(hc_overrides or {})}
    regions.hc = HippocampalRegion(
        input_dim=regions.s1.n_l23_total,
        **hc_cfg,
    )

    circuit = Circuit(encoder)
    circuit.add_region("S1", regions.s1, entry=True, input_region=True)
    circuit.add_region("HC", regions.hc)
    circuit.add_region("BG", regions.bg)
    circuit.add_region("M1", regions.m1, output_region=True)

    # S1 feeds both HC (memory path) and BG (action-selection path).
    circuit.connect(
        regions.s1.output_port, regions.hc.input_port, ConnectionRole.FEEDFORWARD
    )
    circuit.connect(
        regions.s1.output_port, regions.bg.input_port, ConnectionRole.FEEDFORWARD
    )
    # HC emits into M1; no direct S1 → M1 in this arm so M1's drive is
    # HC-mediated. The ablation tests whether that mediation helps.
    circuit.connect(
        regions.hc.output_port, regions.m1.input_port, ConnectionRole.FEEDFORWARD
    )
    circuit.connect(
        regions.bg.output_port, regions.m1.input_port, ConnectionRole.MODULATORY
    )

    if finalize:
        circuit.finalize()
    return circuit
