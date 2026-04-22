"""Circuit factories for MiniGrid benchmarks.

Two minimal topologies sharing as much configuration as possible so the
ARB-118 ablation tests a clean "HC vs no-HC" question.

Baseline (no HC)
----------------
::

    T1 → M1                  (sensorimotor feedforward)
    T1 → BG → M1 (mod)       (action-selection gate)

Hippocampal (with HC)
---------------------
::

    T1 → M1                  (sensorimotor feedforward — same as baseline)
    T1 → BG                  (current-state value path — same as baseline)
    T1 → HC                  (sensory input to HC)
    HC → BG                  (memory-informed value — what HC adds)
    BG → M1 (mod)            (same action-selection gate)

Rationale (per ARB-123): HC's biologically-grounded downstream target
is ventral striatum (a subregion of BG), not M1. HC biases *which
action is selected* by enriching BG's value signal; it does not drive
M1 directly. This topology makes the ARB-118 falsification story clean:
baseline BG sees only the current sensory state and cannot distinguish
two MemoryS13 distractors; the HC arm's BG additionally receives HC's
pattern-completed memory of the stored target, allowing it to bias
action-selection toward the matching distractor.

Shared-region invariants
------------------------
T1 and M1 are identical across arms (same constructor args, same seed).
BG is *intentionally* wider in the HC arm (it receives FF from both T1
and HC, so `input_dim` is the sum). Every other BG configuration field
matches.
"""

from __future__ import annotations

from dataclasses import dataclass

from arbora.basal_ganglia import BasalGangliaRegion
from arbora.cortex import SensoryRegion
from arbora.cortex.circuit import Circuit, ConnectionRole
from arbora.cortex.motor import MotorRegion
from arbora.hippocampus import HippocampalRegion
from examples.minigrid.encoder import MiniGridEncoder

# Shared default dimensions. Both arms use the same T1/M1 so the
# ablation holds everything but HC presence constant on the cortical
# side. BG is also shared-in-config (learning rate, n_actions, seed)
# except for `input_dim`, which differs because HC adds an FF stream
# in the HC arm.
_T1_DEFAULTS = dict(
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

# HC defaults tuned for MiniGrid scale: inputs at T1.n_l23_total (256),
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
class _SharedRegions:
    """T1 and M1 — truly shared across arms (identical construction).

    BG is *not* shared: its `input_dim` depends on whether HC is
    present, so each factory builds its own BG.
    """

    t1: SensoryRegion
    m1: MotorRegion


def _build_shared_regions(
    encoder: MiniGridEncoder,
    t1_overrides: dict | None,
    m1_overrides: dict | None,
) -> _SharedRegions:
    t1_cfg = {**_T1_DEFAULTS, **(t1_overrides or {})}
    t1 = SensoryRegion(
        input_dim=encoder.input_dim,
        encoding_width=encoder.encoding_width,
        **t1_cfg,
    )

    m1_cfg = {**_M1_DEFAULTS, **(m1_overrides or {})}
    m1 = MotorRegion(
        input_dim=t1.n_l23_total,
        **m1_cfg,
    )

    return _SharedRegions(t1=t1, m1=m1)


def _build_bg(input_dim: int, bg_overrides: dict | None) -> BasalGangliaRegion:
    bg_cfg = {**_BG_DEFAULTS, **(bg_overrides or {})}
    return BasalGangliaRegion(input_dim=input_dim, **bg_cfg)


def build_baseline_circuit(
    encoder: MiniGridEncoder,
    *,
    t1_overrides: dict | None = None,
    bg_overrides: dict | None = None,
    m1_overrides: dict | None = None,
    finalize: bool = True,
) -> Circuit:
    """Build `T1 → M1 + BG` — the no-HC baseline for the ARB-118 ablation.

    Matches the topology already used by `examples/minigrid/train.py`.
    BG reads T1 only; without HC, it cannot distinguish memory-gated
    choices on MemoryS13 — that's the whole point of the ablation.
    """
    shared = _build_shared_regions(encoder, t1_overrides, m1_overrides)
    bg = _build_bg(input_dim=shared.t1.n_l23_total, bg_overrides=bg_overrides)

    circuit = Circuit(encoder)
    circuit.add_region("T1", shared.t1, entry=True, input_region=True)
    circuit.add_region("BG", bg)
    circuit.add_region("M1", shared.m1, output_region=True)

    circuit.connect(shared.t1.output_port, bg.input_port, ConnectionRole.FEEDFORWARD)
    circuit.connect(
        shared.t1.output_port, shared.m1.input_port, ConnectionRole.FEEDFORWARD
    )
    circuit.connect(bg.output_port, shared.m1.input_port, ConnectionRole.MODULATORY)

    if finalize:
        circuit.finalize()
    return circuit


def build_hippocampal_circuit(
    encoder: MiniGridEncoder,
    *,
    t1_overrides: dict | None = None,
    hc_overrides: dict | None = None,
    bg_overrides: dict | None = None,
    m1_overrides: dict | None = None,
    finalize: bool = True,
) -> Circuit:
    """Build `T1 → {M1, BG, HC}`, `HC → BG`, `BG → M1 (mod)`.

    HC is the biologically-motivated ventral-striatum-analog path (see
    ARB-123): HC projects to BG to augment the action-selection value
    signal with pattern-completed memory. M1 receives only direct T1 FF
    — the same sensorimotor path the baseline uses — so the two arms
    differ only in whether BG's value signal is memory-informed.

    BG is widened in this arm (`input_dim = T1.n_l23_total +
    HC.output_port.n_total`) so it can accept both FF streams. All
    other BG configuration matches the baseline.
    """
    shared = _build_shared_regions(encoder, t1_overrides, m1_overrides)

    hc_cfg = {**_HC_DEFAULTS, **(hc_overrides or {})}
    hc = HippocampalRegion(
        input_dim=shared.t1.n_l23_total,
        **hc_cfg,
    )

    # BG in the HC arm receives FF from both T1 and HC. This is the
    # intentional shared-region asymmetry (see module docstring).
    bg = _build_bg(
        input_dim=shared.t1.n_l23_total + hc.output_port.n_total,
        bg_overrides=bg_overrides,
    )

    circuit = Circuit(encoder)
    circuit.add_region("T1", shared.t1, entry=True, input_region=True)
    circuit.add_region("HC", hc)
    circuit.add_region("BG", bg)
    circuit.add_region("M1", shared.m1, output_region=True)

    # T1 feeds all three downstream regions.
    circuit.connect(shared.t1.output_port, hc.input_port, ConnectionRole.FEEDFORWARD)
    circuit.connect(shared.t1.output_port, bg.input_port, ConnectionRole.FEEDFORWARD)
    circuit.connect(
        shared.t1.output_port, shared.m1.input_port, ConnectionRole.FEEDFORWARD
    )
    # HC augments BG's input with memory-informed value context. This
    # is the core architectural claim of the HC arm.
    circuit.connect(hc.output_port, bg.input_port, ConnectionRole.FEEDFORWARD)
    # BG gates M1 action selection (identical to baseline).
    circuit.connect(bg.output_port, shared.m1.input_port, ConnectionRole.MODULATORY)

    if finalize:
        circuit.finalize()
    return circuit
