from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arbora.cortex.motor import MotorRegion
    from arbora.cortex.sensory import SensoryRegion


class PlasticityRule(Enum):
    """Feedforward weight learning rule.

    HEBBIAN: immediate LTP/LTD on ff_weights each step (two-factor).
        Used by sensory regions (T1, T2, T3) that learn input statistics.

    THREE_FACTOR: accumulate Hebbian coincidences in eligibility traces,
        consolidate into ff_weights only when apply_reward() is called.
        Used by motor/prefrontal regions where learning must be gated
        by a reward signal (dopaminergic modulation).
    """

    HEBBIAN = "hebbian"
    THREE_FACTOR = "three_factor"


@dataclass
class CortexConfig:
    n_columns: int = 32
    n_l4: int = 4
    n_l23: int = 4
    n_l5: int | None = None  # defaults to n_l23 if None
    k_columns: int = 4
    voltage_decay: float = 0.5
    eligibility_decay: float = 0.95
    synapse_decay: float = 0.999
    learning_rate: float = 0.05
    max_excitability: float = 0.2
    fb_boost: float = 0.4
    ltd_rate: float = 0.2
    burst_learning_scale: float = 3.0
    # Dendritic segment parameters
    n_l4_lat_segments: int = 4
    n_l23_segments: int = 4
    n_synapses_per_segment: int = 24  # shared across all segment types for now
    perm_threshold: float = 0.5
    perm_init: float = 0.6
    perm_increment: float = 0.2
    perm_decrement: float = 0.05
    seg_activation_threshold: int = 2
    prediction_gain: float = 2.5
    n_apical_segments: int = 4
    n_l5_segments: int = 4  # L5 lateral sequence prediction
    l23_prediction_boost: float = 0.0
    pre_trace_decay: float = 0.8
    plasticity_rule: PlasticityRule = PlasticityRule.HEBBIAN
    seed: int = 0


def _default_t1_config() -> CortexConfig:
    """T1 defaults tuned for TinyDialogues char-level input.

    128 columns with k=8 gives ~6.25% activation fraction — the sweet
    spot for dendritic segment learning. Sweep results (30k chars):
      128/k=8: BPC 4.89, den=21.8%, M1=57.0%
      64/k=4:  BPC 4.72, den=17.9%, M1=22.0% (faster but less M1 capacity)
      32/k=4:  BPC 5.22, den=8.2%,  M1=6.8%  (undersized for 65-char vocab)

    synapse_decay=1.0: no passive weight decay. Sparse encoding prevents
    catastrophic forgetting; LTD alone controls weight growth. Decay
    comparison (30k chars) showed decay=1.0 gives best recent BPC (4.72)
    and strongest learning trend across dialogues.

    Use ltd_rate=0.05 for char-level data (default 0.2 is for GPT-2 tokens).
    """
    return CortexConfig(
        n_columns=128,
        k_columns=8,
        ltd_rate=0.05,
        synapse_decay=1.0,
    )


def _default_region2_config() -> CortexConfig:
    """Region 2 defaults: slower temporal dynamics, moderate learning rate.

    Tuned for char-level T1 input (128-dim L2/3 firing rates).
    32 cols with k=2 gives selective columns while maintaining context
    discrimination. lr=0.03 and ltd=0.30 balance weight growth with
    pruning on the higher-dimensional T1 output.

    Use encoding_width=0 (sliding window) when constructing the SensoryRegion,
    since T1's L2/3 output has no character-position structure.
    """
    return CortexConfig(
        n_columns=32,
        k_columns=4,
        voltage_decay=0.8,
        eligibility_decay=0.98,
        synapse_decay=0.9999,
        learning_rate=0.20,
        ltd_rate=0.20,
    )


def _default_region3_config() -> CortexConfig:
    """Region 3 (T3) defaults: association cortex for topic/theme extraction.

    Receives T2's L2/3 firing rates through a temporal buffer.
    T2 changes roughly every word (~4 chars), so a buffer_depth=8
    on T2 output spans ~8 words — approximately one phrase/clause.

    Slower dynamics than T2: higher voltage_decay retains context longer,
    high eligibility_decay for extended temporal credit assignment.
    Same 32c/k4 as T2 — distributed representation works well at this scale.
    """
    return CortexConfig(
        n_columns=32,
        k_columns=4,
        voltage_decay=0.9,
        eligibility_decay=0.99,
        synapse_decay=0.9999,
        learning_rate=0.15,
        ltd_rate=0.15,
    )


@dataclass
class HierarchyConfig:
    """Configuration for a two-region sensory hierarchy."""

    region1: CortexConfig = field(default_factory=CortexConfig)
    region2: CortexConfig = field(default_factory=_default_region2_config)
    surprise_baseline_decay: float = 0.99
    surprise_min_baseline: float = 0.01
    # Apical feedback: disabled by default until T2 representations mature.
    # When enabled, feedback is precision-weighted by T2's confidence
    # (1 - burst_rate), modeling thalamic gating / predictive coding.
    enable_apical_feedback: bool = False
    # Temporal buffer: T2 sees a sliding window of recent T1 states.
    # buffer_depth=1 is direct pass-through (default, backward compatible).
    ff_buffer_depth: int = 1
    # Burst gating: T2 only sees novel/surprising events (bursting columns).
    ff_burst_gate: bool = False
    # Thalamic gating: receiver-side surprise suppresses feedback until stable.
    gate_feedback: bool = False
    # Motor cortex: M1 receives T1 L2/3, predicts next token, feeds back.
    enable_motor: bool = False
    # Reward modulation: M1→T1 dopaminergic signal from turn-taking reward.
    enable_reward: bool = False


def _default_motor_config() -> CortexConfig:
    """Motor region defaults: responsive to current context, moderate learning.

    Receives T1's L2/3 firing rate (128-dim). 32 columns with k=4
    competitive selection gives 16 active L2/3 neurons — enough signal
    for DendriticDecoder to map M1 state → tokens (replaces degenerate
    k=1 column→token frequency mapping). Lower voltage_decay than T2
    for crisper output decisions. Moderate LTD for column specialization.
    """
    return CortexConfig(
        n_columns=32,
        k_columns=4,
        voltage_decay=0.5,
        eligibility_decay=0.95,
        synapse_decay=0.999,
        learning_rate=0.15,
        ltd_rate=0.15,
        plasticity_rule=PlasticityRule.THREE_FACTOR,
    )


def make_sensory_region(
    cfg: CortexConfig,
    input_dim: int,
    encoding_width: int = 0,
    seed: int | None = None,
) -> SensoryRegion:
    """Create a SensoryRegion from a CortexConfig.

    Eliminates the 25-line boilerplate of unpacking every CortexConfig
    field into SensoryRegion constructor kwargs.
    """
    from arbora.cortex.sensory import SensoryRegion

    d = asdict(cfg)
    s = d.pop("seed")
    d.pop("ltd_rate")  # explicit kwarg on SensoryRegion
    d.pop("plasticity_rule")  # sensory regions always use HEBBIAN (base default)
    return SensoryRegion(
        input_dim=input_dim,
        encoding_width=encoding_width,
        ltd_rate=cfg.ltd_rate,
        seed=seed if seed is not None else s,
        **d,
    )


def _default_pfc_config() -> CortexConfig:
    """PFC defaults: slow decay for working memory, slow learning for stability.

    16 columns (small — PFC is fewer columns than sensory regions).
    k=4 gives denser activation than sensory (mixed selectivity).
    voltage_decay=0.97 sustains activity ~30 steps (working memory).
    Slow learning rate — goals should be stable, not rapidly tracking.
    Longer eligibility traces for temporal credit assignment.
    """
    return CortexConfig(
        n_columns=16,
        k_columns=4,
        voltage_decay=0.97,
        eligibility_decay=0.98,
        synapse_decay=0.999,
        learning_rate=0.02,
        ltd_rate=0.02,
        plasticity_rule=PlasticityRule.THREE_FACTOR,
    )


def make_pfc_region(
    cfg: CortexConfig,
    input_dim: int,
    seed: int | None = None,
    source_dims: list[int] | None = None,
):
    """Create a PFCRegion from a CortexConfig."""
    from arbora.cortex.pfc import PFCRegion

    d = asdict(cfg)
    s = d.pop("seed")
    d.pop("ltd_rate")
    d.pop("n_columns")
    d.pop("k_columns")
    plasticity_rule = PlasticityRule(d.pop("plasticity_rule"))
    return PFCRegion(
        input_dim=input_dim,
        n_columns=cfg.n_columns,
        k_columns=cfg.k_columns,
        ltd_rate=cfg.ltd_rate,
        plasticity_rule=plasticity_rule,
        source_dims=source_dims,
        ff_sparsity=0.4 if source_dims else 0.0,
        seed=seed if seed is not None else s,
        **d,
    )


def _default_premotor_config() -> CortexConfig:
    """Premotor (M2) defaults: sequence generation from PFC goals.

    32 columns with k=4, moderate voltage decay (faster than PFC,
    slower than M1 — holds sequence state across a few steps).
    Moderate learning rate for sequence acquisition.
    """
    return CortexConfig(
        n_columns=32,
        k_columns=4,
        voltage_decay=0.7,
        eligibility_decay=0.95,
        synapse_decay=0.999,
        learning_rate=0.05,
        ltd_rate=0.05,
        plasticity_rule=PlasticityRule.THREE_FACTOR,
    )


def make_premotor_region(
    cfg: CortexConfig,
    input_dim: int,
    seed: int | None = None,
    source_dims: list[int] | None = None,
):
    """Create a PremotorRegion from a CortexConfig."""
    from arbora.cortex.premotor import PremotorRegion

    d = asdict(cfg)
    s = d.pop("seed")
    d.pop("ltd_rate")
    d.pop("n_columns")
    d.pop("k_columns")
    plasticity_rule = PlasticityRule(d.pop("plasticity_rule"))
    return PremotorRegion(
        input_dim=input_dim,
        n_columns=cfg.n_columns,
        k_columns=cfg.k_columns,
        ltd_rate=cfg.ltd_rate,
        plasticity_rule=plasticity_rule,
        source_dims=source_dims,
        ff_sparsity=0.4 if source_dims else 0.0,
        seed=seed if seed is not None else s,
        **d,
    )


def make_motor_region(
    cfg: CortexConfig,
    input_dim: int,
    output_threshold: float = 0.3,
    seed: int | None = None,
) -> MotorRegion:
    """Create a MotorRegion from a CortexConfig.

    Motor regions inherit directly from CorticalRegion (not SensoryRegion)
    with full connectivity (no encoding-width spatial structure).
    """
    from arbora.cortex.motor import MotorRegion

    d = asdict(cfg)
    s = d.pop("seed")
    d.pop("ltd_rate")
    d.pop("n_columns")  # explicit kwarg on MotorRegion
    plasticity_rule = PlasticityRule(d.pop("plasticity_rule"))
    return MotorRegion(
        input_dim=input_dim,
        n_columns=cfg.n_columns,
        output_threshold=output_threshold,
        ltd_rate=cfg.ltd_rate,
        plasticity_rule=plasticity_rule,
        seed=seed if seed is not None else s,
        **d,
    )
