from dataclasses import dataclass, field


@dataclass
class CortexConfig:
    n_columns: int = 32
    n_l4: int = 4
    n_l23: int = 4
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
    n_fb_segments: int = 4
    n_lat_segments: int = 4
    n_l23_segments: int = 4
    n_synapses_per_segment: int = 24
    perm_threshold: float = 0.5
    perm_init: float = 0.6
    perm_increment: float = 0.2
    perm_decrement: float = 0.05
    seg_activation_threshold: int = 2
    prediction_gain: float = 2.5
    n_apical_segments: int = 4
    l23_prediction_boost: float = 0.0
    seed: int = 0


def _default_region2_config() -> "CortexConfig":
    """Region 2 defaults: slower temporal dynamics, moderate learning rate.

    Tuned for char-level S1 input (128-dim L2/3 firing rates).
    32 cols with k=2 gives selective columns while maintaining context
    discrimination. lr=0.03 and ltd=0.30 balance weight growth with
    pruning on the higher-dimensional S1 output.

    Use encoding_width=0 (sliding window) when constructing the SensoryRegion,
    since S1's L2/3 output has no character-position structure.
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


@dataclass
class HierarchyConfig:
    """Configuration for a two-region sensory hierarchy."""

    region1: CortexConfig = field(default_factory=CortexConfig)
    region2: CortexConfig = field(default_factory=_default_region2_config)
    surprise_baseline_decay: float = 0.99
    surprise_min_baseline: float = 0.01
    # Apical feedback: disabled by default until S2 representations mature.
    # When enabled, feedback is precision-weighted by S2's confidence
    # (1 - burst_rate), modeling thalamic gating / predictive coding.
    enable_apical_feedback: bool = False
    # Temporal buffer: S2 sees a sliding window of recent S1 states.
    # buffer_depth=1 is direct pass-through (default, backward compatible).
    ff_buffer_depth: int = 1
    # Burst gating: S2 only sees novel/surprising events (bursting columns).
    ff_burst_gate: bool = False
    # Thalamic gating: receiver-side surprise suppresses feedback until stable.
    gate_feedback: bool = False
    # Motor cortex: M1 receives S1 L2/3, predicts next token, feeds back.
    enable_motor: bool = False


def _default_motor_config() -> "CortexConfig":
    """Motor region defaults: responsive to current context, moderate learning.

    Receives S1's L2/3 firing rate (128-dim). 32 columns (one per char)
    with k=4 competitive selection. Lower voltage_decay than S2 for
    crisper output decisions. Moderate LTD for column specialization.
    """
    return CortexConfig(
        n_columns=32,
        k_columns=1,
        voltage_decay=0.5,
        eligibility_decay=0.95,
        synapse_decay=0.999,
        learning_rate=0.15,
        ltd_rate=0.15,
    )
