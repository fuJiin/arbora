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
    prediction_gain: float = 1.0
    seed: int = 0


def _default_region2_config() -> "CortexConfig":
    """Region 2 defaults: slower temporal dynamics, lower learning rate.

    Use encoding_width=0 (sliding window) when constructing the SensoryRegion,
    since R1's L2/3 output has no character-position structure.
    """
    return CortexConfig(
        n_columns=16,
        k_columns=2,
        voltage_decay=0.8,
        eligibility_decay=0.98,
        synapse_decay=0.9999,
        learning_rate=0.01,
        ltd_rate=0.4,
    )


@dataclass
class HierarchyConfig:
    """Configuration for a two-region sensory hierarchy."""

    region1: CortexConfig = field(default_factory=CortexConfig)
    region2: CortexConfig = field(default_factory=_default_region2_config)
    surprise_baseline_decay: float = 0.99
    surprise_min_baseline: float = 0.01
