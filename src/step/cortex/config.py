from dataclasses import dataclass


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
    n_synapses_per_segment: int = 16
    perm_threshold: float = 0.5
    perm_init: float = 0.6
    perm_increment: float = 0.1
    perm_decrement: float = 0.05
    seg_activation_threshold: int = 4
    prediction_gain: float = 1.0
    seed: int = 0
