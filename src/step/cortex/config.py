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
    fb_boost_threshold: float = 0.3
    fb_boost: float = 0.4
    seed: int = 0
