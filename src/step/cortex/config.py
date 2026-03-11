from dataclasses import dataclass


@dataclass
class CortexConfig:
    n_columns: int = 32
    n_l4: int = 4
    n_l23: int = 4
    k_columns: int = 4
    fb_threshold: float = 0.5
    voltage_decay: float = 0.9
    eligibility_decay: float = 0.95
    synapse_decay: float = 0.999
    learning_rate: float = 0.01
    seed: int = 0
