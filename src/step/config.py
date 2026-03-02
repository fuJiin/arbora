from dataclasses import dataclass


@dataclass
class EncoderConfig:
    model_name: str = "gpt2"
    n: int = 2048
    k: int = 40


@dataclass
class ModelConfig:
    n: int = 2048
    k: int = 40
    max_lr: float = 0.5
    weight_decay: float = 0.999
    penalty_factor: float = 0.5
    eligibility_window: int = 101


@dataclass
class TrainingConfig:
    dataset_name: str = "roneneldan/TinyStories"
    dataset_split: str = "train"
    max_tokens: int = 5000
    log_interval: int = 100
    rolling_window: int = 100
