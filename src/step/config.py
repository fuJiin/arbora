from dataclasses import dataclass


@dataclass
class EncoderConfig:
    model_name: str = "gpt2"
    n: int = 2048
    k: int = 40
    vocab_size: int = 50257
    adaptive: bool = False
    context_fraction: float = 0.5
    # "active" (eligibility bits) or "predicted" (model prediction)
    seeding: str = "active"


@dataclass
class ModelConfig:
    n: int = 2048
    k: int = 40
    max_lr: float = 0.5
    weight_decay: float = 0.999
    penalty_factor: float = 0.5
    eligibility_window: int = 101
    # Three-factor gated learning: gate source bits by relevance.
    # 0.0 = disabled (original rule), >0 = relevance threshold
    relevance_gate: float = 0.0
    # Initial weight value (default 0.0 = zero init)
    weight_init: float = 0.0


@dataclass
class TrainingConfig:
    dataset_name: str = "roneneldan/TinyStories"
    dataset_split: str = "train"
    max_tokens: int = 5000
    log_interval: int = 100
    rolling_window: int = 100
