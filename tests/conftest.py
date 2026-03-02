import pytest

from step.config import EncoderConfig, ModelConfig, TrainingConfig


@pytest.fixture
def small_encoder_config():
    return EncoderConfig(model_name="gpt2", n=256, k=10)


@pytest.fixture
def small_model_config():
    return ModelConfig(
        n=256,
        k=10,
        max_lr=0.5,
        weight_decay=0.999,
        penalty_factor=0.5,
        eligibility_window=20,
    )


@pytest.fixture
def small_training_config():
    return TrainingConfig(
        dataset_name="roneneldan/TinyStories",
        dataset_split="train",
        max_tokens=50,
        log_interval=10,
        rolling_window=10,
    )
