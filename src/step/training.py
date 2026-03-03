from collections.abc import Callable, Iterator

from step.config import EncoderConfig, ModelConfig, TrainingConfig
from step.metrics import rolling_mean
from step.model import ModelState, initial_state, learn, observe, predict


def train_step(
    state: ModelState,
    t: int,
    current_sdr: frozenset[int],
    config: ModelConfig,
) -> tuple[ModelState, float | None]:
    iou = None
    if t > 0:
        predicted = predict(state, t, config)
        iou = learn(state, t, current_sdr, predicted, config)

    state = observe(state, t, current_sdr, config)
    return state, iou


def train(
    stream: Iterator[tuple[int, int, frozenset[int]]],
    model_config: ModelConfig,
    encoder_config: EncoderConfig,
    training_config: TrainingConfig,
    log_fn: Callable[[int, float], None] | None = None,
) -> ModelState:
    if log_fn is None:

        def log_fn(t: int, rolling: float) -> None:
            print(f"Token {t} | Rolling IoU: {rolling:.4f}")

    state = initial_state()
    ious: list[float] = []

    for t, _token_id, current_sdr in stream:
        state, iou = train_step(state, t, current_sdr, model_config)

        if iou is not None:
            ious.append(iou)
            if t % training_config.log_interval == 0:
                rolling = rolling_mean(ious, training_config.rolling_window)
                log_fn(t, rolling)

    return state
