"""Experiment runner with seed control and JSON result logging."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from step.config import EncoderConfig, ModelConfig, TrainingConfig
from step.metrics import rolling_mean
from step.model import initial_state, learn, observe, predict
from step.sdr import encode_token


@dataclass
class ExperimentConfig:
    encoder: EncoderConfig
    model: ModelConfig
    training: TrainingConfig
    seed: int = 0
    name: str = "step_default"


@dataclass
class RunResult:
    config: ExperimentConfig
    ious: list[float]
    rolling_ious: list[tuple[int, float]]
    elapsed_seconds: float


def run_step_experiment(config: ExperimentConfig) -> RunResult:
    """Run a single STEP experiment with a fixed token sequence.

    The seed controls the order of token IDs presented (via a shuffled
    sequence), ensuring different seeds produce different training
    trajectories while remaining fully reproducible.
    """
    import numpy as np

    rng = np.random.default_rng(config.seed)
    ec = config.encoder
    mc = config.model
    tc = config.training

    # Generate a reproducible token sequence by sampling token IDs
    vocab_size = 50257
    token_ids = rng.integers(0, vocab_size, size=tc.max_tokens)

    state = initial_state()
    ious: list[float] = []
    rolling_ious: list[tuple[int, float]] = []

    start = time.monotonic()

    for t, tid in enumerate(token_ids):
        current_sdr = encode_token(int(tid), ec)

        if t > 0:
            predicted = predict(state, t, mc)
            iou = learn(state, t, current_sdr, predicted, mc)
            ious.append(iou)

            if t % tc.log_interval == 0:
                rolling = rolling_mean(ious, tc.rolling_window)
                rolling_ious.append((t, rolling))

        state = observe(state, t, current_sdr, mc)

    elapsed = time.monotonic() - start

    return RunResult(
        config=config,
        ious=ious,
        rolling_ious=rolling_ious,
        elapsed_seconds=elapsed,
    )


def run_multi_seed(
    base_config: ExperimentConfig,
    seeds: list[int],
    output_dir: Path | None = None,
) -> list[RunResult]:
    """Run the same experiment across multiple seeds for error bars."""
    results = []
    for seed in seeds:
        config = ExperimentConfig(
            encoder=base_config.encoder,
            model=base_config.model,
            training=base_config.training,
            seed=seed,
            name=base_config.name,
        )
        result = run_step_experiment(config)
        results.append(result)

        if output_dir is not None:
            save_result(result, output_dir)

    return results


def save_result(result: RunResult, output_dir: Path) -> Path:
    """Save a run result as JSON. Returns the path written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"seed{result.config.seed}.json"
    path = output_dir / filename

    data = {
        "config": asdict(result.config),
        "ious": result.ious,
        "rolling_ious": result.rolling_ious,
        "elapsed_seconds": result.elapsed_seconds,
    }

    path.write_text(json.dumps(data, indent=2))
    return path


def load_result(path: Path) -> RunResult:
    """Load a run result from JSON."""
    data = json.loads(path.read_text())
    config = ExperimentConfig(
        encoder=EncoderConfig(**data["config"]["encoder"]),
        model=ModelConfig(**data["config"]["model"]),
        training=TrainingConfig(**data["config"]["training"]),
        seed=data["config"]["seed"],
        name=data["config"]["name"],
    )
    return RunResult(
        config=config,
        ious=data["ious"],
        rolling_ious=[tuple(x) for x in data["rolling_ious"]],
        elapsed_seconds=data["elapsed_seconds"],
    )
