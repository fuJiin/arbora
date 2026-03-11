"""Experiment runner with seed control and JSON result logging."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from step.config import EncoderConfig, ModelConfig, TrainingConfig
from step.data import STORY_BOUNDARY, cached_token_stream, token_stream
from step.metrics import rolling_mean
from step.model import initial_state, learn, observe, predict
from step.sdr import encode_token

if TYPE_CHECKING:
    from step.protocol import Model


@dataclass
class ExperimentConfig:
    encoder: EncoderConfig
    model: ModelConfig
    training: TrainingConfig
    seed: int = 0
    name: str = "step_default"


# --- Legacy RunResult (kept for backward compat with run_step_experiment) ---


@dataclass
class RunResult:
    config: ExperimentConfig
    ious: list[float]
    rolling_ious: list[tuple[int, float]]
    elapsed_seconds: float


# --- New ComparisonRunResult for run_experiment ---


@dataclass
class ComparisonRunResult:
    model_name: str
    accuracies: list[float]
    rolling_accuracies: list[tuple[int, float]]
    native_metrics: list[float]
    rolling_native: list[tuple[int, float]]
    native_metric_name: str
    elapsed_seconds: float


def _make_stream(
    config: ExperimentConfig,
    token_cache: list[tuple[int, frozenset[int]]] | None,
) -> Iterator[tuple[int, int, frozenset[int]]]:
    """Build token stream from cache or live dataset."""
    if token_cache is not None:
        return cached_token_stream(token_cache, config.training.max_tokens)
    return token_stream(config.training, config.encoder)


def pretrain_step_model(
    model: Model,
    config: ExperimentConfig,
    token_cache: list[tuple[int, frozenset[int]]] | None = None,
    on_step: Callable[[int, Model], None] | None = None,
) -> None:
    """Pre-train a STEP model by running observe+learn on a token stream.

    This is STEP's equivalent of MiniGPT's pre-training phase.
    If on_step is provided, it is called with (t, model) each step.
    """
    tc = config.training
    total = tc.max_tokens
    start = time.monotonic()

    after_boundary = False
    for t, token_id, sdr in _make_stream(config, token_cache):
        if token_id == STORY_BOUNDARY:
            model.observe(t, token_id, sdr)
            after_boundary = True
            continue
        encode_fn = getattr(model, "encode_token_sdr", None)
        if encode_fn is not None:
            sdr = encode_fn(token_id, t)
        if t > 0 and not after_boundary:
            predicted_sdr = model.predict_sdr(t)
            model.learn(t, sdr, predicted_sdr)
        after_boundary = False
        model.observe(t, token_id, sdr)

        if on_step is not None:
            on_step(t, model)

        if t > 0 and t % tc.log_interval == 0:
            elapsed = time.monotonic() - start
            print(f"  [pretrain] t={t:,}/{total:,} ({elapsed:.1f}s)")

    # Flush any pending writes (e.g., SQLite batched commits)
    flush_fn = getattr(model, "flush", None)
    if flush_fn is not None:
        flush_fn()

    elapsed = time.monotonic() - start
    print(f"  [pretrain] done in {elapsed:.1f}s")


def run_experiment(
    config: ExperimentConfig,
    model_factory: Callable[[ExperimentConfig], Model],
    model_name: str,
    native_metric_name: str = "iou",
    token_cache: list[tuple[int, frozenset[int]]] | None = None,
    on_eval_step: Callable[[int, int, int, float], None] | None = None,
) -> ComparisonRunResult:
    """Run an experiment using any Model-protocol-conforming model.

    If token_cache is provided, replays from cache (no re-download).
    If on_eval_step is provided, called with (t, pred, actual, metric).
    Loop: predict_token -> predict_sdr -> learn -> observe.
    """
    tc = config.training
    model = model_factory(config)

    accuracies: list[float] = []
    rolling_accuracies: list[tuple[int, float]] = []
    native_metrics: list[float] = []
    rolling_native: list[tuple[int, float]] = []

    start = time.monotonic()

    after_boundary = False
    for t, token_id, sdr in _make_stream(config, token_cache):
        if token_id == STORY_BOUNDARY:
            model.observe(t, token_id, sdr)
            after_boundary = True
            continue
        encode_fn = getattr(model, "encode_token_sdr", None)
        if encode_fn is not None:
            sdr = encode_fn(token_id, t)
        if t > 0 and not after_boundary:
            predicted_token = model.predict_token(t)
            predicted_sdr = model.predict_sdr(t)
            accuracy = 1.0 if predicted_token == token_id else 0.0
            native_metric = model.learn(t, sdr, predicted_sdr)

            accuracies.append(accuracy)
            native_metrics.append(native_metric)

            if on_eval_step is not None:
                on_eval_step(t, predicted_token, token_id, native_metric)

            if t % tc.log_interval == 0:
                rolling_acc = rolling_mean(accuracies, tc.rolling_window)
                rolling_accuracies.append((t, rolling_acc))
                rolling_nat = rolling_mean(native_metrics, tc.rolling_window)
                rolling_native.append((t, rolling_nat))
                elapsed_so_far = time.monotonic() - start
                print(
                    f"  [{model_name}] t={t:,}/{tc.max_tokens:,} "
                    f"acc={rolling_acc:.4f} "
                    f"{native_metric_name}={rolling_nat:.4f} "
                    f"({elapsed_so_far:.1f}s)"
                )

        after_boundary = False
        model.observe(t, token_id, sdr)

    elapsed = time.monotonic() - start

    # Cleanup if model supports it (e.g., SQLite)
    close_fn = getattr(model, "close", None)
    if close_fn is not None:
        close_fn()

    return ComparisonRunResult(
        model_name=model_name,
        accuracies=accuracies,
        rolling_accuracies=rolling_accuracies,
        native_metrics=native_metrics,
        rolling_native=rolling_native,
        native_metric_name=native_metric_name,
        elapsed_seconds=elapsed,
    )


def save_comparison_result(result: ComparisonRunResult, output_dir: Path) -> Path:
    """Save a comparison run result as JSON. Returns the path written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run.json"

    data = {
        "model_name": result.model_name,
        "accuracies": result.accuracies,
        "rolling_accuracies": result.rolling_accuracies,
        "native_metrics": result.native_metrics,
        "rolling_native": result.rolling_native,
        "native_metric_name": result.native_metric_name,
        "elapsed_seconds": result.elapsed_seconds,
    }

    path.write_text(json.dumps(data, indent=2))
    return path


def load_comparison_result(path: Path) -> ComparisonRunResult:
    """Load a comparison run result from JSON."""
    data = json.loads(path.read_text())
    return ComparisonRunResult(
        model_name=data["model_name"],
        accuracies=data["accuracies"],
        rolling_accuracies=[tuple(x) for x in data["rolling_accuracies"]],
        native_metrics=data["native_metrics"],
        rolling_native=[tuple(x) for x in data["rolling_native"]],
        native_metric_name=data["native_metric_name"],
        elapsed_seconds=data["elapsed_seconds"],
    )


# --- Legacy functions (kept for backward compat) ---


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

    state = initial_state(mc)
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
