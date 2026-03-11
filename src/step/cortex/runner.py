"""Cortex training loop with natural prediction measurement."""

import time
from dataclasses import dataclass, field

import numpy as np

from step.config import ModelConfig
from step.cortex.diagnostics import CortexDiagnostics
from step.cortex.sensory import SensoryRegion
from step.decode import DecodeIndex
from step.encoders.charbit import CharbitEncoder

STORY_BOUNDARY = -1


@dataclass
class RunMetrics:
    overlaps: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    synaptic_accuracies: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def run_cortex(
    region: SensoryRegion,
    encoder: CharbitEncoder,
    tokens: list[tuple[int, str]],
    log_interval: int = 100,
    rolling_window: int = 100,
    diagnostics: CortexDiagnostics | None = None,
) -> RunMetrics:
    """Run cortex model on a token sequence, measuring prediction quality.

    tokens: list of (token_id, token_string) pairs.
            token_id == STORY_BOUNDARY signals a story boundary.
    """
    decode_index = DecodeIndex()
    metrics = RunMetrics()
    k = region.k_columns
    start = time.monotonic()

    for t, (token_id, token_str) in enumerate(tokens):
        if token_id == STORY_BOUNDARY:
            region.reset_working_memory()
            continue

        # Prediction: what does the network expect before seeing input?
        predicted_neurons = region.get_prediction(k)
        predicted_set = frozenset(int(i) for i in predicted_neurons)

        # Synaptic decode: walk predicted columns back through ff_weights
        predicted_cols = np.unique(predicted_neurons // region.n_l4)
        reconstructed = region.reconstruct(predicted_cols)
        reconstructed_matrix = reconstructed.reshape(encoder.length, encoder.width)
        synaptic_token = encoder.decode(reconstructed_matrix)

        # Step: feed input, triggers activation + learning
        active_neurons = region.process(encoder.encode(token_str))
        active_set = frozenset(int(i) for i in active_neurons)

        if diagnostics is not None:
            diagnostics.step(t, region)

        if t > 0:
            # Prediction overlap: fraction of predicted neurons that fired
            if active_set:
                overlap = len(predicted_set & active_set) / len(active_set)
            else:
                overlap = 0.0
            metrics.overlaps.append(overlap)

            # Inverted index decode accuracy
            predicted_token = decode_index.decode(predicted_set)
            accuracy = 1.0 if predicted_token == token_id else 0.0
            metrics.accuracies.append(accuracy)

            # Synaptic decode accuracy
            syn_acc = 1.0 if synaptic_token == token_str else 0.0
            metrics.synaptic_accuracies.append(syn_acc)

        decode_index.observe(token_id, active_set)

        if t > 0 and t % log_interval == 0 and metrics.overlaps:
            tail = metrics.overlaps[-rolling_window:]
            roll_overlap = sum(tail) / len(tail)
            tail_acc = metrics.accuracies[-rolling_window:]
            roll_acc = sum(tail_acc) / len(tail_acc)
            tail_syn = metrics.synaptic_accuracies[-rolling_window:]
            roll_syn = sum(tail_syn) / len(tail_syn)
            elapsed = time.monotonic() - start
            print(
                f"  [cortex] t={t:,} "
                f"overlap={roll_overlap:.4f} "
                f"acc={roll_acc:.4f} "
                f"syn_acc={roll_syn:.4f} "
                f"({elapsed:.1f}s)"
            )

    metrics.elapsed_seconds = time.monotonic() - start
    return metrics


def run_step_baseline(
    tokens: list[tuple[int, frozenset[int]]],
    model_config: ModelConfig,
    log_interval: int = 100,
    rolling_window: int = 100,
) -> RunMetrics:
    """Run original STEP model on a token sequence for comparison.

    tokens: list of (token_id, sdr) pairs.
            token_id == STORY_BOUNDARY signals a story boundary.
    """
    from step.model import initial_state, learn, observe, predict

    state = initial_state(model_config)
    decode_index = DecodeIndex()
    metrics = RunMetrics()
    start = time.monotonic()

    after_boundary = False
    for t, (token_id, sdr) in enumerate(tokens):
        if token_id == STORY_BOUNDARY:
            state.history.clear()
            after_boundary = True
            continue

        if t > 0 and not after_boundary:
            predicted_sdr = predict(state, t, model_config)
            iou = learn(state, t, sdr, predicted_sdr, model_config)
            metrics.overlaps.append(iou)

            predicted_token = decode_index.decode(predicted_sdr)
            accuracy = 1.0 if predicted_token == token_id else 0.0
            metrics.accuracies.append(accuracy)

        after_boundary = False
        state = observe(state, t, sdr, model_config)
        decode_index.observe(token_id, sdr)

        if t > 0 and t % log_interval == 0 and metrics.overlaps:
            tail = metrics.overlaps[-rolling_window:]
            roll_iou = sum(tail) / len(tail)
            tail_acc = metrics.accuracies[-rolling_window:]
            roll_acc = sum(tail_acc) / len(tail_acc)
            elapsed = time.monotonic() - start
            print(
                f"  [step]   t={t:,} "
                f"iou={roll_iou:.4f} "
                f"acc={roll_acc:.4f} "
                f"({elapsed:.1f}s)"
            )

    metrics.elapsed_seconds = time.monotonic() - start
    return metrics
