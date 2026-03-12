"""Cortex training loop with natural prediction measurement."""

import time
from dataclasses import dataclass, field

from step.config import ModelConfig
from step.cortex.decoder import SynapticDecoder
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
    column_accuracies: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def run_cortex(
    region: SensoryRegion,
    encoder: CharbitEncoder,
    tokens: list[tuple[int, str]],
    log_interval: int = 100,
    rolling_window: int = 100,
    diagnostics: CortexDiagnostics | None = None,
    show_predictions: int = 0,
) -> RunMetrics:
    """Run cortex model on a token sequence, measuring prediction quality.

    tokens: list of (token_id, token_string) pairs.
            token_id == STORY_BOUNDARY signals a story boundary.
    show_predictions: if > 0, print this many prediction samples at each
                      log interval (actual vs predicted for each decoder).
    """
    decode_index = DecodeIndex()
    syn_decoder = SynapticDecoder()
    metrics = RunMetrics()
    k = region.k_columns
    start = time.monotonic()

    # Buffer for prediction samples (for display)
    prediction_log: list[tuple[str, str, str, str]] = []

    for t, (token_id, token_str) in enumerate(tokens):
        if token_id == STORY_BOUNDARY:
            region.reset_working_memory()
            continue

        # Prediction: what does the network expect before seeing input?
        predicted_neurons = region.get_prediction(k)
        predicted_set = frozenset(int(i) for i in predicted_neurons)

        # Three decode paths
        syn_id, syn_str = syn_decoder.decode_synaptic(predicted_neurons, region)
        col_id, col_str = syn_decoder.decode_columns(predicted_neurons, region.n_l4)
        idx_predicted = decode_index.decode(predicted_set)

        # Look up string for idx decode
        idx_str = ""
        if idx_predicted >= 0 and idx_predicted in syn_decoder._token_id_set:
            for i, tid in enumerate(syn_decoder._token_ids):
                if tid == idx_predicted:
                    idx_str = syn_decoder._token_strs[i]
                    break

        # Step: feed input, triggers activation + learning
        encoding = encoder.encode(token_str)
        active_neurons = region.process(encoding)
        active_set = frozenset(int(i) for i in active_neurons)

        if diagnostics is not None:
            diagnostics.step(t, region)

        if t > 0:
            # Prediction overlap
            if active_set:
                overlap = len(predicted_set & active_set) / len(active_set)
            else:
                overlap = 0.0
            metrics.overlaps.append(overlap)

            # Inverted index decode accuracy
            accuracy = 1.0 if idx_predicted == token_id else 0.0
            metrics.accuracies.append(accuracy)

            # Synaptic decode accuracy
            syn_acc = 1.0 if syn_id == token_id else 0.0
            metrics.synaptic_accuracies.append(syn_acc)

            # Column decode accuracy
            col_acc = 1.0 if col_id == token_id else 0.0
            metrics.column_accuracies.append(col_acc)

            if show_predictions > 0:
                prediction_log.append((token_str, idx_str, col_str, syn_str))

        decode_index.observe(token_id, active_set)
        syn_decoder.observe(token_id, token_str, encoding, region.active_columns)

        if t > 0 and t % log_interval == 0 and metrics.overlaps:
            tail = metrics.overlaps[-rolling_window:]
            roll_overlap = sum(tail) / len(tail)
            tail_acc = metrics.accuracies[-rolling_window:]
            roll_acc = sum(tail_acc) / len(tail_acc)
            tail_syn = metrics.synaptic_accuracies[-rolling_window:]
            roll_syn = sum(tail_syn) / len(tail_syn)
            tail_col = metrics.column_accuracies[-rolling_window:]
            roll_col = sum(tail_col) / len(tail_col)
            elapsed = time.monotonic() - start
            print(
                f"  [cortex] t={t:,} "
                f"syn={roll_syn:.4f} "
                f"col={roll_col:.4f} "
                f"idx={roll_acc:.4f} "
                f"overlap={roll_overlap:.4f} "
                f"({elapsed:.1f}s)"
            )

            # Show prediction samples
            if show_predictions > 0 and prediction_log:
                samples = prediction_log[-show_predictions:]
                hdr = f"{'actual':>12s} | {'idx':>12s} | {'col':>12s} | {'syn':>12s}"
                print(f"    {hdr}")
                print(f"    {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
                for actual, idx_p, col_p, syn_p in samples:
                    fmt = lambda s: repr(s)[:12].ljust(12)  # noqa: E731
                    marks = [
                        "*" if p == actual else " "
                        for p in (idx_p, col_p, syn_p)
                    ]
                    print(
                        f"    {fmt(actual)} "
                        f"|{marks[0]}{fmt(idx_p)} "
                        f"|{marks[1]}{fmt(col_p)} "
                        f"|{marks[2]}{fmt(syn_p)}"
                    )
                prediction_log.clear()

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
