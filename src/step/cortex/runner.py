"""Cortex training loop with natural prediction measurement."""

import time
from dataclasses import dataclass, field

import numpy as np

from step.config import ModelConfig
from step.cortex.diagnostics import CortexDiagnostics
from step.cortex.representation import RepresentationTracker
from step.cortex.sensory import SensoryRegion
from step.cortex.surprise import SurpriseTracker
from step.decoders import InvertedIndexDecoder, SynapticDecoder
from step.encoders.charbit import CharbitEncoder

STORY_BOUNDARY = -1


@dataclass
class RunMetrics:
    overlaps: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    synaptic_accuracies: list[float] = field(default_factory=list)
    column_accuracies: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    representation: dict = field(default_factory=dict)


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
    decode_index = InvertedIndexDecoder()
    syn_decoder = SynapticDecoder()
    rep_tracker = RepresentationTracker(region.n_columns, region.n_l4)
    metrics = RunMetrics()
    k = region.k_columns
    start = time.monotonic()

    # Buffer for prediction samples (for display)
    prediction_log: list[tuple[str, str, str, str]] = []

    for t, (token_id, token_str) in enumerate(tokens):
        if token_id == STORY_BOUNDARY:
            region.reset_working_memory()
            rep_tracker.reset_context()
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

        rep_tracker.observe(token_id, region.active_columns, region.active_l4)

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
            tail_syn = metrics.synaptic_accuracies[-rolling_window:]
            roll_syn = sum(tail_syn) / len(tail_syn)
            tail = metrics.overlaps[-rolling_window:]
            roll_overlap = sum(tail) / len(tail)
            elapsed = time.monotonic() - start

            # Burst rate over recent window
            if diagnostics is not None:
                bc = diagnostics._burst_count
                pc = diagnostics._precise_count
                total = bc + pc
                burst_pct = bc / total if total > 0 else 0.0
            else:
                burst_pct = 0.0

            print(
                f"  [cortex] t={t:,} "
                f"syn={roll_syn:.4f} "
                f"overlap={roll_overlap:.4f} "
                f"burst={burst_pct:.1%} "
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
    rep_summ = rep_tracker.summary(region.ff_weights)
    # Include per-column selectivity for dashboard visualization
    sel = rep_tracker.column_selectivity()
    rep_summ["column_selectivity_per_col"] = sel["per_column"]
    metrics.representation = rep_summ

    # Print representation report
    rep_tracker.print_report(region.ff_weights)

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
    decode_index = InvertedIndexDecoder()
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


@dataclass
class HierarchyMetrics:
    region1: RunMetrics = field(default_factory=RunMetrics)
    region2: RunMetrics = field(default_factory=RunMetrics)
    surprise_modulators: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def run_hierarchy(
    region1: SensoryRegion,
    region2: SensoryRegion,
    encoder: CharbitEncoder,
    tokens: list[tuple[int, str]],
    *,
    surprise_tracker: SurpriseTracker | None = None,
    log_interval: int = 100,
    rolling_window: int = 100,
    diagnostics1: CortexDiagnostics | None = None,
    diagnostics2: CortexDiagnostics | None = None,
) -> HierarchyMetrics:
    """Run two-region hierarchy: Region 1 (sensory) → Region 2 (secondary).

    Region 2 receives Region 1's L2/3 boolean output as its encoding.
    Surprise (Region 1 burst rate) modulates Region 2 learning rate.
    """
    if surprise_tracker is None:
        surprise_tracker = SurpriseTracker()

    decode_index1 = InvertedIndexDecoder()
    syn_decoder1 = SynapticDecoder()
    rep_tracker1 = RepresentationTracker(region1.n_columns, region1.n_l4)
    rep_tracker2 = RepresentationTracker(region2.n_columns, region2.n_l4)
    metrics = HierarchyMetrics()
    k1 = region1.k_columns
    start = time.monotonic()

    for t, (token_id, token_str) in enumerate(tokens):
        if token_id == STORY_BOUNDARY:
            region1.reset_working_memory()
            region2.reset_working_memory()
            rep_tracker1.reset_context()
            rep_tracker2.reset_context()
            continue

        # --- Region 1: prediction + activation ---
        predicted_neurons = region1.get_prediction(k1)
        predicted_set = frozenset(int(i) for i in predicted_neurons)
        syn_id, _ = syn_decoder1.decode_synaptic(predicted_neurons, region1)
        idx_predicted = decode_index1.decode(predicted_set)

        encoding = encoder.encode(token_str)
        active_neurons1 = region1.process(encoding)
        active_set1 = frozenset(int(i) for i in active_neurons1)

        rep_tracker1.observe(token_id, region1.active_columns, region1.active_l4)
        if diagnostics1 is not None:
            diagnostics1.step(t, region1)

        # --- Surprise modulation from Region 1 burst rate ---
        n_active_cols = int(region1.active_columns.sum())
        n_bursting = int(region1.bursting_columns.sum())
        burst_rate = n_bursting / max(n_active_cols, 1)
        modulator = surprise_tracker.update(burst_rate)
        region2.surprise_modulator = modulator
        metrics.surprise_modulators.append(modulator)

        # --- Region 2: receives Region 1 L2/3 output ---
        r2_encoding = region1.active_l23.astype(np.float64)
        region2.process(r2_encoding)

        rep_tracker2.observe(token_id, region2.active_columns, region2.active_l4)
        if diagnostics2 is not None:
            diagnostics2.step(t, region2)

        # --- Region 1 decoder metrics ---
        if t > 0:
            if active_set1:
                overlap = len(predicted_set & active_set1) / len(active_set1)
            else:
                overlap = 0.0
            metrics.region1.overlaps.append(overlap)

            accuracy = 1.0 if idx_predicted == token_id else 0.0
            metrics.region1.accuracies.append(accuracy)

            syn_acc = 1.0 if syn_id == token_id else 0.0
            metrics.region1.synaptic_accuracies.append(syn_acc)

        decode_index1.observe(token_id, active_set1)
        syn_decoder1.observe(token_id, token_str, encoding, region1.active_columns)

        if t > 0 and t % log_interval == 0 and metrics.region1.overlaps:
            tail_syn = metrics.region1.synaptic_accuracies[-rolling_window:]
            roll_syn = sum(tail_syn) / len(tail_syn)
            tail_o = metrics.region1.overlaps[-rolling_window:]
            roll_o = sum(tail_o) / len(tail_o)
            tail_mod = metrics.surprise_modulators[-rolling_window:]
            avg_mod = sum(tail_mod) / len(tail_mod)
            elapsed = time.monotonic() - start

            burst_pct = 0.0
            if diagnostics1 is not None:
                bc = diagnostics1._burst_count
                pc = diagnostics1._precise_count
                total = bc + pc
                burst_pct = bc / total if total > 0 else 0.0

            print(
                f"  [R1] t={t:,} "
                f"syn={roll_syn:.4f} "
                f"overlap={roll_o:.4f} "
                f"burst={burst_pct:.1%} "
                f"mod={avg_mod:.2f} "
                f"({elapsed:.1f}s)"
            )

    elapsed = time.monotonic() - start
    metrics.elapsed_seconds = elapsed

    # Region 1 representation
    rep_summ1 = rep_tracker1.summary(region1.ff_weights)
    sel1 = rep_tracker1.column_selectivity()
    rep_summ1["column_selectivity_per_col"] = sel1["per_column"]
    metrics.region1.representation = rep_summ1

    # Region 2 representation
    rep_summ2 = rep_tracker2.summary(region2.ff_weights)
    sel2 = rep_tracker2.column_selectivity()
    rep_summ2["column_selectivity_per_col"] = sel2["per_column"]
    metrics.region2.representation = rep_summ2

    print("\n--- Region 1 ---")
    rep_tracker1.print_report(region1.ff_weights)
    print("\n--- Region 2 ---")
    rep_tracker2.print_report(region2.ff_weights)

    return metrics
