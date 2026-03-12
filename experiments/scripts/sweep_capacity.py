#!/usr/bin/env python3
"""Sweep neuron capacity configurations: per-neuron ff_weights × neurons per column.

Tests 6 configurations on 10K tokens and reports key metrics.

Usage: uv run experiments/scripts/sweep_capacity.py [--tokens N]
"""

import argparse
import string

from datasets import load_dataset
from transformers import AutoTokenizer

import step.env  # noqa: F401
from step.cortex.config import CortexConfig
from step.cortex.diagnostics import CortexDiagnostics
from step.cortex.runner import STORY_BOUNDARY, run_cortex
from step.cortex.sensory import SensoryRegion
from step.encoders.charbit import CharbitEncoder

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1

CONFIGS = [
    # (name, n_l4, n_l23, per_neuron_ff)
    ("shared-4n", 4, 4, False),
    ("shared-8n", 8, 8, False),
    ("shared-16n", 16, 16, False),
    ("neuron-4n", 4, 4, True),
    ("neuron-8n", 8, 8, True),
    ("neuron-16n", 16, 16, True),
]


def prepare_tokens(max_tokens: int):
    """Load TinyStories tokens for cortex model."""
    print("Loading dataset and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    tokens: list[tuple[int, str]] = []
    t = 0
    first_story = True
    for example in dataset:
        if not first_story:
            tokens.append((STORY_BOUNDARY, ""))
            t += 1
            if t >= max_tokens:
                break
        first_story = False
        for tid in tokenizer.encode(example["text"]):
            tokens.append((tid, tokenizer.decode([tid])))
            t += 1
            if t >= max_tokens:
                break
        if t >= max_tokens:
            break

    unique = len({tid for tid, _ in tokens if tid != STORY_BOUNDARY})
    boundaries = sum(1 for tid, _ in tokens if tid == STORY_BOUNDARY)
    print(f"  {len(tokens):,} tokens, {unique} unique, {boundaries + 1} stories\n")
    return tokens


def run_config(name, n_l4, n_l23, per_neuron_ff, tokens, log_interval):
    """Run a single configuration and return results dict."""
    cortex_cfg = CortexConfig(n_l4=n_l4, n_l23=n_l23)
    charbit = CharbitEncoder(length=CHAR_LENGTH, width=CHAR_WIDTH, chars=CHARS)
    input_dim = CHAR_LENGTH * CHAR_WIDTH

    region = SensoryRegion(
        input_dim=input_dim,
        encoding_width=CHAR_WIDTH,
        n_columns=cortex_cfg.n_columns,
        n_l4=n_l4,
        n_l23=n_l23,
        k_columns=cortex_cfg.k_columns,
        voltage_decay=cortex_cfg.voltage_decay,
        eligibility_decay=cortex_cfg.eligibility_decay,
        synapse_decay=cortex_cfg.synapse_decay,
        learning_rate=cortex_cfg.learning_rate,
        max_excitability=cortex_cfg.max_excitability,
        fb_boost_threshold=cortex_cfg.fb_boost_threshold,
        fb_boost=cortex_cfg.fb_boost,
        ltd_rate=cortex_cfg.ltd_rate,
        burst_learning_scale=cortex_cfg.burst_learning_scale,
        prediction_ltd_rate=cortex_cfg.prediction_ltd_rate,
        per_neuron_ff=per_neuron_ff,
        seed=cortex_cfg.seed,
    )

    n_params = (
        region.ff_weights.size
        + region.fb_weights.size
        + region.lateral_weights.size
        + region.l23_lateral_weights.size
    )

    diag = CortexDiagnostics(snapshot_interval=log_interval)

    print(f"--- {name} ---")
    print(f"  n_l4={n_l4} n_l23={n_l23} per_neuron_ff={per_neuron_ff}")
    print(f"  ff_weights: {region.ff_weights.shape}, params: {n_params:,}")

    metrics = run_cortex(
        region, charbit, tokens,
        log_interval=log_interval,
        diagnostics=diag,
    )

    summ = diag.summary()
    snap = diag.snapshots[-1] if diag.snapshots else None

    # Last-100 rolling
    last_n = 100
    tail_overlap = metrics.overlaps[-last_n:] if metrics.overlaps else []
    tail_idx = metrics.accuracies[-last_n:] if metrics.accuracies else []
    tail_col = metrics.column_accuracies[-last_n:] if metrics.column_accuracies else []
    tail_syn = metrics.synaptic_accuracies[-last_n:] if metrics.synaptic_accuracies else []

    result = {
        "name": name,
        "n_l4": n_l4,
        "n_l23": n_l23,
        "per_neuron_ff": per_neuron_ff,
        "n_params": n_params,
        "time": metrics.elapsed_seconds,
        # Overall averages
        "avg_overlap": sum(metrics.overlaps) / len(metrics.overlaps) if metrics.overlaps else 0,
        "avg_idx_acc": sum(metrics.accuracies) / len(metrics.accuracies) if metrics.accuracies else 0,
        "avg_col_acc": sum(metrics.column_accuracies) / len(metrics.column_accuracies) if metrics.column_accuracies else 0,
        "avg_syn_acc": sum(metrics.synaptic_accuracies) / len(metrics.synaptic_accuracies) if metrics.synaptic_accuracies else 0,
        # Last-100
        "last_overlap": sum(tail_overlap) / len(tail_overlap) if tail_overlap else 0,
        "last_idx": sum(tail_idx) / len(tail_idx) if tail_idx else 0,
        "last_col": sum(tail_col) / len(tail_col) if tail_col else 0,
        "last_syn": sum(tail_syn) / len(tail_syn) if tail_syn else 0,
        # Diagnostics
        "entropy_ratio": summ["column_entropy_ratio"],
        "unique_col_sets": summ["unique_column_sets"],
        "unique_l4": summ["unique_l4_neurons"],
        "burst_rate": summ["burst_rate"],
        "unique_pred_sets": summ["unique_prediction_sets"],
        "pred_hit_neuron": summ["prediction_hit_neuron"],
        "pred_hit_column": summ["prediction_hit_column"],
        "n_predicted": snap.n_predicted_neurons if snap else 0,
        "fb_cosine": snap.fb_row_cosine_mean if snap else 0,
        "lat_cosine": snap.lat_row_cosine_mean if snap else 0,
    }

    print(f"  overlap={result['avg_overlap']:.4f} idx={result['avg_idx_acc']:.1%} "
          f"col={result['avg_col_acc']:.1%} syn={result['avg_syn_acc']:.1%} "
          f"({result['time']:.1f}s)")
    print(f"  entropy={result['entropy_ratio']:.1%} burst={result['burst_rate']:.1%} "
          f"pred_sets={result['unique_pred_sets']} predicted_n={result['n_predicted']}")
    print()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=2000)
    args = parser.parse_args()

    tokens = prepare_tokens(args.tokens)

    results = []
    for name, n_l4, n_l23, per_neuron_ff in CONFIGS:
        result = run_config(name, n_l4, n_l23, per_neuron_ff, tokens, args.log_interval)
        results.append(result)

    # Summary table
    print("\n" + "=" * 120)
    print(f"{'Config':<14} {'Params':>8} {'Time':>6} "
          f"{'Overlap':>8} {'Idx':>6} {'Col':>6} {'Syn':>6} "
          f"{'L-Idx':>6} {'L-Col':>6} {'L-Syn':>6} "
          f"{'Entropy':>8} {'ColSets':>8} {'Burst':>6} "
          f"{'PredSets':>9} {'Pred#':>6} {'FbCos':>6}")
    print("=" * 120)

    for r in results:
        print(f"{r['name']:<14} {r['n_params']:>8,} {r['time']:>5.1f}s "
              f"{r['avg_overlap']:>8.4f} {r['avg_idx_acc']:>5.1%} {r['avg_col_acc']:>5.1%} {r['avg_syn_acc']:>5.1%} "
              f"{r['last_idx']:>5.1%} {r['last_col']:>5.1%} {r['last_syn']:>5.1%} "
              f"{r['entropy_ratio']:>7.1%} {r['unique_col_sets']:>8} {r['burst_rate']:>5.1%} "
              f"{r['unique_pred_sets']:>9} {r['n_predicted']:>6} {r['fb_cosine']:>6.3f}")

    print("=" * 120)

    # Find best config by last-100 idx accuracy
    best_idx = max(results, key=lambda r: r["last_idx"])
    best_col = max(results, key=lambda r: r["last_col"])
    best_syn = max(results, key=lambda r: r["last_syn"])
    print(f"\nBest last-100 idx: {best_idx['name']} ({best_idx['last_idx']:.1%})")
    print(f"Best last-100 col: {best_col['name']} ({best_col['last_col']:.1%})")
    print(f"Best last-100 syn: {best_syn['name']} ({best_syn['last_syn']:.1%})")


if __name__ == "__main__":
    main()
