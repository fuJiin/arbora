#!/usr/bin/env python3
"""PoC: Compare cortex model vs original STEP on TinyStories.

Usage: uv run experiments/scripts/poc_cortex.py [--tokens N]
"""

import argparse
import string

from datasets import load_dataset
from transformers import AutoTokenizer

import step.env  # noqa: F401
from step.config import EncoderConfig, ModelConfig
from step.cortex.config import CortexConfig
from step.cortex.diagnostics import CortexDiagnostics
from step.cortex.runner import STORY_BOUNDARY, run_cortex, run_step_baseline
from step.cortex.sensory import SensoryRegion
from step.encoders.charbit import CharbitEncoder
from step.encoders.random import RandomEncoder

# CharbitEncoder params
CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1  # 101

# STEP baseline params (best from prior experiments)
STEP_N = 256
STEP_K = 10


def prepare_tokens(max_tokens: int):
    """Load TinyStories tokens with both string and SDR representations."""
    print("Loading dataset and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    enc_cfg = EncoderConfig(n=STEP_N, k=STEP_K, vocab_size=50257)
    encoder = RandomEncoder(enc_cfg)

    cortex_tokens: list[tuple[int, str]] = []
    step_tokens: list[tuple[int, frozenset[int]]] = []

    t = 0
    first_story = True
    for example in dataset:
        if not first_story:
            cortex_tokens.append((STORY_BOUNDARY, ""))
            step_tokens.append((STORY_BOUNDARY, frozenset()))
            t += 1
            if t >= max_tokens:
                break
        first_story = False

        token_ids = tokenizer.encode(example["text"])
        for tid in token_ids:
            token_str = tokenizer.decode([tid])
            cortex_tokens.append((tid, token_str))
            step_tokens.append((tid, encoder.encode(tid)))
            t += 1
            if t >= max_tokens:
                break
        if t >= max_tokens:
            break

    print(f"  Prepared {len(cortex_tokens):,} tokens")

    # Count unique tokens and stories
    unique = len({tid for tid, _ in cortex_tokens if tid != STORY_BOUNDARY})
    boundaries = sum(1 for tid, _ in cortex_tokens if tid == STORY_BOUNDARY)
    print(f"  Unique tokens: {unique}, Stories: {boundaries + 1}")

    return cortex_tokens, step_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()

    cortex_tokens, step_tokens = prepare_tokens(args.tokens)

    # --- Cortex model ---
    cortex_cfg = CortexConfig()
    charbit = CharbitEncoder(length=CHAR_LENGTH, width=CHAR_WIDTH, chars=CHARS)
    input_dim = CHAR_LENGTH * CHAR_WIDTH
    region = SensoryRegion(
        input_dim=input_dim,
        encoding_width=CHAR_WIDTH,
        n_columns=cortex_cfg.n_columns,
        n_l4=cortex_cfg.n_l4,
        n_l23=cortex_cfg.n_l23,
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
        seed=cortex_cfg.seed,
    )

    n_params_cortex = (
        region.ff_weights.size
        + region.fb_weights.size
        + region.lateral_weights.size
        + region.l23_lateral_weights.size
    )

    diag = CortexDiagnostics(snapshot_interval=args.log_interval)

    print("\n--- Cortex model ---")
    print(
        f"  Columns={cortex_cfg.n_columns} "
        f"L4={cortex_cfg.n_l4} L2/3={cortex_cfg.n_l23} "
        f"k={cortex_cfg.k_columns}"
    )
    print(f"  Input dim: {input_dim} (CharbitEncoder {CHAR_LENGTH}x{CHAR_WIDTH})")
    print(f"  Parameters: {n_params_cortex:,}")

    cortex_metrics = run_cortex(
        region,
        charbit,
        cortex_tokens,
        log_interval=args.log_interval,
        diagnostics=diag,
    )

    diag.print_report()

    # --- STEP baseline ---
    step_cfg = ModelConfig(
        n=STEP_N,
        k=STEP_K,
        max_lr=0.5,
        weight_decay=0.999,
        penalty_factor=0.5,
        eligibility_window=20,
    )

    n_params_step = STEP_N * STEP_N
    print("\n--- STEP baseline ---")
    print(f"  n={STEP_N} k={STEP_K} window={step_cfg.eligibility_window}")
    print(f"  Parameters: {n_params_step:,}")

    step_metrics = run_step_baseline(
        step_tokens,
        step_cfg,
        log_interval=args.log_interval,
    )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(
        f"{'Model':<12} {'Params':>8} {'Metric':>8} "
        f"{'Idx Acc':>8} {'Syn Acc':>8} {'Time':>8}"
    )
    print(f"{'=' * 60}")

    def summarize(name, metrics, n_params):
        avg_metric = (
            sum(metrics.overlaps) / len(metrics.overlaps) if metrics.overlaps else 0.0
        )
        avg_acc = (
            sum(metrics.accuracies) / len(metrics.accuracies)
            if metrics.accuracies
            else 0.0
        )
        avg_syn = (
            sum(metrics.synaptic_accuracies) / len(metrics.synaptic_accuracies)
            if metrics.synaptic_accuracies
            else 0.0
        )
        syn_str = f"{avg_syn:>7.1%}" if metrics.synaptic_accuracies else "    n/a"
        print(
            f"{name:<12} {n_params:>8,} "
            f"{avg_metric:>8.4f} "
            f"{avg_acc:>7.1%} "
            f"{syn_str} "
            f"{metrics.elapsed_seconds:>7.1f}s"
        )

    summarize("Cortex", cortex_metrics, n_params_cortex)
    summarize("STEP", step_metrics, n_params_step)

    # Rolling last-100 if enough data
    if len(cortex_metrics.overlaps) >= 100:
        print("\nLast-100 rolling:")
        c_tail = cortex_metrics.overlaps[-100:]
        s_tail = step_metrics.overlaps[-100:]
        ca_tail = cortex_metrics.accuracies[-100:]
        sa_tail = step_metrics.accuracies[-100:]
        cs_tail = cortex_metrics.synaptic_accuracies[-100:]
        print(
            f"  Cortex: overlap={sum(c_tail) / 100:.4f} "
            f"idx_acc={sum(ca_tail) / 100:.1%} "
            f"syn_acc={sum(cs_tail) / 100:.1%}"
        )
        print(f"  STEP:   iou={sum(s_tail) / 100:.4f} acc={sum(sa_tail) / 100:.1%}")


if __name__ == "__main__":
    main()
