#!/usr/bin/env python3
"""Evaluate the dendritic decoder: nonlinear next-token prediction.

Runs the cortex and dendritic decoder online — at each step, the decoder
tries to predict the next token from the current L2/3 state, then learns
from the actual token. This mirrors how a downstream cortical region
would read representations in real time.

Compares: dendritic decoder (nonlinear) vs linear probe (baseline).

Usage: uv run experiments/scripts/eval_dendritic_decoder.py [--tokens 20000]
"""

import argparse
import string
import time

import numpy as np

import step.env  # noqa: F401
from step.config import CortexConfig
from step.cortex.sensory import SensoryRegion
from step.data import STORY_BOUNDARY, prepare_tokens
from step.decoders.dendritic import DendriticDecoder
from step.encoders.charbit import CharbitEncoder

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1


def run_online_eval(
    tokens: list[tuple[int, str]],
    cfg: CortexConfig,
    *,
    log_interval: int = 1000,
) -> dict:
    """Run cortex + dendritic decoder in lockstep, measuring online accuracy."""
    charbit = CharbitEncoder(length=CHAR_LENGTH, width=CHAR_WIDTH, chars=CHARS)
    input_dim = CHAR_LENGTH * CHAR_WIDTH
    region = SensoryRegion(
        input_dim=input_dim,
        encoding_width=CHAR_WIDTH,
        n_columns=cfg.n_columns,
        n_l4=cfg.n_l4,
        n_l23=cfg.n_l23,
        k_columns=cfg.k_columns,
        voltage_decay=cfg.voltage_decay,
        eligibility_decay=cfg.eligibility_decay,
        synapse_decay=cfg.synapse_decay,
        learning_rate=cfg.learning_rate,
        max_excitability=cfg.max_excitability,
        fb_boost=cfg.fb_boost,
        ltd_rate=cfg.ltd_rate,
        burst_learning_scale=cfg.burst_learning_scale,
        seed=cfg.seed,
    )

    decoder = DendriticDecoder(source_dim=region.n_l23_total)

    prev_l23: np.ndarray | None = None
    prev_was_boundary = False

    # Online accuracy tracking (windowed)
    top1_hits: list[bool] = []
    top5_hits: list[bool] = []
    window_top1: list[bool] = []
    window_top5: list[bool] = []

    start = time.monotonic()
    for t, (token_id, token_str) in enumerate(tokens):
        if token_id == STORY_BOUNDARY:
            region.reset_working_memory()
            prev_l23 = None
            prev_was_boundary = True
            continue

        # --- Decode: predict this token from previous L2/3 state ---
        if prev_l23 is not None and not prev_was_boundary:
            predictions = decoder.decode(prev_l23, k=5)
            hit1 = len(predictions) > 0 and predictions[0] == token_id
            hit5 = token_id in predictions
            top1_hits.append(hit1)
            top5_hits.append(hit5)
            window_top1.append(hit1)
            window_top5.append(hit5)

        # --- Process token through cortex ---
        encoding = charbit.encode(token_str)
        region.process(encoding)

        # --- Learn: teach decoder about this token ---
        if prev_l23 is not None and not prev_was_boundary:
            decoder.observe(token_id, prev_l23)

        prev_l23 = region.active_l23.copy()
        prev_was_boundary = False

        if t > 0 and t % log_interval == 0:
            elapsed = time.monotonic() - start
            n = len(top1_hits)
            recent = min(log_interval, len(window_top1))
            w_top1 = sum(window_top1[-recent:]) / recent if recent > 0 else 0
            w_top5 = sum(window_top5[-recent:]) / recent if recent > 0 else 0
            cum_top1 = sum(top1_hits) / n if n > 0 else 0
            print(
                f"  t={t:>6d}  "
                f"window top1={w_top1:.3f} top5={w_top5:.3f}  "
                f"cumulative top1={cum_top1:.3f}  "
                f"vocab={decoder.n_tokens}  "
                f"({elapsed:.1f}s)"
            )

    elapsed = time.monotonic() - start
    n = len(top1_hits)
    return {
        "top1": sum(top1_hits) / n if n else 0,
        "top5": sum(top5_hits) / n if n else 0,
        "n_samples": n,
        "n_tokens_seen": decoder.n_tokens,
        "elapsed": elapsed,
        # Per-token breakdown for analysis
        "top1_hits": top1_hits,
        "top5_hits": top5_hits,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dendritic decoder online"
    )
    parser.add_argument("--tokens", type=int, default=20000)
    parser.add_argument("--log-interval", type=int, default=2000)
    args = parser.parse_args()

    tokens = prepare_tokens(args.tokens)
    cfg = CortexConfig()

    unique_tokens = len({tid for tid, _ in tokens if tid != STORY_BOUNDARY})

    print("\nOnline dendritic decoder evaluation")
    print(f"  {len(tokens):,} tokens, {unique_tokens} unique")
    print(f"  Region: {cfg.n_columns} cols x {cfg.n_l4} L4 x {cfg.n_l23} L2/3")
    print(f"  L2/3 dim: {cfg.n_columns * cfg.n_l23}")
    print()

    results = run_online_eval(tokens, cfg, log_interval=args.log_interval)

    # Baselines
    token_ids = [tid for tid, _ in tokens if tid != STORY_BOUNDARY]
    from collections import Counter

    counts = Counter(token_ids)
    majority_frac = counts.most_common(1)[0][1] / len(token_ids)
    uniform_top1 = 1.0 / unique_tokens
    uniform_top5 = min(5.0 / unique_tokens, 1.0)

    print(f"\n{'=' * 50}")
    print("Dendritic Decoder Results (online)")
    print(f"{'=' * 50}")
    print(f"  Top-1 accuracy: {results['top1']:.4f}")
    print(f"  Top-5 accuracy: {results['top5']:.4f}")
    print(f"  Vocab learned:  {results['n_tokens_seen']}")
    print(f"  Samples:        {results['n_samples']:,}")
    print(f"  Time:           {results['elapsed']:.1f}s")
    print()
    print("Baselines:")
    print(f"  Majority class: {majority_frac:.4f}")
    print(f"  Uniform random: top1={uniform_top1:.4f} top5={uniform_top5:.4f}")

    lift = results["top1"] / max(majority_frac, 1e-10)
    print(f"\n  Lift over majority: {lift:.2f}x")

    # Learning curve: compare first half vs second half
    n = results["n_samples"]
    mid = n // 2
    first_half = sum(results["top1_hits"][:mid]) / mid if mid > 0 else 0
    second_half = sum(results["top1_hits"][mid:]) / (n - mid) if n > mid else 0
    print(f"\n  First half top-1:  {first_half:.4f}")
    print(f"  Second half top-1: {second_half:.4f}")
    if second_half > first_half * 1.1:
        print("  -> Learning curve: improving over time")
    elif second_half < first_half * 0.9:
        print("  -> Learning curve: degrading (possible overfitting)")
    else:
        print("  -> Learning curve: flat")


if __name__ == "__main__":
    main()
