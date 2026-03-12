#!/usr/bin/env python3
"""Sweep Region 2 learning parameters at 20k tokens.

Tests learning_rate and ltd_rate combinations to improve R2 column
diversity (cross-col cosine) and FF sparsity. Runs on BabyLM with
CharbitEncoder.

Usage:
  uv run experiments/scripts/sweep_r2_params.py
  uv run experiments/scripts/sweep_r2_params.py --tokens 10000
"""

import argparse
import itertools
import string
import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

import step.env  # noqa: F401
from step.cortex.diagnostics import CortexDiagnostics
from step.cortex.runner import STORY_BOUNDARY, run_hierarchy
from step.cortex.sensory import SensoryRegion
from step.cortex.surprise import SurpriseTracker
from step.encoders.charbit import CharbitEncoder

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1


def prepare_tokens(max_tokens: int):
    print("Loading BabyLM (10M)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("nilq/babylm-10M", split="train")

    tokens: list[tuple[int, str]] = []
    t = 0
    in_doc = False
    for ex in dataset:
        text = ex.get("text", "").strip()
        if not text:
            if in_doc:
                tokens.append((STORY_BOUNDARY, ""))
                t += 1
                in_doc = False
            if t >= max_tokens:
                break
            continue
        in_doc = True
        for tid in tokenizer.encode(text):
            tokens.append((tid, tokenizer.decode([tid])))
            t += 1
            if t >= max_tokens:
                break
        if t >= max_tokens:
            break

    unique = len({tid for tid, _ in tokens if tid != STORY_BOUNDARY})
    print(f"  {len(tokens):,} tokens, {unique} unique\n")
    return tokens


def run_one(tokens, encoder, input_dim, r2_lr, r2_ltd, log_interval):
    """Run hierarchy with given R2 params, return summary dict."""
    region1 = SensoryRegion(
        input_dim=input_dim,
        encoding_width=CHAR_WIDTH,
        n_columns=32,
        n_l4=4,
        n_l23=4,
        k_columns=4,
        seed=42,
    )
    region2 = SensoryRegion(
        input_dim=region1.n_l23_total,
        encoding_width=region1.n_l23,
        n_columns=16,
        n_l4=4,
        n_l23=4,
        k_columns=2,
        voltage_decay=0.8,
        eligibility_decay=0.98,
        synapse_decay=0.9999,
        learning_rate=r2_lr,
        ltd_rate=r2_ltd,
        seed=123,
    )
    surprise = SurpriseTracker()
    diag2 = CortexDiagnostics(snapshot_interval=log_interval)

    start = time.monotonic()
    metrics = run_hierarchy(
        region1,
        region2,
        encoder,
        tokens,
        surprise_tracker=surprise,
        log_interval=log_interval,
        diagnostics2=diag2,
    )
    elapsed = time.monotonic() - start

    r2_rep = metrics.region2.representation
    summ2 = diag2.summary()
    mods = np.array(metrics.surprise_modulators)

    return {
        "r2_lr": r2_lr,
        "r2_ltd": r2_ltd,
        "r2_burst": summ2["burst_rate"],
        "r2_selectivity": r2_rep.get("column_selectivity_mean", 0),
        "r2_ctx_disc": r2_rep.get("context_discrimination", 0),
        "r2_cross_cos": r2_rep.get("ff_cross_col_cosine", 0),
        "r2_ff_sparsity": r2_rep.get("ff_sparsity", 0),
        "r2_similarity": r2_rep.get("similarity_mean", 0),
        "mod_mean": float(mods.mean()),
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=20000)
    parser.add_argument("--log-interval", type=int, default=5000)
    args = parser.parse_args()

    tokens = prepare_tokens(args.tokens)

    encoder = CharbitEncoder(length=CHAR_LENGTH, width=CHAR_WIDTH, chars=CHARS)
    input_dim = CHAR_LENGTH * CHAR_WIDTH

    # Sweep grid: R2 learning_rate and ltd_rate
    lr_values = [0.01, 0.03, 0.05, 0.1]
    ltd_values = [0.05, 0.1, 0.2, 0.4]

    results = []
    total = len(lr_values) * len(ltd_values)

    for i, (lr, ltd) in enumerate(itertools.product(lr_values, ltd_values)):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] R2 lr={lr}, ltd={ltd}")
        print(f"{'='*60}")

        result = run_one(
            tokens, encoder, input_dim, lr, ltd, args.log_interval
        )
        results.append(result)

        print(
            f"  → burst={result['r2_burst']:.1%}"
            f" sel={result['r2_selectivity']:.3f}"
            f" ctx={result['r2_ctx_disc']:.3f}"
            f" cos={result['r2_cross_cos']:.3f}"
            f" spar={result['r2_ff_sparsity']:.3f}"
            f" ({result['elapsed']:.0f}s)"
        )

    # Summary table
    print(f"\n{'='*60}")
    print("SWEEP RESULTS")
    print(f"{'='*60}")
    header = (
        f"{'lr':>6s} {'ltd':>6s} │ "
        f"{'burst':>6s} {'select':>6s} {'ctx_d':>6s} "
        f"{'cosine':>6s} {'spars':>6s} {'sim':>6s}"
    )
    print(header)
    print("─" * len(header))

    # Sort by cross-col cosine (lower = better diversity)
    for r in sorted(results, key=lambda x: x["r2_cross_cos"]):
        print(
            f"{r['r2_lr']:6.3f} {r['r2_ltd']:6.3f} │ "
            f"{r['r2_burst']:5.1%} {r['r2_selectivity']:6.3f} "
            f"{r['r2_ctx_disc']:6.3f} "
            f"{r['r2_cross_cos']:6.3f} {r['r2_ff_sparsity']:6.3f} "
            f"{r['r2_similarity']:6.3f}"
        )

    # Best by each key metric
    print("\nBest by metric:")
    for metric, direction in [
        ("r2_cross_cos", "lowest"),
        ("r2_ff_sparsity", "highest"),
        ("r2_ctx_disc", "highest"),
        ("r2_selectivity", "lowest"),
    ]:
        rev = direction == "highest"
        best = sorted(results, key=lambda x: x[metric], reverse=rev)[0]
        print(
            f"  {metric} ({direction}): "
            f"lr={best['r2_lr']}, ltd={best['r2_ltd']} "
            f"→ {best[metric]:.3f}"
        )


if __name__ == "__main__":
    main()
