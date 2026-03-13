#!/usr/bin/env python3
"""Evaluate two-region sensory hierarchy on BabyLM.

Runs S1 (sensory) → S2 (secondary sensory) with
surprise-modulated learning. Reports per-region representation
metrics and hierarchy-specific diagnostics.

Usage:
  uv run experiments/scripts/eval_hierarchy.py --tokens 5000
  uv run experiments/scripts/eval_hierarchy.py --tokens 20000 --dataset babylm
"""

import argparse
import string
import time

import numpy as np

import step.env  # noqa: F401
from step.cortex.sensory import SensoryRegion
from step.cortex.surprise import SurpriseTracker
from step.data import prepare_tokens
from step.encoders.charbit import CharbitEncoder
from step.probes.diagnostics import CortexDiagnostics
from step.runner import run_cortex, run_hierarchy

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument(
        "--dataset", choices=["tinystories", "babylm"], default="babylm"
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Only run single-region baseline (skip hierarchy)",
    )
    args = parser.parse_args()

    tokens = prepare_tokens(args.tokens, dataset=args.dataset)

    encoder = CharbitEncoder(length=CHAR_LENGTH, width=CHAR_WIDTH, chars=CHARS)
    input_dim = CHAR_LENGTH * CHAR_WIDTH
    encoding_width = CHAR_WIDTH

    # --- Single region baseline ---
    print("=" * 60)
    print("Single-region baseline")
    print("=" * 60)

    region_solo = SensoryRegion(
        input_dim=input_dim,
        encoding_width=encoding_width,
        n_columns=32,
        n_l4=4,
        n_l23=4,
        k_columns=4,
        seed=42,
    )
    diag_solo = CortexDiagnostics(snapshot_interval=args.log_interval)

    start_solo = time.monotonic()
    metrics_solo = run_cortex(
        region_solo,
        encoder,
        tokens,
        log_interval=args.log_interval,
        diagnostics=diag_solo,
    )
    elapsed_solo = time.monotonic() - start_solo

    if args.single_only:
        return

    # --- Two-region hierarchy ---
    print("\n" + "=" * 60)
    print("Two-region hierarchy")
    print("=" * 60)

    region1 = SensoryRegion(
        input_dim=input_dim,
        encoding_width=encoding_width,
        n_columns=32,
        n_l4=4,
        n_l23=4,
        k_columns=4,
        seed=42,
    )
    region2 = SensoryRegion(
        input_dim=region1.n_l23_total,
        encoding_width=0,  # sliding window — no positional structure in L2/3 output
        n_columns=16,
        n_l4=4,
        n_l23=4,
        k_columns=2,
        voltage_decay=0.8,
        eligibility_decay=0.98,
        synapse_decay=0.9999,
        learning_rate=0.01,
        ltd_rate=0.4,
        seed=123,
    )
    surprise = SurpriseTracker()
    diag1 = CortexDiagnostics(snapshot_interval=args.log_interval)
    diag2 = CortexDiagnostics(snapshot_interval=args.log_interval)

    start_hier = time.monotonic()
    metrics_hier = run_hierarchy(
        region1,
        region2,
        encoder,
        tokens,
        surprise_tracker=surprise,
        log_interval=args.log_interval,
        diagnostics1=diag1,
        diagnostics2=diag2,
    )
    elapsed_hier = time.monotonic() - start_hier

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)

    print(
        f"\nRuntime: single={elapsed_solo:.1f}s  hierarchy={elapsed_hier:.1f}s  "
        f"ratio={elapsed_hier / max(elapsed_solo, 0.01):.2f}x"
    )

    # Region 1 decoder accuracy
    if metrics_solo.synaptic_accuracies:
        tail = metrics_solo.synaptic_accuracies[-100:]
        print(f"\nSingle region syn accuracy (last 100): {sum(tail) / len(tail):.4f}")
    if metrics_hier.region1.synaptic_accuracies:
        tail = metrics_hier.region1.synaptic_accuracies[-100:]
        print(f"Hierarchy S1 syn accuracy (last 100):  {sum(tail) / len(tail):.4f}")

    # Surprise modulator distribution
    mods = metrics_hier.surprise_modulators
    if mods:
        mods_arr = np.array(mods)
        print(
            f"\nSurprise modulator: "
            f"mean={mods_arr.mean():.3f} "
            f"std={mods_arr.std():.3f} "
            f"min={mods_arr.min():.3f} "
            f"max={mods_arr.max():.3f}"
        )

    # S2 representation summary
    r2_rep = metrics_hier.region2.representation
    if r2_rep:
        print("\nS2 representation:")
        print(
            f"  selectivity={r2_rep.get('column_selectivity_mean', 0):.3f} "
            f"similarity={r2_rep.get('similarity_mean', 0):.3f} "
            f"ctx_disc={r2_rep.get('context_discrimination', 0):.3f}"
        )

    # Burst rates
    summ1 = diag1.summary()
    summ2 = diag2.summary()
    print(f"\nBurst rates: S1={summ1['burst_rate']:.1%}  S2={summ2['burst_rate']:.1%}")


if __name__ == "__main__":
    main()
