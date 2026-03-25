#!/usr/bin/env python3
"""Compare synapse_decay values to diagnose forgetting.

Tests whether passive weight decay is preventing long-term consolidation
across dialogue boundaries. Runs S1=128/k=8 with full hierarchy on
TinyDialogues, varying only synapse_decay.

Usage:
    uv run experiments/scripts/decay_comparison.py
    uv run experiments/scripts/decay_comparison.py --tokens 50000
"""

import argparse
import time

import numpy as np

import step.env  # noqa: F401
from step.config import (
    CortexConfig,
    _default_motor_config,
    _default_region2_config,
    make_motor_region,
    make_sensory_region,
)
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.circuit import Circuit, ConnectionRole
from step.cortex.modulators import SurpriseTracker, ThalamicGate
from step.data import prepare_tokens_tinydialogues
from step.encoders.positional import PositionalCharEncoder


def run_decay(synapse_decay, tokens, encoder):
    """Run one config and return per-dialogue BPC breakdown."""
    s1_cfg = CortexConfig(
        n_columns=128,
        k_columns=8,
        ltd_rate=0.05,
        synapse_decay=synapse_decay,
    )
    region1 = make_sensory_region(
        s1_cfg,
        encoder.input_dim,
        encoder.encoding_width,
    )

    r2_cfg = _default_region2_config()
    r2_input_dim = region1.n_l23_total * 4
    region2 = make_sensory_region(r2_cfg, r2_input_dim, seed=123)

    motor = make_motor_region(
        _default_motor_config(),
        region1.n_l23_total,
        seed=456,
    )

    bg = BasalGanglia(
        context_dim=region1.n_columns + 1,
        learning_rate=0.1,
        seed=789,
    )

    cortex = Circuit(encoder, diagnostics_interval=10000)
    cortex.add_region("S1", region1, entry=True)
    cortex.add_region("S2", region2)
    cortex.connect(
        "S1",
        "S2",
        ConnectionRole.FEEDFORWARD,
        buffer_depth=4,
        burst_gate=True,
        surprise_tracker=SurpriseTracker(),
    )
    cortex.connect("S2", "S1", ConnectionRole.APICAL, thalamic_gate=ThalamicGate())
    cortex.add_region("M1", motor, basal_ganglia=bg)
    cortex.connect(
        "S1", "M1", ConnectionRole.FEEDFORWARD, surprise_tracker=SurpriseTracker()
    )
    if motor.n_l23_total == region2.n_l23_total:
        cortex.connect("M1", "S1", ConnectionRole.APICAL, thalamic_gate=ThalamicGate())

    t0 = time.time()
    result = cortex.run(tokens, log_interval=100000)
    elapsed = time.time() - t0

    s1 = result.per_region["S1"]
    m1 = result.per_region["M1"]

    # M1 accuracy (last 20%)
    accs = m1.motor_accuracies
    tail = accs[int(len(accs) * 0.8) :] if accs else []
    m1_acc = sum(tail) / len(tail) if tail else 0.0

    return {
        "synapse_decay": synapse_decay,
        "bpc": s1.bpc,
        "bpc_recent": s1.bpc_recent,
        "m1_acc": m1_acc,
        "burst": s1.representation.get("burst_rate", 0),
        "elapsed": elapsed,
        "dialogue_bpcs": s1.bpc_per_dialogue,
        "boundary_bpcs": s1.bpc_boundary,
        "steady_bpcs": s1.bpc_steady,
    }


def print_results(results):
    """Print comparison table and per-dialogue trends."""
    print(f"\n{'=' * 90}")
    print(
        f"{'decay':>10} | {'BPC':>5} {'recent':>6} {'M1%':>5} "
        f"| {'bdry_mean':>9} {'stdy_mean':>9} {'gap':>5} "
        f"| {'stdy_trend':>10} | {'time':>5}"
    )
    print("-" * 90)

    for r in results:
        bdry = r["boundary_bpcs"]
        stdy = r["steady_bpcs"]
        bdry_mean = np.mean(bdry) if bdry else 0
        stdy_mean = np.mean(stdy) if stdy else 0

        # Trend: compare first half vs second half of steady BPCs
        if len(stdy) >= 6:
            half = len(stdy) // 2
            first = np.mean(stdy[:half])
            second = np.mean(stdy[half:])
            trend = second - first
            trend_str = f"{trend:+.3f}"
        else:
            trend_str = "n/a"

        print(
            f"{r['synapse_decay']:>10} "
            f"| {r['bpc']:>5.2f} {r['bpc_recent']:>6.2f} {r['m1_acc']:>5.1%} "
            f"| {bdry_mean:>9.2f} {stdy_mean:>9.2f} "
            f"{bdry_mean - stdy_mean:>5.2f} "
            f"| {trend_str:>10} "
            f"| {r['elapsed']:>5.0f}s"
        )

    print("=" * 90)

    # Per-dialogue steady BPC trajectory (first and last 5)
    for r in results:
        stdy = r["steady_bpcs"]
        if len(stdy) >= 10:
            first5 = [f"{v:.2f}" for v in stdy[:5]]
            last5 = [f"{v:.2f}" for v in stdy[-5:]]
            print(
                f"\n  decay={r['synapse_decay']}: first5={first5}  ...  last5={last5}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Compare synapse_decay values for forgetting diagnosis",
    )
    parser.add_argument("--tokens", type=int, default=30000)
    args = parser.parse_args()

    tokens = prepare_tokens_tinydialogues(args.tokens, speak_window=10)
    alphabet = sorted({ch for _, ch in tokens if _ >= 0})
    encoder = PositionalCharEncoder("".join(alphabet), max_positions=8)
    print(f"Vocab: {len(alphabet)} chars, {len(tokens)} tokens\n")

    decay_values = [0.999, 0.9999, 1.0]
    results = []

    for decay in decay_values:
        print(f"--- synapse_decay={decay} ---")
        r = run_decay(decay, tokens, encoder)
        results.append(r)
        print(
            f"  BPC={r['bpc']:.2f} recent={r['bpc_recent']:.2f} "
            f"M1={r['m1_acc']:.1%} ({r['elapsed']:.0f}s)\n"
        )

    print_results(results)


if __name__ == "__main__":
    main()
