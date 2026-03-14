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
from step.config import CortexConfig, _default_motor_config, _default_region2_config
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.modulators import SurpriseTracker, ThalamicGate
from step.cortex.motor import MotorRegion
from step.cortex.sensory import SensoryRegion
from step.cortex.topology import Topology
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
    region1 = SensoryRegion(
        input_dim=encoder.input_dim,
        n_columns=s1_cfg.n_columns,
        n_l4=s1_cfg.n_l4,
        n_l23=s1_cfg.n_l23,
        k_columns=s1_cfg.k_columns,
        voltage_decay=s1_cfg.voltage_decay,
        eligibility_decay=s1_cfg.eligibility_decay,
        synapse_decay=s1_cfg.synapse_decay,
        learning_rate=s1_cfg.learning_rate,
        ltd_rate=s1_cfg.ltd_rate,
        encoding_width=encoder.encoding_width,
        burst_learning_scale=s1_cfg.burst_learning_scale,
        n_fb_segments=s1_cfg.n_fb_segments,
        n_lat_segments=s1_cfg.n_lat_segments,
        n_synapses_per_segment=s1_cfg.n_synapses_per_segment,
        perm_threshold=s1_cfg.perm_threshold,
        perm_init=s1_cfg.perm_init,
        perm_increment=s1_cfg.perm_increment,
        perm_decrement=s1_cfg.perm_decrement,
        seg_activation_threshold=s1_cfg.seg_activation_threshold,
        prediction_gain=s1_cfg.prediction_gain,
        n_apical_segments=s1_cfg.n_apical_segments,
        seed=s1_cfg.seed,
    )

    r2_cfg = _default_region2_config()
    r2_input_dim = region1.n_l23_total * 4
    region2 = SensoryRegion(
        input_dim=r2_input_dim,
        encoding_width=0,
        n_columns=r2_cfg.n_columns,
        n_l4=r2_cfg.n_l4,
        n_l23=r2_cfg.n_l23,
        k_columns=r2_cfg.k_columns,
        voltage_decay=r2_cfg.voltage_decay,
        eligibility_decay=r2_cfg.eligibility_decay,
        synapse_decay=r2_cfg.synapse_decay,
        learning_rate=r2_cfg.learning_rate,
        ltd_rate=r2_cfg.ltd_rate,
        seed=123,
    )

    m1_base = _default_motor_config()
    motor = MotorRegion(
        input_dim=region1.n_l23_total,
        n_columns=m1_base.n_columns,
        n_l4=m1_base.n_l4,
        n_l23=m1_base.n_l23,
        k_columns=1,
        voltage_decay=m1_base.voltage_decay,
        eligibility_decay=m1_base.eligibility_decay,
        synapse_decay=m1_base.synapse_decay,
        learning_rate=m1_base.learning_rate,
        ltd_rate=m1_base.ltd_rate,
        seed=456,
    )

    bg = BasalGanglia(
        context_dim=region1.n_columns + 1,
        learning_rate=0.1,
        seed=789,
    )

    cortex = Topology(encoder, diagnostics_interval=10000)
    cortex.add_region("S1", region1, entry=True)
    cortex.add_region("S2", region2)
    cortex.connect("S1", "S2", "feedforward", buffer_depth=4, burst_gate=True)
    cortex.connect("S1", "S2", "surprise", surprise_tracker=SurpriseTracker())
    cortex.connect("S2", "S1", "apical", thalamic_gate=ThalamicGate())
    cortex.add_region("M1", motor, basal_ganglia=bg)
    cortex.connect("S1", "M1", "feedforward")
    cortex.connect("S1", "M1", "surprise", surprise_tracker=SurpriseTracker())
    if motor.n_l23_total == region2.n_l23_total:
        cortex.connect("M1", "S1", "apical", thalamic_gate=ThalamicGate())

    t0 = time.time()
    result = cortex.run(tokens, log_interval=100000)
    elapsed = time.time() - t0

    s1 = result.per_region["S1"]
    m1 = result.per_region["M1"]

    # M1 accuracy (last 20%)
    accs = m1.motor_accuracies
    tail = accs[int(len(accs) * 0.8):] if accs else []
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
                f"\n  decay={r['synapse_decay']}: "
                f"first5={first5}  ...  last5={last5}"
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
