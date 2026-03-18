#!/usr/bin/env python3
"""Sweep motor cortex parameters, one axis at a time.

Usage:
    uv run experiments/scripts/motor_sweep.py --tokens 5000
"""

import argparse
import time

import step.env  # noqa: F401
from step.config import CortexConfig, _default_region2_config
from step.cortex.modulators import SurpriseTracker, ThalamicGate
from step.cortex.motor import MotorRegion
from step.cortex.sensory import SensoryRegion
from step.cortex.topology import Topology
from step.data import prepare_tokens_charlevel
from step.encoders.positional import PositionalCharEncoder

# Baseline motor params
BASELINE = {
    "k_columns": 4,
    "learning_rate": 0.15,
    "ltd_rate": 0.15,
    "voltage_decay": 0.5,
    "output_threshold": 0.3,
}

# Sweep axes: param -> values to try
SWEEPS = {
    "k_columns": [1, 2, 4],
    "learning_rate": [0.10, 0.20, 0.30],
    "ltd_rate": [0.10, 0.20, 0.30],
    "voltage_decay": [0.3, 0.5, 0.7],
    "output_threshold": [0.1, 0.2, 0.3],
}


def run_one(tokens, encoder, cortex_cfg, motor_params):
    """Run hierarchy with given motor params, return summary dict."""
    r1 = SensoryRegion(
        input_dim=encoder.input_dim,
        encoding_width=encoder.encoding_width,
        n_columns=cortex_cfg.n_columns,
        n_l4=cortex_cfg.n_l4,
        n_l23=cortex_cfg.n_l23,
        k_columns=cortex_cfg.k_columns,
        voltage_decay=cortex_cfg.voltage_decay,
        learning_rate=cortex_cfg.learning_rate,
        ltd_rate=cortex_cfg.ltd_rate,
        seed=0,
    )
    r2_cfg = _default_region2_config()
    r2 = SensoryRegion(
        input_dim=r1.n_l23_total * 4,
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
    m1 = MotorRegion(
        input_dim=r1.n_l23_total,
        n_columns=32,
        n_l4=4,
        n_l23=4,
        k_columns=motor_params["k_columns"],
        voltage_decay=motor_params["voltage_decay"],
        learning_rate=motor_params["learning_rate"],
        ltd_rate=motor_params["ltd_rate"],
        output_threshold=motor_params["output_threshold"],
        seed=456,
    )

    cortex = Topology(encoder, diagnostics_interval=10000)
    cortex.add_region("S1", r1, entry=True)
    cortex.add_region("S2", r2)
    cortex.add_region("M1", m1)
    cortex.connect("S1", "S2", "feedforward", buffer_depth=4, burst_gate=True)
    cortex.connect("S1", "S2", "surprise", surprise_tracker=SurpriseTracker())
    cortex.connect("S2", "S1", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("S1", "M1", "feedforward")
    cortex.connect("S1", "M1", "surprise", surprise_tracker=SurpriseTracker())
    cortex.connect("M1", "S1", "apical", thalamic_gate=ThalamicGate())

    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        result = cortex.run(tokens, log_interval=999999)

    m1_metrics = result.per_region["M1"]
    s1_metrics = result.per_region["S1"]

    # M1 accuracy (last half only — skip warmup)
    half = len(m1_metrics.motor_accuracies) // 2
    m_acc_all = m1_metrics.motor_accuracies
    m_acc_last = m_acc_all[half:] if half > 0 else m_acc_all
    acc = sum(m_acc_last) / max(len(m_acc_last), 1)

    # Silence rate (last half)
    m_conf_all = m1_metrics.motor_confidences
    m_conf_last = m_conf_all[half:] if half > 0 else m_conf_all
    sil = sum(1 for c in m_conf_last if c == 0.0) / max(len(m_conf_last), 1)

    # M1 selectivity
    m1_sel = m1_metrics.representation.get("column_selectivity_mean", 0)

    # S1 dendritic accuracy (last half)
    den_all = s1_metrics.dendritic_accuracies
    den_last = den_all[half:] if half > 0 else den_all
    s1_den = sum(den_last) / max(len(den_last), 1)

    # S1 overlap (last half)
    ovl_all = s1_metrics.overlaps
    ovl_last = ovl_all[half:] if half > 0 else ovl_all
    s1_ovl = sum(ovl_last) / max(len(ovl_last), 1)

    # N times M1 spoke
    n_spoke = len(m_acc_all)

    return {
        "m1_acc": acc,
        "m1_sil": sil,
        "m1_sel": m1_sel,
        "s1_den": s1_den,
        "s1_ovl": s1_ovl,
        "n_spoke": n_spoke,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=5000)
    args = parser.parse_args()

    tokens = prepare_tokens_charlevel(args.tokens)
    alphabet = sorted({ch for _, ch in tokens if _ != -1})
    encoder = PositionalCharEncoder("".join(alphabet), max_positions=8)
    cortex_cfg = CortexConfig(ltd_rate=0.05)

    # Build all configs: baseline + one-axis sweeps
    configs = []
    for param, values in SWEEPS.items():
        for val in values:
            cfg = dict(BASELINE)
            cfg[param] = val
            # Deduplicate: skip if identical to another config
            label = f"{param}={val}"
            if cfg == BASELINE:
                label += " (baseline)"
            configs.append((label, cfg))

    # Deduplicate
    seen = set()
    unique_configs = []
    for label, cfg in configs:
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            unique_configs.append((label, cfg))

    print(f"Running {len(unique_configs)} configs on {len(tokens):,} tokens\n")

    # Header
    print(
        f"{'config':<28s} {'M1acc':>6s} {'M1sil':>6s} {'M1sel':>6s} "
        f"{'S1den':>6s} {'S1ovl':>6s} {'spoke':>6s} {'time':>5s}"
    )
    print("-" * 80)

    results = []
    for label, cfg in unique_configs:
        t0 = time.monotonic()
        r = run_one(tokens, encoder, cortex_cfg, cfg)
        elapsed = time.monotonic() - t0
        results.append((label, cfg, r))
        print(
            f"{label:<28s} {r['m1_acc']:6.1%} {r['m1_sil']:6.1%} {r['m1_sel']:6.3f} "
            f"{r['s1_den']:6.1%} {r['s1_ovl']:6.3f} {r['n_spoke']:6d} {elapsed:5.1f}s"
        )

    # Best config by M1 accuracy
    best = max(results, key=lambda x: x[2]["m1_acc"])
    print(f"\nBest M1 accuracy: {best[0]} -> {best[2]['m1_acc']:.1%}")


if __name__ == "__main__":
    main()
