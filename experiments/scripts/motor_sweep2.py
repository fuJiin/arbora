#!/usr/bin/env python3
"""Focused sweep: k=1 with other param variations."""

import argparse
import contextlib
import io
import time

import step.env  # noqa: F401
from step.config import CortexConfig, _default_region2_config
from step.cortex.modulators import SurpriseTracker, ThalamicGate
from step.cortex.motor import MotorRegion
from step.cortex.sensory import SensoryRegion
from step.cortex.topology import Topology
from step.data import prepare_tokens_charlevel
from step.encoders.positional import PositionalCharEncoder


def run_one(tokens, encoder, cortex_cfg, motor_params):
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

    with contextlib.redirect_stdout(io.StringIO()):
        result = cortex.run(tokens, log_interval=999999)

    m1_metrics = result.per_region["M1"]
    s1_metrics = result.per_region["S1"]

    half = len(m1_metrics.motor_accuracies) // 2
    m_acc = m1_metrics.motor_accuracies
    m_acc_last = m_acc[half:] if half > 0 else m_acc
    acc = sum(m_acc_last) / max(len(m_acc_last), 1)

    m_conf = m1_metrics.motor_confidences
    m_conf_last = m_conf[half:] if half > 0 else m_conf
    sil = sum(1 for c in m_conf_last if c == 0.0) / max(len(m_conf_last), 1)

    m1_sel = m1_metrics.representation.get("column_selectivity_mean", 0)

    den = s1_metrics.dendritic_accuracies
    den_last = den[half:] if half > 0 else den
    s1_den = sum(den_last) / max(len(den_last), 1)

    ovl = s1_metrics.overlaps
    ovl_last = ovl[half:] if half > 0 else ovl
    s1_ovl = sum(ovl_last) / max(len(ovl_last), 1)

    return {
        "m1_acc": acc,
        "m1_sil": sil,
        "m1_sel": m1_sel,
        "s1_den": s1_den,
        "s1_ovl": s1_ovl,
        "n_spoke": len(m_acc),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=5000)
    args = parser.parse_args()

    tokens = prepare_tokens_charlevel(args.tokens)
    alphabet = sorted({ch for _, ch in tokens if _ != -1})
    encoder = PositionalCharEncoder("".join(alphabet), max_positions=8)
    cortex_cfg = CortexConfig(ltd_rate=0.05)

    # k=1 with variations
    configs = [
        ("k1 baseline",        {"k_columns": 1, "learning_rate": 0.15, "ltd_rate": 0.15, "voltage_decay": 0.5, "output_threshold": 0.3}),
        ("k1 lr=0.10",         {"k_columns": 1, "learning_rate": 0.10, "ltd_rate": 0.15, "voltage_decay": 0.5, "output_threshold": 0.3}),
        ("k1 lr=0.20",         {"k_columns": 1, "learning_rate": 0.20, "ltd_rate": 0.15, "voltage_decay": 0.5, "output_threshold": 0.3}),
        ("k1 lr=0.30",         {"k_columns": 1, "learning_rate": 0.30, "ltd_rate": 0.15, "voltage_decay": 0.5, "output_threshold": 0.3}),
        ("k1 ltd=0.10",        {"k_columns": 1, "learning_rate": 0.15, "ltd_rate": 0.10, "voltage_decay": 0.5, "output_threshold": 0.3}),
        ("k1 ltd=0.20",        {"k_columns": 1, "learning_rate": 0.15, "ltd_rate": 0.20, "voltage_decay": 0.5, "output_threshold": 0.3}),
        ("k1 ltd=0.30",        {"k_columns": 1, "learning_rate": 0.15, "ltd_rate": 0.30, "voltage_decay": 0.5, "output_threshold": 0.3}),
        ("k1 vd=0.3",          {"k_columns": 1, "learning_rate": 0.15, "ltd_rate": 0.15, "voltage_decay": 0.3, "output_threshold": 0.3}),
        ("k1 vd=0.7",          {"k_columns": 1, "learning_rate": 0.15, "ltd_rate": 0.15, "voltage_decay": 0.7, "output_threshold": 0.3}),
        ("k1 thresh=0.1",      {"k_columns": 1, "learning_rate": 0.15, "ltd_rate": 0.15, "voltage_decay": 0.5, "output_threshold": 0.1}),
        # Combos
        ("k1 lr=0.20 ltd=0.20", {"k_columns": 1, "learning_rate": 0.20, "ltd_rate": 0.20, "voltage_decay": 0.5, "output_threshold": 0.3}),
        ("k1 lr=0.20 vd=0.7",  {"k_columns": 1, "learning_rate": 0.20, "ltd_rate": 0.15, "voltage_decay": 0.7, "output_threshold": 0.3}),
        ("k2 baseline",        {"k_columns": 2, "learning_rate": 0.15, "ltd_rate": 0.15, "voltage_decay": 0.5, "output_threshold": 0.3}),
        ("k2 lr=0.20 ltd=0.20", {"k_columns": 2, "learning_rate": 0.20, "ltd_rate": 0.20, "voltage_decay": 0.5, "output_threshold": 0.3}),
    ]

    print(f"Running {len(configs)} configs on {len(tokens):,} tokens\n")
    print(
        f"{'config':<28s} {'M1acc':>6s} {'M1sil':>6s} {'M1sel':>6s} "
        f"{'S1den':>6s} {'S1ovl':>6s} {'spoke':>6s} {'time':>5s}"
    )
    print("-" * 80)

    results = []
    for label, cfg in configs:
        t0 = time.monotonic()
        r = run_one(tokens, encoder, cortex_cfg, cfg)
        elapsed = time.monotonic() - t0
        results.append((label, cfg, r))
        print(
            f"{label:<28s} {r['m1_acc']:6.1%} {r['m1_sil']:6.1%} {r['m1_sel']:6.3f} "
            f"{r['s1_den']:6.1%} {r['s1_ovl']:6.3f} {r['n_spoke']:6d} {elapsed:5.1f}s"
        )

    best = max(results, key=lambda x: x[2]["m1_acc"])
    print(f"\nBest M1 accuracy: {best[0]} -> {best[2]['m1_acc']:.1%}")


if __name__ == "__main__":
    main()
