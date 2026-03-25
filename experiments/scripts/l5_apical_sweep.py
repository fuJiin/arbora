#!/usr/bin/env python3
"""Compare L5 apical segments vs linear gain baseline.

Runs 50k sensory stage with:
1. Baseline: linear gain (default)
2. L5 apical segments on sensory regions (S1, S2, S3)
"""

import time
from dataclasses import replace

import numpy as np

import step.env  # noqa: F401
from step.config import (
    _default_motor_config,
    _default_pfc_config,
    _default_premotor_config,
    _default_region2_config,
    _default_region3_config,
    _default_s1_config,
    make_motor_region,
    make_pfc_region,
    make_premotor_region,
    make_sensory_region,
)
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.circuit import Circuit, ConnectionRole
from step.cortex.modulators import SurpriseTracker, ThalamicGate
from step.data import inject_eom_tokens, prepare_tokens_charlevel
from step.encoders.positional import PositionalCharEncoder

N_TOKENS = 50_000
LOG_INTERVAL = 10_000


def load_data():
    all_tokens = prepare_tokens_charlevel(1_000_000, dataset="babylm")
    alphabet = sorted({ch for _, ch in all_tokens if _ >= 0})
    encoder = PositionalCharEncoder("".join(alphabet), max_positions=8)
    tokens = all_tokens[:N_TOKENS]
    tokens = inject_eom_tokens(tokens, segment_length=200)
    return tokens, encoder


def build_circuit(encoder, use_l5_apical=False):
    s1_cfg = _default_s1_config()
    if use_l5_apical:
        s1_cfg = replace(s1_cfg, use_l5_apical_segments=True)
    s1 = make_sensory_region(s1_cfg, encoder.input_dim, encoder.encoding_width)

    r2_cfg = _default_region2_config()
    if use_l5_apical:
        r2_cfg = replace(r2_cfg, use_l5_apical_segments=True)
    s2 = make_sensory_region(r2_cfg, s1.n_l23_total * 4, seed=123)

    r3_cfg = _default_region3_config()
    if use_l5_apical:
        r3_cfg = replace(r3_cfg, use_l5_apical_segments=True)
    s3 = make_sensory_region(r3_cfg, s2.n_l23_total * 8, seed=789)

    m2_cfg = _default_premotor_config()
    m2_n_l23 = m2_cfg.n_columns * m2_cfg.n_l23

    m1_cfg = _default_motor_config()
    output_vocab = [ord(ch) for ch in encoder._char_to_idx]
    m1 = make_motor_region(m1_cfg, m2_n_l23, seed=456)
    m1._output_vocab = np.array(output_vocab, dtype=np.int64)
    m1.n_output_tokens = len(output_vocab)
    n_l5 = m1.n_l5_total
    m1.output_weights = m1._rng.uniform(0, 0.01, size=(n_l5, len(output_vocab)))
    m1.output_mask = (m1._rng.random((n_l5, len(output_vocab))) < 0.5).astype(
        np.float64
    )
    m1.output_weights *= m1.output_mask
    m1._output_eligibility = np.zeros((n_l5, len(output_vocab)))

    cortex = Circuit(
        encoder,
        enable_timeline=False,
        diagnostics_interval=LOG_INTERVAL,
    )

    cortex.add_region("S1", s1, entry=True)
    cortex.add_region("S2", s2)
    cortex.add_region("S3", s3)
    bg = BasalGanglia(context_dim=s1.n_columns + 1, learning_rate=0.05, seed=789)
    cortex.add_region("M1", m1, basal_ganglia=bg)

    pfc_cfg = _default_pfc_config()
    pfc = make_pfc_region(
        pfc_cfg,
        s2.n_l23_total + s3.n_l23_total,
        seed=999,
        source_dims=[s2.n_l23_total, s3.n_l23_total],
    )
    cortex.add_region("PFC", pfc)

    m2 = make_premotor_region(
        m2_cfg,
        s2.n_l23_total + pfc.n_l23_total,
        seed=321,
        source_dims=[s2.n_l23_total, pfc.n_l23_total],
    )
    cortex.add_region("M2", m2)

    # Feedforward
    cortex.connect(
        "S1",
        "S2",
        ConnectionRole.FEEDFORWARD,
        buffer_depth=4,
        burst_gate=True,
        surprise_tracker=SurpriseTracker(),
    )
    cortex.connect(
        "S2",
        "S3",
        ConnectionRole.FEEDFORWARD,
        buffer_depth=8,
        burst_gate=True,
        surprise_tracker=SurpriseTracker(),
    )
    cortex.connect("S2", "PFC", ConnectionRole.FEEDFORWARD)
    cortex.connect("S3", "PFC", ConnectionRole.FEEDFORWARD)
    cortex.connect("S2", "M2", ConnectionRole.FEEDFORWARD)
    cortex.connect("PFC", "M2", ConnectionRole.FEEDFORWARD)
    cortex.connect("M2", "M1", ConnectionRole.FEEDFORWARD)

    # Apical
    cortex.connect("S2", "S1", ConnectionRole.APICAL, thalamic_gate=ThalamicGate())
    cortex.connect("S3", "S2", ConnectionRole.APICAL, thalamic_gate=ThalamicGate())
    cortex.connect("M1", "M2", ConnectionRole.APICAL, thalamic_gate=ThalamicGate())
    cortex.connect("M2", "PFC", ConnectionRole.APICAL, thalamic_gate=ThalamicGate())
    cortex.connect(
        "S1",
        "M1",
        ConnectionRole.APICAL,
        thalamic_gate=ThalamicGate(),
        surprise_tracker=SurpriseTracker(),
    )
    cortex.connect("M1", "S1", ConnectionRole.APICAL, thalamic_gate=ThalamicGate())

    return cortex


def run_config(name, tokens, encoder, use_l5_apical):
    print(f"\n{'=' * 60}")
    print(f"Config: {name} (l5_apical_segments={use_l5_apical})")
    print(f"{'=' * 60}")

    cortex = build_circuit(encoder, use_l5_apical=use_l5_apical)
    cortex.finalize()

    t0 = time.time()
    result = cortex.run(tokens, log_interval=LOG_INTERVAL)
    elapsed = time.time() - t0

    # Extract summary metrics
    summary = {}
    for rname, metrics in result.per_region.items():
        rep = metrics.representation
        if rep:
            summary[rname] = {
                "ctx_disc": rep.get("context_discrimination", 0),
                "selectivity": rep.get("column_selectivity_mean", 0),
            }

    print(f"\n{name} completed in {elapsed:.1f}s")
    print(f"{'Region':<8} {'ctx_disc':>10} {'selectivity':>12}")
    print("-" * 32)
    for rname in ["S1", "S2", "S3", "M1", "PFC", "M2"]:
        if rname in summary:
            s = summary[rname]
            print(f"{rname:<8} {s['ctx_disc']:>10.4f} {s['selectivity']:>12.4f}")

    return summary, elapsed


def main():
    print("Loading data...")
    tokens, encoder = load_data()
    print(f"Loaded {len(tokens)} tokens, {len(encoder._char_to_idx)} chars")

    results = {}

    # Baseline: linear gain (default)
    results["baseline"], t1 = run_config(
        "baseline", tokens, encoder, use_l5_apical=False
    )

    # L5 apical segments on sensory regions
    results["l5_apical"], t2 = run_config(
        "l5_apical_segments", tokens, encoder, use_l5_apical=True
    )

    # Comparison
    print(f"\n{'=' * 60}")
    print("COMPARISON: L5 Apical Segments vs Baseline")
    print(f"{'=' * 60}")
    print(f"{'Region':<8} {'Baseline':>10} {'L5 Apical':>10} {'Delta':>8}")
    print("-" * 38)
    for rname in ["S1", "S2", "S3", "M1", "PFC", "M2"]:
        b = results["baseline"].get(rname, {}).get("ctx_disc", 0)
        a = results["l5_apical"].get(rname, {}).get("ctx_disc", 0)
        delta = a - b
        sign = "+" if delta > 0 else ""
        print(f"{rname:<8} {b:>10.4f} {a:>10.4f} {sign}{delta:>7.4f}")

    print(f"\nTiming: baseline={t1:.0f}s, l5_apical={t2:.0f}s")


if __name__ == "__main__":
    main()
