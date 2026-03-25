#!/usr/bin/env python3
"""Compare L5 apical segments vs linear gain baseline.

Runs 50k sensory stage with:
1. Baseline: linear gain (default)
2. L5 apical segments on sensory regions (S1, S2, S3)
"""

import time

import step.env  # noqa: F401
from step.cortex.canonical import build_canonical_circuit
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
    apical_override = {"use_l5_apical_segments": True} if use_l5_apical else None
    return build_canonical_circuit(
        encoder,
        log_interval=LOG_INTERVAL,
        timeline_interval=0,
        s1_overrides=apical_override,
        s2_overrides=apical_override,
        s3_overrides=apical_override,
        finalize=False,
    )


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
