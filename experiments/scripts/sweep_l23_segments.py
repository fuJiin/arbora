#!/usr/bin/env python3
"""Sweep L2/3 lateral segment parameters.

Tests whether L2/3 segments benefit from independent tuning vs L4 segments.
Key parameter: l23_prediction_boost (separate from fb_boost which only
affects L4). Also tests segment count, synapse count, and activation threshold.

Usage:
  uv run experiments/scripts/sweep_l23_segments.py
  uv run experiments/scripts/sweep_l23_segments.py --tokens 20000
"""

import argparse
import string
import time

import numpy as np

import step.env  # noqa: F401
from step.cortex.sensory import SensoryRegion
from step.data import STORY_BOUNDARY, prepare_tokens
from step.encoders.charbit import CharbitEncoder
from step.probes.diagnostics import CortexDiagnostics
from step.probes.representation import RepresentationTracker

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1


def run_config(
    name: str,
    tokens: list[tuple[int, str]],
    encoder: CharbitEncoder,
    input_dim: int,
    log_interval: int,
    *,
    n_l23_segments: int = 4,
    l23_prediction_boost: float = 0.0,
    l23_n_synapses: int = 0,
):
    region = SensoryRegion(
        input_dim=input_dim,
        encoding_width=CHAR_WIDTH,
        n_columns=32,
        n_l4=4,
        n_l23=4,
        k_columns=4,
        n_l23_segments=n_l23_segments,
        l23_prediction_boost=l23_prediction_boost,
        seed=42,
    )

    # Override L2/3 synapse count if specified
    if l23_n_synapses:
        n23 = region.n_l23_total
        region.l23_seg_indices = np.zeros(
            (n23, n_l23_segments, l23_n_synapses), dtype=np.int32
        )
        region.l23_seg_perm = np.zeros((n23, n_l23_segments, l23_n_synapses))
        rng = np.random.default_rng(42)
        for i in range(n23):
            col = i // region.n_l23
            pool = region._l23_col_pools[col]
            for s in range(n_l23_segments):
                region.l23_seg_indices[i, s] = rng.choice(
                    pool, l23_n_synapses, replace=len(pool) < l23_n_synapses
                )

    diag = CortexDiagnostics(snapshot_interval=log_interval)
    rep_tracker = RepresentationTracker(n_columns=32, n_l4=4)

    start = time.monotonic()
    print(f"--- {name} ---")

    for t_step, (token_id, token_str) in enumerate(tokens):
        if token_id == STORY_BOUNDARY:
            region.reset_working_memory()
            continue

        encoding = encoder.encode(token_str)
        region.process(encoding)

        diag.step(t_step, region)
        rep_tracker.observe(token_id, region.active_columns, region.active_l4)

        if t_step > 0 and t_step % log_interval == 0:
            elapsed = time.monotonic() - start
            bc = diag._burst_count
            pc = diag._precise_count
            snap = diag.snapshots[-1] if diag.snapshots else None
            l23_conn = snap.l23_seg_connected_frac if snap else 0
            l23_pred = snap.n_predicted_l23 if snap else 0
            print(
                f"  t={t_step:,} "
                f"burst={bc / (bc + pc):.1%} "
                f"l23_conn={l23_conn:.1%} "
                f"l23_pred={l23_pred} "
                f"({elapsed:.1f}s)"
            )

    elapsed = time.monotonic() - start
    summ = diag.summary()
    rep = rep_tracker.summary(ff_weights=region.ff_weights)
    snap = diag.snapshots[-1] if diag.snapshots else None

    result = {
        "name": name,
        "time": elapsed,
        "burst_rate": summ["burst_rate"],
        "selectivity": rep["column_selectivity_mean"],
        "ctx_disc": rep["context_discrimination"],
        "cross_cos": rep.get("ff_cross_col_cosine", 0),
        # L2/3 segment health
        "l23_conn": snap.l23_seg_connected_frac if snap else 0,
        "l23_perm": snap.l23_seg_perm_mean if snap else 0,
        "l23_active": snap.n_active_l23_segments if snap else 0,
        "l23_pred": snap.n_predicted_l23 if snap else 0,
        # L4 segment health (should stay stable across L2/3 changes)
        "fb_conn": snap.fb_seg_connected_frac if snap else 0,
        "pred_sets": summ["unique_prediction_sets"],
    }

    print(
        f"  DONE: burst={result['burst_rate']:.1%} "
        f"sel={result['selectivity']:.3f} "
        f"ctx={result['ctx_disc']:.3f} "
        f"l23_conn={result['l23_conn']:.1%} "
        f"l23_pred={result['l23_pred']} "
        f"({elapsed:.1f}s)\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=2000)
    args = parser.parse_args()

    tokens = prepare_tokens(args.tokens)
    encoder = CharbitEncoder(length=CHAR_LENGTH, width=CHAR_WIDTH, chars=CHARS)
    input_dim = CHAR_LENGTH * CHAR_WIDTH

    configs = [
        # Baseline: L2/3 segments with default boost (= fb_boost = 0.4)
        ("baseline", {}),
        # No L2/3 segments
        ("no_l23_seg", {"n_l23_segments": 1}),  # 1 segment ~= minimal
        # L2/3 prediction boost (independent of L4 fb_boost)
        ("boost=0.2", {"l23_prediction_boost": 0.2}),
        ("boost=0.4", {"l23_prediction_boost": 0.4}),
        ("boost=0.6", {"l23_prediction_boost": 0.6}),
        ("boost=0.8", {"l23_prediction_boost": 0.8}),
        ("boost=1.0", {"l23_prediction_boost": 1.0}),
        # Segment count
        ("2seg", {"n_l23_segments": 2}),
        ("8seg", {"n_l23_segments": 8}),
        # Synapse count
        ("12syn", {"l23_n_synapses": 12}),
        ("16syn", {"l23_n_synapses": 16}),
        # Best boost + segment variations
        ("b0.8+2seg", {"l23_prediction_boost": 0.8, "n_l23_segments": 2}),
        ("b0.8+8seg", {"l23_prediction_boost": 0.8, "n_l23_segments": 8}),
    ]

    results = []
    for name, overrides in configs:
        result = run_config(
            name, tokens, encoder, input_dim, args.log_interval, **overrides
        )
        results.append(result)

    # Summary table
    print("\n" + "=" * 140)
    print(
        f"{'Config':<12} {'Time':>5} "
        f"{'Burst':>6} {'Select':>7} {'CtxDsc':>7} {'XCos':>6} "
        f"{'L23Con':>7} {'L23Prm':>7} {'L23Act':>7} {'L23Prd':>7} "
        f"{'FbCon':>6} {'PrdSet':>7}"
    )
    print("=" * 140)

    for r in results:
        print(
            f"{r['name']:<12} {r['time']:>4.0f}s "
            f"{r['burst_rate']:>5.1%} {r['selectivity']:>7.3f} "
            f"{r['ctx_disc']:>7.3f} {r['cross_cos']:>6.3f} "
            f"{r['l23_conn']:>6.1%} {r['l23_perm']:>7.4f} "
            f"{r['l23_active']:>7} {r['l23_pred']:>7} "
            f"{r['fb_conn']:>5.1%} {r['pred_sets']:>7}"
        )

    print("=" * 140)

    best_ctx = max(results, key=lambda r: r["ctx_disc"])
    best_sel = min(results, key=lambda r: r["selectivity"])
    best_burst = min(results, key=lambda r: r["burst_rate"])
    print(f"\nBest ctx disc:    {best_ctx['name']} ({best_ctx['ctx_disc']:.3f})")
    print(f"Best selectivity: {best_sel['name']} ({best_sel['selectivity']:.3f})")
    print(f"Best burst rate:  {best_burst['name']} ({best_burst['burst_rate']:.1%})")


if __name__ == "__main__":
    main()
