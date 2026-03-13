#!/usr/bin/env python3
"""Interactive cortex diagnostics dashboard served on port 80.

Runs the cortex PoC, captures per-step timeline data, and serves
interactive Plotly charts showing column activation dynamics, ff_weight
divergence, voltage/excitability balance, and column drive distributions.

Usage: uv run experiments/scripts/cortex_dashboard.py [--tokens N] [--port 80]
"""

import argparse
import string
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from tempfile import TemporaryDirectory

import step.env  # noqa: F401
from step.config import CortexConfig
from step.cortex.sensory import SensoryRegion
from step.cortex.surprise import SurpriseTracker
from step.data import prepare_tokens
from step.encoders.charbit import CharbitEncoder
from step.probes.diagnostics import CortexDiagnostics
from step.probes.timeline import Timeline
from step.runner import run_cortex, run_hierarchy
from step.viz import (
    build_burst_rate_over_time,
    build_column_activation_heatmap,
    build_column_drive_histogram,
    build_column_entropy_over_time,
    build_column_selectivity_bar,
    build_dashboard_html,
    build_dual_burst_rate,
    build_ff_weight_divergence,
    build_hierarchy_summary_cards,
    build_representation_summary_cards,
    build_surprise_modulator_over_time,
    build_voltage_excitability,
)

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1


def run_with_timeline(tokens, region, encoder, log_interval):
    """Run cortex and capture per-step timeline + representation tracking."""
    timeline = Timeline()
    diag = CortexDiagnostics(snapshot_interval=log_interval)

    # Patch: capture timeline after each process call
    original_process = region.process

    def instrumented_process(encoding):
        result = original_process(encoding)
        timeline.capture(len(timeline.frames), region, region.last_column_drive)
        return result

    region.process = instrumented_process

    metrics = run_cortex(
        region, encoder, tokens, log_interval=log_interval, diagnostics=diag
    )

    # Backfill representation tracker from metrics
    # (run_cortex already tracks this, but we need it for dashboard)
    # Re-run observe from token data — cheaper than double-patching
    rep_region = region  # use final state for ff_convergence
    diag.print_report()

    # Restore
    region.process = original_process
    return metrics, timeline, diag, rep_region


def main():
    parser = argparse.ArgumentParser(description="Cortex diagnostics dashboard")
    parser.add_argument("--tokens", type=int, default=1000)
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--save-only", action="store_true", help="Save HTML without serving"
    )
    parser.add_argument(
        "--hierarchy", action="store_true", help="Run two-region hierarchy"
    )
    args = parser.parse_args()

    # Run the PoC with timeline capture
    tokens = prepare_tokens(args.tokens)

    cortex_cfg = CortexConfig()
    charbit = CharbitEncoder(length=CHAR_LENGTH, width=CHAR_WIDTH, chars=CHARS)
    input_dim = CHAR_LENGTH * CHAR_WIDTH

    if args.hierarchy:
        html = _run_hierarchy_dashboard(tokens, cortex_cfg, charbit, input_dim, args)
    else:
        html = _run_single_dashboard(tokens, cortex_cfg, charbit, input_dim, args)

    _serve_or_save(html, args)


def _make_region(cortex_cfg, input_dim):
    return SensoryRegion(
        input_dim=input_dim,
        n_columns=cortex_cfg.n_columns,
        n_l4=cortex_cfg.n_l4,
        n_l23=cortex_cfg.n_l23,
        k_columns=cortex_cfg.k_columns,
        voltage_decay=cortex_cfg.voltage_decay,
        eligibility_decay=cortex_cfg.eligibility_decay,
        synapse_decay=cortex_cfg.synapse_decay,
        learning_rate=cortex_cfg.learning_rate,
        max_excitability=cortex_cfg.max_excitability,
        fb_boost=cortex_cfg.fb_boost,
        ltd_rate=cortex_cfg.ltd_rate,
        encoding_width=CHAR_WIDTH,
        burst_learning_scale=cortex_cfg.burst_learning_scale,
        n_fb_segments=cortex_cfg.n_fb_segments,
        n_lat_segments=cortex_cfg.n_lat_segments,
        n_synapses_per_segment=cortex_cfg.n_synapses_per_segment,
        perm_threshold=cortex_cfg.perm_threshold,
        perm_init=cortex_cfg.perm_init,
        perm_increment=cortex_cfg.perm_increment,
        perm_decrement=cortex_cfg.perm_decrement,
        seg_activation_threshold=cortex_cfg.seg_activation_threshold,
        prediction_gain=cortex_cfg.prediction_gain,
        seed=cortex_cfg.seed,
    )


def _run_single_dashboard(tokens, cortex_cfg, charbit, input_dim, args) -> str:
    region = _make_region(cortex_cfg, input_dim)

    print(f"\nRunning cortex on {len(tokens):,} tokens...")
    metrics, timeline, diag, _rep_region = run_with_timeline(
        tokens, region, charbit, args.log_interval
    )

    print(f"\nCaptured {len(timeline.frames)} timeline frames")
    print("Building dashboard...")

    rep_summary = metrics.representation
    summ = diag.summary()
    burst_rate = summ["burst_rate"]

    cards_html = build_representation_summary_cards(rep_summary, burst_rate)

    n_cols = cortex_cfg.n_columns
    figures = [
        ("Surprise Rate", build_burst_rate_over_time(timeline)),
        ("Column Selectivity", build_column_selectivity_bar(rep_summary)),
        ("Column Usage", build_column_entropy_over_time(timeline, n_cols)),
        ("Column Activation", build_column_activation_heatmap(timeline, n_cols)),
        ("Feature Differentiation", build_ff_weight_divergence(timeline, n_cols)),
        ("Signal Balance", build_voltage_excitability(timeline, n_cols)),
        ("Input Distribution", build_column_drive_histogram(timeline)),
    ]

    return build_dashboard_html(figures, cards_html)


def _run_hierarchy_dashboard(tokens, cortex_cfg, charbit, input_dim, args) -> str:
    region1 = _make_region(cortex_cfg, input_dim)
    region2 = SensoryRegion(
        input_dim=region1.n_l23_total,
        encoding_width=0,  # sliding window — no positional structure in L2/3
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
    timeline1 = Timeline()
    timeline2 = Timeline()

    # Instrument both regions for timeline capture
    orig_r1_process = region1.process
    orig_r2_process = region2.process

    def instrumented_r1(encoding):
        result = orig_r1_process(encoding)
        timeline1.capture(len(timeline1.frames), region1, region1.last_column_drive)
        return result

    def instrumented_r2(encoding):
        result = orig_r2_process(encoding)
        timeline2.capture(len(timeline2.frames), region2, region2.last_column_drive)
        return result

    region1.process = instrumented_r1
    region2.process = instrumented_r2

    print(f"\nRunning hierarchy on {len(tokens):,} tokens...")
    hier_metrics = run_hierarchy(
        region1,
        region2,
        charbit,
        tokens,
        surprise_tracker=surprise,
        log_interval=args.log_interval,
        diagnostics1=diag1,
        diagnostics2=diag2,
    )

    region1.process = orig_r1_process
    region2.process = orig_r2_process

    print(f"\nCaptured {len(timeline1.frames)} R1 + {len(timeline2.frames)} R2 frames")
    print("Building hierarchy dashboard...")

    rep1 = hier_metrics.region1.representation
    rep2 = hier_metrics.region2.representation
    summ1 = diag1.summary()
    summ2 = diag2.summary()

    cards_html = build_hierarchy_summary_cards(
        rep1,
        rep2,
        summ1["burst_rate"],
        summ2["burst_rate"],
        hier_metrics.surprise_modulators,
    )

    n_cols_r1 = cortex_cfg.n_columns
    n_cols_r2 = 16
    figures = [
        ("Dual Burst Rate", build_dual_burst_rate(timeline1, timeline2)),
        (
            "Surprise Modulator",
            build_surprise_modulator_over_time(hier_metrics.surprise_modulators),
        ),
        ("R1 Column Selectivity", build_column_selectivity_bar(rep1)),
        ("R2 Column Selectivity", build_column_selectivity_bar(rep2)),
        ("R1 Surprise Rate", build_burst_rate_over_time(timeline1)),
        ("R1 Column Usage", build_column_entropy_over_time(timeline1, n_cols_r1)),
        ("R1 Column Activation", build_column_activation_heatmap(timeline1, n_cols_r1)),
        ("R2 Column Activation", build_column_activation_heatmap(timeline2, n_cols_r2)),
        (
            "R1 Feature Differentiation",
            build_ff_weight_divergence(timeline1, n_cols_r1),
        ),
        (
            "R2 Feature Differentiation",
            build_ff_weight_divergence(timeline2, n_cols_r2),
        ),
        ("R1 Signal Balance", build_voltage_excitability(timeline1, n_cols_r1)),
    ]

    return build_dashboard_html(figures, cards_html, title="Cortex Hierarchy Dashboard")


def _serve_or_save(html: str, args):
    if args.save_only:
        out_path = Path("experiments/figures/cortex_dashboard.html")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html)
        print(f"Saved to {out_path}")
        return

    # Serve on requested port
    with TemporaryDirectory() as tmpdir:
        index = Path(tmpdir) / "index.html"
        index.write_text(html)

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=tmpdir, **kw)

            def log_message(self, format, *a):
                pass  # quiet

        print(f"\nServing dashboard on http://0.0.0.0:{args.port}")
        print("Press Ctrl+C to stop")
        server = HTTPServer(("0.0.0.0", args.port), Handler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
