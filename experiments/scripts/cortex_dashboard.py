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
from step.config import CortexConfig, _default_region2_config
from step.cortex.sensory import SensoryRegion
from step.cortex.surprise import SurpriseTracker
from step.cortex.topology import Topology
from step.data import prepare_tokens, prepare_tokens_charlevel
from step.encoders.charbit import CharbitEncoder
from step.encoders.positional import PositionalCharEncoder
from step.viz import (
    build_apical_prediction_over_time,
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
    build_segment_health_over_time,
    build_surprise_modulator_over_time,
    build_voltage_excitability,
)

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1


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
    parser.add_argument(
        "--char-level", action="store_true",
        help="Use character-level tokenization with positional encoding",
    )
    parser.add_argument(
        "--buffer-depth", type=int, default=1,
        help="Temporal buffer depth for S1→S2 feedforward (default: 1 = direct)",
    )
    parser.add_argument(
        "--burst-gate", action="store_true",
        help="Gate feedforward signal by bursting columns (novel events only)",
    )
    args = parser.parse_args()

    cortex_cfg = CortexConfig()

    if args.char_level:
        tokens = prepare_tokens_charlevel(args.tokens)
        alphabet = sorted(
            {ch for _, ch in tokens if _ != -1}
        )
        encoder = PositionalCharEncoder("".join(alphabet), max_positions=8)
        input_dim = encoder.input_dim
        encoding_width = encoder.encoding_width
        # Lower LTD for char-level (default 0.2 is too aggressive)
        cortex_cfg = CortexConfig(ltd_rate=0.05)
    else:
        tokens = prepare_tokens(args.tokens)
        encoder = CharbitEncoder(length=CHAR_LENGTH, width=CHAR_WIDTH, chars=CHARS)
        input_dim = CHAR_LENGTH * CHAR_WIDTH
        encoding_width = CHAR_WIDTH

    # Build config banner
    enc_name = type(encoder).__name__
    r1_dim = cortex_cfg.n_columns * cortex_cfg.n_l23
    config_parts = [
        f'<span><span class="cfg-label">Encoder:</span> {enc_name} '
        f'({input_dim}-dim)</span>',
        f'<span><span class="cfg-label">S1:</span> '
        f'{cortex_cfg.n_columns} cols, k={cortex_cfg.k_columns}, '
        f'{cortex_cfg.n_l4} L4, {cortex_cfg.n_l23} L2/3 '
        f'(dim={r1_dim}), '
        f'lr={cortex_cfg.learning_rate}, ltd={cortex_cfg.ltd_rate}</span>',
        f'<span><span class="cfg-label">Tokens:</span> {len(tokens):,}</span>',
    ]
    if args.hierarchy:
        r2_cfg = _default_region2_config()
        r2_dim = r2_cfg.n_columns * r2_cfg.n_l23
        ff_extras = ""
        if args.buffer_depth > 1:
            ff_extras += f", buffer_depth={args.buffer_depth}"
        if args.burst_gate:
            ff_extras += ", burst_gate=True"
        config_parts.insert(2,
            f'<span><span class="cfg-label">S2:</span> '
            f'{r2_cfg.n_columns} cols, k={r2_cfg.k_columns}, '
            f'{r2_cfg.n_l4} L4, {r2_cfg.n_l23} L2/3 '
            f'(dim={r2_dim}), '
            f'lr={r2_cfg.learning_rate}, ltd={r2_cfg.ltd_rate}, '
            f'v_decay={r2_cfg.voltage_decay}{ff_extras}</span>'
        )
    config_html = (
        '<div class="config-banner">'
        + "".join(config_parts)
        + "</div>"
    )

    if args.hierarchy:
        html = _run_hierarchy_dashboard(
            tokens, cortex_cfg, encoder, input_dim, args,
            encoding_width=encoding_width,
            config_html=config_html,
        )
    else:
        html = _run_single_dashboard(
            tokens, cortex_cfg, encoder, input_dim, args,
            encoding_width=encoding_width,
            config_html=config_html,
        )

    _serve_or_save(html, args)


def _make_region(cortex_cfg, input_dim, encoding_width=CHAR_WIDTH):
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
        encoding_width=encoding_width,
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
        n_apical_segments=cortex_cfg.n_apical_segments,
        seed=cortex_cfg.seed,
    )


def _run_single_dashboard(
    tokens, cortex_cfg, encoder, input_dim, args,
    encoding_width=CHAR_WIDTH,
    config_html="",
) -> str:
    region = _make_region(cortex_cfg, input_dim, encoding_width)

    cortex = Topology(
        encoder,
        enable_timeline=True,
        diagnostics_interval=args.log_interval,
    )
    cortex.add_region("S1", region, entry=True)

    print(f"\nRunning cortex on {len(tokens):,} tokens...")
    result = cortex.run(tokens, log_interval=args.log_interval)

    timeline = cortex.timelines["S1"]
    diag = cortex.diagnostics["S1"]
    metrics = result.per_region["S1"]

    diag.print_report()

    print(f"\nCaptured {len(timeline.frames)} timeline frames")
    print("Building dashboard...")

    rep_summary = metrics.representation
    summ = diag.summary()
    burst_rate = summ["burst_rate"]

    cards_html = build_representation_summary_cards(rep_summary, burst_rate)

    n_cols = cortex_cfg.n_columns
    tabs = {
        "Activity": [
            ("Surprise Rate", build_burst_rate_over_time(timeline)),
            ("Column Activation", build_column_activation_heatmap(timeline, n_cols)),
            ("Column Usage", build_column_entropy_over_time(timeline, n_cols)),
            ("Input Distribution", build_column_drive_histogram(timeline)),
        ],
        "Representations": [
            ("Column Selectivity", build_column_selectivity_bar(rep_summary)),
            ("Feature Differentiation", build_ff_weight_divergence(timeline, n_cols)),
            ("Signal Balance", build_voltage_excitability(timeline, n_cols)),
        ],
        "Segments": [
            ("Segment Health", build_segment_health_over_time(diag, region_label="S1")),
            ("Apical Predictions", build_apical_prediction_over_time(timeline)),
        ],
    }

    return build_dashboard_html(
        [],
        cards_html,
        title="Cortex Dashboard",
        tabs=tabs,
        config_html=config_html,
    )


def _run_hierarchy_dashboard(
    tokens, cortex_cfg, encoder, input_dim, args,
    encoding_width=CHAR_WIDTH,
    config_html="",
) -> str:
    region1 = _make_region(cortex_cfg, input_dim, encoding_width)
    r2_cfg = _default_region2_config()
    r2_input_dim = region1.n_l23_total * args.buffer_depth
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

    surprise = SurpriseTracker()

    cortex = Topology(
        encoder,
        enable_timeline=True,
        diagnostics_interval=args.log_interval,
    )
    cortex.add_region("S1", region1, entry=True)
    cortex.add_region("S2", region2)
    cortex.connect(
        "S1", "S2", "feedforward",
        buffer_depth=args.buffer_depth, burst_gate=args.burst_gate,
    )
    cortex.connect("S1", "S2", "surprise", surprise_tracker=surprise)

    print(f"\nRunning hierarchy on {len(tokens):,} tokens...")
    result = cortex.run(tokens, log_interval=args.log_interval)

    timeline1 = cortex.timelines["S1"]
    timeline2 = cortex.timelines["S2"]
    diag1 = cortex.diagnostics["S1"]
    diag2 = cortex.diagnostics["S2"]

    print(
        f"\nCaptured {len(timeline1.frames)} S1"
        f" + {len(timeline2.frames)} S2 frames"
    )
    print("Building hierarchy dashboard...")

    rep1 = result.per_region["S1"].representation
    rep2 = result.per_region["S2"].representation
    summ1 = diag1.summary()
    summ2 = diag2.summary()

    cards_html = build_hierarchy_summary_cards(
        rep1,
        rep2,
        summ1["burst_rate"],
        summ2["burst_rate"],
        result.surprise_modulators.get("S2", []),
        diag1=diag1,
    )

    n_cols_r1 = cortex_cfg.n_columns
    n_cols_r2 = r2_cfg.n_columns

    tabs = {
        "Overview": [
            ("Dual Burst Rate", build_dual_burst_rate(timeline1, timeline2)),
            (
                "Surprise Modulator",
                build_surprise_modulator_over_time(
                    result.surprise_modulators.get("S2", []),
                ),
            ),
            (
                "S1 Column Selectivity",
                build_column_selectivity_bar(rep1, region_label="S1 (Sensory)"),
            ),
            (
                "S2 Column Selectivity",
                build_column_selectivity_bar(rep2, region_label="S2 (Secondary)"),
            ),
        ],
        "Region 1": [
            ("S1 Surprise Rate", build_burst_rate_over_time(timeline1)),
            (
                "S1 Column Usage",
                build_column_entropy_over_time(timeline1, n_cols_r1),
            ),
            (
                "S1 Column Activation",
                build_column_activation_heatmap(timeline1, n_cols_r1),
            ),
            (
                "S1 Feature Differentiation",
                build_ff_weight_divergence(timeline1, n_cols_r1),
            ),
            (
                "S1 Signal Balance",
                build_voltage_excitability(timeline1, n_cols_r1),
            ),
        ],
        "Region 2": [
            ("S2 Surprise Rate", build_burst_rate_over_time(timeline2)),
            (
                "S2 Column Activation",
                build_column_activation_heatmap(timeline2, n_cols_r2),
            ),
            (
                "S2 Feature Differentiation",
                build_ff_weight_divergence(timeline2, n_cols_r2),
            ),
            (
                "S2 Column Usage",
                build_column_entropy_over_time(timeline2, n_cols_r2),
            ),
        ],
        "Feedback": [
            (
                "S1 Segment Health",
                build_segment_health_over_time(diag1, region_label="S1"),
            ),
            ("Apical Predictions", build_apical_prediction_over_time(timeline1)),
            (
                "S2 Segment Health",
                build_segment_health_over_time(diag2, region_label="S2"),
            ),
        ],
    }

    return build_dashboard_html(
        [],
        cards_html,
        title="Cortex Hierarchy Dashboard",
        tabs=tabs,
        config_html=config_html,
    )


def _serve_or_save(html: str, args):
    if args.save_only:
        out_path = Path("experiments/figures/cortex_dashboard.html")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html)
        print(f"Saved to {out_path}")
        return

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
