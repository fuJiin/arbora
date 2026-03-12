#!/usr/bin/env python3
"""Interactive cortex diagnostics dashboard served on port 80.

Runs the cortex PoC, captures per-step timeline data, and serves
interactive Plotly charts showing column activation dynamics, ff_weight
divergence, voltage/excitability balance, and column drive distributions.

Usage: uv run experiments/scripts/cortex_dashboard.py [--tokens N] [--port 80]
"""

import argparse
import json
import string
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import step.env  # noqa: F401
from step.cortex.config import CortexConfig
from step.cortex.diagnostics import CortexDiagnostics
from step.cortex.runner import STORY_BOUNDARY, run_cortex
from step.cortex.sensory import SensoryRegion
from step.cortex.timeline import Timeline
from step.encoders.charbit import CharbitEncoder

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1


def prepare_cortex_tokens(max_tokens: int):
    """Load TinyStories tokens for cortex model."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading dataset and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    tokens: list[tuple[int, str]] = []
    t = 0
    first_story = True
    for example in dataset:
        if not first_story:
            tokens.append((STORY_BOUNDARY, ""))
            t += 1
            if t >= max_tokens:
                break
        first_story = False
        for tid in tokenizer.encode(example["text"]):
            tokens.append((tid, tokenizer.decode([tid])))
            t += 1
            if t >= max_tokens:
                break
        if t >= max_tokens:
            break

    unique = len({tid for tid, _ in tokens if tid != STORY_BOUNDARY})
    boundaries = sum(1 for tid, _ in tokens if tid == STORY_BOUNDARY)
    print(f"  {len(tokens):,} tokens, {unique} unique, {boundaries + 1} stories")
    return tokens


def run_with_timeline(tokens, region, encoder, log_interval):
    """Run cortex and capture per-step timeline."""
    timeline = Timeline()
    diag = CortexDiagnostics(snapshot_interval=log_interval)

    # Patch: capture timeline after each process call
    original_process = region.process

    def instrumented_process(encoding):
        result = original_process(encoding)
        timeline.capture(len(timeline.frames), region, region.last_column_drive)
        return result

    region.process = instrumented_process

    metrics = run_cortex(region, encoder, tokens, log_interval=log_interval, diagnostics=diag)
    diag.print_report()

    # Restore
    region.process = original_process
    return metrics, timeline, diag


def build_column_activation_heatmap(timeline: Timeline, n_columns: int) -> go.Figure:
    """Heatmap: rows=columns, cols=timesteps, color=activated."""
    n_steps = len(timeline.frames)
    matrix = np.zeros((n_columns, n_steps), dtype=np.float32)

    for i, frame in enumerate(timeline.frames):
        for col in frame.active_columns:
            matrix[col, i] = 1.0

    # Compute cumulative activation rate for column ordering
    cum_rate = matrix.sum(axis=1)
    order = np.argsort(-cum_rate)  # most active at top

    fig = go.Figure(
        go.Heatmap(
            z=matrix[order],
            x=list(range(n_steps)),
            y=[f"col {c}" for c in order],
            colorscale=[[0, "#1a1a2e"], [1, "#e94560"]],
            showscale=False,
        )
    )
    fig.update_layout(
        title="Column Activation Over Time (sorted by frequency)",
        xaxis_title="Timestep",
        yaxis_title="Column",
        height=500,
        template="plotly_dark",
    )
    return fig


def build_ff_weight_divergence(timeline: Timeline, n_columns: int) -> go.Figure:
    """Per-column ff_weight L2 norm over time — divergence = differentiation."""
    n_steps = len(timeline.frames)
    norms = np.zeros((n_columns, n_steps))
    for i, frame in enumerate(timeline.frames):
        norms[:, i] = frame.ff_weight_norms

    fig = go.Figure()
    for col in range(n_columns):
        fig.add_trace(
            go.Scatter(
                x=list(range(n_steps)),
                y=norms[col],
                mode="lines",
                name=f"col {col}",
                line=dict(width=1),
                opacity=0.6,
            )
        )

    # Add std band
    norm_std = norms.std(axis=0)
    norm_mean = norms.mean(axis=0)
    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=norm_std,
            mode="lines",
            name="cross-column std",
            line=dict(color="white", width=3),
        )
    )

    fig.update_layout(
        title="FF Weight Norms Per Column (spread = differentiation)",
        xaxis_title="Timestep",
        yaxis_title="L2 Norm",
        height=450,
        template="plotly_dark",
        showlegend=False,
    )
    return fig


def build_voltage_excitability(timeline: Timeline, n_columns: int) -> go.Figure:
    """Voltage vs excitability balance over time."""
    n_steps = len(timeline.frames)
    voltage_max = np.zeros(n_steps)
    voltage_mean = np.zeros(n_steps)
    excitability_max = np.zeros(n_steps)
    excitability_mean = np.zeros(n_steps)
    drive_spread = np.zeros(n_steps)

    for i, frame in enumerate(timeline.frames):
        voltage_max[i] = frame.voltage_l4_by_col.max()
        voltage_mean[i] = frame.voltage_l4_by_col.mean()
        excitability_max[i] = frame.excitability_l4_by_col.max()
        excitability_mean[i] = frame.excitability_l4_by_col.mean()
        drive_spread[i] = frame.column_drive.std()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["Voltage vs Excitability (max)", "Column Drive Spread (std)"])

    fig.add_trace(go.Scatter(x=list(range(n_steps)), y=voltage_max,
                             name="voltage max", line=dict(color="#06d6a0")), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(n_steps)), y=voltage_mean,
                             name="voltage mean", line=dict(color="#06d6a0", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(n_steps)), y=excitability_max,
                             name="excitability max", line=dict(color="#ffd166")), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(n_steps)), y=excitability_mean,
                             name="excitability mean", line=dict(color="#ffd166", dash="dot")), row=1, col=1)

    fig.add_trace(go.Scatter(x=list(range(n_steps)), y=drive_spread,
                             name="drive std", line=dict(color="#118ab2")), row=2, col=1)

    fig.update_layout(
        height=600,
        template="plotly_dark",
        xaxis2_title="Timestep",
    )
    return fig


def build_column_drive_histogram(timeline: Timeline) -> go.Figure:
    """Snapshots of column_drive distribution at different points in training."""
    n_steps = len(timeline.frames)
    sample_points = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
    sample_points = [p for p in sample_points if 0 <= p < n_steps]

    fig = go.Figure()
    colors = ["#ef476f", "#ffd166", "#06d6a0", "#118ab2", "#073b4c"]
    for i, t in enumerate(sample_points):
        frame = timeline.frames[t]
        fig.add_trace(
            go.Histogram(
                x=frame.column_drive,
                name=f"t={t}",
                opacity=0.6,
                marker_color=colors[i % len(colors)],
                nbinsx=30,
            )
        )

    fig.update_layout(
        title="Column Drive Distribution Over Training",
        xaxis_title="Column Drive (input @ ff_weights)",
        yaxis_title="Count",
        barmode="overlay",
        height=400,
        template="plotly_dark",
    )
    return fig


def build_column_entropy_over_time(timeline: Timeline, n_columns: int, window: int = 50) -> go.Figure:
    """Rolling column entropy over time — shows when monopoly develops."""
    from collections import Counter

    n_steps = len(timeline.frames)
    entropies = []
    unique_counts = []

    for i in range(n_steps):
        start = max(0, i - window + 1)
        counts = Counter()
        for j in range(start, i + 1):
            for col in timeline.frames[j].active_columns:
                counts[col] += 1

        total = sum(counts.values()) or 1
        probs = np.array([counts.get(c, 0) / total for c in range(n_columns)])
        probs = probs[probs > 0]
        entropy = -float(np.sum(probs * np.log2(probs)))
        max_entropy = np.log2(n_columns)
        entropies.append(entropy / max_entropy)
        unique_counts.append(len(counts))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[
                            f"Rolling Column Entropy (window={window})",
                            f"Unique Active Columns (window={window})",
                        ])

    fig.add_trace(go.Scatter(x=list(range(n_steps)), y=entropies,
                             name="entropy ratio", line=dict(color="#e94560")), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(n_steps)), y=[1.0] * n_steps,
                             name="perfect", line=dict(color="gray", dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=list(range(n_steps)), y=unique_counts,
                             name="unique cols", line=dict(color="#06d6a0")), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(n_steps)), y=[n_columns] * n_steps,
                             name="max possible", line=dict(color="gray", dash="dash")), row=2, col=1)

    fig.update_layout(height=500, template="plotly_dark", xaxis2_title="Timestep")
    return fig


def build_dashboard_html(figures: list[tuple[str, go.Figure]]) -> str:
    """Combine all figures into a single HTML page."""
    charts_html = []
    for i, (title, fig) in enumerate(figures):
        div_id = f"chart-{i}"
        chart_json = fig.to_json()
        charts_html.append(f"""
        <div class="chart-container">
            <div id="{div_id}"></div>
            <script>
                Plotly.newPlot('{div_id}', ...JSON.parse('{chart_json}').data ?
                    [JSON.parse('{chart_json}')].map(c => ({{data: c.data, layout: c.layout}})) :
                    [{{data: [], layout: {{}}}}]
                );
                var chartData = JSON.parse(`{chart_json}`);
                Plotly.newPlot('{div_id}', chartData.data, chartData.layout, {{responsive: true}});
            </script>
        </div>
        """)

    # Simpler approach: use plotly's to_html
    chart_divs = []
    for i, (title, fig) in enumerate(figures):
        chart_divs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Cortex Diagnostics Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
    <style>
        body {{
            background: #0d1117;
            color: #c9d1d9;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            color: #e94560;
            border-bottom: 1px solid #30363d;
            padding-bottom: 10px;
        }}
        .chart-container {{
            margin: 20px 0;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 10px;
            background: #161b22;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #e94560;
        }}
        .stat-label {{
            color: #8b949e;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>Cortex Column Monopoly Dashboard</h1>
    <div id="summary"></div>
    {''.join(f'<div class="chart-container">{div}</div>' for div in chart_divs)}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Cortex diagnostics dashboard")
    parser.add_argument("--tokens", type=int, default=1000)
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-only", action="store_true", help="Save HTML without serving")
    args = parser.parse_args()

    # Run the PoC with timeline capture
    tokens = prepare_cortex_tokens(args.tokens)

    cortex_cfg = CortexConfig()
    charbit = CharbitEncoder(length=CHAR_LENGTH, width=CHAR_WIDTH, chars=CHARS)
    input_dim = CHAR_LENGTH * CHAR_WIDTH
    region = SensoryRegion(
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
        fb_boost_threshold=cortex_cfg.fb_boost_threshold,
        fb_boost=cortex_cfg.fb_boost,
        ltd_rate=cortex_cfg.ltd_rate,
        encoding_width=CHAR_WIDTH,
        seed=cortex_cfg.seed,
    )

    print(f"\nRunning cortex on {len(tokens):,} tokens...")
    metrics, timeline, diag = run_with_timeline(tokens, region, charbit, args.log_interval)

    print(f"\nCaptured {len(timeline.frames)} timeline frames")
    print("Building dashboard...")

    # Build all charts
    figures = [
        ("Column Entropy", build_column_entropy_over_time(timeline, cortex_cfg.n_columns)),
        ("Column Activation", build_column_activation_heatmap(timeline, cortex_cfg.n_columns)),
        ("FF Weight Divergence", build_ff_weight_divergence(timeline, cortex_cfg.n_columns)),
        ("Voltage vs Excitability", build_voltage_excitability(timeline, cortex_cfg.n_columns)),
        ("Column Drive Distribution", build_column_drive_histogram(timeline)),
    ]

    html = build_dashboard_html(figures)

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
