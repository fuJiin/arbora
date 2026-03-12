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

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import step.env  # noqa: F401
from step.cortex.config import CortexConfig
from step.cortex.diagnostics import CortexDiagnostics
from step.cortex.runner import STORY_BOUNDARY, run_cortex, run_hierarchy
from step.cortex.sensory import SensoryRegion
from step.cortex.surprise import SurpriseTracker
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
    """Run cortex and capture per-step timeline + representation tracking."""
    timeline = Timeline()
    diag = CortexDiagnostics(snapshot_interval=log_interval)

    # Patch: capture timeline after each process call
    original_process = region.process

    def instrumented_process(encoding):
        result = original_process(encoding)
        timeline.capture(
            len(timeline.frames), region, region.last_column_drive
        )
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
        title="Which Columns Fire? (sorted by how often each column activates)",
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
        # ff_weight_norms is per-neuron; aggregate to per-column (max)
        per_neuron = frame.ff_weight_norms
        n_per_col = len(per_neuron) // n_columns
        norms[:, i] = per_neuron.reshape(n_columns, n_per_col).max(axis=1)

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

    # Add spread line (how different the columns are from each other)
    norm_std = norms.std(axis=0)
    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=norm_std,
            mode="lines",
            name="spread between columns",
            line=dict(color="white", width=3),
        )
    )

    fig.update_layout(
        title=(
            "Are Columns Learning Different Features?"
            " (each line = one column's learned weight strength;"
            " white = spread between columns)"
        ),
        xaxis_title="Timestep",
        yaxis_title="Weight Strength",
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

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Input Signal vs Homeostatic Boost"
            " (green should stay above yellow)",
            "How Much Does Input Vary Across Columns?"
            " (higher = more discriminating)",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=voltage_max,
            name="input signal (max)",
            line=dict(color="#06d6a0"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=voltage_mean,
            name="input signal (mean)",
            line=dict(color="#06d6a0", dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=excitability_max,
            name="homeostatic boost (max)",
            line=dict(color="#ffd166"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=excitability_mean,
            name="homeostatic boost (mean)",
            line=dict(color="#ffd166", dash="dot"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=drive_spread,
            name="input spread",
            line=dict(color="#118ab2"),
        ),
        row=2,
        col=1,
    )

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
        title=(
            "How Strongly Does Input Activate Each Column?"
            " (snapshots over training)"
        ),
        xaxis_title="Activation Strength",
        yaxis_title="Number of Columns",
        barmode="overlay",
        height=400,
        template="plotly_dark",
    )
    return fig


def build_column_entropy_over_time(
    timeline: Timeline, n_columns: int, window: int = 50
) -> go.Figure:
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

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            f"Column Usage Fairness (window={window})"
            " — 1.0 = all columns used equally",
            f"How Many Different Columns Are Active (window={window})",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=entropies,
            name="fairness",
            line=dict(color="#e94560"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=[1.0] * n_steps,
            name="perfect",
            line=dict(color="gray", dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=unique_counts,
            name="unique cols",
            line=dict(color="#06d6a0"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=[n_columns] * n_steps,
            name="max possible",
            line=dict(color="gray", dash="dash"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=500, template="plotly_dark", xaxis2_title="Timestep")
    return fig


def build_burst_rate_over_time(
    timeline: Timeline, window: int = 50
) -> go.Figure:
    """Rolling burst rate over time — primary temporal prediction metric."""
    n_steps = len(timeline.frames)
    burst_rates = []

    for i in range(n_steps):
        start = max(0, i - window + 1)
        total_burst = 0
        total_active = 0
        for j in range(start, i + 1):
            f = timeline.frames[j]
            total_burst += f.n_bursting
            total_active += f.n_active
        rate = total_burst / total_active if total_active > 0 else 0.0
        burst_rates.append(rate)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=burst_rates,
            name="surprise rate",
            line=dict(color="#e94560", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=[1.0] * n_steps,
            name="100% (completely surprised)",
            line=dict(color="gray", dash="dash"),
        )
    )
    fig.update_layout(
        title=(
            f"Surprise Rate (window={window})"
            " — how often the cortex is caught off guard"
            " (lower = learning to anticipate)"
        ),
        xaxis_title="Timestep",
        yaxis_title="Surprise Rate",
        yaxis_range=[0, 1.05],
        height=400,
        template="plotly_dark",
    )
    return fig


def build_column_selectivity_bar(
    rep_summary: dict,
) -> go.Figure:
    """Per-column selectivity bar chart from representation metrics."""
    sel = rep_summary.get("column_selectivity_per_col", [])
    if not sel:
        return go.Figure()

    n = len(sel)
    colors = [
        "#06d6a0" if s < 0.3
        else "#ffd166" if s < 0.6
        else "#e94560"
        for s in sel
    ]

    fig = go.Figure(
        go.Bar(
            x=list(range(n)),
            y=sel,
            marker_color=colors,
        )
    )
    fig.update_layout(
        title=(
            "How Picky Is Each Column?"
            " (lower = responds to fewer tokens = better feature detector)"
        ),
        xaxis_title="Column",
        yaxis_title="Response Breadth (0=specialist, 1=responds to everything)",
        yaxis_range=[0, 1.05],
        height=350,
        template="plotly_dark",
    )
    return fig


def build_surprise_modulator_over_time(
    modulators: list[float], window: int = 50
) -> go.Figure:
    """Surprise modulator time series — how much Region 2 learning is boosted."""
    n = len(modulators)
    if n == 0:
        return go.Figure()

    # Rolling average
    rolling = []
    for i in range(n):
        start = max(0, i - window + 1)
        rolling.append(sum(modulators[start : i + 1]) / (i - start + 1))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(n)),
            y=modulators,
            name="raw",
            line=dict(color="#118ab2", width=1),
            opacity=0.3,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(n)),
            y=rolling,
            name=f"rolling avg (w={window})",
            line=dict(color="#118ab2", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(n)),
            y=[1.0] * n,
            name="baseline (no modulation)",
            line=dict(color="gray", dash="dash"),
        )
    )
    fig.update_layout(
        title=(
            "Surprise Modulator (Region 2 Learning Rate Scale)"
            " — >1 = R1 surprised, R2 learns faster"
        ),
        xaxis_title="Timestep",
        yaxis_title="Modulator",
        yaxis_range=[0, 2.1],
        height=400,
        template="plotly_dark",
    )
    return fig


def build_dual_burst_rate(
    timeline1: Timeline, timeline2: Timeline, window: int = 50
) -> go.Figure:
    """Side-by-side burst rate for Region 1 and Region 2."""

    def _rolling_burst(tl: Timeline) -> list[float]:
        n = len(tl.frames)
        rates = []
        for i in range(n):
            start = max(0, i - window + 1)
            total_burst = sum(tl.frames[j].n_bursting for j in range(start, i + 1))
            total_active = sum(tl.frames[j].n_active for j in range(start, i + 1))
            rates.append(total_burst / total_active if total_active > 0 else 0.0)
        return rates

    r1 = _rolling_burst(timeline1)
    r2 = _rolling_burst(timeline2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(r1))),
            y=r1,
            name="Region 1 (sensory)",
            line=dict(color="#e94560", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(r2))),
            y=r2,
            name="Region 2 (secondary)",
            line=dict(color="#06d6a0", width=2),
        )
    )
    fig.update_layout(
        title=(
            f"Surprise Rate by Region (window={window})"
            " — lower = better predictions"
        ),
        xaxis_title="Timestep",
        yaxis_title="Surprise Rate",
        yaxis_range=[0, 1.05],
        height=400,
        template="plotly_dark",
    )
    return fig


def build_hierarchy_summary_cards(
    rep1: dict, rep2: dict, burst1: float, burst2: float, modulators: list[float]
) -> str:
    """Build HTML stat cards comparing Region 1 and Region 2."""
    mod_arr = np.array(modulators) if modulators else np.array([1.0])

    def _card(value: str, label: str, hint: str, color: str) -> str:
        return f"""
        <div class="stat-card" style="border-color: {color}">
            <div class="stat-value" style="color: {color}">{value}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-label" style="font-size:0.75em">{hint}</div>
        </div>"""

    cards = [
        _card(
            f"{burst1:.0%} → {burst2:.0%}",
            "Surprise Rate (R1 → R2)",
            "R2 should be lower if it learns R1 patterns",
            _health_color(burst2, (0, 0.4), (0, 0.7)),
        ),
        _card(
            f"{rep1.get('column_selectivity_mean', 0):.2f}",
            "R1 Feature Selectivity",
            "how picky R1 columns are",
            _health_color(
                rep1.get("column_selectivity_mean", 1), (0.05, 0.5), (0.02, 0.8)
            ),
        ),
        _card(
            f"{rep2.get('column_selectivity_mean', 0):.2f}",
            "R2 Feature Selectivity",
            "how picky R2 columns are",
            _health_color(
                rep2.get("column_selectivity_mean", 1), (0.05, 0.7), (0.02, 0.9)
            ),
        ),
        _card(
            f"{rep1.get('context_discrimination', 0):.2f}"
            f" → {rep2.get('context_discrimination', 0):.2f}",
            "Context Sensitivity (R1 → R2)",
            "R2 should discriminate context better",
            _health_color(
                rep2.get("context_discrimination", 0), (0.1, 0.95), (0.05, 0.98)
            ),
        ),
        _card(
            f"{mod_arr.mean():.2f}",
            "Avg Surprise Modulator",
            f"range [{mod_arr.min():.2f}, {mod_arr.max():.2f}]",
            _health_color(mod_arr.mean(), (0.8, 1.2), (0.5, 1.5)),
        ),
        _card(
            f"{rep2.get('ff_cross_col_cosine', 0):.2f}",
            "R2 Column Diversity",
            "are R2 columns learning different things? (lower = yes)",
            _health_color(rep2.get("ff_cross_col_cosine", 1), (0, 0.3), (0, 0.6)),
        ),
    ]

    return '<div class="summary">' + "".join(cards) + "</div>"


def _health_color(value: float, green_range: tuple, yellow_range: tuple) -> str:
    """Return CSS color based on whether value is in healthy range.

    green_range: (low, high) — green if value is in this range
    yellow_range: (low, high) — yellow if in this but not green
    Otherwise red.
    """
    if green_range[0] <= value <= green_range[1]:
        return "#06d6a0"  # green
    if yellow_range[0] <= value <= yellow_range[1]:
        return "#ffd166"  # yellow
    return "#e94560"  # red


def build_representation_summary_cards(
    rep_summary: dict, burst_rate: float
) -> str:
    """Build HTML stat cards with health-based color coding.

    Color coding detects degenerate states rather than defining
    "optimal" — healthy representations have a range of acceptable
    values, not a single target.
    """
    sel = rep_summary.get("column_selectivity_mean", 0)
    sim = rep_summary.get("similarity_mean", 0)
    ctx = rep_summary.get("context_discrimination", 0)
    spar = rep_summary.get("ff_sparsity", 0)
    div = rep_summary.get("ff_cross_col_cosine", 0)

    # (value_str, label, hint, color)
    cards = [
        (
            f"{burst_rate:.0%}",
            "Surprise Rate",
            "how often the cortex can't anticipate what's next",
            _health_color(burst_rate, (0, 0.4), (0, 0.7)),
        ),
        (
            f"{sel:.2f}",
            "Feature Selectivity",
            "how picky columns are (lower = more specialized)",
            _health_color(sel, (0.05, 0.5), (0.02, 0.8)),
        ),
        (
            f"{sim:.2f}",
            "Token Similarity",
            "do similar tokens share columns? (want > 0, < 1)",
            _health_color(sim, (0.01, 0.8), (0.001, 0.95)),
        ),
        (
            f"{ctx:.2f}",
            "Context Sensitivity",
            "same token, different context = different neurons?",
            _health_color(ctx, (0.1, 0.95), (0.05, 0.98)),
        ),
        (
            f"{spar:.2f}",
            "Receptive Field Focus",
            "how sharp each column's input filter is",
            _health_color(spar, (0.9, 1.0), (0.7, 1.0)),
        ),
        (
            f"{div:.2f}",
            "Column Diversity",
            "are columns learning different things? (lower = yes)",
            _health_color(div, (0, 0.1), (0, 0.5)),
        ),
    ]

    html = '<div class="summary">'
    for value, label, hint, color in cards:
        html += f"""
        <div class="stat-card" style="border-color: {color}">
            <div class="stat-value" style="color: {color}">
                {value}
            </div>
            <div class="stat-label">{label}</div>
            <div class="stat-label" style="font-size:0.75em">
                {hint}
            </div>
        </div>"""
    html += "</div>"
    return html


def build_dashboard_html(
    figures: list[tuple[str, go.Figure]],
    summary_cards_html: str = "",
    title: str = "Cortex Representation Dashboard",
) -> str:
    """Combine all figures into a single HTML page."""
    chart_divs = []
    for _title, fig in figures:
        chart_divs.append(
            fig.to_html(full_html=False, include_plotlyjs=False)
        )

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
    <h1>{title}</h1>
    {summary_cards_html}
    {"".join(f'<div class="chart-container">{div}</div>' for div in chart_divs)}
</body>
</html>"""


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
    tokens = prepare_cortex_tokens(args.tokens)

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
        encoding_width=region1.n_l23,
        n_columns=16,
        n_l4=4,
        n_l23=4,
        k_columns=2,
        voltage_decay=0.8,
        eligibility_decay=0.98,
        synapse_decay=0.9999,
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
        region1, region2, charbit, tokens,
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
        rep1, rep2, summ1["burst_rate"], summ2["burst_rate"],
        hier_metrics.surprise_modulators,
    )

    n_cols_r1 = cortex_cfg.n_columns
    n_cols_r2 = 16
    figures = [
        ("Dual Burst Rate", build_dual_burst_rate(timeline1, timeline2)),
        ("Surprise Modulator", build_surprise_modulator_over_time(
            hier_metrics.surprise_modulators
        )),
        ("R1 Column Selectivity", build_column_selectivity_bar(rep1)),
        ("R2 Column Selectivity", build_column_selectivity_bar(rep2)),
        ("R1 Surprise Rate", build_burst_rate_over_time(timeline1)),
        ("R1 Column Usage", build_column_entropy_over_time(timeline1, n_cols_r1)),
        ("R1 Column Activation", build_column_activation_heatmap(timeline1, n_cols_r1)),
        ("R2 Column Activation", build_column_activation_heatmap(timeline2, n_cols_r2)),
        ("R1 Feature Differentiation", build_ff_weight_divergence(
            timeline1, n_cols_r1
        )),
        ("R2 Feature Differentiation", build_ff_weight_divergence(
            timeline2, n_cols_r2
        )),
        ("R1 Signal Balance", build_voltage_excitability(timeline1, n_cols_r1)),
    ]

    return build_dashboard_html(
        figures, cards_html, title="Cortex Hierarchy Dashboard"
    )


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
