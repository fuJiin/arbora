"""Plotly chart builders for cortex diagnostics dashboards."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from step.probes.timeline import Timeline

if TYPE_CHECKING:
    from step.probes.diagnostics import CortexDiagnostics


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
    n_dead = int((cum_rate == 0).sum())
    n_rare = int(((cum_rate > 0) & (cum_rate < n_steps * 0.01)).sum())

    fig = go.Figure(
        go.Heatmap(
            z=matrix[order],
            x=list(range(n_steps)),
            y=[f"col {c}" for c in order],
            colorscale=[[0, "#1a1a2e"], [1, "#e94560"]],
            showscale=False,
        )
    )
    dead_text = f"{n_dead} dead columns (never fired)"
    if n_rare:
        dead_text += f", {n_rare} rare (<1% of steps)"
    fig.update_layout(
        title=(
            f"Which Columns Fire? ({dead_text})"
        ),
        xaxis_title="Timestep",
        yaxis_title="Column",
        height=500,
        template="plotly_dark",
    )
    return fig


def build_ff_weight_divergence(timeline: Timeline, n_columns: int) -> go.Figure:
    """Per-column ff_weight change from initial over time."""
    n_steps = len(timeline.frames)
    norms = np.zeros((n_columns, n_steps))
    for i, frame in enumerate(timeline.frames):
        # ff_weight_norms is per-neuron; aggregate to per-column (max)
        per_neuron = frame.ff_weight_norms
        n_per_col = len(per_neuron) // n_columns
        norms[:, i] = per_neuron.reshape(n_columns, n_per_col).max(axis=1)

    # Relative change from initial values so small movements are visible
    initial = norms[:, 0:1]
    safe_initial = np.where(initial > 1e-8, initial, 1.0)
    rel_change = (norms - initial) / safe_initial

    fig = go.Figure()
    for col in range(n_columns):
        fig.add_trace(
            go.Scatter(
                x=list(range(n_steps)),
                y=rel_change[col],
                mode="lines",
                name=f"col {col}",
                line=dict(width=1),
                opacity=0.6,
            )
        )

    # Spread: std of relative changes across columns
    rel_std = rel_change.std(axis=0)
    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=rel_std,
            mode="lines",
            name="spread between columns",
            line=dict(color="white", width=3),
        )
    )

    fig.update_layout(
        title=(
            "Are Columns Learning Different Features?"
            " (each line = one column's weight change from initial;"
            " white = divergence between columns)"
        ),
        xaxis_title="Timestep",
        yaxis_title="Relative Weight Change",
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
            "Input Signal vs Homeostatic Boost (green should stay above yellow)",
            "How Much Does Input Vary Across Columns? (higher = more discriminating)",
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
            "How Strongly Does Input Activate Each Column? (snapshots over training)"
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
    n_steps = len(timeline.frames)
    entropies = []
    unique_counts = []

    for i in range(n_steps):
        start = max(0, i - window + 1)
        counts: Counter[int] = Counter()
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
            f"Column Usage Fairness (window={window}) — 1.0 = all columns used equally",
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


def build_burst_rate_over_time(timeline: Timeline, window: int = 50) -> go.Figure:
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
    region_label: str = "",
) -> go.Figure:
    """Per-column selectivity bar chart from representation metrics."""
    sel = rep_summary.get("column_selectivity_per_col", [])
    if not sel:
        return go.Figure()

    n = len(sel)
    colors = [
        "#06d6a0" if s < 0.3 else "#ffd166" if s < 0.6 else "#e94560" for s in sel
    ]

    fig = go.Figure(
        go.Bar(
            x=list(range(n)),
            y=sel,
            marker_color=colors,
        )
    )
    prefix = f"{region_label}: " if region_label else ""
    fig.update_layout(
        title=(
            f"{prefix}How Picky Is Each Column?"
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
            f"Surprise Rate by Region (window={window}) — lower = better predictions"
        ),
        xaxis_title="Timestep",
        yaxis_title="Surprise Rate",
        yaxis_range=[0, 1.05],
        height=400,
        template="plotly_dark",
    )
    return fig


def build_segment_health_over_time(
    diag: CortexDiagnostics,
    region_label: str = "",
) -> go.Figure:
    """Segment learning progress: active segments and mean permanence."""
    if not diag.snapshots:
        return go.Figure()

    steps = [s.t for s in diag.snapshots]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=[
            "Active Segments Per Type (firing given current activity)",
            "Mean Permanence Per Type (learned synapse strength)",
        ],
    )

    # Row 1: active segment counts (more sensitive than connected fraction)
    for name, attr, color in [
        ("Feedback (L2/3->L4)", "n_active_fb_segments", "#e94560"),
        ("Lateral (L4->L4)", "n_active_lat_segments", "#ffd166"),
        ("L2/3 Lateral", "n_active_l23_segments", "#06d6a0"),
        ("Apical (R2->R1)", "n_apical_predicted_cols", "#118ab2"),
    ]:
        data = [getattr(s, attr) for s in diag.snapshots]
        if any(d > 0 for d in data):
            fig.add_trace(
                go.Scatter(
                    x=steps, y=data, name=name,
                    line=dict(color=color, width=2),
                ),
                row=1, col=1,
            )

    # Row 2: mean permanence (only for non-zero, shows learning progress)
    for name, attr, color in [
        ("Feedback", "fb_seg_perm_mean", "#e94560"),
        ("Lateral", "lat_seg_perm_mean", "#ffd166"),
        ("L2/3 Lateral", "l23_seg_perm_mean", "#06d6a0"),
        ("Apical", "apical_seg_perm_mean", "#118ab2"),
    ]:
        data = [getattr(s, attr) for s in diag.snapshots]
        if any(d > 0 for d in data):
            fig.add_trace(
                go.Scatter(
                    x=steps, y=data, name=name,
                    line=dict(color=color, width=2),
                    showlegend=False,
                ),
                row=2, col=1,
            )

    prefix = f"{region_label} " if region_label else ""
    fig.update_layout(
        title=f"{prefix}Segment Learning Progress",
        height=550,
        template="plotly_dark",
        xaxis2_title="Step",
    )
    return fig


def build_apical_prediction_over_time(
    timeline: Timeline, window: int = 50
) -> go.Figure:
    """Rolling apical predicted column count and hit rate over time."""
    n_steps = len(timeline.frames)
    pred_counts: list[float] = []
    hit_rates: list[float] = []

    for i in range(n_steps):
        start = max(0, i - window + 1)
        total_pred = 0
        total_hits = 0
        for j in range(start, i + 1):
            f = timeline.frames[j]
            pred_set = set(f.apical_predicted_columns)
            active_set = set(f.active_columns)
            total_pred += len(pred_set)
            total_hits += len(pred_set & active_set)
        pred_counts.append(total_pred / (i - start + 1))
        hit_rates.append(total_hits / total_pred if total_pred > 0 else 0.0)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            f"Avg Apically-Predicted Columns (window={window})",
            f"Apical Prediction Hit Rate (window={window})",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=pred_counts,
            name="predicted cols",
            line=dict(color="#118ab2", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps)),
            y=hit_rates,
            name="hit rate",
            line=dict(color="#06d6a0", width=2),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=500,
        template="plotly_dark",
        xaxis2_title="Timestep",
    )
    return fig
