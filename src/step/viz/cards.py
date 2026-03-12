"""HTML summary card builders for cortex dashboards."""

import numpy as np

from step.probes.diagnostics import CortexDiagnostics


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


def build_hierarchy_summary_cards(
    rep1: dict,
    rep2: dict,
    burst1: float,
    burst2: float,
    modulators: list[float],
    diag1: CortexDiagnostics | None = None,
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

    # Get apical stats from last diagnostic snapshot
    apical_conn = 0.0
    apical_pred = 0
    if diag1 and diag1.snapshots:
        snap = diag1.snapshots[-1]
        apical_conn = snap.apical_seg_connected_frac
        apical_pred = snap.n_apical_predicted_cols

    # Surprise: green if R2 improves on R1, yellow if similar, red if worse
    burst_delta = burst2 - burst1
    if burst_delta < -0.03:
        burst_color = "#06d6a0"  # R2 improved
    elif burst_delta < 0.03:
        burst_color = "#ffd166"  # similar
    else:
        burst_color = "#e94560"  # R2 worse

    # Context: green if R2 improves on R1
    ctx1 = rep1.get("context_discrimination", 0)
    ctx2 = rep2.get("context_discrimination", 0)
    ctx_improved = ctx2 > ctx1
    ctx_color = (
        "#06d6a0" if ctx_improved and ctx2 > 0.5
        else "#ffd166" if ctx_improved
        else "#e94560"
    )

    cards = [
        _card(
            f"{burst1:.0%} -> {burst2:.0%}",
            "Surprise Rate (R1 -> R2)",
            "R2 should be lower if it learns R1 patterns",
            burst_color,
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
            f"{ctx1:.2f} -> {ctx2:.2f}",
            "Context Sensitivity (R1 -> R2)",
            "R2 should discriminate context better",
            ctx_color,
        ),
        _card(
            f"{mod_arr.mean():.2f}",
            "Avg Surprise Modulator",
            f"range [{mod_arr.min():.2f}, {mod_arr.max():.2f}]",
            _health_color(mod_arr.mean(), (0.8, 1.2), (0.5, 1.5)),
        ),
        _card(
            f"{apical_conn:.1%}",
            "Apical Connectivity",
            f"{apical_pred} cols predicted -- R2->R1 feedback strength",
            _health_color(apical_conn, (0.02, 0.15), (0.005, 0.3)),
        ),
    ]

    return '<div class="summary">' + "".join(cards) + "</div>"


def build_representation_summary_cards(rep_summary: dict, burst_rate: float) -> str:
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
