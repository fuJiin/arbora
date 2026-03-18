"""Dashboard HTML layout and page assembly."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from step.runs import list_runs, load_meta


def build_dashboard_html(
    figures: list[tuple[str, go.Figure]],
    summary_cards_html: str = "",
    title: str = "Cortex Representation Dashboard",
    tabs: dict[str, list[tuple[str, go.Figure]]] | None = None,
    config_html: str = "",
) -> str:
    """Combine all figures into a single HTML page.

    If tabs is provided, renders a tabbed layout. Each key is a tab name,
    each value is a list of (title, figure) pairs. The figures param is
    ignored when tabs is set.
    """
    css = """
        body {
            background: #0d1117;
            color: #c9d1d9;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
            margin: 0;
            padding: 20px;
        }
        h1 {
            color: #e94560;
            border-bottom: 1px solid #30363d;
            padding-bottom: 10px;
        }
        .chart-container {
            margin: 20px 0;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 10px;
            background: #161b22;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #e94560;
        }
        .stat-label {
            color: #8b949e;
            font-size: 0.9em;
        }
        .tab-bar {
            display: flex;
            gap: 0;
            border-bottom: 2px solid #30363d;
            margin: 20px 0 0 0;
        }
        .tab-btn {
            padding: 10px 24px;
            background: none;
            border: none;
            color: #8b949e;
            cursor: pointer;
            font-size: 1em;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
        }
        .tab-btn.active {
            color: #e94560;
            border-bottom-color: #e94560;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .config-banner {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 10px 16px;
            margin: 10px 0;
            font-size: 0.85em;
            color: #8b949e;
            display: flex;
            flex-wrap: wrap;
            gap: 8px 24px;
        }
        .config-banner span {
            white-space: nowrap;
        }
        .config-banner .cfg-label {
            color: #c9d1d9;
            font-weight: 600;
        }
    """

    if tabs:
        return _build_tabbed_html(tabs, summary_cards_html, title, css, config_html)

    chart_divs = []
    for _title, fig in figures:
        chart_divs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
    <style>{css}</style>
</head>
<body>
    <h1>{title}</h1>
    {config_html}
    {summary_cards_html}
    {"".join(f'<div class="chart-container">{div}</div>' for div in chart_divs)}
</body>
</html>"""


def _build_tabbed_html(
    tabs: dict[str, list[tuple[str, go.Figure]]],
    summary_cards_html: str,
    title: str,
    css: str,
    config_html: str = "",
) -> str:
    tab_names = list(tabs.keys())

    # Build tab buttons
    buttons_html = ""
    for i, name in enumerate(tab_names):
        active = " active" if i == 0 else ""
        buttons_html += (
            f'<button class="tab-btn{active}" '
            f"onclick=\"switchTab('{name}')\">{name}</button>"
        )

    # Build tab content
    content_html = ""
    for i, (name, figs) in enumerate(tabs.items()):
        active = " active" if i == 0 else ""
        chart_divs = "".join(
            f'<div class="chart-container">'
            f"{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"
            for _t, fig in figs
        )
        content_html += (
            f'<div class="tab-content{active}" id="tab-{name}">{chart_divs}</div>'
        )

    js = """
    function switchTab(name) {
        document.querySelectorAll('.tab-btn').forEach(
            b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(
            c => c.classList.remove('active'));
        event.target.classList.add('active');
        document.getElementById('tab-' + name).classList.add('active');
        // Trigger Plotly resize for newly visible charts
        window.dispatchEvent(new Event('resize'));
    }
    """

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
    <style>{css}</style>
</head>
<body>
    <h1>{title}</h1>
    {config_html}
    {summary_cards_html}
    <div class="tab-bar">{buttons_html}</div>
    {content_html}
    <script>{js}</script>
</body>
</html>"""


_INDEX_CSS = """
    body {
        background: #0d1117;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
        margin: 0;
        padding: 20px;
    }
    h1 {
        color: #e94560;
        border-bottom: 1px solid #30363d;
        padding-bottom: 10px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }
    th, td {
        text-align: left;
        padding: 10px 14px;
        border-bottom: 1px solid #30363d;
    }
    th {
        color: #8b949e;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    tr:hover {
        background: #161b22;
    }
    a {
        color: #58a6ff;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    .tag {
        display: inline-block;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 0.78em;
        margin: 1px 2px;
        color: #8b949e;
    }
    .metric {
        color: #e94560;
        font-weight: 600;
    }
    .empty {
        color: #484f58;
        padding: 40px;
        text-align: center;
    }
"""


def build_index_html(runs_dir: Path | None = None) -> str:
    """Build an HTML index page listing all saved runs.

    Reads meta.json from each run directory. Links point to
    {run_dir_name}/dashboard.html (generated separately).
    """
    from step.runs import RUNS_DIR

    runs_dir = runs_dir or RUNS_DIR
    run_dirs = list_runs(runs_dir)

    if not run_dirs:
        return f"""<!DOCTYPE html>
<html>
<head><title>STEP Runs</title><style>{_INDEX_CSS}</style></head>
<body>
    <h1>STEP Experiment Runs</h1>
    <div class="empty">No runs found. Use cortex_run.py to create one.</div>
</body>
</html>"""

    rows = []
    for d in run_dirs:
        try:
            meta = load_meta(d)
        except Exception:
            continue

        name = meta.get("name", d.name)
        created = meta.get("created", "")[:19].replace("T", " ")
        tags = meta.get("tags", [])
        n_tokens = meta.get("n_tokens", "")
        elapsed = meta.get("elapsed_seconds", 0)
        regions = meta.get("regions", [])

        # Summary stats
        summary = meta.get("summary", {})
        stat_parts = []
        for rname, stats in summary.items():
            ctx = stats.get("ctx_disc", 0)
            sel = stats.get("selectivity", 0)
            stat_parts.append(f"{rname}: ctx={ctx:.2f} sel={sel:.2f}")
        stats_html = " | ".join(stat_parts) if stat_parts else "-"

        tags_html = "".join(f'<span class="tag">{t}</span>' for t in tags)
        regions_html = ", ".join(regions) if regions else "-"
        tokens_str = f"{n_tokens:,}" if isinstance(n_tokens, int) else str(n_tokens)
        elapsed_str = f"{elapsed:.1f}s" if elapsed else "-"

        link = f"{d.name}/dashboard.html"
        rows.append(
            f"<tr>"
            f'<td><a href="{link}">{name}</a></td>'
            f"<td>{created}</td>"
            f"<td>{regions_html}</td>"
            f"<td>{tokens_str}</td>"
            f"<td>{elapsed_str}</td>"
            f'<td><span class="metric">{stats_html}</span></td>'
            f"<td>{tags_html}</td>"
            f"</tr>"
        )

    rows_html = "\n".join(rows)
    return f"""<!DOCTYPE html>
<html>
<head><title>STEP Runs</title><style>{_INDEX_CSS}</style></head>
<body>
    <h1>STEP Experiment Runs</h1>
    <table>
        <thead>
            <tr>
                <th>Name</th>
                <th>Created</th>
                <th>Regions</th>
                <th>Tokens</th>
                <th>Time</th>
                <th>Metrics</th>
                <th>Tags</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>"""
