"""Dashboard HTML layout and page assembly."""

import plotly.graph_objects as go


def build_dashboard_html(
    figures: list[tuple[str, go.Figure]],
    summary_cards_html: str = "",
    title: str = "Cortex Representation Dashboard",
    tabs: dict[str, list[tuple[str, go.Figure]]] | None = None,
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
    """

    if tabs:
        return _build_tabbed_html(tabs, summary_cards_html, title, css)

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
    {summary_cards_html}
    {"".join(f'<div class="chart-container">{div}</div>' for div in chart_divs)}
</body>
</html>"""


def _build_tabbed_html(
    tabs: dict[str, list[tuple[str, go.Figure]]],
    summary_cards_html: str,
    title: str,
    css: str,
) -> str:
    tab_names = list(tabs.keys())

    # Build tab buttons
    buttons_html = ""
    for i, name in enumerate(tab_names):
        active = " active" if i == 0 else ""
        buttons_html += (
            f'<button class="tab-btn{active}" '
            f'onclick="switchTab(\'{name}\')">{name}</button>'
        )

    # Build tab content
    content_html = ""
    for i, (name, figs) in enumerate(tabs.items()):
        active = " active" if i == 0 else ""
        chart_divs = "".join(
            f'<div class="chart-container">'
            f'{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'
            for _t, fig in figs
        )
        content_html += f'<div class="tab-content{active}" id="tab-{name}">{chart_divs}</div>'

    js = """
    function switchTab(name) {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
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
    {summary_cards_html}
    <div class="tab-bar">{buttons_html}</div>
    {content_html}
    <script>{js}</script>
</body>
</html>"""
