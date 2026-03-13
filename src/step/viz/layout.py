"""Dashboard HTML layout and page assembly."""

import plotly.graph_objects as go


def build_dashboard_html(
    figures: list[tuple[str, go.Figure]],
    summary_cards_html: str = "",
    title: str = "Cortex Representation Dashboard",
) -> str:
    """Combine all figures into a single HTML page."""
    chart_divs = []
    for _title, fig in figures:
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
    <h1>{title}</h1>
    {summary_cards_html}
    {"".join(f'<div class="chart-container">{div}</div>' for div in chart_divs)}
</body>
</html>"""
