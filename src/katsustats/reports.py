"""
katsustats.reports — Report generation combining metrics and plots.

Primary entry points:
    katsustats.reports.full(pnl, base_pnl)
    katsustats.reports.html(pnl, base_pnl, output="report.html")
"""

from __future__ import annotations

import base64
import io

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

from . import plots, stats
from ._dataframe import DataFrameLike, ensure_polars


def _print_df(df: pl.DataFrame, title: str = "") -> None:
    """Pretty-print a Polars DataFrame as a formatted table."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    # Get column widths
    cols = df.columns
    data = df.to_dict(as_series=False)
    widths = {}
    for col in cols:
        max_w = len(col)
        for val in data[col]:
            max_w = max(max_w, len(str(val)) if val is not None else 4)
        widths[col] = max_w + 2

    # Header
    header = "  ".join(str(col).rjust(widths[col]) for col in cols)
    print(header)
    print("  ".join("-" * widths[col] for col in cols))

    # Rows
    n_rows = len(data[cols[0]])
    for i in range(n_rows):
        row = "  ".join(
            str(data[col][i] if data[col][i] is not None else "—").rjust(widths[col])
            for col in cols
        )
        print(row)
    print()


def _fig_to_base64(fig: plt.Figure, dpi: int = 150) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _df_to_html_table(df: pl.DataFrame, *, css_class: str = "metrics") -> str:
    """Convert a Polars DataFrame to a styled HTML <table> string."""
    cols = df.columns
    data = df.to_dict(as_series=False)
    n_rows = len(data[cols[0]])

    rows_html: list[str] = []
    # Header
    header_cells = "".join(f"<th>{col}</th>" for col in cols)
    rows_html.append(f"<tr>{header_cells}</tr>")
    # Body
    for i in range(n_rows):
        cells = "".join(
            f"<td>{data[col][i] if data[col][i] is not None else '—'}</td>"
            for col in cols
        )
        rows_html.append(f"<tr>{cells}</tr>")

    return f'<table class="{css_class}">{"".join(rows_html)}</table>'


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title} — Backtest Report</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d29;
    --surface2: #242837;
    --border: #2e3348;
    --text: #e8eaed;
    --text2: #9aa0b4;
    --accent: #6c8cff;
    --accent2: #4ecdc4;
    --positive: #4caf50;
    --negative: #ef5350;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 32px 40px;
    line-height: 1.6;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}

  /* Header */
  .header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 24px;
    margin-bottom: 32px;
  }}
  .header h1 {{
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
  }}
  .header .subtitle {{
    font-size: 14px;
    color: var(--text2);
  }}
  .badge {{
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 12px;
    color: var(--accent);
    margin-right: 8px;
  }}

  /* Highlight cards */
  .highlights {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 32px;
  }}
  .highlight-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
  }}
  .highlight-card .label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text2);
    margin-bottom: 4px;
  }}
  .highlight-card .value {{
    font-size: 22px;
    font-weight: 700;
  }}
  .highlight-card .value.pos {{ color: var(--positive); }}
  .highlight-card .value.neg {{ color: var(--negative); }}

  /* Section */
  .section {{
    margin-bottom: 36px;
  }}
  .section h2 {{
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}

  /* Tables */
  table.metrics {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  table.metrics th {{
    text-align: left;
    padding: 10px 14px;
    background: var(--surface2);
    color: var(--text2);
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid var(--border);
  }}
  table.metrics td {{
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }}
  table.metrics tr:hover td {{
    background: var(--surface);
  }}
  table.metrics td:first-child {{
    color: var(--text2);
    font-weight: 500;
  }}

  /* Charts */
  .chart-img {{
    width: 100%;
    border-radius: 8px;
    margin-bottom: 20px;
  }}

  /* Footer */
  .footer {{
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    font-size: 12px;
    color: var(--text2);
    text-align: center;
  }}
  .footer a {{
    color: var(--accent);
    text-decoration: none;
  }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <h1>{title}</h1>
    <div class="subtitle">
      <span class="badge">Backtest Report</span>
      {date_range} &nbsp;·&nbsp; {n_days} trading days
    </div>
  </div>

  <!-- Highlight Cards -->
  <div class="highlights">
    {highlight_cards}
  </div>

  <!-- Performance Metrics -->
  <div class="section">
    <h2>Performance Metrics</h2>
    {metrics_table}
  </div>

  <!-- Top Drawdowns -->
  {drawdown_section}

  <!-- Regime Analysis -->
  {regime_section}

  <!-- Day-of-Week Statistics -->
  <div class="section">
    <h2>Day-of-Week Statistics</h2>
    {dow_table}
  </div>

  <!-- Charts -->
  {chart_sections}

  <div class="footer">
    Generated by <a href="https://github.com/katsu1110/katsustats">katsustats</a>
  </div>

</div>
</body>
</html>
"""


def full(
    pnl: DataFrameLike,
    base_pnl: DataFrameLike | None = None,
    rf: float = 0.0,
    periods: int = 252,
    figsize_main: tuple = (12, 5),
    figsize_small: tuple = (12, 4),
    show: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Generate a full backtest report with metrics and plots.

    Args:
        pnl: Polars or pandas DataFrame with ["date", "pnl"] columns (daily returns).
        base_pnl: Optional benchmark DataFrame with same schema.
        rf: Risk-free rate (annualized, default 0.0).
        periods: Trading days per year (default 252).
        figsize_main: Figure size for main charts.
        figsize_small: Figure size for smaller charts.
        show: Whether to display plots inline (default True).
        verbose: Whether to print metric tables to stdout (default True).

    Returns:
        dict with keys: "metrics", "drawdowns", "dow_stats", "figures"
    """
    # Validate inputs
    pnl = ensure_polars(pnl, name="pnl")
    assert "date" in pnl.columns, "pnl must have a 'date' column"
    assert "pnl" in pnl.columns, "pnl must have a 'pnl' column"
    if base_pnl is not None:
        base_pnl = ensure_polars(base_pnl, name="base_pnl")
        assert "date" in base_pnl.columns, "base_pnl must have a 'date' column"
        assert "pnl" in base_pnl.columns, "base_pnl must have a 'pnl' column"

    # Sort by date and assert one row per date
    pnl = pnl.sort("date")
    assert pnl["date"].n_unique() == pnl.height, (
        "pnl must have one row per date; pass pre-aggregated portfolio-level returns"
    )
    if base_pnl is not None:
        base_pnl = base_pnl.sort("date")

    # ── 1. Metrics Summary ──────────────────────────────────────────
    summary = stats.summary_metrics_raw(pnl, base_pnl, rf, periods)
    metrics = stats.summary_metrics(pnl, base_pnl, rf, periods)
    if verbose:
        _print_df(metrics, "Performance Metrics")

    # ── 2. Top Drawdowns ────────────────────────────────────────────
    dd = stats.drawdown_details(pnl)
    if verbose and dd.height > 0:
        _print_df(dd, "Top 5 Drawdowns")

    # ── 3. Day-of-Week Stats ────────────────────────────────────────
    dow = stats.day_of_week_stats(pnl)
    if verbose:
        _print_df(dow, "Day-of-Week Statistics")

    # ── 4. Plots ────────────────────────────────────────────────────
    figures: dict[str, plt.Figure] = {}

    # Cumulative Returns
    fig = plots.plot_cumulative_returns(pnl, base_pnl, figsize=figsize_main)
    figures["cumulative_returns"] = fig
    if show:
        fig.show()

    # Drawdown
    fig = plots.plot_drawdown(pnl, figsize=figsize_small)
    figures["drawdown"] = fig
    if show:
        fig.show()

    # Monthly Heatmap
    fig = plots.plot_monthly_heatmap(pnl, figsize=figsize_main)
    figures["monthly_heatmap"] = fig
    if show:
        fig.show()

    # Yearly Returns
    fig = plots.plot_yearly_returns(pnl, base_pnl, figsize=figsize_main)
    figures["yearly_returns"] = fig
    if show:
        fig.show()

    # Return Distribution
    fig = plots.plot_return_distribution(pnl, base_pnl, figsize=figsize_main)
    figures["distribution"] = fig
    if show:
        fig.show()

    # Rolling Sharpe
    fig = plots.plot_rolling_sharpe(pnl, base_pnl, figsize=figsize_small)
    figures["rolling_sharpe"] = fig
    if show:
        fig.show()

    # Rolling Volatility
    fig = plots.plot_rolling_volatility(pnl, base_pnl, figsize=figsize_small)
    figures["rolling_volatility"] = fig
    if show:
        fig.show()

    # Day-of-Week Analysis
    fig = plots.plot_dow_returns(pnl)
    figures["dow_returns"] = fig
    if show:
        fig.show()

    return {
        "summary": summary,
        "metrics": metrics,
        "drawdowns": dd,
        "dow_stats": dow,
        "figures": figures,
    }


def html(
    pnl: DataFrameLike,
    base_pnl: DataFrameLike | None = None,
    rf: float = 0.0,
    periods: int = 252,
    title: str = "Strategy",
    output: str | None = None,
) -> str:
    """
    Generate a self-contained HTML backtest report.

    Args:
        pnl: Polars or pandas DataFrame with ["date", "pnl"] columns (daily returns).
        base_pnl: Optional benchmark DataFrame with same schema.
        rf: Risk-free rate (annualized, default 0.0).
        periods: Trading days per year (default 252).
        title: Report title (default "Strategy").
        output: File path to write HTML. If None, only returns the HTML string.

    Returns:
        The rendered HTML string.
    """
    # Use plt.switch_backend() (safe after pyplot import) instead of matplotlib.use()
    orig_backend = matplotlib.get_backend()
    plt.switch_backend("agg")
    try:
        return _build_html(pnl, base_pnl, rf, periods, title, output)
    finally:
        plt.switch_backend(orig_backend)


def _build_html(
    pnl: DataFrameLike,
    base_pnl: DataFrameLike | None,
    rf: float,
    periods: int,
    title: str,
    output: str | None,
) -> str:
    """Internal: build the HTML report string."""
    # Validate
    pnl = ensure_polars(pnl, name="pnl")
    assert "date" in pnl.columns, "pnl must have a 'date' column"
    assert "pnl" in pnl.columns, "pnl must have a 'pnl' column"
    if base_pnl is not None:
        base_pnl = ensure_polars(base_pnl, name="base_pnl")
        assert "date" in base_pnl.columns, "base_pnl must have a 'date' column"
        assert "pnl" in base_pnl.columns, "base_pnl must have a 'pnl' column"

    pnl = pnl.sort("date")
    assert pnl["date"].n_unique() == pnl.height, (
        "pnl must have one row per date; pass pre-aggregated portfolio-level returns"
    )
    if base_pnl is not None:
        base_pnl = base_pnl.sort("date")

    # ── Metadata ────────────────────────────────────────────────────
    dates = pnl.get_column("date")
    date_start = str(dates.min())
    date_end = str(dates.max())
    n_days = len(dates)
    date_range = f"{date_start} → {date_end}"

    # ── Metrics ─────────────────────────────────────────────────────
    summary = stats.summary_metrics_raw(pnl, base_pnl, rf, periods)
    metrics_df = stats.summary_metrics(pnl, base_pnl, rf, periods)
    metrics_table = _df_to_html_table(metrics_df)

    # ── Headline cards ──────────────────────────────────────────────
    highlight_defs = [
        ("Total Return", summary["total_return"], True),
        ("CAGR", summary["cagr"], True),
        ("Sharpe", summary["sharpe"], False),
        ("Max Drawdown", summary["max_drawdown"], True),
        ("Volatility", summary["volatility"], True),
        ("Win Rate", summary["win_rate"], True),
    ]
    cards: list[str] = []
    for label, val, is_pct in highlight_defs:
        fmt_val = f"{val:.2%}" if is_pct else f"{val:.2f}"
        css_cls = "pos" if val > 0 else ("neg" if val < 0 else "")
        # Max drawdown is always negative, invert color logic
        if label == "Max Drawdown":
            css_cls = "neg" if val < -0.1 else "pos"
        cards.append(
            f'<div class="highlight-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value {css_cls}">{fmt_val}</div>'
            f"</div>"
        )
    highlight_cards = "\n    ".join(cards)

    # ── Drawdowns ───────────────────────────────────────────────────
    dd_df = stats.drawdown_details(pnl)
    if dd_df.height > 0:
        dd_table = _df_to_html_table(dd_df)
        drawdown_section = (
            f'<div class="section"><h2>Top Drawdowns</h2>{dd_table}</div>'
        )
    else:
        drawdown_section = ""

    # ── Regime Analysis ─────────────────────────────────────────────
    if base_pnl is not None:
        regime_df = stats.regime_stats(pnl, base_pnl, periods=periods)
        regime_table = _df_to_html_table(regime_df)
        regime_section = (
            f'<div class="section"><h2>Regime Analysis</h2>{regime_table}</div>'
        )
    else:
        regime_section = ""

    # ── Day-of-Week ─────────────────────────────────────────────────
    dow_df = stats.day_of_week_stats(pnl)
    dow_table = _df_to_html_table(dow_df)

    # ── Charts ──────────────────────────────────────────────────────
    chart_specs = [
        ("Cumulative Returns", plots.plot_cumulative_returns(pnl, base_pnl)),
        ("Drawdown", plots.plot_drawdown(pnl)),
        ("Monthly Returns Heatmap", plots.plot_monthly_heatmap(pnl)),
        ("Yearly Returns", plots.plot_yearly_returns(pnl, base_pnl)),
        ("Daily Return Distribution", plots.plot_return_distribution(pnl, base_pnl)),
        ("Rolling Sharpe", plots.plot_rolling_sharpe(pnl, base_pnl)),
        ("Rolling Volatility", plots.plot_rolling_volatility(pnl, base_pnl)),
        ("Day-of-Week Analysis", plots.plot_dow_returns(pnl)),
    ]

    chart_html_parts: list[str] = []
    for chart_title, fig in chart_specs:
        b64 = _fig_to_base64(fig)
        chart_html_parts.append(
            f'<div class="section">'
            f"<h2>{chart_title}</h2>"
            f'<img class="chart-img" src="data:image/png;base64,{b64}" '
            f'alt="{chart_title}"/>'
            f"</div>"
        )
    chart_sections = "\n  ".join(chart_html_parts)

    # ── Render ──────────────────────────────────────────────────────
    rendered = _HTML_TEMPLATE.format(
        title=title,
        date_range=date_range,
        n_days=n_days,
        highlight_cards=highlight_cards,
        metrics_table=metrics_table,
        drawdown_section=drawdown_section,
        regime_section=regime_section,
        dow_table=dow_table,
        chart_sections=chart_sections,
    )

    if output is not None:
        with open(output, "w", encoding="utf-8") as f:
            f.write(rendered)

    return rendered
