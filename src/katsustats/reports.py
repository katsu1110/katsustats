"""
katsustats.reports — Report generation combining metrics and plots.

Primary entry points:
    katsustats.reports.full(pnl, base_pnl)
    katsustats.reports.html(pnl, base_pnl, output="report.html")
"""

from __future__ import annotations

import base64
import datetime as _dt
import io
import math

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
            max_w = max(max_w, len(_format_cell(col, val)))
        widths[col] = max_w + 2

    # Header
    header = "  ".join(str(col).rjust(widths[col]) for col in cols)
    print(header)
    print("  ".join("-" * widths[col] for col in cols))

    # Rows
    n_rows = len(data[cols[0]])
    for i in range(n_rows):
        row = "  ".join(
            _format_cell(col, data[col][i]).rjust(widths[col]) for col in cols
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


_PCT_COLS = {
    "cagr",
    "max_drawdown",
    "max_dd",
    "win_rate",
    "mean_return",
    "total_return",
    "volatility",
}
_INT_COLS = {"n_days", "count", "dow", "drawdown_days", "recovery_days"}


def _format_cell(col: str, val: object) -> str:
    """Format a table cell value based on its column name."""
    if val is None:
        return "—"
    if isinstance(val, float) and math.isnan(val):
        return "—"
    if isinstance(val, (_dt.datetime, _dt.date)):
        return val.isoformat()[:10]
    if col in _PCT_COLS:
        return f"{val:.2%}"
    if col in _INT_COLS:
        return str(int(val))  # type: ignore[arg-type]
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _df_to_html_table(df: pl.DataFrame, *, css_class: str = "metrics") -> str:
    """Convert a Polars DataFrame to a styled HTML <table> string."""
    cols = df.columns
    data = df.to_dict(as_series=False)
    n_rows = len(data[cols[0]])

    rows_html: list[str] = []
    header_cells = "".join(f"<th>{col}</th>" for col in cols)
    rows_html.append(f"<tr>{header_cells}</tr>")
    for i in range(n_rows):
        cells = "".join(f"<td>{_format_cell(col, data[col][i])}</td>" for col in cols)
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
    --bg: #f8f9fa;
    --surface: #ffffff;
    --surface2: #f1f3f5;
    --border: #dee2e6;
    --text: #212529;
    --text2: #6c757d;
    --accent: #2563eb;
    --accent2: #0d9488;
    --positive: #2e7d32;
    --negative: #c62828;
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

  /* Sub-heading inside grid cells */
  .sub-h2 {{
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 12px;
    padding-bottom: 6px;
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
    margin-bottom: 0;
  }}

  /* Two-column layout for Key Performance block */
  .grid-keyperf {{
    display: grid;
    grid-template-columns: minmax(0, 0.85fr) minmax(0, 1.15fr);
    gap: 24px;
    align-items: start;
  }}

  /* Stacked items in right column */
  .stack > * + * {{
    margin-top: 16px;
  }}

  /* Two-column chart grid */
  .charts-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 24px;
    margin-bottom: 36px;
  }}
  .charts-grid .section {{
    margin-bottom: 0;
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

  @media (max-width: 900px) {{
    .grid-keyperf, .charts-grid {{ grid-template-columns: 1fr; }}
    body {{ padding: 20px 16px; }}
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

  <!-- Hero: Cumulative Returns (full-width) -->
  {hero_chart}

  <!-- Key Performance: Metrics table (left) | Period + Yearly + Drawdown (right) -->
  {key_performance_block}

  <!-- Top Drawdowns (full-width) -->
  {top_drawdowns_section}

  <!-- Charts Grid: paired two-column layout -->
  {charts_grid_block}

  <!-- Regime Analysis (full-width, benchmark only) -->
  {regime_section}

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

    # Sort by date after normalization.
    # ensure_polars() already compounds duplicate same-date rows with a warning,
    # so this uniqueness check is a defensive sanity check.
    pnl = pnl.sort("date")
    assert pnl["date"].n_unique() == pnl.height, (
        "pnl dates are expected to be unique after ensure_polars() normalization "
        "and compounding of duplicate same-date rows; if duplicates still exist, "
        "this indicates an unexpected post-normalization state or a bug. "
        "Please report this issue if it persists."
    )
    if base_pnl is not None:
        base_pnl = base_pnl.sort("date")

    # ── 1. Metrics Summary ──────────────────────────────────────────
    summary = stats.summary_metrics_raw(pnl, base_pnl, rf, periods)
    metrics = stats.summary_metrics(pnl, base_pnl, rf, periods)
    if verbose:
        _print_df(metrics, "Performance Metrics")
        _print_df(stats.period_performance(pnl, base_pnl), "Period Performance")

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

    def _handle_fig(name: str, fg: plt.Figure) -> None:
        figures[name] = fg
        if show:
            fg.show()
        else:
            plt.close(fg)

    _handle_fig(
        "cumulative_returns",
        plots.plot_cumulative_returns(pnl, base_pnl, figsize=figsize_main),
    )
    _handle_fig("drawdown", plots.plot_drawdown(pnl, figsize=figsize_small))
    _handle_fig(
        "monthly_heatmap", plots.plot_monthly_heatmap(pnl, figsize=figsize_main)
    )
    _handle_fig(
        "yearly_returns", plots.plot_yearly_returns(pnl, base_pnl, figsize=figsize_main)
    )
    _handle_fig(
        "distribution",
        plots.plot_return_distribution(pnl, base_pnl, figsize=figsize_main),
    )
    _handle_fig(
        "rolling_sharpe",
        plots.plot_rolling_sharpe(pnl, base_pnl, figsize=figsize_small),
    )
    _handle_fig(
        "rolling_volatility",
        plots.plot_rolling_volatility(pnl, base_pnl, figsize=figsize_small),
    )
    _handle_fig("dow_returns", plots.plot_dow_returns(pnl))

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

    # Sort by date after normalization.
    # ensure_polars() already compounds duplicate same-date rows with a warning,
    # so this uniqueness check is a defensive sanity check.
    pnl = pnl.sort("date")
    assert pnl["date"].n_unique() == pnl.height, (
        "Expected `pnl` to have unique dates after `ensure_polars()` "
        "normalization/compounding. If this fails, check the input for "
        "duplicate same-date rows or investigate whether normalization "
        "did not run as expected."
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

    # ── Hero chart: Cumulative Returns (full-width) ─────────────────
    hero_b64 = _fig_to_base64(
        plots.plot_cumulative_returns(pnl, base_pnl, figsize=(12, 5))
    )
    hero_chart = (
        f'<div class="section">'
        f"<h2>Cumulative Returns</h2>"
        f'<img class="chart-img" src="data:image/png;base64,{hero_b64}" '
        f'alt="Cumulative Returns"/>'
        f"</div>"
    )

    # ── Key Performance block (2-column) ────────────────────────────
    # Left: summary metrics table  |  Right: period table + compact charts
    period_df = stats.period_performance(pnl, base_pnl)
    period_html = _df_to_html_table(period_df)

    yearly_b64 = _fig_to_base64(
        plots.plot_yearly_returns(pnl, base_pnl, figsize=(8, 3))
    )
    dd_compact_b64 = _fig_to_base64(plots.plot_drawdown(pnl, figsize=(8, 3)))
    rolling_sharpe_compact_b64 = _fig_to_base64(
        plots.plot_rolling_sharpe(pnl, base_pnl, figsize=(8, 3))
    )

    key_performance_block = (
        f'<div class="section">'
        f'<div class="grid-keyperf">'
        f'<div>'
        f'<h3 class="sub-h2">Performance Metrics</h3>'
        f"{metrics_table}"
        f"</div>"
        f'<div class="stack">'
        f'<div><h3 class="sub-h2">Period Performance</h3>{period_html}</div>'
        f'<div><img class="chart-img" src="data:image/png;base64,{yearly_b64}"'
        f' alt="Yearly Returns"/></div>'
        f'<div><img class="chart-img" src="data:image/png;base64,{dd_compact_b64}"'
        f' alt="Drawdown"/></div>'
        f'<div><img class="chart-img" src="data:image/png;base64,{rolling_sharpe_compact_b64}"'
        f' alt="Rolling Sharpe"/></div>'
        f"</div>"
        f"</div>"
        f"</div>"
    )

    # ── Top Drawdowns (full-width) ───────────────────────────────────
    dd_df = stats.drawdown_details(pnl)
    if dd_df.height > 0:
        dd_table = _df_to_html_table(dd_df)
        top_drawdowns_section = (
            f'<div class="section"><h2>Top Drawdowns</h2>{dd_table}</div>'
        )
    else:
        top_drawdowns_section = ""

    # ── Charts grid (2-column pairs) ────────────────────────────────
    def _grid_section(heading: str, b64: str) -> str:
        return (
            f'<div class="section">'
            f"<h2>{heading}</h2>"
            f'<img class="chart-img" src="data:image/png;base64,{b64}" alt="{heading}"/>'
            f"</div>"
        )

    heatmap_b64 = _fig_to_base64(plots.plot_monthly_heatmap(pnl, figsize=(8, 4)))
    dist_b64 = _fig_to_base64(
        plots.plot_return_distribution(pnl, base_pnl, figsize=(8, 4))
    )
    sharpe_b64 = _fig_to_base64(
        plots.plot_rolling_sharpe(pnl, base_pnl, figsize=(8, 4))
    )
    vol_b64 = _fig_to_base64(
        plots.plot_rolling_volatility(pnl, base_pnl, figsize=(8, 4))
    )
    dow_df = stats.day_of_week_stats(pnl)
    dow_table_html = _df_to_html_table(dow_df)
    dow_chart_b64 = _fig_to_base64(plots.plot_dow_returns(pnl, figsize=(8, 4)))

    charts_grid_block = (
        f'<div class="charts-grid">'
        f"{_grid_section('Monthly Returns Heatmap', heatmap_b64)}"
        f"{_grid_section('Daily Return Distribution', dist_b64)}"
        f"{_grid_section('Rolling Sharpe', sharpe_b64)}"
        f"{_grid_section('Rolling Volatility', vol_b64)}"
        f'<div class="section"><h2>Day-of-Week Statistics</h2>{dow_table_html}</div>'
        f"{_grid_section('Day-of-Week Analysis', dow_chart_b64)}"
        f"</div>"
    )

    # ── Regime Analysis (full-width, benchmark only) ─────────────────
    if base_pnl is not None:
        regime_df = stats.regime_stats(pnl, base_pnl, periods=periods)
        regime_df = regime_df.filter(pl.col("n_days") > 0)
        if regime_df.height > 0:
            regime_table = _df_to_html_table(regime_df)
            regime_section = (
                f'<div class="section"><h2>Regime Analysis</h2>{regime_table}</div>'
            )
        else:
            regime_section = ""
    else:
        regime_section = ""

    # ── Render ──────────────────────────────────────────────────────
    rendered = _HTML_TEMPLATE.format(
        title=title,
        date_range=date_range,
        n_days=n_days,
        highlight_cards=highlight_cards,
        hero_chart=hero_chart,
        key_performance_block=key_performance_block,
        top_drawdowns_section=top_drawdowns_section,
        charts_grid_block=charts_grid_block,
        regime_section=regime_section,
    )

    if output is not None:
        with open(output, "w", encoding="utf-8") as f:
            f.write(rendered)

    return rendered
