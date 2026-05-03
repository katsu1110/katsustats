"""
katsustats.reports — Report generation combining metrics and plots.

Primary entry points:
    katsustats.reports.full(returns, benchmark)
    katsustats.reports.html(returns, benchmark, output="report.html")
"""

from __future__ import annotations

import base64
import datetime as _dt
import io
import json as _json
import math

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

from . import plots, stats
from ._dataframe import DataFrameLike, ensure_polars

_COMPARISON_KEYS = {
    "alpha",
    "beta",
    "correlation",
    "information_ratio",
    "excess_return",
}

_MARKDOWN_HEADLINE_SPECS = [
    ("Total Return", "total_return", "pct"),
    ("CAGR", "cagr", "pct"),
    ("Sharpe", "sharpe", "float"),
    ("Max Drawdown", "max_drawdown", "pct"),
    ("Volatility", "volatility", "pct"),
    ("Win Rate", "win_rate", "pct"),
]


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


def _validate_and_sort(
    returns: DataFrameLike,
    benchmark: DataFrameLike | None,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Normalise, validate, and sort inputs; return (returns, benchmark) as Polars frames."""
    returns = ensure_polars(returns, name="returns")
    assert "date" in returns.columns, "returns must have a 'date' column"
    assert "returns" in returns.columns, "returns must have a 'returns' column"
    if benchmark is not None:
        benchmark = ensure_polars(benchmark, name="benchmark")
        assert "date" in benchmark.columns, "benchmark must have a 'date' column"
        assert "returns" in benchmark.columns, "benchmark must have a 'returns' column"
    returns = returns.sort("date")
    assert returns["date"].n_unique() == returns.height, (
        "Expected `returns` to have unique dates after `ensure_polars()` "
        "normalization/compounding. If this fails, check the input for "
        "duplicate same-date rows or investigate whether normalization "
        "did not run as expected."
    )
    if benchmark is not None:
        benchmark = benchmark.sort("date")
    return returns, benchmark


def _json_safe_value(value: object) -> object:
    """Convert Python / Polars values into JSON-safe primitives."""
    if value is None:
        return None
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _df_to_records(df: pl.DataFrame) -> list[dict[str, object]]:
    """Convert a Polars DataFrame to a list of JSON-safe row dicts."""
    return [
        {key: _json_safe_value(value) for key, value in row.items()}
        for row in df.to_dicts()
    ]


def _json_dumps(payload: dict[str, object]) -> str:
    """Serialize report payload to pretty JSON."""
    return _json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)


def _markdown_escape(value: object) -> str:
    """Escape Markdown table delimiters in string values."""
    return str(value).replace("|", r"\|")


def _format_markdown_value(value: object, fmt: str) -> str:
    """Format a value for Markdown output."""
    if value is None:
        return "—"
    if fmt == "pct":
        return f"{float(value):.2%}"
    if fmt == "float":
        return f"{float(value):.2f}"
    if fmt == "int":
        return str(int(value))
    return _markdown_escape(value)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a Markdown table from headers and preformatted rows."""
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def _markdown_report(payload: dict[str, object]) -> str:
    """Render a structured payload as a Markdown backtest summary."""
    metadata = payload["metadata"]
    strategy = payload["strategy"]
    benchmark = payload["benchmark"]
    comparison = payload["comparison"]
    drawdowns = payload["drawdowns"]
    dow_stats = payload["day_of_week_stats"]
    regime_analysis = payload["regime_analysis"]

    summary = strategy["summary"]
    lines = [
        f"# {_markdown_escape(metadata['title'])} Backtest Summary",
        "",
        "## Overview",
        f"- Period: {_markdown_escape(metadata['start_date'])} to {_markdown_escape(metadata['end_date'])}",
        f"- Trading days: {_format_markdown_value(metadata['n_days'], 'int')}",
        f"- Benchmark: {'Yes' if metadata['has_benchmark'] else 'No'}",
        f"- Risk-free rate: {_format_markdown_value(metadata['rf'], 'pct')}",
        f"- Periods per year: {_format_markdown_value(metadata['periods'], 'int')}",
        "",
        "## Headline Metrics",
    ]
    for label, key, fmt in _MARKDOWN_HEADLINE_SPECS:
        lines.append(f"- {label}: {_format_markdown_value(summary.get(key), fmt)}")

    lines.extend(["", "## Performance Metrics"])
    perf_headers = ["Metric", "Strategy"]
    if benchmark is not None:
        perf_headers.append("Benchmark")
    perf_rows: list[list[str]] = []
    benchmark_summary = None if benchmark is None else benchmark["summary"]
    for label, key, fmt in stats._SUMMARY_METRIC_SPECS:
        row = [
            label,
            _format_markdown_value(summary.get(key), fmt),
        ]
        if benchmark_summary is not None:
            row.append(_format_markdown_value(benchmark_summary.get(key), fmt))
        perf_rows.append(row)
    if comparison is not None:
        for label, key, fmt in stats._COMPARISON_METRIC_SPECS:
            perf_rows.append(
                [
                    label,
                    _format_markdown_value(comparison.get(key), fmt),
                    "—",
                ]
            )
    lines.extend(
        [_markdown_table(perf_headers, perf_rows), "", "## Period Performance"]
    )

    period_headers = ["Period", "Strategy"]
    if benchmark is not None:
        period_headers.append("Benchmark")
    period_rows: list[list[str]] = []
    benchmark_periods = None if benchmark is None else benchmark["period_performance"]
    for label in stats._PERIOD_LABELS:
        row = [
            label,
            _format_markdown_value(
                strategy["period_performance"][label].get("strategy"), "pct"
            ),
        ]
        if benchmark_periods is not None:
            row.append(
                _format_markdown_value(benchmark_periods[label].get("benchmark"), "pct")
            )
        period_rows.append(row)
    lines.extend([_markdown_table(period_headers, period_rows), ""])

    if drawdowns:
        lines.extend(["## Top Drawdowns"])
        drawdown_rows = [
            [
                _format_markdown_value(row.get("start"), "str"),
                _format_markdown_value(row.get("trough"), "str"),
                _format_markdown_value(row.get("recovery"), "str"),
                _format_markdown_value(row.get("max_dd"), "pct"),
                _format_markdown_value(row.get("drawdown_days"), "int"),
                _format_markdown_value(row.get("recovery_days"), "int"),
            ]
            for row in drawdowns
        ]
        lines.extend(
            [
                _markdown_table(
                    [
                        "Start",
                        "Trough",
                        "Recovery",
                        "Max DD",
                        "Drawdown Days",
                        "Recovery Days",
                    ],
                    drawdown_rows,
                ),
                "",
            ]
        )

    if dow_stats:
        lines.extend(["## Day-of-Week Statistics"])
        dow_rows = [
            [
                _format_markdown_value(row.get("dow_name"), "str"),
                _format_markdown_value(row.get("mean_return"), "pct"),
                _format_markdown_value(row.get("win_rate"), "pct"),
                _format_markdown_value(row.get("total_return"), "pct"),
                _format_markdown_value(row.get("count"), "int"),
            ]
            for row in dow_stats
        ]
        lines.extend(
            [
                _markdown_table(
                    ["Day", "Mean Return", "Win Rate", "Total Return", "Count"],
                    dow_rows,
                ),
                "",
            ]
        )

    if regime_analysis:
        lines.extend(["## Regime Analysis"])
        regime_rows = [
            [
                _format_markdown_value(row.get("regime"), "str"),
                _format_markdown_value(row.get("n_days"), "int"),
                _format_markdown_value(row.get("cagr"), "pct"),
                _format_markdown_value(row.get("sharpe"), "float"),
                _format_markdown_value(row.get("max_drawdown"), "pct"),
                _format_markdown_value(row.get("win_rate"), "pct"),
            ]
            for row in regime_analysis
        ]
        lines.extend(
            [
                _markdown_table(
                    ["Regime", "Days", "CAGR", "Sharpe", "Max Drawdown", "Win Rate"],
                    regime_rows,
                ),
                "",
            ]
        )

    lines.append("Generated by katsustats")
    return "\n".join(lines).strip() + "\n"


def _metadata_payload(
    returns: pl.DataFrame,
    benchmark: pl.DataFrame | None,
    *,
    title: str,
    rf: float,
    periods: int,
) -> dict[str, object]:
    """Build shared metadata for structured report outputs."""
    dates = returns.get_column("date")
    return {
        "title": title,
        "start_date": _json_safe_value(dates.min()),
        "end_date": _json_safe_value(dates.max()),
        "n_days": returns.height,
        "rf": rf,
        "periods": periods,
        "has_benchmark": benchmark is not None,
    }


def _report_payload(
    returns: pl.DataFrame,
    benchmark: pl.DataFrame | None,
    *,
    title: str,
    rf: float,
    periods: int,
) -> dict[str, object]:
    """Build an AI-friendly structured report payload."""
    benchmark_summary: dict[str, float] | None = None
    benchmark_periods: dict[str, dict[str, float]] | None = None
    comparison: dict[str, float] | None = None
    regime_analysis: list[dict[str, object]] = []

    if benchmark is not None:
        combined_summary = stats.summary_metrics_raw(returns, benchmark, rf, periods)
        strategy_summary = {
            k: v for k, v in combined_summary.items() if k not in _COMPARISON_KEYS
        }
        comparison = {
            key: _json_safe_value(value)
            for key, value in combined_summary.items()
            if key in _COMPARISON_KEYS
        }
        benchmark_summary = stats.summary_metrics_raw(benchmark, None, rf, periods)
        aligned_periods = stats.period_performance_raw(returns, benchmark)
        strategy_periods: dict[str, dict[str, object]] = {}
        benchmark_periods = {}
        for label, values in aligned_periods.items():
            strategy_periods[label] = {"strategy": values["strategy"]}
            benchmark_periods[label] = {"benchmark": values["benchmark"]}
        regime_df = stats.regime_stats(returns, benchmark, periods=periods)
        regime_analysis = _df_to_records(regime_df.filter(pl.col("n_days") > 0))
    else:
        strategy_summary = stats.summary_metrics_raw(returns, None, rf, periods)
        strategy_periods = stats.period_performance_raw(returns, None)

    return {
        "metadata": _metadata_payload(
            returns,
            benchmark,
            title=title,
            rf=rf,
            periods=periods,
        ),
        "strategy": {
            "summary": {
                key: _json_safe_value(value) for key, value in strategy_summary.items()
            },
            "period_performance": {
                label: {key: _json_safe_value(value) for key, value in values.items()}
                for label, values in strategy_periods.items()
            },
        },
        "benchmark": (
            None
            if benchmark is None
            else {
                "summary": {
                    key: _json_safe_value(value)
                    for key, value in benchmark_summary.items()
                },
                "period_performance": {
                    label: {
                        key: _json_safe_value(value)
                        for key, value in values.items()
                        if key == "benchmark"
                    }
                    for label, values in benchmark_periods.items()
                },
            }
        ),
        "comparison": comparison,
        "drawdowns": _df_to_records(stats.drawdown_details(returns)),
        "day_of_week_stats": _df_to_records(stats.day_of_week_stats(returns)),
        "regime_analysis": regime_analysis,
    }


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
  .section-sub-heading {{
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
    returns: DataFrameLike,
    benchmark: DataFrameLike | None = None,
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
        returns: Polars or pandas DataFrame with ["date", "returns"] columns (daily returns).
        benchmark: Optional benchmark DataFrame with same schema.
        rf: Risk-free rate (annualized, default 0.0).
        periods: Trading days per year (default 252).
        figsize_main: Figure size for main charts.
        figsize_small: Figure size for smaller charts.
        show: Whether to display plots inline (default True).
        verbose: Whether to print metric tables to stdout (default True).

    Returns:
        dict with keys: "metrics", "drawdowns", "dow_stats", "figures"
    """
    returns, benchmark = _validate_and_sort(returns, benchmark)

    # ── 1. Metrics Summary ──────────────────────────────────────────
    summary = stats.summary_metrics_raw(returns, benchmark, rf, periods)
    metrics = stats.summary_metrics(returns, benchmark, rf, periods)
    if verbose:
        _print_df(metrics, "Performance Metrics")
        _print_df(stats.period_performance(returns, benchmark), "Period Performance")

    # ── 2. Top Drawdowns ────────────────────────────────────────────
    dd = stats.drawdown_details(returns)
    if verbose and dd.height > 0:
        _print_df(dd, "Top 5 Drawdowns")

    # ── 3. Day-of-Week Stats ────────────────────────────────────────
    dow = stats.day_of_week_stats(returns)
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
        plots.plot_cumulative_returns(returns, benchmark, figsize=figsize_main),
    )
    _handle_fig("drawdown", plots.plot_drawdown(returns, figsize=figsize_small))
    _handle_fig(
        "monthly_heatmap", plots.plot_monthly_heatmap(returns, figsize=figsize_main)
    )
    _handle_fig(
        "yearly_returns",
        plots.plot_yearly_returns(returns, benchmark, figsize=figsize_main),
    )
    _handle_fig(
        "distribution",
        plots.plot_return_distribution(returns, benchmark, figsize=figsize_main),
    )
    _handle_fig(
        "rolling_sharpe",
        plots.plot_rolling_sharpe(returns, benchmark, figsize=figsize_small),
    )
    _handle_fig(
        "rolling_volatility",
        plots.plot_rolling_volatility(returns, benchmark, figsize=figsize_small),
    )
    _handle_fig("dow_returns", plots.plot_dow_returns(returns))

    return {
        "summary": summary,
        "metrics": metrics,
        "drawdowns": dd,
        "dow_stats": dow,
        "figures": figures,
    }


def html(
    returns: DataFrameLike,
    benchmark: DataFrameLike | None = None,
    rf: float = 0.0,
    periods: int = 252,
    title: str = "Strategy",
    output: str | None = None,
) -> str:
    """
    Generate a self-contained HTML backtest report.

    Args:
        returns: Polars or pandas DataFrame with ["date", "returns"] columns (daily returns).
        benchmark: Optional benchmark DataFrame with same schema.
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
        return _build_html(returns, benchmark, rf, periods, title, output)
    finally:
        plt.switch_backend(orig_backend)


def json(
    returns: DataFrameLike,
    benchmark: DataFrameLike | None = None,
    rf: float = 0.0,
    periods: int = 252,
    title: str = "Strategy",
    output: str | None = None,
) -> str:
    """
    Generate an AI-friendly JSON backtest report.

    Args:
        returns: Polars or pandas DataFrame with ["date", "returns"] columns.
        benchmark: Optional benchmark DataFrame with the same schema.
        rf: Risk-free rate (annualized, default 0.0).
        periods: Trading days per year (default 252).
        title: Report title (default "Strategy").
        output: File path to write JSON. If None, only returns the JSON string.

    Returns:
        The rendered JSON string.

        Top-level keys:
            metadata: Report metadata including title, date range, trading-day
                count, risk-free rate, periods, and whether a benchmark exists.
            strategy: Dict with raw numeric `summary` metrics and
                `period_performance` keyed by period label.
            benchmark: None when no benchmark is provided; otherwise a dict with
                benchmark `summary` metrics and benchmark-aligned
                `period_performance` values keyed by period label.
            comparison: None when no benchmark is provided; otherwise raw
                strategy-vs-benchmark comparison metrics such as `alpha`,
                `beta`, `correlation`, `information_ratio`, and
                `excess_return`.
            drawdowns: List of top drawdown rows.
            day_of_week_stats: List of day-of-week summary rows.
            regime_analysis: Empty list when no benchmark is provided or no
                regime rows are available; otherwise a list of regime summary
                rows.

        When a benchmark is provided, both `strategy.period_performance` and
        `benchmark.period_performance` are aligned to the common date overlap so
        the period windows remain directly comparable.
    """
    returns, benchmark = _validate_and_sort(returns, benchmark)

    rendered = _json_dumps(
        _report_payload(
            returns,
            benchmark,
            title=title,
            rf=rf,
            periods=periods,
        )
    )

    if output is not None:
        with open(output, "w", encoding="utf-8") as f:
            f.write(rendered)

    return rendered


def markdown(
    returns: DataFrameLike,
    benchmark: DataFrameLike | None = None,
    rf: float = 0.0,
    periods: int = 252,
    title: str = "Strategy",
    output: str | None = None,
) -> str:
    """
    Generate a Markdown backtest summary for humans and AI agents.

    Args:
        returns: Polars or pandas DataFrame with ["date", "returns"] columns.
        benchmark: Optional benchmark DataFrame with the same schema.
        rf: Risk-free rate (annualized, default 0.0).
        periods: Trading days per year (default 252).
        title: Report title (default "Strategy").
        output: File path to write Markdown. If None, only returns the Markdown.

    Returns:
        A Markdown string with overview, headline metrics, performance tables,
        period performance, drawdowns, day-of-week statistics, and optional
        regime analysis when a benchmark is provided.
    """
    returns, benchmark = _validate_and_sort(returns, benchmark)

    rendered = _markdown_report(
        _report_payload(
            returns,
            benchmark,
            title=title,
            rf=rf,
            periods=periods,
        )
    )

    if output is not None:
        with open(output, "w", encoding="utf-8") as f:
            f.write(rendered)

    return rendered


def _build_html(
    returns: DataFrameLike,
    benchmark: DataFrameLike | None,
    rf: float,
    periods: int,
    title: str,
    output: str | None,
) -> str:
    """Internal: build the HTML report string."""
    returns, benchmark = _validate_and_sort(returns, benchmark)

    # ── Metadata ────────────────────────────────────────────────────
    dates = returns.get_column("date")
    date_start = str(dates.min())
    date_end = str(dates.max())
    n_days = len(dates)
    date_range = f"{date_start} → {date_end}"

    # ── Metrics ─────────────────────────────────────────────────────
    summary = stats.summary_metrics_raw(returns, benchmark, rf, periods)
    metrics_df = stats.summary_metrics(returns, benchmark, rf, periods)
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
        plots.plot_cumulative_returns(returns, benchmark, figsize=(12, 5))
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
    period_df = stats.period_performance(returns, benchmark)
    period_html = _df_to_html_table(period_df)

    yearly_b64 = _fig_to_base64(
        plots.plot_yearly_returns(returns, benchmark, figsize=(8, 3))
    )
    dd_compact_b64 = _fig_to_base64(plots.plot_drawdown(returns, figsize=(8, 3)))
    dd_periods_b64 = _fig_to_base64(
        plots.plot_drawdown_periods(returns, figsize=(8, 3))
    )

    key_performance_block = (
        f'<div class="section">'
        f"<h2>Key Performance</h2>"
        f'<div class="grid-keyperf">'
        f"<div>"
        f'<h3 class="section-sub-heading">Performance Metrics</h3>'
        f"{metrics_table}"
        f"</div>"
        f'<div class="stack">'
        f'<div><h3 class="section-sub-heading">Period Performance</h3>{period_html}</div>'
        f'<div><img class="chart-img" src="data:image/png;base64,{yearly_b64}"'
        f' alt="Yearly Returns"/></div>'
        f'<div><img class="chart-img" src="data:image/png;base64,{dd_compact_b64}"'
        f' alt="Drawdown"/></div>'
        f'<div><img class="chart-img" src="data:image/png;base64,{dd_periods_b64}"'
        f' alt="Drawdown Periods"/></div>'
        f"</div>"
        f"</div>"
        f"</div>"
    )

    # ── Top Drawdowns (full-width) ───────────────────────────────────
    dd_df = stats.drawdown_details(returns)
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

    heatmap_b64 = _fig_to_base64(plots.plot_monthly_heatmap(returns, figsize=(8, 4)))
    dist_b64 = _fig_to_base64(
        plots.plot_return_distribution(returns, benchmark, figsize=(8, 4))
    )
    sharpe_b64 = _fig_to_base64(
        plots.plot_rolling_sharpe(returns, benchmark, figsize=(8, 4))
    )
    vol_b64 = _fig_to_base64(
        plots.plot_rolling_volatility(returns, benchmark, figsize=(8, 4))
    )
    dow_df = stats.day_of_week_stats(returns)
    dow_table_html = _df_to_html_table(dow_df)
    dow_chart_b64 = _fig_to_base64(plots.plot_dow_returns(returns, figsize=(8, 4)))

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
    if benchmark is not None:
        regime_df = stats.regime_stats(returns, benchmark, periods=periods)
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
