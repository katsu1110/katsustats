# Katsustats

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/katsu1110/katsustats/actions/workflows/ci.yml/badge.svg)](https://github.com/katsu1110/katsustats/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Sponsor](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/katsu1110)

A simple backtest tool for your return series, inspired by [quantstats](https://github.com/ranaroussi/quantstats). 

Put your return series, and you get backtest results with visualizations and key metrics.

# How to use

## Installation

```bash
pip install katsustats
```

Or with `uv`:

```bash
uv add katsustats
```

## Data format

`katsustats` accepts either a [Polars](https://pola.rs/) or pandas DataFrame
with two required columns:

| column | type | description |
|--------|------|-------------|
| `date` | `pl.Date` | Trading date |
| `pnl`  | `pl.Float64` | Daily return (e.g. `0.01` = +1%) |

When a pandas DataFrame is passed, `katsustats` converts it to Polars at the
start of processing.

## Basic usage

```python
import polars as pl
import katsustats

# Build your return series
pnl = pl.DataFrame({
    "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2023, 12, 31), "1d", eager=True),
    "pnl": your_daily_returns,   # list / numpy array of floats
})

# Generate the full report (prints metrics + shows all plots)
results = katsustats.reports.full(pnl)
```

Pandas inputs work too:

```python
import pandas as pd

pnl = pd.DataFrame({
    "date": dates,
    "pnl": your_daily_returns,
})

results = katsustats.reports.full(pnl)
```

`results` is a dict with the following keys:

| key | type | description |
|-----|------|-------------|
| `metrics` | `pl.DataFrame` | Summary metrics table |
| `drawdowns` | `pl.DataFrame` | Top-5 drawdown periods |
| `dow_stats` | `pl.DataFrame` | Day-of-week statistics |
| `figures` | `dict[str, Figure]` | All matplotlib figures |

## With a benchmark

```python
benchmark = pl.DataFrame({
    "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2023, 12, 31), "1d", eager=True),
    "pnl": benchmark_daily_returns,
})

results = katsustats.reports.full(pnl, base_pnl=benchmark)
```

When a benchmark is provided, the metrics table also includes **Alpha**, **Beta**, **Correlation**, **Information Ratio**, and **Excess Return**.

## Advanced options

```python
results = katsustats.reports.full(
    pnl,
    base_pnl=benchmark,
    rf=0.04,          # annualized risk-free rate (default 0.0)
    periods=252,      # trading days per year (default 252)
    show=False,       # suppress inline plot display
)
```

## HTML report

Generate a self-contained HTML report (similar to `qs.reports.html()`):

```python
# Save to file
katsustats.reports.html(pnl, base_pnl=benchmark, title="My Strategy", output="report.html")

# Or get HTML string
html_str = katsustats.reports.html(pnl, title="My Strategy")
```

The report includes headline metric cards, performance tables, drawdown analysis, day-of-week statistics, and all 8 charts embedded as images — all in a single `.html` file that works offline.

## Using individual modules

```python
import katsustats

# --- Stats ---
katsustats.stats.total_return(pnl)
katsustats.stats.cagr(pnl)
katsustats.stats.sharpe(pnl, rf=0.0)
katsustats.stats.sortino(pnl)
katsustats.stats.max_drawdown(pnl)
katsustats.stats.calmar(pnl)
katsustats.stats.volatility(pnl)
katsustats.stats.win_rate(pnl)
katsustats.stats.profit_factor(pnl)
katsustats.stats.value_at_risk(pnl, alpha=0.05)

katsustats.stats.drawdown_details(pnl, top_n=5)      # pl.DataFrame
katsustats.stats.day_of_week_stats(pnl)              # pl.DataFrame
katsustats.stats.summary_metrics(pnl, base_pnl)     # pl.DataFrame

# --- Plots ---
katsustats.plots.plot_cumulative_returns(pnl, base_pnl)
katsustats.plots.plot_drawdown(pnl)
katsustats.plots.plot_monthly_heatmap(pnl)
katsustats.plots.plot_yearly_returns(pnl, base_pnl)
katsustats.plots.plot_return_distribution(pnl, base_pnl)
katsustats.plots.plot_rolling_sharpe(pnl, base_pnl)
katsustats.plots.plot_rolling_volatility(pnl, base_pnl)
katsustats.plots.plot_dow_returns(pnl)
```

## Metrics produced

| metric | description |
|--------|-------------|
| Total Return | Compounded return over the full period |
| CAGR | Compound Annual Growth Rate |
| Sharpe Ratio | Annualized risk-adjusted return |
| Sortino Ratio | Sharpe using only downside deviation |
| Max Drawdown | Largest peak-to-trough decline |
| Calmar Ratio | CAGR / \|Max Drawdown\| |
| Volatility (ann.) | Annualized standard deviation |
| Win Rate | % of days with positive returns |
| Profit Factor | Gross profit / gross loss |
| Best / Worst Day | Largest single-day gain / loss |
| Avg Win / Avg Loss | Mean return on winning / losing days |
| Daily VaR (95%) | 5th-percentile daily return |
| Recovery Factor | Total return / \|Max Drawdown\| |
| Skewness / Kurtosis | Distribution shape statistics |
