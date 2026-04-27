"""
katsustats — A modernized backtest report module powered by Polars.

Usage:
    import katsustats
    katsustats.reports.full(returns, benchmark)              # console + plots
    katsustats.reports.html(returns, output="report.html")   # HTML report

    # Flat imports also work:
    from katsustats import sharpe, plot_cumulative_returns, html
"""

from __future__ import annotations

from . import plots, reports, stats  # noqa: F401

# plots
from .plots import (
    plot_cumulative_returns,
    plot_dow_returns,
    plot_drawdown,
    plot_drawdown_periods,
    plot_eoy_returns,
    plot_monthly_heatmap,
    plot_return_distribution,
    plot_returns_vs_benchmark,
    plot_rolling_sharpe,
    plot_rolling_volatility,
    plot_yearly_returns,
)

# reports
from .reports import full, html

# stats
from .stats import (
    alpha_beta,
    avg_loss,
    avg_win,
    best_day,
    best_month,
    best_year,
    cagr,
    calmar,
    common_sense_ratio,
    consecutive_losses,
    consecutive_wins,
    correlation,
    cvar,
    day_of_week_stats,
    drawdown_details,
    excess_return,
    exposure,
    information_ratio,
    kurtosis,
    max_drawdown,
    period_performance,
    period_performance_raw,
    positive_months_pct,
    positive_years_pct,
    profit_factor,
    recovery_factor,
    regime_stats,
    risk_of_ruin,
    rolling_sharpe,
    rolling_volatility,
    sharpe,
    skewness,
    sortino,
    summary_metrics,
    summary_metrics_raw,
    tail_ratio,
    total_return,
    value_at_risk,
    volatility,
    win_rate,
    worst_day,
    worst_month,
    worst_year,
)

__version__ = "0.2.4"

__all__ = [
    # submodules
    "plots",
    "reports",
    "stats",
    # stats
    "alpha_beta",
    "avg_loss",
    "avg_win",
    "best_day",
    "best_month",
    "best_year",
    "cagr",
    "calmar",
    "common_sense_ratio",
    "consecutive_losses",
    "consecutive_wins",
    "correlation",
    "cvar",
    "day_of_week_stats",
    "drawdown_details",
    "excess_return",
    "exposure",
    "information_ratio",
    "kurtosis",
    "max_drawdown",
    "period_performance",
    "period_performance_raw",
    "positive_months_pct",
    "positive_years_pct",
    "profit_factor",
    "regime_stats",
    "recovery_factor",
    "risk_of_ruin",
    "rolling_sharpe",
    "rolling_volatility",
    "sharpe",
    "skewness",
    "sortino",
    "summary_metrics",
    "tail_ratio",
    "summary_metrics_raw",
    "total_return",
    "value_at_risk",
    "volatility",
    "win_rate",
    "worst_day",
    "worst_month",
    "worst_year",
    # plots
    "plot_cumulative_returns",
    "plot_dow_returns",
    "plot_drawdown",
    "plot_drawdown_periods",
    "plot_eoy_returns",
    "plot_monthly_heatmap",
    "plot_return_distribution",
    "plot_returns_vs_benchmark",
    "plot_rolling_sharpe",
    "plot_rolling_volatility",
    "plot_yearly_returns",
    # reports
    "full",
    "html",
]
