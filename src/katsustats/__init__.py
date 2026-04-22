"""
katsustats — A modernized backtest report module powered by Polars.

Usage:
    import katsustats
    katsustats.reports.full(pnl, base_pnl)              # console + plots
    katsustats.reports.html(pnl, output="report.html")   # HTML report

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
    plot_monthly_heatmap,
    plot_return_distribution,
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
    cagr,
    calmar,
    correlation,
    day_of_week_stats,
    drawdown_details,
    excess_return,
    information_ratio,
    kurtosis,
    max_drawdown,
    profit_factor,
    recovery_factor,
    regime_stats,
    rolling_sharpe,
    rolling_volatility,
    sharpe,
    skewness,
    sortino,
    summary_metrics,
    summary_metrics_raw,
    total_return,
    value_at_risk,
    volatility,
    win_rate,
    worst_day,
)

__version__ = "0.1.0"

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
    "cagr",
    "calmar",
    "correlation",
    "day_of_week_stats",
    "drawdown_details",
    "excess_return",
    "information_ratio",
    "kurtosis",
    "max_drawdown",
    "profit_factor",
    "regime_stats",
    "recovery_factor",
    "rolling_sharpe",
    "rolling_volatility",
    "sharpe",
    "skewness",
    "sortino",
    "summary_metrics",
    "summary_metrics_raw",
    "total_return",
    "value_at_risk",
    "volatility",
    "win_rate",
    "worst_day",
    # plots
    "plot_cumulative_returns",
    "plot_dow_returns",
    "plot_drawdown",
    "plot_drawdown_periods",
    "plot_monthly_heatmap",
    "plot_return_distribution",
    "plot_rolling_sharpe",
    "plot_rolling_volatility",
    "plot_yearly_returns",
    # reports
    "full",
    "html",
]
