"""katsustats — A modernized backtest report module powered by Polars.

Usage:
    import katsustats
    katsustats.reports.full(returns, benchmark)              # console + plots
    katsustats.reports.html(returns, output="report.html")   # HTML report
    katsustats.reports.json(returns, output="report.json")   # JSON report
    katsustats.reports.markdown(returns, output="report.md") # Markdown report

    # Lower-level modules:
    katsustats.stats.sharpe(returns)
    katsustats.plots.plot_snapshot(returns)
"""

from __future__ import annotations

from . import plots, reports, stats

__version__ = "0.10.0"

__all__ = ["plots", "reports", "stats"]
