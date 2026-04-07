"""
katsustats — A modernized backtest report module powered by Polars.

Usage:
    import katsustats
    katsustats.reports.full(pnl, base_pnl)              # console + plots
    katsustats.reports.html(pnl, output="report.html")   # HTML report
"""

from . import reports, stats, plots  # noqa: F401

__version__ = "0.1.0"
