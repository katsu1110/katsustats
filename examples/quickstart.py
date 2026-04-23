"""
Quickstart: generate a full backtest report from a synthetic return series.

Run with:
    uv run python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

import katsustats

# Reproducible synthetic daily returns: ~+15% CAGR, 1% daily vol
rng = np.random.default_rng(seed=0)
dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2022, 12, 31), "1d", eager=True)
returns = rng.normal(loc=0.0006, scale=0.01, size=len(dates))

pnl = pl.DataFrame({"date": dates, "pnl": returns.tolist()})

# Generate full report (prints metrics tables; pass show=True to open chart windows)
results = katsustats.reports.full(pnl, show=False)

print("\nKeys in results:", list(results.keys()))
print("\nMetrics shape:", results["metrics"].shape)
print("\nTop drawdown:")
print(results["drawdowns"].head(1))
