"""
Benchmark comparison: alpha, beta, correlation, and information ratio.

Passing a benchmark to reports.full() adds benchmark-relative metrics to the
summary table (Alpha, Beta, Correlation, Information Ratio, Excess Return).

Run with:
    uv run python examples/with_benchmark.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

import katsustats


def main() -> None:
    rng = np.random.default_rng(seed=0)
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2022, 12, 31), "1d", eager=True)
    n = len(dates)

    # Strategy: higher return, higher vol
    strategy_returns = rng.normal(loc=0.0006, scale=0.012, size=n)
    # Benchmark: lower return, lower vol (simulate a broad index)
    bench_returns = rng.normal(loc=0.0003, scale=0.008, size=n)

    returns = pl.DataFrame({"date": dates, "returns": strategy_returns.tolist()})
    benchmark = pl.DataFrame({"date": dates, "returns": bench_returns.tolist()})

    results = katsustats.reports.full(returns, benchmark=benchmark, show=False)

    # The metrics DataFrame includes benchmark-relative columns when benchmark is given
    print("\nAll metrics:")
    print(results["metrics"])


if __name__ == "__main__":
    main()
