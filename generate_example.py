from datetime import date, timedelta

import numpy as np
import polars as pl

import katsustats.reports as reports

# 1. Generate 3 years of daily returns
rng = np.random.default_rng(42)
n = 756  # ~3 years
start = date(2021, 1, 1)
dates = [start + timedelta(days=int(i)) for i in range(n)]

# Strategy: ~12% annualized return, 15% volatility
returns = pl.DataFrame({"date": dates, "returns": rng.normal(0.0005, 0.010, n)})

# Benchmark: ~8% annualized return, 12% volatility
bm_rng = np.random.default_rng(99)
benchmark = pl.DataFrame({"date": dates, "returns": bm_rng.normal(0.0003, 0.008, n)})

# 2. Generate HTML Report
output_path = "example_report.html"
reports.html(
    returns,
    benchmark=benchmark,
    title="Alpha Strategy V1",
    output=output_path,
    monte_carlo=True,
)
print(f"Report successfully generated at: {output_path}")
