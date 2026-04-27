"""
HTML report: generate a self-contained HTML file you can open in any browser.

The output is a single .html file with embedded charts (no external dependencies).

Run with:
    uv run python examples/html_report.py
    open report.html   # macOS — or just double-click the file
"""

from __future__ import annotations

import numpy as np
import polars as pl

import katsustats


def main() -> None:
    rng = np.random.default_rng(seed=42)
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2022, 12, 31), "1d", eager=True)
    n = len(dates)

    returns = pl.DataFrame(
        {"date": dates, "returns": rng.normal(loc=0.0005, scale=0.011, size=n).tolist()}
    )
    benchmark = pl.DataFrame(
        {"date": dates, "returns": rng.normal(loc=0.0003, scale=0.008, size=n).tolist()}
    )

    katsustats.reports.html(
        returns,
        benchmark=benchmark,
        title="Demo Strategy",
        output="report.html",
    )

    print("Report written to report.html")


if __name__ == "__main__":
    main()
