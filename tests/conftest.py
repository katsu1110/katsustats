"""Shared fixtures for katsustats tests."""

from __future__ import annotations

import datetime

import matplotlib
import pandas as pd
import polars as pl
import pytest

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WEEKDAYS = [
    datetime.date(2023, 1, 2),  # Mon
    datetime.date(2023, 1, 3),  # Tue
    datetime.date(2023, 1, 4),  # Wed
    datetime.date(2023, 1, 5),  # Thu
    datetime.date(2023, 1, 6),  # Fri
    datetime.date(2023, 1, 9),  # Mon
    datetime.date(2023, 1, 10),  # Tue
    datetime.date(2023, 1, 11),  # Wed
    datetime.date(2023, 1, 12),  # Thu
    datetime.date(2023, 1, 13),  # Fri
    datetime.date(2023, 1, 16),  # Mon
    datetime.date(2023, 1, 17),  # Tue
    datetime.date(2023, 1, 18),  # Wed
    datetime.date(2023, 1, 19),  # Thu
    datetime.date(2023, 1, 20),  # Fri
    datetime.date(2023, 1, 23),  # Mon
    datetime.date(2023, 1, 24),  # Tue
    datetime.date(2023, 1, 25),  # Wed
    datetime.date(2023, 1, 26),  # Thu
    datetime.date(2023, 1, 27),  # Fri
]

_RETURNS = [
    0.01,
    -0.005,
    0.008,
    -0.012,
    0.003,
    0.015,
    -0.007,
    0.002,
    -0.004,
    0.009,
    -0.011,
    0.006,
    0.013,
    -0.002,
    0.004,
    0.007,
    -0.008,
    0.001,
    -0.003,
    0.011,
]

_BENCH_RETURNS = [
    0.005,
    -0.003,
    0.006,
    -0.008,
    0.002,
    0.009,
    -0.004,
    0.001,
    -0.006,
    0.007,
    -0.006,
    0.004,
    0.008,
    -0.001,
    0.003,
    0.005,
    -0.005,
    0.003,
    -0.002,
    0.006,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """20 weekdays, mix of positive and negative returns."""
    return pl.DataFrame({"date": _WEEKDAYS, "returns": _RETURNS}).with_columns(
        pl.col("date").cast(pl.Date)
    )


@pytest.fixture
def benchmark_df() -> pl.DataFrame:
    """20 weekdays, benchmark returns."""
    return pl.DataFrame({"date": _WEEKDAYS, "returns": _BENCH_RETURNS}).with_columns(
        pl.col("date").cast(pl.Date)
    )


@pytest.fixture
def sample_pandas_df() -> pd.DataFrame:
    """20 weekdays, mix of positive and negative returns, as pandas."""
    return pd.DataFrame({"date": _WEEKDAYS, "returns": _RETURNS})


@pytest.fixture
def benchmark_pandas_df() -> pd.DataFrame:
    """20 weekdays, benchmark returns, as pandas."""
    return pd.DataFrame({"date": _WEEKDAYS, "returns": _BENCH_RETURNS})


@pytest.fixture
def sample_pandas_df_indexed() -> pd.DataFrame:
    """20 weekdays, mix of positive and negative returns — DatetimeIndex form."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in _WEEKDAYS])
    return pd.DataFrame({"returns": _RETURNS}, index=idx)


@pytest.fixture
def benchmark_pandas_df_indexed() -> pd.DataFrame:
    """20 weekdays, benchmark returns — DatetimeIndex form."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in _WEEKDAYS])
    return pd.DataFrame({"returns": _BENCH_RETURNS}, index=idx)


@pytest.fixture
def sample_pandas_series() -> pd.Series:
    """20 weekdays, mix of positive and negative returns — pd.Series form."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in _WEEKDAYS])
    return pd.Series(_RETURNS, index=idx, name="returns")


@pytest.fixture
def benchmark_pandas_series() -> pd.Series:
    """20 weekdays, benchmark returns — pd.Series form."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in _WEEKDAYS])
    return pd.Series(_BENCH_RETURNS, index=idx, name="returns")


@pytest.fixture
def grouped_sample_pandas_df() -> pd.DataFrame:
    """20 weekdays of grouped portfolio PnL contributions, as pandas."""
    groups = ["Tech", "Energy", "Financials"]
    rows: list[dict[str, object]] = []
    for date, ret in zip(_WEEKDAYS, _RETURNS):
        rows.extend(
            [
                {"date": date, "group": groups[0], "returns": ret * 0.5},
                {"date": date, "group": groups[1], "returns": ret * 0.3},
                {"date": date, "group": groups[2], "returns": ret * 0.2},
            ]
        )

    return pd.DataFrame(rows)


@pytest.fixture
def empty_df() -> pl.DataFrame:
    """0-row DataFrame with correct schema."""
    return pl.DataFrame(
        {
            "date": pl.Series([], dtype=pl.Date),
            "returns": pl.Series([], dtype=pl.Float64),
        }
    )


@pytest.fixture
def single_row_df() -> pl.DataFrame:
    """1-row DataFrame."""
    return pl.DataFrame(
        {"date": [datetime.date(2023, 1, 2)], "returns": [0.01]}
    ).with_columns(pl.col("date").cast(pl.Date))


@pytest.fixture
def all_positive_df() -> pl.DataFrame:
    """10 weekdays, all positive returns."""
    return pl.DataFrame(
        {"date": _WEEKDAYS[:10], "returns": [0.005, 0.010, 0.003, 0.007, 0.002] * 2}
    ).with_columns(pl.col("date").cast(pl.Date))


@pytest.fixture
def all_negative_df() -> pl.DataFrame:
    """10 weekdays, all negative returns."""
    return pl.DataFrame(
        {
            "date": _WEEKDAYS[:10],
            "returns": [-0.005, -0.010, -0.003, -0.007, -0.002] * 2,
        }
    ).with_columns(pl.col("date").cast(pl.Date))
