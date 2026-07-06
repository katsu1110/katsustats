"""Tests for input DataFrame normalization."""

from __future__ import annotations

import datetime

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from katsustats._dataframe import ensure_polars


def test_ensure_polars_with_polars_df(sample_df: pl.DataFrame) -> None:
    """Test that a valid Polars DataFrame is returned unchanged."""
    result = ensure_polars(sample_df)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["date", "returns"]
    assert result.schema["date"] == pl.Date
    assert_frame_equal(result, sample_df)


def test_ensure_polars_with_pandas_df(
    sample_pandas_df: pd.DataFrame, sample_df: pl.DataFrame
) -> None:
    """Test that a valid pandas DataFrame is converted correctly."""
    result = ensure_polars(sample_pandas_df)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["date", "returns"]
    assert result.schema["date"] == pl.Date
    assert_frame_equal(result, sample_df)


def test_ensure_polars_with_pandas_df_indexed(
    sample_pandas_df_indexed: pd.DataFrame, sample_df: pl.DataFrame
) -> None:
    """Test that a pandas DataFrame with a DatetimeIndex is converted correctly."""
    result = ensure_polars(sample_pandas_df_indexed)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["date", "returns"]
    assert result.schema["date"] == pl.Date
    assert_frame_equal(result, sample_df)


def test_ensure_polars_with_pandas_series(
    sample_pandas_series: pd.Series, sample_df: pl.DataFrame
) -> None:
    """Test that a pandas Series with a DatetimeIndex is converted correctly."""
    result = ensure_polars(sample_pandas_series)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["date", "returns"]
    assert result.schema["date"] == pl.Date
    assert_frame_equal(result, sample_df)


def test_ensure_polars_invalid_type() -> None:
    """Test that passing a list raises a TypeError."""
    with pytest.raises(
        TypeError,
        match="df must be a Polars DataFrame, pandas DataFrame, or pandas Series, got list",
    ):
        ensure_polars([])  # type: ignore


def test_ensure_polars_missing_columns() -> None:
    """Test that a pandas DataFrame missing columns raises an AssertionError."""
    df = pd.DataFrame({"price": [1, 2]})
    with pytest.raises(AssertionError, match="df must have a 'date' column"):
        ensure_polars(df)


def test_ensure_polars_invalid_pandas_type() -> None:
    """Test that passing an unsupported pandas object raises a TypeError."""
    idx = pd.Index([1, 2, 3])
    with pytest.raises(
        TypeError,
        match="df must be a Polars DataFrame, pandas DataFrame, or pandas Series, got Index",
    ):
        ensure_polars(idx)


def test_ensure_polars_duplicate_dates() -> None:
    """Test that duplicate dates are correctly compounded and issue a warning."""
    df = pl.DataFrame(
        {
            "date": [
                datetime.date(2023, 1, 2),
                datetime.date(2023, 1, 2),
                datetime.date(2023, 1, 3),
            ],
            "returns": [0.1, 0.2, 0.05],
        }
    ).with_columns(pl.col("date").cast(pl.Date))

    with pytest.warns(UserWarning, match="df has duplicate dates"):
        result = ensure_polars(df)

    expected = pl.DataFrame(
        {
            "date": [datetime.date(2023, 1, 2), datetime.date(2023, 1, 3)],
            "returns": [(1.1 * 1.2 - 1.0), 0.05],
        }
    ).with_columns(pl.col("date").cast(pl.Date))

    assert_frame_equal(result, expected)
