from __future__ import annotations

import warnings
from typing import Any, Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class DataFrameLike(Protocol):
    def to_pandas(self) -> Any: ...


def _is_pandas_dataframe(obj: Any) -> bool:
    return type(obj).__module__.startswith("pandas")


def _compound_by_date(df: pl.DataFrame) -> pl.DataFrame:
    """Compound all rows with the same date into one daily return per date.

    This is the core compounding helper shared by duplicate-date normalization
    (``_compound_duplicate_dates``) and ``stats._daily_returns``.
    """
    return (
        df.group_by("date")
        .agg(((pl.col("pnl") + 1).product() - 1).alias("pnl"))
        .sort("date")
    )


def _compound_duplicate_dates(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Compound duplicate same-date returns into one daily return per date."""
    if df.height == 0 or df.get_column("date").n_unique() == df.height:
        return df

    warnings.warn(
        (
            f"{name} has duplicate dates; compounding same-date pnl values into "
            "one daily return per date."
        ),
        UserWarning,
        stacklevel=3,
    )
    return _compound_by_date(df)


def ensure_polars(df: Any, name: str = "df") -> pl.DataFrame:
    """Convert a pandas or Polars DataFrame to a Polars DataFrame.

    Always returns a DataFrame with exactly the columns ``["date", "pnl"]``.
    Validates that the input has those columns, casts ``date`` to ``pl.Date``
    if needed, and compounds same-date ``pnl`` rows into one daily return with
    a warning when duplicate dates are detected.
    """
    if isinstance(df, pl.DataFrame):
        polars_df = df
    elif _is_pandas_dataframe(df):
        try:
            polars_df = pl.from_pandas(df)
        except ImportError:
            polars_df = pl.DataFrame({col: df[col].tolist() for col in df.columns})
    else:
        raise TypeError(
            f"{name} must be a Polars or pandas DataFrame, got {type(df).__name__}"
        )
    missing = {"date", "pnl"} - set(polars_df.columns)
    assert not missing, f"{name} is missing columns: {missing}"
    if polars_df.schema["date"] != pl.Date:
        polars_df = polars_df.with_columns(pl.col("date").cast(pl.Date))
    polars_df = polars_df.select(["date", "pnl"])
    return _compound_duplicate_dates(polars_df, name)
