from __future__ import annotations

import warnings
from typing import Any, Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class DataFrameLike(Protocol):
    def to_pandas(self) -> Any: ...


def _is_pandas_object(obj: Any) -> bool:
    return type(obj).__module__.startswith("pandas")


def _normalize_pandas_input(obj: Any, name: str = "df") -> Any:
    """Normalize quantstats-style pandas inputs to a column-form DataFrame.

    Accepts:
    - ``pd.Series`` with a DatetimeIndex: converted to a DataFrame with a
      ``returns`` column, then treated as a DatetimeIndex DataFrame.
    - ``pd.DataFrame`` with no ``date`` column and a datetime-like index:
      the index is promoted to a ``date`` column.
    - ``pd.DataFrame`` that already has a ``date`` column: returned unchanged.

    Raises ``TypeError`` for other pandas objects (e.g. ``pd.Timestamp``,
    ``pd.Index``) so callers get a clear message instead of a Polars error.
    """
    import pandas as pd  # safe: only called when pandas is confirmed importable

    if isinstance(obj, pd.Series):
        obj = obj.to_frame(name="returns")
    elif not isinstance(obj, pd.DataFrame):
        raise TypeError(
            f"{name} must be a Polars DataFrame, pandas DataFrame, or pandas Series, "
            f"got {type(obj).__name__}"
        )

    if "date" not in obj.columns and pd.api.types.is_datetime64_any_dtype(obj.index):
        obj = obj.reset_index()
        # rename whatever the index column is called to "date"
        first_col = obj.columns[0]
        if first_col != "date":
            obj = obj.rename(columns={first_col: "date"})

    return obj


def _compound_by_date(df: pl.DataFrame) -> pl.DataFrame:
    """Compound all rows with the same date into one daily return per date.

    This is the core compounding helper shared by duplicate-date normalization
    (``_compound_duplicate_dates``) and ``stats._daily_returns``.
    """
    return (
        df.group_by("date")
        .agg(((pl.col("returns") + 1).product() - 1).alias("returns"))
        .sort("date")
    )


def _compound_duplicate_dates(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Compound duplicate same-date returns into one daily return per date."""
    if df.height == 0 or df.get_column("date").n_unique() == df.height:
        return df

    warnings.warn(
        (
            f"{name} has duplicate dates; compounding same-date returns values into "
            "one daily return per date."
        ),
        UserWarning,
        stacklevel=3,
    )
    return _compound_by_date(df)


def ensure_polars(df: Any, name: str = "df") -> pl.DataFrame:
    """Convert a pandas or Polars DataFrame to a Polars DataFrame.

    Always returns a DataFrame with exactly the columns ``["date", "returns"]``.
    Validates that the input has those columns, casts ``date`` to ``pl.Date``
    if needed, and compounds same-date ``returns`` rows into one daily return with
    a warning when duplicate dates are detected.

    Accepted pandas shapes (in addition to the standard ``["date", "returns"]``
    column form):

    - ``pd.Series`` with a ``DatetimeIndex`` — values become the ``returns``
      column; the index becomes the ``date`` column.
    - ``pd.DataFrame`` with a ``DatetimeIndex`` and a ``returns`` column but
      no ``date`` column — the index is promoted to a ``date`` column.
    """
    if isinstance(df, pl.DataFrame):
        polars_df = df
    elif _is_pandas_object(df):
        df = _normalize_pandas_input(df, name=name)
        try:
            polars_df = pl.from_pandas(df)
        except ImportError:
            polars_df = pl.DataFrame({col: df[col].tolist() for col in df.columns})
    else:
        raise TypeError(
            f"{name} must be a Polars DataFrame, pandas DataFrame, or pandas Series, "
            f"got {type(df).__name__}"
        )
    missing = {"date", "returns"} - set(polars_df.columns)
    assert not missing, f"{name} is missing columns: {missing}"
    if polars_df.schema["date"] != pl.Date:
        polars_df = polars_df.with_columns(pl.col("date").cast(pl.Date))
    polars_df = polars_df.select(["date", "returns"])
    return _compound_duplicate_dates(polars_df, name)
