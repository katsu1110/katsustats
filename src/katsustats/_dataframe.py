"""Private helpers for normalizing accepted dataframe inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import pandas as pd

    DataFrameLike = pl.DataFrame | pd.DataFrame
else:
    DataFrameLike = pl.DataFrame


def _is_pandas_dataframe(df: object) -> bool:
    """Return True when the object is a pandas DataFrame."""
    df_type = type(df)
    return df_type.__name__ == "DataFrame" and df_type.__module__.startswith("pandas")


def ensure_polars(df: DataFrameLike, *, name: str = "df") -> pl.DataFrame:
    """Convert supported dataframe inputs to a Polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df

    if _is_pandas_dataframe(df):
        result = pl.DataFrame(df.to_dict(orient="list"))
        if "date" in result.columns:
            result = result.with_columns(pl.col("date").cast(pl.Date))
        return result

    raise AssertionError(f"{name} must be a Polars or pandas DataFrame")
