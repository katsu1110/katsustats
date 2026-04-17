"""Private helpers for normalizing accepted dataframe inputs."""

from __future__ import annotations

from typing import Any, Protocol

import polars as pl


class _PandasDataFrameLike(Protocol):
    """Structural type for pandas-style dataframe inputs."""

    def to_dict(self, orient: str = ...) -> dict[str, list[Any]]: ...


DataFrameLike = pl.DataFrame | _PandasDataFrameLike
def _is_pandas_dataframe(df: object) -> bool:
    """Return True when the object is a pandas DataFrame."""
    df_type = type(df)
    return df_type.__name__ == "DataFrame" and df_type.__module__.startswith("pandas")


def ensure_polars(df: DataFrameLike, *, name: str = "df") -> pl.DataFrame:
    """Convert supported dataframe inputs to a Polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df

    if _is_pandas_dataframe(df):
        result = pl.from_pandas(df)
        if "date" in result.columns:
            result = result.with_columns(pl.col("date").cast(pl.Date))
        return result

    raise TypeError(f"{name} must be a Polars or pandas DataFrame")
