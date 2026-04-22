from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class DataFrameLike(Protocol):
    def to_pandas(self) -> Any: ...


def _is_pandas_dataframe(obj: Any) -> bool:
    return type(obj).__module__.startswith("pandas")


def ensure_polars(df: Any, name: str = "df") -> pl.DataFrame:
    """Convert a pandas or Polars DataFrame to a Polars DataFrame.

    Validates that the result has the required ["date", "pnl"] columns.
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
    return polars_df
