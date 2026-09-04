"""katsustats.utils — Data conversion helpers (quantstats parity).

Polars-first equivalents of quantstats.utils conversions:
prices <-> returns, log returns, period aggregation, monthly matrix,
and optional ``download_returns`` via yfinance (optional dependency).
"""

from __future__ import annotations

import math

import polars as pl

from ._constants import COL_DATE, COL_RETURNS
from ._dataframe import DataFrameLike, ensure_polars


def to_returns(
    prices: DataFrameLike, price_col: str = "close", base_col: str = "date"
) -> pl.DataFrame:
    """Convert a price frame to ``[date, returns]`` via simple pct change."""
    from ._dataframe import _is_pandas_object

    if _is_pandas_object(prices):
        import pandas as pd

        pdf = prices.copy() if isinstance(prices, pd.DataFrame) else prices.to_frame()
        df = pl.from_pandas(pdf)
    elif isinstance(prices, pl.DataFrame):
        df = prices
    else:
        raise TypeError(
            f"prices must be a Polars/pandas DataFrame, got {type(prices).__name__}"
        )
    if price_col not in df.columns:
        raise ValueError(f"price_col={price_col!r} not in columns {df.columns}")
    date = base_col if base_col in df.columns else COL_DATE
    px = df.get_column(price_col).cast(pl.Float64)
    prev = px.shift(1)
    rets = (px / prev - 1).fill_null(0.0).fill_nan(0.0)
    return pl.DataFrame({COL_DATE: df.get_column(date), COL_RETURNS: rets}).sort(
        COL_DATE
    )


def to_prices(
    df: DataFrameLike, base: float = 1e5, out_col: str = "price"
) -> pl.DataFrame:
    """Convert ``[date, returns]`` to prices rebased at ``base``."""
    frame = ensure_polars(df)
    r = frame.get_column(COL_RETURNS).cast(pl.Float64).fill_null(0.0)
    prices = base * (1 + r).cum_prod()
    return pl.DataFrame({COL_DATE: frame.get_column(COL_DATE), out_col: prices}).sort(
        COL_DATE
    )


def to_log_returns(df: DataFrameLike) -> pl.DataFrame:
    """Convert ``[date, returns]`` to log returns ``ln(1 + r)``."""
    frame = ensure_polars(df)
    r = frame.get_column(COL_RETURNS).cast(pl.Float64)
    logr = r.map_elements(
        lambda x: math.log1p(x) if x is not None and x > -1 else float("nan"),
        return_dtype=pl.Float64,
    )
    return pl.DataFrame({COL_DATE: frame.get_column(COL_DATE), COL_RETURNS: logr}).sort(
        COL_DATE
    )


log_returns = to_log_returns


def aggregate_returns(df: DataFrameLike, period: str = "month") -> pl.DataFrame:
    """Compound daily returns into calendar buckets (month/quarter/year/week)."""
    frame = ensure_polars(df).with_columns(pl.col(COL_DATE).cast(pl.Date))
    p = period.lower()
    if "month" in p or p in ("me", "eom"):
        key = pl.col(COL_DATE).dt.strftime("%Y-%m-01")
    elif "quarter" in p or p in ("qe", "eoq"):
        key = (
            pl.col(COL_DATE).dt.year().cast(pl.String)
            + "-Q"
            + pl.col(COL_DATE).dt.quarter().cast(pl.String)
        )
    elif "year" in p or p in ("ye", "eoy", "yoy"):
        key = pl.col(COL_DATE).dt.year().cast(pl.String)
    elif "week" in p or p == "w":
        key = pl.col(COL_DATE).dt.strftime("%Y-%W")
    else:
        raise ValueError(f"Unknown period {period!r}: use month/quarter/year/week")
    return (
        frame.with_columns(key.alias("_bucket"))
        .group_by("_bucket", maintain_order=True)
        .agg(((pl.col(COL_RETURNS) + 1).product() - 1).alias(COL_RETURNS))
        .sort("_bucket")
    )


def monthly_returns(df: DataFrameLike) -> pl.DataFrame:
    """Monthly returns matrix (years × Jan..Dec + EOY), quantstats-style pivot."""
    from . import stats as _stats

    frame = (
        ensure_polars(df).with_columns(pl.col(COL_DATE).cast(pl.Date)).sort(COL_DATE)
    )
    months_order = [
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    ]
    years = sorted(frame.get_column(COL_DATE).dt.year().unique().to_list())
    rows: list[dict] = []
    for y in years:
        row: dict = {"YEAR": y}
        year_df = frame.filter(pl.col(COL_DATE).dt.year() == y)
        for i, up in enumerate(months_order):
            m_df = year_df.filter(pl.col(COL_DATE).dt.month() == i + 1)
            row[up] = float(_stats.total_return(m_df)) if m_df.height else 0.0
        row["EOY"] = float(_stats.total_return(year_df))
        rows.append(row)
    if not rows:
        return pl.DataFrame({"YEAR": []})
    return pl.DataFrame(rows, schema=["YEAR", *months_order, "EOY"])


def download_returns(ticker: str, period: str = "max") -> pl.DataFrame:
    """Download daily returns for ``ticker`` via yfinance (optional dep)."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "download_returns requires yfinance: pip install yfinance"
        ) from e
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for ticker {ticker!r}")
    close = (
        df["Close"].iloc[:, 0]
        if hasattr(df["Close"], "iloc") and df["Close"].ndim > 1
        else df["Close"]
    )
    import pandas as pd

    rets = close.pct_change().fillna(0)
    return pl.DataFrame(
        {
            COL_DATE: pd.to_datetime(rets.index).date,
            COL_RETURNS: rets.to_numpy(dtype=float),
        }
    ).sort(COL_DATE)
