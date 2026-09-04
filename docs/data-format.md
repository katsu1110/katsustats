# Data format

`katsustats` accepts a [Polars](https://pola.rs/) or pandas DataFrame with two
required columns:

| column | type | description |
|--------|------|-------------|
| `date` | date-like | Trading date |
| `returns` | float-like | Daily return (e.g. `0.01` = +1%) |

## Conversions

When a pandas DataFrame or Series is passed, `katsustats` converts it to
Polars at the start of processing.

If `date` is datetime-like, it is normalized to `pl.Date` before analysis.

If multiple rows share the same `date`, `katsustats` compounds those same-day
`returns` values into one daily return, emits a warning, and continues.

Quantstats-style inputs (`pd.Series` with a `DatetimeIndex`, or a
`pd.DataFrame` with a `DatetimeIndex` and a `returns` column) are accepted
automatically — the index is promoted to the `date` column.

## Building inputs from prices

`katsustats.utils` converts between prices and returns:

```python
from katsustats import utils

returns = utils.to_returns(prices_df, price_col="close")  # -> ["date", "returns"]
prices = utils.to_prices(returns)                         # -> ["date", "price"]
log_rets = utils.to_log_returns(returns)                  # ln(1 + r)
monthly = utils.aggregate_returns(returns, period="month")
matrix = utils.monthly_returns(returns)                   # years x months pivot
```

`utils.download_returns("AAPL")` fetches daily returns via yfinance
(requires `pip install yfinance`).
