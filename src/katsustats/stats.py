"""
katsustats.stats — Financial metrics computed with Polars.

All functions accept a Polars or pandas DataFrame with columns ["date", "pnl"]
where "pnl" represents daily P&L (profit/loss) values, and return
scalar metric values or Polars DataFrames.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ._dataframe import DataFrameLike, ensure_polars

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_returns(df: DataFrameLike, name: str = "df") -> pl.Series:
    """Extract the pnl column as a Polars Series."""
    df = ensure_polars(df, name=name)
    return df.get_column("pnl")


def _cumulative(returns: pl.Series) -> pl.Series:
    """Cumulative compounded returns: (1+r1)*(1+r2)*... - 1."""
    return (returns + 1).cum_prod() - 1


def _cumulative_value(returns: pl.Series) -> pl.Series:
    """Cumulative value curve starting from 1.0."""
    return (returns + 1).cum_prod()


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------


def total_return(df: DataFrameLike) -> float:
    """Total compounded return over the full period."""
    r = _to_returns(df)
    return float((r + 1).product() - 1)


def cagr(df: DataFrameLike, periods: int = 252) -> float:
    """Compound Annual Growth Rate."""
    r = _to_returns(df)
    n = r.len()
    if n == 0:
        return 0.0
    total = float((r + 1).product())
    years = n / periods
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def volatility(df: DataFrameLike, periods: int = 252) -> float:
    """Annualized volatility (standard deviation of returns)."""
    r = _to_returns(df)
    std = r.std()
    if std is None:
        return float("nan")
    return float(std * np.sqrt(periods))


def sharpe(df: DataFrameLike, rf: float = 0.0, periods: int = 252) -> float:
    """Annualized Sharpe ratio."""
    r = _to_returns(df)
    rf_per_period = rf / periods
    excess = r - rf_per_period
    std = float(excess.std())
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods))


def sortino(df: DataFrameLike, rf: float = 0.0, periods: int = 252) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    r = _to_returns(df)
    rf_per_period = rf / periods
    excess = r - rf_per_period
    # Downside deviation: RMS of min(excess, 0) over ALL days (not just negative ones)
    downside_sq_mean = (excess.clip(upper_bound=0.0) ** 2).mean()
    if downside_sq_mean is None or downside_sq_mean == 0.0:
        return float("inf")
    return float(excess.mean() / (downside_sq_mean**0.5) * np.sqrt(periods))


def max_drawdown(df: DataFrameLike) -> float:
    """Maximum drawdown (returned as a negative value)."""
    r = _to_returns(df)
    cumval = _cumulative_value(r)
    running_max = cumval.cum_max()
    dd = (cumval - running_max) / running_max
    return float(dd.min())


def calmar(df: DataFrameLike, periods: int = 252) -> float:
    """Calmar ratio: CAGR / |Max Drawdown|."""
    mdd = max_drawdown(df)
    if mdd == 0:
        return 0.0
    return cagr(df, periods) / abs(mdd)


def win_rate(df: DataFrameLike) -> float:
    """Percentage of days with positive returns."""
    r = _to_returns(df)
    n = r.len()
    if n == 0:
        return 0.0
    return float((r > 0).sum() / n)


def profit_factor(df: DataFrameLike) -> float:
    """Gross profits / gross losses."""
    r = _to_returns(df)
    gains = r.filter(r > 0).sum()
    losses = r.filter(r < 0).abs().sum()
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def best_day(df: DataFrameLike) -> float:
    """Best single-day return."""
    return float(_to_returns(df).max())


def worst_day(df: DataFrameLike) -> float:
    """Worst single-day return."""
    return float(_to_returns(df).min())


def avg_win(df: DataFrameLike) -> float:
    """Average return on winning days."""
    r = _to_returns(df)
    wins = r.filter(r > 0)
    if wins.len() == 0:
        return 0.0
    return float(wins.mean())


def avg_loss(df: DataFrameLike) -> float:
    """Average return on losing days."""
    r = _to_returns(df)
    losses = r.filter(r < 0)
    if losses.len() == 0:
        return 0.0
    return float(losses.mean())


def value_at_risk(df: DataFrameLike, alpha: float = 0.05) -> float:
    """Daily Value at Risk at the given confidence level."""
    r = _to_returns(df)
    return float(r.quantile(alpha, interpolation="linear"))


def recovery_factor(df: DataFrameLike) -> float:
    """Total return / |Max Drawdown|."""
    mdd = max_drawdown(df)
    if mdd == 0:
        return 0.0
    return total_return(df) / abs(mdd)


def skewness(df: DataFrameLike) -> float:
    """Skewness of daily returns."""
    r = _to_returns(df)
    return float(r.skew())


def kurtosis(df: DataFrameLike) -> float:
    """Excess kurtosis of daily returns."""
    r = _to_returns(df)
    return float(r.kurtosis())


# ---------------------------------------------------------------------------
# Drawdown Analysis
# ---------------------------------------------------------------------------


def drawdown_details(df: DataFrameLike, top_n: int = 5) -> pl.DataFrame:
    """
    Top-N drawdowns with start, trough, recovery dates, max drawdown, and
    duration in days.

    Returns a Polars DataFrame with columns:
        [start, trough, recovery, max_dd, drawdown_days, recovery_days]
    """
    df = ensure_polars(df)
    r = _to_returns(df)
    dates = df.get_column("date")
    cumval = _cumulative_value(r)
    running_max = cumval.cum_max()
    dd = (cumval - running_max) / running_max

    # Identify drawdown periods (contiguous blocks where dd < 0)
    dd_np = dd.to_numpy()
    dates_np = dates.to_numpy()
    n = len(dd_np)

    periods: list[dict] = []
    i = 0
    while i < n:
        if dd_np[i] < 0:
            start_idx = i - 1 if i > 0 else 0  # peak before drawdown
            trough_idx = i
            min_dd = dd_np[i]
            while i < n and dd_np[i] < 0:
                if dd_np[i] < min_dd:
                    min_dd = dd_np[i]
                    trough_idx = i
                i += 1
            recovery_idx = i if i < n else None
            periods.append(
                {
                    "start": dates_np[start_idx],
                    "trough": dates_np[trough_idx],
                    "recovery": dates_np[recovery_idx]
                    if recovery_idx is not None
                    else None,
                    "max_dd": min_dd,
                    "drawdown_days": (trough_idx - start_idx),
                    "recovery_days": (recovery_idx - trough_idx)
                    if recovery_idx is not None
                    else None,
                }
            )
        else:
            i += 1

    if not periods:
        return pl.DataFrame(
            {
                "start": pl.Series([], dtype=pl.Date),
                "trough": pl.Series([], dtype=pl.Date),
                "recovery": pl.Series([], dtype=pl.Date),
                "max_dd": pl.Series([], dtype=pl.Float64),
                "drawdown_days": pl.Series([], dtype=pl.Int64),
                "recovery_days": pl.Series([], dtype=pl.Int64),
            }
        )

    result = pl.DataFrame(periods)
    result = result.sort("max_dd").head(top_n)
    return result


# ---------------------------------------------------------------------------
# Benchmark Comparison Metrics
# ---------------------------------------------------------------------------


def alpha_beta(
    df: DataFrameLike, base_df: DataFrameLike, periods: int = 252
) -> tuple[float, float]:
    """Annualized alpha and beta vs benchmark using OLS."""
    df = ensure_polars(df)
    base_df = ensure_polars(base_df, "base_df")
    joined = df.join(base_df.rename({"pnl": "_base_pnl"}), on="date", how="inner")
    r = joined.get_column("pnl").to_numpy()
    b = joined.get_column("_base_pnl").to_numpy()

    cov = np.cov(r, b)
    var_b = cov[1, 1]
    if var_b == 0:
        return 0.0, 0.0
    beta = float(cov[0, 1] / var_b)
    alpha_daily = float(np.mean(r) - beta * np.mean(b))
    alpha_annual = alpha_daily * periods
    return alpha_annual, beta


def correlation(df: DataFrameLike, base_df: DataFrameLike) -> float:
    """Pearson correlation between strategy and benchmark returns."""
    df = ensure_polars(df)
    base_df = ensure_polars(base_df, "base_df")
    joined = df.join(base_df.rename({"pnl": "_base_pnl"}), on="date", how="inner")
    r = joined.get_column("pnl").to_numpy()
    b = joined.get_column("_base_pnl").to_numpy()
    return float(np.corrcoef(r, b)[0, 1])


def information_ratio(
    df: DataFrameLike, base_df: DataFrameLike, periods: int = 252
) -> float:
    """Annualized information ratio (excess return / tracking error)."""
    df = ensure_polars(df)
    base_df = ensure_polars(base_df, "base_df")
    joined = df.join(base_df.rename({"pnl": "_base_pnl"}), on="date", how="inner")
    excess = joined.get_column("pnl") - joined.get_column("_base_pnl")
    te = float(excess.std())
    if te == 0:
        return 0.0
    return float(excess.mean() / te * np.sqrt(periods))


def excess_return(df: DataFrameLike, base_df: DataFrameLike) -> float:
    """Total compounded excess return vs benchmark (aligned on common dates)."""
    df = ensure_polars(df)
    base_df = ensure_polars(base_df, "base_df")
    joined = df.join(base_df.rename({"pnl": "_base_pnl"}), on="date", how="inner")
    strat_ret = float((joined.get_column("pnl") + 1).product() - 1)
    bench_ret = float((joined.get_column("_base_pnl") + 1).product() - 1)
    return strat_ret - bench_ret


# ---------------------------------------------------------------------------
# Day-of-Week Analysis (New Feature)
# ---------------------------------------------------------------------------


def day_of_week_stats(df: DataFrameLike) -> pl.DataFrame:
    """
    Returns a DataFrame with day-of-week level statistics:
        [dow, dow_name, mean_return, win_rate, total_return, count]

    dow: 1=Monday .. 7=Sunday (ISO weekday)
    """
    df = ensure_polars(df)
    result = (
        df.with_columns(pl.col("date").cast(pl.Date).dt.weekday().alias("dow"))
        .group_by("dow")
        .agg(
            pl.col("pnl").mean().alias("mean_return"),
            (pl.col("pnl") > 0).mean().alias("win_rate"),
            ((pl.col("pnl") + 1).product() - 1).alias("total_return"),
            pl.col("pnl").count().alias("count"),
        )
        .sort("dow")
        .with_columns(
            pl.col("dow")
            .replace_strict(
                {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"},
                return_dtype=pl.Utf8,
            )
            .alias("dow_name")
        )
    )
    return result


# ---------------------------------------------------------------------------
# Rolling Metrics
# ---------------------------------------------------------------------------


def rolling_sharpe(
    df: DataFrameLike, window: int = 126, periods: int = 252
) -> pl.DataFrame:
    """Rolling Sharpe ratio over a given window."""
    df = ensure_polars(df)
    return (
        df.sort("date")
        .with_columns(
            [
                pl.col("pnl").rolling_mean(window_size=window).alias("_rm"),
                pl.col("pnl").rolling_std(window_size=window, ddof=1).alias("_rs"),
            ]
        )
        .with_columns(
            pl.when(pl.col("_rs") > 0)
            .then(pl.col("_rm") / pl.col("_rs") * float(periods**0.5))
            .alias("rolling_sharpe")
        )
        .select(["date", "rolling_sharpe"])
    )


def rolling_volatility(
    df: DataFrameLike, window: int = 126, periods: int = 252
) -> pl.DataFrame:
    """Rolling annualized volatility over a given window."""
    df = ensure_polars(df)
    return (
        df.sort("date")
        .with_columns(
            (
                pl.col("pnl").rolling_std(window_size=window, ddof=1)
                * float(periods**0.5)
            ).alias("rolling_vol")
        )
        .select(["date", "rolling_vol"])
    )


# ---------------------------------------------------------------------------
# Summary Table Builder
# ---------------------------------------------------------------------------


_SUMMARY_METRIC_SPECS = [
    ("Total Return", "total_return", "pct"),
    ("CAGR", "cagr", "pct"),
    ("Sharpe Ratio", "sharpe", "float"),
    ("Sortino Ratio", "sortino", "float"),
    ("Max Drawdown", "max_drawdown", "pct"),
    ("Calmar Ratio", "calmar", "float"),
    ("Volatility (ann.)", "volatility", "pct"),
    ("Win Rate", "win_rate", "pct"),
    ("Profit Factor", "profit_factor", "float"),
    ("Best Day", "best_day", "pct"),
    ("Worst Day", "worst_day", "pct"),
    ("Avg Win", "avg_win", "pct"),
    ("Avg Loss", "avg_loss", "pct"),
    ("Daily VaR (95%)", "value_at_risk", "pct"),
    ("Recovery Factor", "recovery_factor", "float"),
    ("Skewness", "skewness", "float"),
    ("Kurtosis", "kurtosis", "float"),
]

_COMPARISON_METRIC_SPECS = [
    ("Alpha", "alpha", "pct"),
    ("Beta", "beta", "float"),
    ("Correlation", "correlation", "float"),
    ("Information Ratio", "information_ratio", "float"),
    ("Excess Return", "excess_return", "pct"),
]


def _summary_metric_values(
    df: DataFrameLike, rf: float = 0.0, periods: int = 252
) -> dict[str, float]:
    """
    Compute the raw numeric summary metrics for a single return series.

    Returned keys:
        total_return, cagr, sharpe, sortino, max_drawdown, calmar,
        volatility, win_rate, profit_factor, best_day, worst_day,
        avg_win, avg_loss, value_at_risk, recovery_factor, skewness, kurtosis
    """
    df = ensure_polars(df)
    return {
        "total_return": float(total_return(df)),
        "cagr": float(cagr(df, periods)),
        "sharpe": float(sharpe(df, rf, periods)),
        "sortino": float(sortino(df, rf, periods)),
        "max_drawdown": float(max_drawdown(df)),
        "calmar": float(calmar(df, periods)),
        "volatility": float(volatility(df, periods)),
        "win_rate": float(win_rate(df)),
        "profit_factor": float(profit_factor(df)),
        "best_day": float(best_day(df)),
        "worst_day": float(worst_day(df)),
        "avg_win": float(avg_win(df)),
        "avg_loss": float(avg_loss(df)),
        "value_at_risk": float(value_at_risk(df)),
        "recovery_factor": float(recovery_factor(df)),
        "skewness": float(skewness(df)),
        "kurtosis": float(kurtosis(df)),
    }


def _comparison_metric_values(
    df: DataFrameLike, base_df: DataFrameLike, periods: int = 252
) -> dict[str, float]:
    """
    Compute raw numeric metrics that compare a strategy to a benchmark.

    Returned keys:
        alpha, beta, correlation, information_ratio, excess_return
    """
    df = ensure_polars(df)
    base_df = ensure_polars(base_df, name="base_df")
    a, b = alpha_beta(df, base_df, periods)
    return {
        "alpha": float(a),
        "beta": float(b),
        "correlation": float(correlation(df, base_df)),
        "information_ratio": float(information_ratio(df, base_df, periods)),
        "excess_return": float(excess_return(df, base_df)),
    }


def _format_summary_value(value: float, fmt: str) -> str:
    """Format a raw numeric summary metric for display."""
    if fmt == "pct":
        return f"{value:.2%}"
    return f"{value:.2f}"


def summary_metrics_raw(
    df: DataFrameLike,
    base_df: DataFrameLike | None = None,
    rf: float = 0.0,
    periods: int = 252,
) -> dict[str, float]:
    """
    Return summary metrics as raw numeric values.

    Args:
        df: Polars or pandas DataFrame with ["date", "pnl"] columns.
        base_df: Optional benchmark DataFrame with the same schema. When
            provided, comparison metrics are added to the returned dict.
        rf: Annualized risk-free rate used by risk-adjusted metrics.
        periods: Number of return periods per year.

    Returns:
        A dict[str, float] with these base keys:
            total_return, cagr, sharpe, sortino, max_drawdown, calmar,
            volatility, win_rate, profit_factor, best_day, worst_day,
            avg_win, avg_loss, value_at_risk, recovery_factor, skewness,
            kurtosis

        When base_df is provided, these additional keys are included:
            alpha, beta, correlation, information_ratio, excess_return

    Example:
        {
            "total_return": 0.131,
            "cagr": 0.127,
            "sharpe": 1.42,
            "max_drawdown": -0.187,
            ...
        }
    """
    raw = _summary_metric_values(df, rf, periods)
    if base_df is not None:
        raw.update(_comparison_metric_values(df, base_df, periods))
    return raw


def summary_metrics(
    df: DataFrameLike,
    base_df: DataFrameLike | None = None,
    rf: float = 0.0,
    periods: int = 252,
) -> pl.DataFrame:
    """
    Build a summary metrics table. Returns a Polars DataFrame with columns:
        [metric, strategy, benchmark] (benchmark only if base_df provided)
    """

    df = ensure_polars(df)
    if base_df is not None:
        base_df = ensure_polars(base_df, name="base_df")

    strat = _summary_metric_values(df, rf, periods)
    data: dict[str, list[str]] = {
        "metric": [label for label, _, _ in _SUMMARY_METRIC_SPECS],
        "strategy": [
            _format_summary_value(strat[key], fmt)
            for label, key, fmt in _SUMMARY_METRIC_SPECS
        ],
    }

    if base_df is not None:
        bench = _summary_metric_values(base_df, rf, periods)
        data["benchmark"] = [
            _format_summary_value(bench[key], fmt)
            for label, key, fmt in _SUMMARY_METRIC_SPECS
        ]
        comparison = _comparison_metric_values(df, base_df, periods)
        comparison_metrics = [label for label, _, _ in _COMPARISON_METRIC_SPECS]
        comparison_strat = [
            _format_summary_value(comparison[key], fmt)
            for label, key, fmt in _COMPARISON_METRIC_SPECS
        ]
        comparison_bench = ["—"] * len(_COMPARISON_METRIC_SPECS)

        data["metric"].extend(comparison_metrics)
        data["strategy"].extend(comparison_strat)
        data["benchmark"].extend(comparison_bench)

    return pl.DataFrame(data)
