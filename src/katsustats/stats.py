"""
katsustats.stats — Financial metrics computed with Polars.

All functions accept a Polars DataFrame with columns ["date", "pnl"]
where "pnl" represents daily P&L (profit/loss) values, and return
scalar metric values or Polars DataFrames.
"""

from __future__ import annotations

import polars as pl
import numpy as np
from datetime import date


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_returns(df: pl.DataFrame) -> pl.Series:
    """Extract the pnl column as a Polars Series."""
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

def total_return(df: pl.DataFrame) -> float:
    """Total compounded return over the full period."""
    r = _to_returns(df)
    return float((r + 1).product() - 1)


def cagr(df: pl.DataFrame, periods: int = 252) -> float:
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


def volatility(df: pl.DataFrame, periods: int = 252) -> float:
    """Annualized volatility (standard deviation of returns)."""
    r = _to_returns(df)
    return float(r.std() * np.sqrt(periods))


def sharpe(df: pl.DataFrame, rf: float = 0.0, periods: int = 252) -> float:
    """Annualized Sharpe ratio."""
    r = _to_returns(df)
    rf_per_period = rf / periods
    excess = r - rf_per_period
    std = float(excess.std())
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods))


def sortino(df: pl.DataFrame, rf: float = 0.0, periods: int = 252) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    r = _to_returns(df)
    rf_per_period = rf / periods
    excess = r - rf_per_period
    downside = excess.filter(excess < 0)
    if downside.len() == 0:
        return float("inf")
    downside_std = float((downside ** 2).mean() ** 0.5)
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods))


def max_drawdown(df: pl.DataFrame) -> float:
    """Maximum drawdown (returned as a negative value)."""
    r = _to_returns(df)
    cumval = _cumulative_value(r)
    running_max = cumval.cum_max()
    dd = (cumval - running_max) / running_max
    return float(dd.min())


def calmar(df: pl.DataFrame, periods: int = 252) -> float:
    """Calmar ratio: CAGR / |Max Drawdown|."""
    mdd = max_drawdown(df)
    if mdd == 0:
        return 0.0
    return cagr(df, periods) / abs(mdd)


def win_rate(df: pl.DataFrame) -> float:
    """Percentage of days with positive returns."""
    r = _to_returns(df)
    n = r.len()
    if n == 0:
        return 0.0
    return float((r > 0).sum() / n)


def profit_factor(df: pl.DataFrame) -> float:
    """Gross profits / gross losses."""
    r = _to_returns(df)
    gains = r.filter(r > 0).sum()
    losses = r.filter(r < 0).abs().sum()
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def best_day(df: pl.DataFrame) -> float:
    """Best single-day return."""
    return float(_to_returns(df).max())


def worst_day(df: pl.DataFrame) -> float:
    """Worst single-day return."""
    return float(_to_returns(df).min())


def avg_win(df: pl.DataFrame) -> float:
    """Average return on winning days."""
    r = _to_returns(df)
    wins = r.filter(r > 0)
    if wins.len() == 0:
        return 0.0
    return float(wins.mean())


def avg_loss(df: pl.DataFrame) -> float:
    """Average return on losing days."""
    r = _to_returns(df)
    losses = r.filter(r < 0)
    if losses.len() == 0:
        return 0.0
    return float(losses.mean())


def value_at_risk(df: pl.DataFrame, alpha: float = 0.05) -> float:
    """Daily Value at Risk at the given confidence level."""
    r = _to_returns(df)
    return float(r.quantile(alpha, interpolation="linear"))


def recovery_factor(df: pl.DataFrame) -> float:
    """Total return / |Max Drawdown|."""
    mdd = max_drawdown(df)
    if mdd == 0:
        return 0.0
    return total_return(df) / abs(mdd)


def skewness(df: pl.DataFrame) -> float:
    """Skewness of daily returns."""
    r = _to_returns(df)
    return float(r.skew())


def kurtosis(df: pl.DataFrame) -> float:
    """Excess kurtosis of daily returns."""
    r = _to_returns(df)
    return float(r.kurtosis())


# ---------------------------------------------------------------------------
# Drawdown Analysis
# ---------------------------------------------------------------------------

def drawdown_details(df: pl.DataFrame, top_n: int = 5) -> pl.DataFrame:
    """
    Top-N drawdowns with start, trough, recovery dates, max drawdown, and
    duration in days.

    Returns a Polars DataFrame with columns:
        [start, trough, recovery, max_dd, days]
    """
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
            periods.append({
                "start": dates_np[start_idx],
                "trough": dates_np[trough_idx],
                "recovery": dates_np[recovery_idx] if recovery_idx is not None else None,
                "max_dd": min_dd,
                "days": (trough_idx - start_idx),
            })
        else:
            i += 1

    if not periods:
        return pl.DataFrame({
            "start": pl.Series([], dtype=pl.Date),
            "trough": pl.Series([], dtype=pl.Date),
            "recovery": pl.Series([], dtype=pl.Date),
            "max_dd": pl.Series([], dtype=pl.Float64),
            "days": pl.Series([], dtype=pl.Int64),
        })

    result = pl.DataFrame(periods)
    result = result.sort("max_dd").head(top_n)
    return result


# ---------------------------------------------------------------------------
# Benchmark Comparison Metrics
# ---------------------------------------------------------------------------

def alpha_beta(
    df: pl.DataFrame, base_df: pl.DataFrame, periods: int = 252
) -> tuple[float, float]:
    """Annualized alpha and beta vs benchmark using OLS."""
    r = _to_returns(df).to_numpy()
    b = _to_returns(base_df).to_numpy()
    min_len = min(len(r), len(b))
    r, b = r[:min_len], b[:min_len]

    cov = np.cov(r, b)
    var_b = cov[1, 1]
    if var_b == 0:
        return 0.0, 0.0
    beta = float(cov[0, 1] / var_b)
    alpha_daily = float(np.mean(r) - beta * np.mean(b))
    alpha_annual = (1 + alpha_daily) ** periods - 1
    return alpha_annual, beta


def correlation(df: pl.DataFrame, base_df: pl.DataFrame) -> float:
    """Pearson correlation between strategy and benchmark returns."""
    r = _to_returns(df).to_numpy()
    b = _to_returns(base_df).to_numpy()
    min_len = min(len(r), len(b))
    return float(np.corrcoef(r[:min_len], b[:min_len])[0, 1])


def information_ratio(
    df: pl.DataFrame, base_df: pl.DataFrame, periods: int = 252
) -> float:
    """Annualized information ratio (excess return / tracking error)."""
    r = _to_returns(df)
    b = _to_returns(base_df)
    min_len = min(r.len(), b.len())
    excess = r.head(min_len) - b.head(min_len)
    te = float(excess.std())
    if te == 0:
        return 0.0
    return float(excess.mean() / te * np.sqrt(periods))


def excess_return(df: pl.DataFrame, base_df: pl.DataFrame) -> float:
    """Total compounded excess return vs benchmark."""
    return total_return(df) - total_return(base_df)


# ---------------------------------------------------------------------------
# Day-of-Week Analysis (New Feature)
# ---------------------------------------------------------------------------

def day_of_week_stats(df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns a DataFrame with day-of-week level statistics:
        [dow, dow_name, mean_return, win_rate, total_return, count]

    dow: 1=Monday .. 7=Sunday (ISO weekday)
    """
    result = (
        df
        .with_columns(
            pl.col("date").cast(pl.Date).dt.weekday().alias("dow")
        )
        .group_by("dow")
        .agg(
            pl.col("pnl").mean().alias("mean_return"),
            (pl.col("pnl") > 0).mean().alias("win_rate"),
            ((pl.col("pnl") + 1).product() - 1).alias("total_return"),
            pl.col("pnl").count().alias("count"),
        )
        .sort("dow")
        .with_columns(
            pl.col("dow").replace_strict(
                {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"},
                return_dtype=pl.Utf8,
            ).alias("dow_name")
        )
    )
    return result


# ---------------------------------------------------------------------------
# Rolling Metrics
# ---------------------------------------------------------------------------

def rolling_sharpe(df: pl.DataFrame, window: int = 126, periods: int = 252) -> pl.DataFrame:
    """Rolling Sharpe ratio over a given window."""
    r = _to_returns(df)
    dates = df.get_column("date")

    r_np = r.to_numpy().astype(np.float64)
    n = len(r_np)
    sharpe_vals = np.full(n, np.nan)

    for i in range(window, n):
        chunk = r_np[i - window : i]
        std = chunk.std(ddof=1)
        if std > 0:
            sharpe_vals[i] = (chunk.mean() / std) * np.sqrt(periods)

    return pl.DataFrame({
        "date": dates,
        "rolling_sharpe": sharpe_vals,
    })


def rolling_volatility(df: pl.DataFrame, window: int = 126, periods: int = 252) -> pl.DataFrame:
    """Rolling annualized volatility over a given window."""
    r = _to_returns(df)
    dates = df.get_column("date")

    r_np = r.to_numpy().astype(np.float64)
    n = len(r_np)
    vol_vals = np.full(n, np.nan)

    for i in range(window, n):
        chunk = r_np[i - window : i]
        vol_vals[i] = chunk.std(ddof=1) * np.sqrt(periods)

    return pl.DataFrame({
        "date": dates,
        "rolling_vol": vol_vals,
    })


# ---------------------------------------------------------------------------
# Summary Table Builder
# ---------------------------------------------------------------------------

def summary_metrics(
    df: pl.DataFrame,
    base_df: pl.DataFrame | None = None,
    rf: float = 0.0,
    periods: int = 252,
) -> pl.DataFrame:
    """
    Build a summary metrics table. Returns a Polars DataFrame with columns:
        [metric, strategy, benchmark] (benchmark only if base_df provided)
    """
    def _compute(d: pl.DataFrame) -> dict[str, str]:
        return {
            "Total Return": f"{total_return(d):.2%}",
            "CAGR": f"{cagr(d, periods):.2%}",
            "Sharpe Ratio": f"{sharpe(d, rf, periods):.2f}",
            "Sortino Ratio": f"{sortino(d, rf, periods):.2f}",
            "Max Drawdown": f"{max_drawdown(d):.2%}",
            "Calmar Ratio": f"{calmar(d, periods):.2f}",
            "Volatility (ann.)": f"{volatility(d, periods):.2%}",
            "Win Rate": f"{win_rate(d):.2%}",
            "Profit Factor": f"{profit_factor(d):.2f}",
            "Best Day": f"{best_day(d):.2%}",
            "Worst Day": f"{worst_day(d):.2%}",
            "Avg Win": f"{avg_win(d):.2%}",
            "Avg Loss": f"{avg_loss(d):.2%}",
            "Daily VaR (95%)": f"{value_at_risk(d):.2%}",
            "Recovery Factor": f"{recovery_factor(d):.2f}",
            "Skewness": f"{skewness(d):.2f}",
            "Kurtosis": f"{kurtosis(d):.2f}",
        }

    strat = _compute(df)
    data: dict[str, list[str]] = {
        "metric": list(strat.keys()),
        "strategy": list(strat.values()),
    }

    if base_df is not None:
        bench = _compute(base_df)
        data["benchmark"] = list(bench.values())

        # Add comparison metrics
        a, b = alpha_beta(df, base_df, periods)
        corr = correlation(df, base_df)
        ir = information_ratio(df, base_df, periods)
        ex_ret = excess_return(df, base_df)

        comparison_metrics = ["Alpha", "Beta", "Correlation", "Information Ratio", "Excess Return"]
        comparison_strat = [f"{a:.2%}", f"{b:.2f}", f"{corr:.2f}", f"{ir:.2f}", f"{ex_ret:.2%}"]
        comparison_bench = ["—", "—", "—", "—", "—"]

        data["metric"].extend(comparison_metrics)
        data["strategy"].extend(comparison_strat)
        data["benchmark"].extend(comparison_bench)

    return pl.DataFrame(data)
