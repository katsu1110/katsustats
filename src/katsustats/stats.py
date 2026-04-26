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
    std = excess.std()
    if std is None:
        return float("nan")
    std = float(std)
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


def cvar(df: DataFrameLike, alpha: float = 0.05) -> float:
    """CVaR / Expected Shortfall: mean of returns at or below the VaR threshold."""
    r = _to_returns(df)
    if r.len() == 0:
        return float("nan")
    threshold = r.quantile(alpha, interpolation="linear")
    if threshold is None:
        return float("nan")
    tail = r.filter(r <= threshold)
    if tail.len() == 0:
        return float("nan")
    return float(tail.mean())


def tail_ratio(df: DataFrameLike, cutoff: float = 0.95) -> float:
    """Upper-tail / lower-tail magnitude: |q_cutoff| / |q_{1-cutoff}|.

    Values > 1 indicate a fatter right tail; < 1 indicate a fatter left tail.
    """
    r = _to_returns(df)
    upper = r.quantile(cutoff, interpolation="linear")
    lower = r.quantile(1.0 - cutoff, interpolation="linear")
    if upper is None or lower is None:
        return float("nan")
    if upper == 0.0 and lower == 0.0:
        return float("nan")
    if lower == 0.0:
        return float("inf")
    return float(abs(upper) / abs(lower))


def common_sense_ratio(df: DataFrameLike) -> float:
    """Profit factor × tail ratio — combines consistency with tail asymmetry."""
    return float(profit_factor(df) * tail_ratio(df))


def risk_of_ruin(df: DataFrameLike, ruin_threshold: float = -0.5) -> float:
    """Estimated probability of reaching ruin_threshold cumulative loss.

    ruin_threshold must be negative (e.g. -0.5 means a 50% drawdown).
    Uses the classic formula: ((1 - edge) / (1 + edge)) ^ n_units, where
    edge = win_rate - (1 - win_rate) / payoff_ratio and n_units scales the
    threshold by the average losing return.
    """
    if ruin_threshold > 0:
        raise ValueError("ruin_threshold must be <= 0 (e.g. -0.5 for a 50% loss)")
    wr = win_rate(df)
    aw = avg_win(df)
    al = abs(avg_loss(df))
    if al == 0.0:
        return 0.0  # no losing days — ruin impossible
    if aw == 0.0:
        return 1.0  # no winning days — ruin certain
    pr = aw / al
    edge = wr - (1.0 - wr) / pr
    if edge <= 0.0:
        return 1.0
    n_units = abs(ruin_threshold) / al
    return float(((1.0 - edge) / (1.0 + edge)) ** n_units)


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
# Streaks, Period Extrema & Exposure
# ---------------------------------------------------------------------------


def _longest_streak(r: pl.Series, positive: bool) -> int:
    """Return the longest run of positive (positive=True) or negative values."""
    is_streak = ((r > 0) if positive else (r < 0)).fill_null(False)
    streak_lengths = (
        pl.DataFrame({"is_streak": is_streak})
        .with_columns(
            (pl.col("is_streak") != pl.col("is_streak").shift(1).fill_null(False))
            .cum_sum()
            .alias("grp")
        )
        .filter(pl.col("is_streak"))
        .group_by("grp")
        .len()
        .get_column("len")
    )
    max_len = streak_lengths.max()
    return int(max_len) if max_len is not None else 0


def _period_returns(df: pl.DataFrame, every: str) -> pl.Series:
    """Compounded returns aggregated by calendar period ('1mo' or '1y')."""
    return (
        df.with_columns(pl.col("date").cast(pl.Date))
        .sort("date")
        .group_by_dynamic("date", every=every)
        .agg((pl.col("pnl") + 1).product() - 1)
        .get_column("pnl")
    )


def consecutive_wins(df: DataFrameLike) -> int:
    """Longest streak of consecutive positive daily returns."""
    r = _to_returns(df)
    if r.len() == 0:
        return 0
    return _longest_streak(r, positive=True)


def consecutive_losses(df: DataFrameLike) -> int:
    """Longest streak of consecutive negative daily returns."""
    r = _to_returns(df)
    if r.len() == 0:
        return 0
    return _longest_streak(r, positive=False)


def positive_months_pct(df: DataFrameLike) -> float:
    """Fraction of calendar months with a positive compounded return."""
    df = ensure_polars(df)
    monthly = _period_returns(df, "1mo")
    return float((monthly > 0).mean()) if monthly.len() > 0 else float("nan")


def positive_years_pct(df: DataFrameLike) -> float:
    """Fraction of calendar years with a positive compounded return."""
    df = ensure_polars(df)
    yearly = _period_returns(df, "1y")
    return float((yearly > 0).mean()) if yearly.len() > 0 else float("nan")


def best_month(df: DataFrameLike) -> float:
    """Compounded return of the best calendar month."""
    df = ensure_polars(df)
    if df.height == 0:
        return float("nan")
    return float(_period_returns(df, "1mo").max())


def worst_month(df: DataFrameLike) -> float:
    """Compounded return of the worst calendar month."""
    df = ensure_polars(df)
    if df.height == 0:
        return float("nan")
    return float(_period_returns(df, "1mo").min())


def best_year(df: DataFrameLike) -> float:
    """Compounded return of the best calendar year."""
    df = ensure_polars(df)
    if df.height == 0:
        return float("nan")
    return float(_period_returns(df, "1y").max())


def worst_year(df: DataFrameLike) -> float:
    """Compounded return of the worst calendar year."""
    df = ensure_polars(df)
    if df.height == 0:
        return float("nan")
    return float(_period_returns(df, "1y").min())


def exposure(df: DataFrameLike) -> float:
    """Fraction of days with non-zero pnl (active market exposure)."""
    r = _to_returns(df)
    n = r.len()
    if n == 0:
        return float("nan")
    return float((r != 0).sum() / n)


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
                    "start": dates_np[start_idx].item(),
                    "trough": dates_np[trough_idx].item(),
                    "recovery": dates_np[recovery_idx].item()
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

    result = pl.DataFrame(
        {
            "start": pl.Series([p["start"] for p in periods], dtype=pl.Date),
            "trough": pl.Series([p["trough"] for p in periods], dtype=pl.Date),
            "recovery": pl.Series([p["recovery"] for p in periods], dtype=pl.Date),
            "max_dd": [p["max_dd"] for p in periods],
            "drawdown_days": [p["drawdown_days"] for p in periods],
            "recovery_days": [p["recovery_days"] for p in periods],
        }
    )
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
# Regime Analysis
# ---------------------------------------------------------------------------


def regime_stats(
    df: DataFrameLike,
    base_df: DataFrameLike | None,
    periods: int = 252,
    trend_window: int = 200,
    vol_window: int = 60,
) -> pl.DataFrame:
    """
    Break down strategy performance by benchmark-defined market regime.

    Returns one row for each of:
        bull_low_vol, bull_high_vol, bear_low_vol, bear_high_vol
    """
    assert base_df is not None, "base_df is required for regime_stats"
    assert trend_window > 0, "trend_window must be positive"
    assert vol_window > 0, "vol_window must be positive"

    df = ensure_polars(df, name="df")
    base_df = ensure_polars(base_df, name="base_df")
    assert "date" in df.columns, "df must have a 'date' column"
    assert "pnl" in df.columns, "df must have a 'pnl' column"
    assert "date" in base_df.columns, "base_df must have a 'date' column"
    assert "pnl" in base_df.columns, "base_df must have a 'pnl' column"

    df = df.sort("date")
    base_df = base_df.sort("date")
    assert df["date"].n_unique() == df.height, "df must have one row per date"
    assert base_df["date"].n_unique() == base_df.height, (
        "base_df must have one row per date"
    )

    base_features = base_df.with_columns(
        ((pl.col("pnl") + 1).cum_prod() - 1).alias("_cumret"),
        pl.col("pnl").rolling_std(window_size=vol_window, ddof=1).alias("_rolling_vol"),
    ).with_columns(
        pl.col("_cumret").rolling_mean(window_size=trend_window).alias("_trend_ma")
    )

    vol_median = base_features.get_column("_rolling_vol").median()
    if vol_median is None or np.isnan(vol_median):
        aligned = df.head(0).with_columns(pl.Series("regime", [], dtype=pl.String))
    else:
        base_regimes = (
            base_features.drop_nulls(["_trend_ma", "_rolling_vol"])
            .with_columns(
                pl.when(pl.col("_cumret") > pl.col("_trend_ma"))
                .then(pl.lit("bull"))
                .otherwise(pl.lit("bear"))
                .alias("_trend"),
                pl.when(pl.col("_rolling_vol") > float(vol_median))
                .then(pl.lit("high_vol"))
                .otherwise(pl.lit("low_vol"))
                .alias("_vol_regime"),
            )
            .with_columns(
                pl.concat_str(["_trend", "_vol_regime"], separator="_").alias("regime")
            )
            .select(["date", "regime"])
        )
        aligned = df.join(base_regimes, on="date", how="inner")

    regimes = [
        "bull_low_vol",
        "bull_high_vol",
        "bear_low_vol",
        "bear_high_vol",
    ]
    rows: list[dict[str, float | int | str]] = []
    for regime in regimes:
        subset = aligned.filter(pl.col("regime") == regime).select(["date", "pnl"])
        n_days = subset.height
        if n_days == 0:
            rows.append(
                {
                    "regime": regime,
                    "n_days": 0,
                    "cagr": float("nan"),
                    "sharpe": float("nan"),
                    "max_drawdown": float("nan"),
                    "win_rate": float("nan"),
                }
            )
            continue

        rows.append(
            {
                "regime": regime,
                "n_days": n_days,
                "cagr": cagr(subset, periods),
                "sharpe": sharpe(subset, periods=periods),
                "max_drawdown": max_drawdown(subset),
                "win_rate": win_rate(subset),
            }
        )

    return pl.DataFrame(rows).cast(
        {
            "regime": pl.String,
            "n_days": pl.Int64,
            "cagr": pl.Float64,
            "sharpe": pl.Float64,
            "max_drawdown": pl.Float64,
            "win_rate": pl.Float64,
        }
    )


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
    ("CVaR (95%)", "cvar", "pct"),
    ("Recovery Factor", "recovery_factor", "float"),
    ("Skewness", "skewness", "float"),
    ("Kurtosis", "kurtosis", "float"),
    ("Best Month", "best_month", "pct"),
    ("Worst Month", "worst_month", "pct"),
    ("Best Year", "best_year", "pct"),
    ("Worst Year", "worst_year", "pct"),
    ("Positive Months", "positive_months_pct", "pct"),
    ("Positive Years", "positive_years_pct", "pct"),
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
        "cvar": float(cvar(df)),
        "recovery_factor": float(recovery_factor(df)),
        "skewness": float(skewness(df)),
        "kurtosis": float(kurtosis(df)),
        "best_month": float(best_month(df)),
        "worst_month": float(worst_month(df)),
        "best_year": float(best_year(df)),
        "worst_year": float(worst_year(df)),
        "positive_months_pct": float(positive_months_pct(df)),
        "positive_years_pct": float(positive_years_pct(df)),
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


# ---------------------------------------------------------------------------
# Period Performance
# ---------------------------------------------------------------------------

_PERIOD_LABELS = ["MTD", "QTD", "YTD", "1Y", "3Y", "5Y", "SI"]


def _period_cutoff(anchor: pl.Date, label: str) -> pl.Date | None:
    """Return the start date for a named period, or None if insufficient data."""
    import datetime as dt

    a: dt.date = anchor
    if label == "MTD":
        return dt.date(a.year, a.month, 1)
    if label == "QTD":
        q_start_month = ((a.month - 1) // 3) * 3 + 1
        return dt.date(a.year, q_start_month, 1)
    if label == "YTD":
        return dt.date(a.year, 1, 1)

    def _subtract_years(d: dt.date, n: int) -> dt.date:
        try:
            return dt.date(d.year - n, d.month, d.day)
        except ValueError:  # Feb 29 on non-leap year
            return dt.date(d.year - n, d.month, d.day - 1)

    if label == "1Y":
        return _subtract_years(a, 1)
    if label == "3Y":
        return _subtract_years(a, 3)
    if label == "5Y":
        return _subtract_years(a, 5)
    return None  # SI — caller uses full series


_TRAILING_LABELS = {"1Y", "3Y", "5Y"}


def _trailing_return(
    df: pl.DataFrame, cutoff: pl.Date | None, *, require_full_window: bool = False
) -> float:
    """Compounded return from cutoff to end of df.  None means full series.

    require_full_window=True (used for 1Y/3Y/5Y): return NaN when the cutoff
    predates the first available data point — the history is genuinely too short.
    require_full_window=False (used for MTD/QTD/YTD): compute from the first
    available row on or after the cutoff, so a month/quarter/year that starts on
    a weekend or holiday still returns a value rather than "—".
    """
    if df.height == 0:
        return float("nan")
    if cutoff is None:
        return float(total_return(df))
    first = df.get_column("date").min()
    if require_full_window and cutoff <= first:
        return float("nan")
    subset = df.filter(pl.col("date") >= cutoff)
    return float(total_return(subset)) if subset.height > 0 else float("nan")


def period_performance_raw(
    df: DataFrameLike,
    base_df: DataFrameLike | None = None,
) -> dict[str, dict[str, float]]:
    """
    Trailing-period compounded returns for MTD, QTD, YTD, 1Y, 3Y, 5Y, SI.

    Returns a dict keyed by period label.  Each value is a dict with key
    "strategy" (always present) and "benchmark" (when base_df provided).
    1Y/3Y/5Y return float("nan") when the series is shorter than the window.
    Strategy and benchmark are aligned to common dates before computing so
    both columns in the table always reflect the same date range.
    """
    df = ensure_polars(df)
    df = df.sort("date")
    if base_df is not None:
        base_df = ensure_polars(base_df, name="base_df").sort("date")
        # Align to common dates so strategy and benchmark use the same anchor.
        joined = df.join(
            base_df.rename({"pnl": "_base_pnl"}), on="date", how="inner"
        ).sort("date")
        df = joined.select(["date", "pnl"])
        base_df = joined.select([pl.col("date"), pl.col("_base_pnl").alias("pnl")])

    if df.height == 0:
        row: dict[str, float] = {"strategy": float("nan")}
        if base_df is not None:
            row["benchmark"] = float("nan")
        return {lbl: dict(row) for lbl in _PERIOD_LABELS}

    anchor = df.get_column("date").max()
    result: dict[str, dict[str, float]] = {}
    for lbl in _PERIOD_LABELS:
        cutoff = _period_cutoff(anchor, lbl)  # None for SI
        full_window = lbl in _TRAILING_LABELS
        entry: dict[str, float] = {
            "strategy": _trailing_return(df, cutoff, require_full_window=full_window)
        }
        if base_df is not None:
            entry["benchmark"] = _trailing_return(
                base_df, cutoff, require_full_window=full_window
            )
        result[lbl] = entry

    return result


def period_performance(
    df: DataFrameLike,
    base_df: DataFrameLike | None = None,
) -> pl.DataFrame:
    """
    Formatted period-performance table (MTD, QTD, YTD, 1Y, 3Y, 5Y, SI).

    Returns a Polars DataFrame with columns [period, strategy] and optionally
    [benchmark] when base_df is provided.  Values are pre-formatted strings
    (e.g. "12.34%" or "—") to match the summary_metrics rendering contract.
    """
    raw = period_performance_raw(df, base_df)
    rows_period: list[str] = []
    rows_strat: list[str] = []
    rows_bench: list[str] = []

    for lbl in _PERIOD_LABELS:
        v = raw[lbl]
        rows_period.append(lbl)
        sv = v["strategy"]
        rows_strat.append("—" if (sv != sv) else f"{sv:.2%}")
        if base_df is not None:
            bv = v.get("benchmark", float("nan"))
            rows_bench.append("—" if (bv != bv) else f"{bv:.2%}")

    data: dict[str, list[str]] = {"period": rows_period, "strategy": rows_strat}
    if base_df is not None:
        data["benchmark"] = rows_bench
    return pl.DataFrame(data)
