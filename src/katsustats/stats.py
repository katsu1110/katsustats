"""
katsustats.stats — Financial metrics computed with Polars.

All functions accept a Polars or pandas DataFrame with columns ["date", "returns"]
where "returns" represents daily P&L (profit/loss) values, and return
scalar metric values or Polars DataFrames.
"""

from __future__ import annotations

import datetime as dt
import enum
import math

import numpy as np
import polars as pl

from ._constants import COL_DATE, COL_RETURNS
from ._dataframe import DataFrameLike, _compound_by_date, ensure_polars

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_returns(df: DataFrameLike, name: str = "df") -> pl.Series:
    """Extract the returns column as a Polars Series."""
    df = ensure_polars(df, name=name)
    return df.get_column(COL_RETURNS)


def _cumulative(returns: pl.Series) -> pl.Series:
    """Cumulative compounded returns: (1+r1)*(1+r2)*... - 1."""
    return (returns + 1).cum_prod() - 1


def _cumulative_value(returns: pl.Series) -> pl.Series:
    """Cumulative value curve starting from 1.0."""
    return (returns + 1).cum_prod()


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the complementary error function."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------


def total_return(df: DataFrameLike) -> float:
    """Total compounded return over the full period."""
    r = _to_returns(df)
    if r.len() == 0:
        return float("nan")
    return float((r + 1).product() - 1)


def cagr(df: DataFrameLike, periods: int = 252) -> float:
    """Compound Annual Growth Rate."""
    r = _to_returns(df)
    n = r.len()
    if n == 0:
        return float("nan")
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


def _float_or_nan(value: object) -> float:
    """Convert a Polars aggregate result to float, returning NaN for None."""
    if value is None:
        return float("nan")
    return float(value)


def max_drawdown(df: DataFrameLike) -> float:
    """Maximum drawdown (returned as a negative value)."""
    r = _to_returns(df)
    cumval = _cumulative_value(r)
    running_max = cumval.cum_max().clip(lower_bound=1.0)
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
    return _float_or_nan(_to_returns(df).max())


def worst_day(df: DataFrameLike) -> float:
    """Worst single-day return."""
    return _float_or_nan(_to_returns(df).min())


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
    return _float_or_nan(r.quantile(alpha, interpolation="linear"))


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
    return _float_or_nan(r.skew())


def kurtosis(df: DataFrameLike) -> float:
    """Excess kurtosis of daily returns."""
    r = _to_returns(df)
    return _float_or_nan(r.kurtosis())


# ---------------------------------------------------------------------------
# Risk-Adjusted Ratios
# ---------------------------------------------------------------------------


def omega_ratio(df: DataFrameLike, threshold: float = 0.0) -> float:
    """Omega ratio: sum(positive excess) / sum(|negative excess|).
    Values > 1 indicate more upside than downside relative to *threshold*.
    """
    r = _to_returns(df)
    excess = r - threshold
    numerator = excess.clip(lower_bound=0.0).sum()
    denominator = (-excess).clip(lower_bound=0.0).sum()
    if denominator is None or denominator == 0:
        if numerator is None or numerator == 0:
            return float("nan")
        return float("inf")
    return float(numerator / denominator)


def ulcer_index(df: DataFrameLike) -> float:
    """Ulcer index: sqrt(mean of squared running drawdowns).

    Measures downside risk as the depth and duration of drawdowns.
    Lower values are better.
    """
    r = _to_returns(df)
    if r.len() == 0:
        return float("nan")
    cumval = _cumulative_value(r)
    running_max = cumval.cum_max().clip(lower_bound=1.0)
    dd = (cumval - running_max) / running_max
    return float((dd**2).mean() ** 0.5)


def martin_ratio(df: DataFrameLike, rf: float = 0.0, periods: int = 252) -> float:
    """Martin (Ulcer Performance Index): annualized excess return / ulcer index."""
    r = _to_returns(df)
    if r.len() == 0:
        return float("nan")
    rf_per_period = rf / periods
    annualized_excess = (float(r.mean()) - rf_per_period) * periods
    ui = ulcer_index(df)
    if ui == 0:
        if annualized_excess == 0:
            return 0.0
        return float("inf") if annualized_excess > 0 else float("-inf")
    return annualized_excess / ui


def gain_to_pain(df: DataFrameLike) -> float:
    """Gain-to-pain ratio: sum(all returns) / sum(|negative returns|)."""
    r = _to_returns(df)
    total_ret = r.sum()
    if total_ret is None:
        return float("nan")
    losses = r.filter(r < 0).abs().sum()
    if losses is None or losses == 0:
        if total_ret == 0:
            return float("nan")
        return float("inf")
    return float(total_ret / losses)


def kelly_criterion(df: DataFrameLike, rf: float = 0.0, periods: int = 252) -> float:
    """Half-Kelly criterion: 0.5 × mean(excess return) / variance(excess return).

    Returns the optimal fraction of capital to allocate per trade / period
    under the Kelly framework (half-Kelly used as a conservative estimate).
    """
    r = _to_returns(df)
    if r.len() < 2:
        return float("nan")
    rf_per_period = rf / periods
    excess = r - rf_per_period
    variance = float(excess.var())
    if variance == 0:
        mean_excess = float(excess.mean())
        if mean_excess == 0:
            return 0.0
        return float("inf") if mean_excess > 0 else float("-inf")
    return float(0.5 * excess.mean() / variance)


def payoff_ratio(df: DataFrameLike) -> float:
    """Payoff ratio: average win / |average loss|."""
    r = _to_returns(df)
    wins = r.filter(r > 0)
    losses = r.filter(r < 0)
    if losses.len() == 0:
        if wins.len() == 0:
            return float("nan")
        return float("inf")
    al = abs(float(losses.mean()))
    if al == 0:
        return float("inf")
    if wins.len() == 0:
        return 0.0
    return float(wins.mean() / al)


def probabilistic_sharpe(
    df: DataFrameLike,
    benchmark_sharpe: float = 0.0,
    rf: float = 0.0,
    periods: int = 252,
) -> float:
    """Probabilistic Sharpe ratio (Bailey-López de Prado).

    The probability that the observed (annualised) Sharpe ratio is greater than
    a given benchmark Sharpe (default 0.0).
    """
    r = _to_returns(df)
    n = r.len()
    if n < 2:
        return float("nan")
    rf_per_period = rf / periods
    excess = r - rf_per_period
    std = float(excess.std())
    if std == 0:
        return float("nan")
    sr = float(excess.mean()) / std * math.sqrt(periods)
    skew = excess.skew()
    excess_kurt = excess.kurtosis()
    if skew is None or excess_kurt is None:
        return float("nan")
    skew = float(skew)
    kurt = float(excess_kurt + 3.0)
    var_est = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (n - 1)
    if var_est <= 0:
        return float("nan")
    z = (sr - benchmark_sharpe) / math.sqrt(var_est)
    return _norm_cdf(z)


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
        df.with_columns(pl.col(COL_DATE).cast(pl.Date))
        .sort(COL_DATE)
        .group_by_dynamic(COL_DATE, every=every)
        .agg((pl.col(COL_RETURNS) + 1).product() - 1)
        .get_column(COL_RETURNS)
    )


def _daily_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Compound returns to one row per calendar date."""
    normalised = df.with_columns(pl.col(COL_DATE).cast(pl.Date)).sort(COL_DATE)
    return _compound_by_date(normalised)


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
    """Fraction of days with non-zero returns (active market exposure)."""
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
    dates = df.get_column(COL_DATE).to_list()
    cumval = _cumulative_value(r)
    running_max = cumval.cum_max().clip(lower_bound=1.0)
    dd = (cumval - running_max) / running_max

    # Identify drawdown periods (contiguous blocks where dd < 0)
    dd_np = dd.to_numpy()
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
                    "start": dates[start_idx],
                    "trough": dates[trough_idx],
                    "recovery": dates[recovery_idx]
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
    joined = df.join(
        base_df.rename({COL_RETURNS: "_base_returns"}), on=COL_DATE, how="inner"
    )
    r = joined.get_column(COL_RETURNS).to_numpy()
    b = joined.get_column("_base_returns").to_numpy()

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
    joined = df.join(
        base_df.rename({COL_RETURNS: "_base_returns"}), on=COL_DATE, how="inner"
    )
    r = joined.get_column(COL_RETURNS).to_numpy()
    b = joined.get_column("_base_returns").to_numpy()
    return float(np.corrcoef(r, b)[0, 1])


def information_ratio(
    df: DataFrameLike, base_df: DataFrameLike, periods: int = 252
) -> float:
    """Annualized information ratio (excess return / tracking error)."""
    df = ensure_polars(df)
    base_df = ensure_polars(base_df, "base_df")
    joined = df.join(
        base_df.rename({COL_RETURNS: "_base_returns"}), on=COL_DATE, how="inner"
    )
    excess = joined.get_column(COL_RETURNS) - joined.get_column("_base_returns")
    te = excess.std()
    if te is None:
        return float("nan")
    te_f = float(te)
    if te_f == 0:
        return 0.0
    return float(excess.mean() / te_f * np.sqrt(periods))


def excess_return(df: DataFrameLike, base_df: DataFrameLike) -> float:
    """Total compounded excess return vs benchmark (aligned on common dates)."""
    df = ensure_polars(df)
    base_df = ensure_polars(base_df, "base_df")
    joined = df.join(
        base_df.rename({COL_RETURNS: "_base_returns"}), on=COL_DATE, how="inner"
    )
    strat_ret = float((joined.get_column(COL_RETURNS) + 1).product() - 1)
    bench_ret = float((joined.get_column("_base_returns") + 1).product() - 1)
    return strat_ret - bench_ret


def treynor_ratio(
    df: DataFrameLike, base_df: DataFrameLike, rf: float = 0.0, periods: int = 252
) -> float:
    """Treynor ratio: annualized excess return / beta.

    Measures risk-adjusted return per unit of systematic (market) risk.
    """
    _, beta = alpha_beta(df, base_df, periods)
    if beta == 0:
        return 0.0

    df = ensure_polars(df)
    base_df = ensure_polars(base_df, "base_df")
    joined = df.join(
        base_df.rename({COL_RETURNS: "_base_returns"}), on=COL_DATE, how="inner"
    )
    r = joined.get_column(COL_RETURNS)
    rf_per_period = rf / periods
    annualized_excess = (float(r.mean()) - rf_per_period) * periods
    return annualized_excess / beta


def r_squared(df: DataFrameLike, base_df: DataFrameLike) -> float:
    """R-squared: coefficient of determination from regression vs benchmark.

    Proportion of strategy return variance explained by the benchmark.
    """
    c = correlation(df, base_df)
    return c * c


def up_capture(df: DataFrameLike, base_df: DataFrameLike) -> float:
    """Up-market capture ratio.

    Average strategy return on benchmark up-days / average benchmark return
    on those same days.  Values > 1 mean the strategy captures more upside.
    """
    df = ensure_polars(df)
    base_df = ensure_polars(base_df, "base_df")
    joined = df.join(
        base_df.rename({COL_RETURNS: "_base_returns"}), on=COL_DATE, how="inner"
    )
    up_mask = joined.get_column("_base_returns") > 0
    strat_up = joined.get_column(COL_RETURNS).filter(up_mask)
    base_up = joined.get_column("_base_returns").filter(up_mask)
    if base_up.len() == 0:
        return float("nan")
    base_mean = float(base_up.mean())
    if base_mean == 0:
        return float("nan")
    return float(strat_up.mean() / base_mean)


def down_capture(df: DataFrameLike, base_df: DataFrameLike) -> float:
    """Down-market capture ratio.

    Average strategy return on benchmark down-days / average benchmark return
    on those same days.  Values < 1 mean the strategy loses less in down markets.
    """
    df = ensure_polars(df)
    base_df = ensure_polars(base_df, "base_df")
    joined = df.join(
        base_df.rename({COL_RETURNS: "_base_returns"}), on=COL_DATE, how="inner"
    )
    down_mask = joined.get_column("_base_returns") < 0
    strat_down = joined.get_column(COL_RETURNS).filter(down_mask)
    base_down = joined.get_column("_base_returns").filter(down_mask)
    if base_down.len() == 0:
        return float("nan")
    base_mean = float(base_down.mean())
    if base_mean == 0:
        return float("nan")
    return float(strat_down.mean() / base_mean)


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

    df = df.sort(COL_DATE)
    base_df = base_df.sort(COL_DATE)
    assert df[COL_DATE].n_unique() == df.height, "df must have one row per date"
    assert base_df[COL_DATE].n_unique() == base_df.height, (
        "base_df must have one row per date"
    )

    base_features = base_df.with_columns(
        ((pl.col(COL_RETURNS) + 1).cum_prod() - 1).alias("_cumret"),
        pl.col(COL_RETURNS)
        .rolling_std(window_size=vol_window, ddof=1)
        .alias("_rolling_vol"),
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
            .select([COL_DATE, "regime"])
        )
        aligned = df.join(base_regimes, on=COL_DATE, how="inner")

    regimes = [
        "bull_low_vol",
        "bull_high_vol",
        "bear_low_vol",
        "bear_high_vol",
    ]
    rows: list[dict[str, float | int | str]] = []
    for regime in regimes:
        subset = aligned.filter(pl.col("regime") == regime).select(["date", "returns"])
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
        df.with_columns(pl.col(COL_DATE).cast(pl.Date).dt.weekday().alias("dow"))
        .group_by("dow")
        .agg(
            pl.col(COL_RETURNS).mean().alias("mean_return"),
            (pl.col(COL_RETURNS) > 0).mean().alias("win_rate"),
            ((pl.col(COL_RETURNS) + 1).product() - 1).alias("total_return"),
            pl.col(COL_RETURNS).count().alias("count"),
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
    df: DataFrameLike, window: int = 126, periods: int = 252, rf: float = 0.0
) -> pl.DataFrame:
    """Rolling Sharpe ratio over a given window."""
    df = ensure_polars(df)
    rf_per_period = rf / periods
    return (
        df.sort(COL_DATE)
        .with_columns(
            [
                pl.col(COL_RETURNS).rolling_mean(window_size=window).alias("_rm"),
                pl.col(COL_RETURNS)
                .rolling_std(window_size=window, ddof=1)
                .alias("_rs"),
            ]
        )
        .with_columns(
            pl.when(pl.col("_rs") > 0)
            .then((pl.col("_rm") - rf_per_period) / pl.col("_rs") * float(periods**0.5))
            .alias("rolling_sharpe")
        )
        .select([COL_DATE, "rolling_sharpe"])
    )


def rolling_sortino(
    df: DataFrameLike,
    window: int = 126,
    periods: int = 252,
    rf: float = 0.0,
) -> pl.DataFrame:
    """Rolling Sortino ratio over a given window."""
    df = ensure_polars(df)
    rf_per_period = rf / periods
    return (
        df.sort(COL_DATE)
        .with_columns((pl.col(COL_RETURNS) - rf_per_period).alias("_excess"))
        .with_columns(
            pl.col("_excess").rolling_mean(window_size=window).alias("_mean_excess"),
            (
                pl.col("_excess")
                .clip(upper_bound=0.0)
                .pow(2)
                .rolling_mean(window_size=window)
                .sqrt()
            ).alias("_downside"),
        )
        .with_columns(
            pl.when(pl.col("_mean_excess").is_null())
            .then(pl.lit(None))
            .otherwise(
                pl.when(pl.col("_downside") > 0)
                .then(
                    pl.col("_mean_excess") / pl.col("_downside") * float(periods**0.5)
                )
                .otherwise(pl.lit(None))
            )
            .alias("rolling_sortino")
        )
        .select([COL_DATE, "rolling_sortino"])
    )


def rolling_beta(
    df: DataFrameLike, base_df: DataFrameLike, window: int = 126
) -> pl.DataFrame:
    """Rolling beta (market exposure) vs benchmark over a given window."""
    df = ensure_polars(df)
    base_df = ensure_polars(base_df, "base_df")
    joined = df.join(
        base_df.rename({COL_RETURNS: "_base_returns"}), on=COL_DATE, how="inner"
    ).sort(COL_DATE)
    n = float(window)
    return (
        joined.with_columns(
            (pl.col(COL_RETURNS) * pl.col("_base_returns"))
            .rolling_mean(window_size=window)
            .alias("_mean_prod"),
            pl.col(COL_RETURNS).rolling_mean(window_size=window).alias("_mean_r"),
            pl.col("_base_returns").rolling_mean(window_size=window).alias("_mean_b"),
            pl.col("_base_returns")
            .rolling_std(window_size=window, ddof=1)
            .alias("_std_b"),
        )
        .with_columns(
            pl.when(pl.col("_std_b").is_null())
            .then(pl.lit(None))
            .otherwise(
                pl.when(pl.col("_std_b") > 0)
                .then(
                    (pl.col("_mean_prod") - pl.col("_mean_r") * pl.col("_mean_b"))
                    * n
                    / (n - 1.0)
                    / pl.col("_std_b").pow(2)
                )
                .otherwise(pl.lit(None))
            )
            .alias("rolling_beta")
        )
        .select([COL_DATE, "rolling_beta"])
    )


def rolling_correlation(
    df: DataFrameLike, base_df: DataFrameLike, window: int = 126
) -> pl.DataFrame:
    """Rolling Pearson correlation vs benchmark over a given window."""
    df = ensure_polars(df)
    base_df = ensure_polars(base_df, "base_df")
    joined = df.join(
        base_df.rename({COL_RETURNS: "_base_returns"}), on=COL_DATE, how="inner"
    ).sort(COL_DATE)
    n = float(window)
    return (
        joined.with_columns(
            (pl.col(COL_RETURNS) * pl.col("_base_returns"))
            .rolling_mean(window_size=window)
            .alias("_mean_prod"),
            pl.col(COL_RETURNS).rolling_mean(window_size=window).alias("_mean_r"),
            pl.col("_base_returns").rolling_mean(window_size=window).alias("_mean_b"),
            pl.col(COL_RETURNS).rolling_std(window_size=window, ddof=1).alias("_std_r"),
            pl.col("_base_returns")
            .rolling_std(window_size=window, ddof=1)
            .alias("_std_b"),
        )
        .with_columns(
            pl.when(pl.col("_mean_r").is_null())
            .then(pl.lit(None))
            .otherwise(
                pl.when((pl.col("_std_r") > 0) & (pl.col("_std_b") > 0))
                .then(
                    (pl.col("_mean_prod") - pl.col("_mean_r") * pl.col("_mean_b"))
                    * n
                    / (n - 1.0)
                    / (pl.col("_std_r") * pl.col("_std_b"))
                )
                .otherwise(pl.lit(None))
            )
            .alias("rolling_correlation")
        )
        .select([COL_DATE, "rolling_correlation"])
    )


def _window_max_drawdown(s: pl.Series) -> float:
    """Max drawdown over a single window (helper for rolling_map)."""
    cumval = (s + 1).cum_prod()
    running_max = cumval.cum_max().clip(lower_bound=1.0)
    dd = (cumval - running_max) / running_max
    return float(dd.min())


def rolling_drawdown(df: DataFrameLike, window: int = 126) -> pl.DataFrame:
    """Rolling max drawdown over a given window."""
    df = ensure_polars(df)
    return (
        df.sort(COL_DATE)
        .with_columns(
            pl.col(COL_RETURNS)
            .rolling_map(_window_max_drawdown, window_size=window)
            .alias("rolling_drawdown")
        )
        .select([COL_DATE, "rolling_drawdown"])
    )


def rolling_volatility_ratio(
    df: DataFrameLike, base_df: DataFrameLike, window: int = 126, periods: int = 252
) -> pl.DataFrame:
    """Rolling volatility ratio: strategy vol / benchmark vol over a given window."""
    df = ensure_polars(df)
    base_df = ensure_polars(base_df, "base_df")
    joined = df.join(
        base_df.rename({COL_RETURNS: "_base_returns"}), on=COL_DATE, how="inner"
    ).sort(COL_DATE)
    return (
        joined.with_columns(
            (
                pl.col(COL_RETURNS).rolling_std(window_size=window, ddof=1)
                * float(periods**0.5)
            ).alias("_vol_r"),
            (
                pl.col("_base_returns").rolling_std(window_size=window, ddof=1)
                * float(periods**0.5)
            ).alias("_vol_b"),
        )
        .with_columns(
            pl.when(pl.col("_vol_b").is_null())
            .then(pl.lit(None))
            .otherwise(
                pl.when(pl.col("_vol_b") > 0)
                .then(pl.col("_vol_r") / pl.col("_vol_b"))
                .otherwise(pl.lit(None))
            )
            .alias("rolling_vol_ratio")
        )
        .select([COL_DATE, "rolling_vol_ratio"])
    )


def rolling_volatility(
    df: DataFrameLike, window: int = 126, periods: int = 252
) -> pl.DataFrame:
    """Rolling annualized volatility over a given window."""
    df = ensure_polars(df)
    return (
        df.sort(COL_DATE)
        .with_columns(
            (
                pl.col(COL_RETURNS).rolling_std(window_size=window, ddof=1)
                * float(periods**0.5)
            ).alias("rolling_vol")
        )
        .select([COL_DATE, "rolling_vol"])
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
    ("Omega Ratio", "omega_ratio", "float"),
    ("Ulcer Index", "ulcer_index", "pct"),
    ("Martin Ratio", "martin_ratio", "float"),
    ("Gain-to-Pain Ratio", "gain_to_pain", "float"),
    ("Kelly Criterion", "kelly_criterion", "float"),
    ("Probabilistic Sharpe", "probabilistic_sharpe", "float"),
    ("Payoff Ratio", "payoff_ratio", "float"),
]

_COMPARISON_METRIC_SPECS = [
    ("Alpha", "alpha", "pct"),
    ("Beta", "beta", "float"),
    ("Correlation", "correlation", "float"),
    ("Information Ratio", "information_ratio", "float"),
    ("Excess Return", "excess_return", "pct"),
    ("Treynor Ratio", "treynor_ratio", "float"),
    ("R-Squared", "r_squared", "pct"),
    ("Up Capture", "up_capture", "pct"),
    ("Down Capture", "down_capture", "pct"),
]


def _summary_metric_values(
    df: DataFrameLike, rf: float = 0.0, periods: int = 252
) -> dict[str, float]:
    """
    Compute the raw numeric summary metrics for a single return series.

    Returned keys:
        total_return, cagr, sharpe, sortino, max_drawdown, calmar,
        volatility, win_rate, profit_factor, best_day, worst_day,
        avg_win, avg_loss, value_at_risk, recovery_factor, skewness,
        kurtosis, omega_ratio, ulcer_index, martin_ratio, gain_to_pain,
        kelly_criterion, probabilistic_sharpe, payoff_ratio
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
        "omega_ratio": float(omega_ratio(df)),
        "ulcer_index": float(ulcer_index(df)),
        "martin_ratio": float(martin_ratio(df, rf, periods)),
        "gain_to_pain": float(gain_to_pain(df)),
        "kelly_criterion": float(kelly_criterion(df, rf, periods)),
        "probabilistic_sharpe": float(probabilistic_sharpe(df, rf=rf, periods=periods)),
        "payoff_ratio": float(payoff_ratio(df)),
    }


def _comparison_metric_values(
    df: DataFrameLike,
    base_df: DataFrameLike,
    rf: float = 0.0,
    periods: int = 252,
) -> dict[str, float]:
    """
    Compute raw numeric metrics that compare a strategy to a benchmark.

    Returned keys:
        alpha, beta, correlation, information_ratio, excess_return,
        treynor_ratio, r_squared, up_capture, down_capture
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
        "treynor_ratio": float(treynor_ratio(df, base_df, rf, periods)),
        "r_squared": float(r_squared(df, base_df)),
        "up_capture": float(up_capture(df, base_df)),
        "down_capture": float(down_capture(df, base_df)),
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
        df: Polars or pandas DataFrame with ["date", "returns"] columns.
        base_df: Optional benchmark DataFrame with the same schema. When
            provided, comparison metrics are added to the returned dict.
        rf: Annualized risk-free rate used by risk-adjusted metrics.
        periods: Number of return periods per year.

    Returns:
        A dict[str, float] with these base keys:
            total_return, cagr, sharpe, sortino, max_drawdown, calmar,
            volatility, win_rate, profit_factor, best_day, worst_day,
            avg_win, avg_loss, value_at_risk, recovery_factor, skewness,
            kurtosis, omega_ratio, ulcer_index, martin_ratio, gain_to_pain,
            kelly_criterion, probabilistic_sharpe, payoff_ratio

        When base_df is provided, these additional keys are included:
            alpha, beta, correlation, information_ratio, excess_return,
            treynor_ratio, r_squared, up_capture, down_capture

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
        raw.update(_comparison_metric_values(df, base_df, rf, periods))
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
        comparison = _comparison_metric_values(df, base_df, rf, periods)
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


class PeriodLabel(str, enum.Enum):
    MTD = "MTD"
    QTD = "QTD"
    YTD = "YTD"
    ONE_YEAR = "1Y"
    THREE_YEAR = "3Y"
    FIVE_YEAR = "5Y"
    SI = "SI"


_PERIOD_LABELS = [
    PeriodLabel.MTD,
    PeriodLabel.QTD,
    PeriodLabel.YTD,
    PeriodLabel.ONE_YEAR,
    PeriodLabel.THREE_YEAR,
    PeriodLabel.FIVE_YEAR,
    PeriodLabel.SI,
]


def _subtract_years(d: dt.date, n: int) -> dt.date:
    try:
        return dt.date(d.year - n, d.month, d.day)
    except ValueError:  # Feb 29 on non-leap year
        return dt.date(d.year - n, d.month, d.day - 1)


def _period_cutoff(anchor: pl.Date, label: PeriodLabel) -> pl.Date | None:
    """Return the start date for a named period, or None if insufficient data."""
    a: dt.date = anchor
    if label == PeriodLabel.MTD:
        return dt.date(a.year, a.month, 1)
    if label == PeriodLabel.QTD:
        q_start_month = ((a.month - 1) // 3) * 3 + 1
        return dt.date(a.year, q_start_month, 1)
    if label == PeriodLabel.YTD:
        return dt.date(a.year, 1, 1)

    if label == PeriodLabel.ONE_YEAR:
        return _subtract_years(a, 1)
    if label == PeriodLabel.THREE_YEAR:
        return _subtract_years(a, 3)
    if label == PeriodLabel.FIVE_YEAR:
        return _subtract_years(a, 5)
    return None  # SI — caller uses full series


_TRAILING_LABELS = {PeriodLabel.ONE_YEAR, PeriodLabel.THREE_YEAR, PeriodLabel.FIVE_YEAR}


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
    first = df.get_column(COL_DATE).min()
    if require_full_window and cutoff <= first:
        return float("nan")
    subset = df.filter(pl.col(COL_DATE) >= cutoff)
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
    df = _daily_returns(ensure_polars(df))
    has_benchmark = base_df is not None
    if has_benchmark:
        base_df = _daily_returns(ensure_polars(base_df, name="base_df"))
        # Align to common dates so strategy and benchmark use the same anchor.
        df = df.join(
            base_df.rename({COL_RETURNS: "_base_returns"}), on=COL_DATE, how="inner"
        ).sort(COL_DATE)

    if df.height == 0:
        row: dict[str, float] = {"strategy": float("nan")}
        if has_benchmark:
            row["benchmark"] = float("nan")
        return {lbl.value: dict(row) for lbl in _PERIOD_LABELS}

    anchor = df.get_column(COL_DATE).max()
    first = df.get_column(COL_DATE).min()

    exprs = []
    cols = [(COL_RETURNS, "strategy")]
    if has_benchmark:
        cols.append(("_base_returns", "benchmark"))

    for lbl in _PERIOD_LABELS:
        cutoff = _period_cutoff(anchor, lbl)  # None for SI
        full_window = lbl in _TRAILING_LABELS

        if cutoff is not None:
            mask = pl.col(COL_DATE) >= cutoff

        for col_name, prefix in cols:
            out_key = f"{prefix}_{lbl.value}"
            if cutoff is None:
                expr = (pl.col(col_name) + 1).product() - 1
            elif full_window and cutoff <= first:
                expr = pl.lit(float("nan"))
            else:
                filtered = pl.col(col_name).filter(mask)
                expr = (
                    pl.when(filtered.count() > 0)
                    .then((filtered + 1).product() - 1)
                    .otherwise(pl.lit(float("nan")))
                )

            exprs.append(expr.alias(out_key))

    res_df = df.select(exprs)
    res_dict = res_df.to_dicts()[0]

    result: dict[str, dict[str, float]] = {}
    for lbl in _PERIOD_LABELS:
        entry: dict[str, float] = {"strategy": float(res_dict[f"strategy_{lbl.value}"])}
        if has_benchmark:
            entry["benchmark"] = float(res_dict[f"benchmark_{lbl.value}"])
        result[lbl.value] = entry

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
        v = raw[lbl.value]
        rows_period.append(lbl.value)
        sv = v["strategy"]
        rows_strat.append("—" if (sv != sv) else f"{sv:.2%}")
        if base_df is not None:
            bv = v.get("benchmark", float("nan"))
            rows_bench.append("—" if (bv != bv) else f"{bv:.2%}")

    data: dict[str, list[str]] = {"period": rows_period, "strategy": rows_strat}
    if base_df is not None:
        data["benchmark"] = rows_bench
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------------------------


def _distribution_stats(
    arr: np.ndarray, *, with_quartiles: bool = False
) -> dict[str, float]:
    """Summary statistics of a 1-D array, dropping NaNs."""
    a = arr[~np.isnan(arr)]
    nan = float("nan")
    if len(a) == 0:
        result: dict[str, float] = {
            "min": nan,
            "max": nan,
            "mean": nan,
            "median": nan,
            "std": nan,
            "percentile_5": nan,
            "percentile_95": nan,
        }
        if with_quartiles:
            result["percentile_25"] = nan
            result["percentile_75"] = nan
        return result
    result = {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "percentile_5": float(np.percentile(a, 5)),
        "percentile_95": float(np.percentile(a, 95)),
    }
    if with_quartiles:
        result["percentile_25"] = float(np.percentile(a, 25))
        result["percentile_75"] = float(np.percentile(a, 75))
    return result


def _build_sim_returns(
    arr: np.ndarray, sims: int, seed: int | None, method: str = "bootstrap"
) -> np.ndarray:
    """Return (n_periods, sims) raw returns matrix. Column 0 = original."""
    if method not in ("bootstrap", "shuffle"):
        raise ValueError(f"method must be 'bootstrap' or 'shuffle', got {method!r}")
    n = len(arr)
    rng = np.random.default_rng(seed)
    sim_returns = np.empty((n, sims))
    sim_returns[:, 0] = arr
    for i in range(1, sims):
        if method == "bootstrap":
            sim_returns[:, i] = arr[rng.integers(0, n, n)]
        else:
            sim_returns[:, i] = rng.permutation(arr)
    return sim_returns


def _sim_max_drawdowns(cum_paths: np.ndarray) -> np.ndarray:
    """Max drawdown per simulation path (column), anchored to initial price 1."""
    prices = 1.0 + cum_paths
    running_max = np.maximum(np.maximum.accumulate(prices, axis=0), 1.0)
    return ((prices - running_max) / running_max).min(axis=0)


def _simulate_paths(
    r: pl.Series, sims: int, seed: int | None, method: str = "bootstrap"
) -> np.ndarray:
    """Return (n_periods, sims) cumulative-returns array. Column 0 = original."""
    if sims < 1:
        raise ValueError("sims must be >= 1")
    arr = r.drop_nulls().to_numpy()
    if len(arr) == 0:
        raise ValueError("monte carlo requires at least one return")
    return np.cumprod(1 + _build_sim_returns(arr, sims, seed, method), axis=0) - 1


def monte_carlo_paths(
    df: DataFrameLike,
    sims: int = 1000,
    seed: int | None = None,
    method: str = "bootstrap",
) -> pl.DataFrame:
    """Simulate return paths by resampling historical returns.

    Returns a wide Polars DataFrame with columns ['step', 'sim_0', 'sim_1', ...]
    of cumulative compounded returns. 'sim_0' is the original (unshuffled) path.

    Args:
        df: Return series.
        sims: Number of simulation paths.
        seed: Random seed for reproducibility.
        method: Resampling method — ``"bootstrap"`` (default, with replacement)
            or ``"shuffle"`` (without replacement).
    """
    r = _to_returns(df)
    cum_paths = _simulate_paths(r, sims, seed, method)
    n = cum_paths.shape[0]
    data: dict[str, object] = {"step": np.arange(1, n + 1)}
    for i in range(sims):
        data[f"sim_{i}"] = cum_paths[:, i]
    return pl.DataFrame(data)


def monte_carlo_summary(
    df: DataFrameLike,
    sims: int = 1000,
    bust: float | None = None,
    goal: float | None = None,
    rf: float = 0.0,
    periods: int = 252,
    seed: int | None = None,
    method: str = "bootstrap",
) -> dict:
    """Probabilistic risk summary via Monte Carlo path simulation.

    Resamples historical returns sims times, then returns distributions of
    terminal value, max drawdown, Sharpe ratio, and CAGR, plus optional
    bust and goal probabilities.

    Args:
        df: Return series.
        sims: Number of simulation paths.
        bust: Drawdown threshold for bust probability.
        goal: Return threshold for goal probability.
        rf: Annualized risk-free rate.
        periods: Trading days per year.
        seed: Random seed for reproducibility.
        method: Resampling method — ``"bootstrap"`` (default) or ``"shuffle"``.

    Returns a dict with keys: terminal, maxdd, sharpe, cagr,
    bust_probability, goal_probability, sims, seed.
    """
    r = _to_returns(df)
    arr = r.drop_nulls().to_numpy()
    if len(arr) == 0:
        raise ValueError("monte carlo requires at least one return")
    if sims < 1:
        raise ValueError("sims must be >= 1")
    n = len(arr)
    sim_returns = _build_sim_returns(arr, sims, seed, method)
    cum_paths = np.cumprod(1 + sim_returns, axis=0) - 1
    terminal = cum_paths[-1, :]

    maxdd_per_path = _sim_max_drawdowns(cum_paths)

    # Sharpe per path
    excess = sim_returns.mean(axis=0) - rf / periods
    std_per_path = sim_returns.std(axis=0, ddof=1)
    sharpe_per_path = np.where(
        std_per_path > 0, excess / std_per_path * np.sqrt(float(periods)), 0.0
    )

    # CAGR per path
    years = n / periods
    cagr_per_path = np.where(
        terminal > -1, (1.0 + terminal) ** (1.0 / years) - 1.0, np.nan
    )

    bust_prob: float | None = None
    if bust is not None:
        bust_prob = float((maxdd_per_path <= bust).sum() / sims)
    goal_prob: float | None = None
    if goal is not None:
        goal_prob = float((terminal >= goal).sum() / sims)

    return {
        "terminal": _distribution_stats(terminal, with_quartiles=True),
        "maxdd": _distribution_stats(maxdd_per_path),
        "sharpe": _distribution_stats(sharpe_per_path),
        "cagr": _distribution_stats(cagr_per_path),
        "bust_probability": bust_prob,
        "goal_probability": goal_prob,
        "sims": sims,
        "seed": seed,
    }
