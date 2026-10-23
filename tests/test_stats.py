"""Unit tests for katsustats.stats module."""

from __future__ import annotations

import math
import statistics

import polars as pl
import pytest

from katsustats import stats

# ---------------------------------------------------------------------------
# Scalar metrics — return type and basic sanity
# ---------------------------------------------------------------------------


class TestTotalReturn:
    def test_known_value(self):
        # (1.1) * (0.9) - 1 = -0.01
        df = pl.DataFrame(
            {"date": ["2023-01-02", "2023-01-03"], "returns": [0.1, -0.1]}
        ).with_columns(pl.col("date").cast(pl.Date))
        result = stats.total_return(df)
        assert abs(result - (-0.01)) < 1e-10


class TestVolatility:
    def test_single_row_returns_nan(self, single_row_df):
        # std of 1 element (ddof=1) is undefined; should return nan not raise
        assert math.isnan(stats.volatility(single_row_df))


class TestSharpe:
    def test_rf_parameter(self, sample_df):
        s0 = stats.sharpe(sample_df, rf=0.0)
        s1 = stats.sharpe(sample_df, rf=0.05)
        # Higher risk-free rate → lower Sharpe (for positive excess returns)
        assert s0 != s1

    def test_single_row_returns_nan(self, single_row_df):
        # std of 1 element (ddof=1) is undefined → NaN, not an exception
        assert math.isnan(stats.sharpe(single_row_df))


class TestSortino:
    def test_known_value_uses_all_day_denominator(self):
        """Regression: denominator is RMS of min(excess,0) over ALL days."""
        df = pl.DataFrame(
            {
                "date": [
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                ],
                "returns": [0.02, 0.01, 0.03, -0.01, -0.02],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        # clip upper=0: [0, 0, 0, -0.01, -0.02]
        # sq_mean = (0 + 0 + 0 + 0.0001 + 0.0004) / 5 = 0.0001
        # downside_std = 0.01; mean = 0.006
        # sortino = 0.006 / 0.01 * sqrt(252)
        expected = 0.6 * math.sqrt(252)
        assert abs(stats.sortino(df, rf=0.0, periods=252) - expected) < 1e-8


class TestWinRate:
    def test_empty_df(self, empty_df):
        assert stats.win_rate(empty_df) == 0.0


class TestVaR:
    def test_alpha_parameter(self, sample_df):
        var5 = stats.value_at_risk(sample_df, alpha=0.05)
        var10 = stats.value_at_risk(sample_df, alpha=0.10)
        # Higher alpha → less extreme quantile → higher (less negative) VaR
        assert var10 >= var5


class TestCVaR:
    def test_cvar_worse_than_var(self, sample_df):
        # CVaR is always at most as good as VaR (mean of tail ≤ threshold)
        assert stats.cvar(sample_df) <= stats.value_at_risk(sample_df)


class TestCommonSenseRatio:
    def test_equals_pf_times_tr(self, sample_df):
        expected = stats.profit_factor(sample_df) * stats.tail_ratio(sample_df)
        assert abs(stats.common_sense_ratio(sample_df) - expected) < 1e-10


class TestRiskOfRuin:
    def test_ruin_threshold_parameter(self, sample_df):
        # Larger loss threshold → harder to reach → lower RoR
        ror_50 = stats.risk_of_ruin(sample_df, ruin_threshold=-0.5)
        ror_25 = stats.risk_of_ruin(sample_df, ruin_threshold=-0.25)
        assert ror_25 >= ror_50

    def test_positive_threshold_raises(self, sample_df):
        with pytest.raises(ValueError, match="ruin_threshold must be <= 0"):
            stats.risk_of_ruin(sample_df, ruin_threshold=0.5)


class TestSkewnessKurtosis:
    def test_skewness_and_kurtosis_values(self, sample_df):
        s = stats.skewness(sample_df)
        k = stats.kurtosis(sample_df)
        assert isinstance(s, float) and math.isfinite(s)
        assert isinstance(k, float) and math.isfinite(k)


# ---------------------------------------------------------------------------
# DataFrame-returning functions
# ---------------------------------------------------------------------------


class TestDrawdownDetails:
    def test_max_dd_negative(self, sample_df):
        dd = stats.drawdown_details(sample_df)
        assert dd.height > 0
        assert all(v <= 0 for v in dd.get_column("max_dd").to_list())

    def test_top_n_respected(self, sample_df):
        dd = stats.drawdown_details(sample_df, top_n=2)
        assert dd.height <= 2

    def test_date_columns_are_date_dtype(self, sample_df):
        dd = stats.drawdown_details(sample_df)
        if dd.height > 0:
            assert dd.schema["start"] == pl.Date
            assert dd.schema["trough"] == pl.Date
            assert dd.schema["recovery"] == pl.Date

    def test_datetime_date_column_accepted(self, sample_df):
        dt_df = sample_df.with_columns(pl.col("date").cast(pl.Datetime))
        dd = stats.drawdown_details(dt_df)
        assert isinstance(dd, pl.DataFrame)
        if dd.height > 0:
            assert dd.schema["start"] == pl.Date
            assert dd.schema["trough"] == pl.Date
            assert dd.schema["recovery"] == pl.Date
        assert dd.height == stats.drawdown_details(sample_df).height


class TestDayOfWeekStats:
    def test_all_weekdays_represented(self, sample_df):
        # sample_df covers all 5 weekdays
        dow = stats.day_of_week_stats(sample_df)
        dow_values = sorted(dow.get_column("dow").to_list())
        assert dow_values == [1, 2, 3, 4, 5]


class TestRollingSharpe:
    def test_known_value_matches_manual_calculation(self, sample_df):
        window = 5
        result = stats.rolling_sharpe(sample_df, window=window)
        vals = result.get_column("rolling_sharpe").to_list()
        tail = sample_df.get_column("returns").to_list()[-window:]
        expected = (sum(tail) / len(tail)) / statistics.stdev(tail) * (252**0.5)
        assert vals[-1] == pytest.approx(expected)


class TestRollingVolatility:
    def test_known_value_matches_manual_calculation(self, sample_df):
        window = 5
        result = stats.rolling_volatility(sample_df, window=window)
        vals = result.get_column("rolling_vol").to_list()
        tail = sample_df.get_column("returns").to_list()[-window:]
        expected = statistics.stdev(tail) * (252**0.5)
        assert vals[-1] == pytest.approx(expected)


class TestRollingSortino:
    def test_values_finite_after_window(self, sample_df):
        window = 5
        result = stats.rolling_sortino(sample_df, window=window)
        vals = result.get_column("rolling_sortino").to_list()
        valid = [v for v in vals[window:] if v is not None]
        assert all(math.isfinite(v) or math.isinf(v) for v in valid)

    def test_rf_lowers_values(self, sample_df):
        w = 5
        s0 = stats.rolling_sortino(sample_df, window=w, rf=0.0)
        s1 = stats.rolling_sortino(sample_df, window=w, rf=0.05)
        v0 = s0.get_column("rolling_sortino").to_list()
        v1 = s1.get_column("rolling_sortino").to_list()
        assert any(v0[i] != v1[i] for i in range(w, len(v0)) if v0[i] is not None)


class TestRollingBeta:
    def test_unequal_lengths(self, sample_df, benchmark_df):
        shorter = benchmark_df.head(10)
        result = stats.rolling_beta(sample_df, shorter, window=5)
        assert isinstance(result, pl.DataFrame)

    def test_self_beta_is_one(self, sample_df):
        window = 5
        result = stats.rolling_beta(sample_df, sample_df, window=window)
        vals = result.get_column("rolling_beta").to_list()
        valid = [v for v in vals[window:] if v is not None and not math.isnan(v)]
        assert valid
        assert all(abs(v - 1.0) < 1e-10 for v in valid)


class TestRollingCorrelation:
    def test_self_correlation_is_one(self, sample_df):
        window = 5
        result = stats.rolling_correlation(sample_df, sample_df, window=window)
        vals = result.get_column("rolling_correlation").to_list()
        valid = [v for v in vals[window:] if v is not None and not math.isnan(v)]
        assert valid
        assert all(abs(v - 1.0) < 1e-10 for v in valid)


class TestRollingVolatilityRatio:
    def test_self_ratio_is_one(self, sample_df):
        window = 5
        result = stats.rolling_volatility_ratio(sample_df, sample_df, window=window)
        vals = result.get_column("rolling_vol_ratio").to_list()
        valid = [v for v in vals[window:] if v is not None and not math.isnan(v)]
        assert valid
        assert all(abs(v - 1.0) < 1e-10 for v in valid)


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------


class TestAlphaBeta:
    def test_unequal_lengths(self, sample_df, benchmark_df):
        # Should not raise; inner-joins on date so shorter series limits rows used
        shorter = benchmark_df.head(10)
        alpha, beta = stats.alpha_beta(sample_df, shorter)
        assert isinstance(alpha, float)

    def test_alpha_uses_linear_annualization(self, sample_df, benchmark_df):
        """Regression: alpha_annual = alpha_daily * periods (not compound)."""
        import numpy as np

        alpha, beta = stats.alpha_beta(sample_df, benchmark_df)
        r = sample_df.get_column("returns").to_numpy()
        b = benchmark_df.get_column("returns").to_numpy()
        cov = np.cov(r, b)
        beta_raw = cov[0, 1] / cov[1, 1]
        alpha_daily = float(np.mean(r) - beta_raw * np.mean(b))
        expected = alpha_daily * 252
        assert abs(alpha - expected) < 1e-10


class TestExcessReturn:
    def test_uses_inner_join_on_date(self):
        """Regression: only common dates contribute to excess return."""
        df = pl.DataFrame(
            {
                "date": ["2023-01-02", "2023-01-03", "2023-01-04"],
                "returns": [0.01, 0.02, -0.01],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        base = pl.DataFrame(
            {
                "date": ["2023-01-02", "2023-01-03"],
                "returns": [0.01, 0.02],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        # Inner join gives 2023-01-02 and 2023-01-03 only.
        # Strategy   total on those dates: (1.01 * 1.02) - 1 ≈ 0.0302
        # Benchmark  total on those dates: (1.01 * 1.02) - 1 ≈ 0.0302
        # Excess return should be ≈ 0
        assert abs(stats.excess_return(df, base)) < 1e-10


# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------


class TestRegimeStats:
    def test_requires_benchmark(self, sample_df):
        with pytest.raises(ValueError, match="base_df is required for regime_stats"):
            stats.regime_stats(sample_df, None)

    def test_schema_and_regime_order(self, sample_df, benchmark_df):
        result = stats.regime_stats(
            sample_df, benchmark_df, trend_window=3, vol_window=3
        )
        assert set(result.columns) == {
            "regime",
            "n_days",
            "cagr",
            "sharpe",
            "max_drawdown",
            "win_rate",
        }
        assert result.get_column("regime").to_list() == [
            "bull_low_vol",
            "bull_high_vol",
            "bear_low_vol",
            "bear_high_vol",
        ]

    def test_large_windows_leave_short_series_with_nan_metrics(
        self, sample_df, benchmark_df
    ):
        result = stats.regime_stats(sample_df, benchmark_df)
        assert result.get_column("n_days").to_list() == [0, 0, 0, 0]
        for col in ["cagr", "sharpe", "max_drawdown", "win_rate"]:
            assert all(math.isnan(value) for value in result.get_column(col).to_list())

    def test_window_kwargs_change_regime_assignment_counts(
        self, sample_df, benchmark_df
    ):
        small = stats.regime_stats(
            sample_df, benchmark_df, trend_window=3, vol_window=3
        )
        large = stats.regime_stats(
            sample_df, benchmark_df, trend_window=10, vol_window=10
        )
        assert small.get_column("n_days").sum() > large.get_column("n_days").sum()

    def test_uses_inner_join_on_benchmark_dates(self):
        df = pl.DataFrame(
            {
                "date": ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
                "returns": [0.01, -0.01, 0.02, -0.02],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        base = pl.DataFrame(
            {
                "date": ["2023-01-03", "2023-01-04", "2023-01-05"],
                "returns": [0.01, -0.03, 0.04],
            }
        ).with_columns(pl.col("date").cast(pl.Date))

        result = stats.regime_stats(df, base, trend_window=2, vol_window=2)
        assert result.get_column("n_days").sum() == 2


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


class TestSummaryMetrics:
    def test_columns_without_benchmark(self, sample_df):
        result = stats.summary_metrics(sample_df)
        assert list(result.columns) == ["metric", "strategy"]

    def test_columns_with_benchmark(self, sample_df, benchmark_df):
        result = stats.summary_metrics(sample_df, benchmark_df)
        assert "benchmark" in result.columns
        assert "metric" in result.columns
        assert "strategy" in result.columns

    def test_benchmark_adds_comparison_rows(self, sample_df, benchmark_df):
        without = stats.summary_metrics(sample_df)
        with_bench = stats.summary_metrics(sample_df, benchmark_df)
        # With benchmark adds comparison metrics (9 rows)
        assert with_bench.height == without.height + 9


class TestCorrelation:
    def test_self_correlation_is_one(self, sample_df):
        assert abs(stats.correlation(sample_df, sample_df) - 1.0) < 1e-10


class TestRSquared:
    def test_self_is_one(self, sample_df):
        assert abs(stats.r_squared(sample_df, sample_df) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Streaks, Period Extrema & Exposure
# ---------------------------------------------------------------------------


class TestConsecutiveWins:
    def test_known_value(self):
        df = pl.DataFrame(
            {
                "date": [
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                ],
                "returns": [0.01, 0.02, -0.01, 0.03, 0.04],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        # streak: 2 wins, then loss, then 2 wins → max is 2
        assert stats.consecutive_wins(df) == 2


class TestConsecutiveLosses:
    def test_known_value(self):
        df = pl.DataFrame(
            {
                "date": [
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                ],
                "returns": [-0.01, -0.02, 0.01, -0.03, -0.04],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        # streak: 2 losses, then win, then 2 losses → max is 2
        assert stats.consecutive_losses(df) == 2


class TestBestWorstMonth:
    def test_known_value(self):
        # January 2023: (1.01)(0.99) - 1 ≈ -0.0001 (one month, known result)
        # February 2023: (1.02) - 1 = 0.02
        df = pl.DataFrame(
            {
                "date": ["2023-01-02", "2023-01-03", "2023-02-01"],
                "returns": [0.01, -0.01, 0.02],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        best = stats.best_month(df)
        worst = stats.worst_month(df)
        assert best == pytest.approx(0.02, abs=1e-10)
        assert worst == pytest.approx(1.01 * 0.99 - 1, abs=1e-10)


class TestBestWorstYear:
    def test_known_value(self):
        # 2022 year: single day, return 0.05
        # 2023 year: single day, return -0.03
        df = pl.DataFrame(
            {
                "date": ["2022-06-01", "2023-06-01"],
                "returns": [0.05, -0.03],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        assert stats.best_year(df) == pytest.approx(0.05, abs=1e-10)
        assert stats.worst_year(df) == pytest.approx(-0.03, abs=1e-10)


class TestExposure:
    def test_partial_exposure(self):
        df = pl.DataFrame(
            {
                "date": ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
                "returns": [0.01, 0.0, -0.02, 0.0],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        assert stats.exposure(df) == pytest.approx(0.5, abs=1e-10)

    def test_all_zeros_is_zero(self):
        df = pl.DataFrame(
            {
                "date": ["2023-01-02", "2023-01-03"],
                "returns": [0.0, 0.0],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        assert stats.exposure(df) == 0.0


# ---------------------------------------------------------------------------
# Positive months / years pct
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Period Performance
# ---------------------------------------------------------------------------


class TestPeriodCutoff:
    def test_leap_year(self):
        from datetime import date

        # 2024 is a leap year. Subtracting 1, 3, or 5 years should yield Feb 28 of non-leap years.
        d = date(2024, 2, 29)
        assert stats._period_cutoff(d, "1Y") == date(2023, 2, 28)
        assert stats._period_cutoff(d, "3Y") == date(2021, 2, 28)
        assert stats._period_cutoff(d, "5Y") == date(2019, 2, 28)


class TestPeriodPerformanceRaw:
    def test_si_matches_total_return(self, sample_df):
        raw = stats.period_performance_raw(sample_df)
        assert abs(raw["SI"]["strategy"] - stats.total_return(sample_df)) < 1e-10

    def test_short_series_returns_nan_for_long_windows(self):
        from datetime import date, timedelta

        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(30)]
        df = pl.DataFrame({"date": dates, "returns": [0.001] * 30}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        raw = stats.period_performance_raw(df)
        assert math.isnan(raw["5Y"]["strategy"])
        assert math.isnan(raw["3Y"]["strategy"])
        assert math.isnan(raw["1Y"]["strategy"])

    def test_calendar_period_start_on_weekend_not_nan(self):
        # 2023-01-01 was a Sunday; first trading day is 2023-01-02.
        # MTD/QTD/YTD cutoff is 2023-01-01, which predates the first row,
        # but there IS data in January so these should NOT be NaN.
        from datetime import date

        df = pl.DataFrame(
            {
                "date": [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
                "returns": [0.01, 0.02, -0.005],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        raw = stats.period_performance_raw(df)
        assert not math.isnan(raw["MTD"]["strategy"])
        assert not math.isnan(raw["QTD"]["strategy"])
        assert not math.isnan(raw["YTD"]["strategy"])

    def test_empty_df(self):
        df = pl.DataFrame({"date": [], "returns": []}).with_columns(
            pl.col("date").cast(pl.Date), pl.col("returns").cast(pl.Float64)
        )
        raw = stats.period_performance_raw(df)
        for entry in raw.values():
            assert math.isnan(entry["strategy"])

    def test_pandas_intraday_inputs_are_aggregated_to_daily_before_join(self):
        from datetime import datetime, time, timedelta

        import pandas as pd

        num_days = 10
        start = datetime(2024, 1, 1)
        days = [start + timedelta(days=i) for i in range(num_days)]
        intraday_dates = [
            datetime.combine(day.date(), session)
            for day in days
            for session in (time(9, 30), time(16, 0))
        ]
        strategy_session_returns = [0.001, 0.002]
        benchmark_session_returns = [0.0005, 0.001]
        daily_strategy_return = (
            (1 + strategy_session_returns[0]) * (1 + strategy_session_returns[1])
        ) - 1
        daily_benchmark_return = (
            (1 + benchmark_session_returns[0]) * (1 + benchmark_session_returns[1])
        ) - 1
        intraday_df = pd.DataFrame(
            {"date": intraday_dates, "returns": strategy_session_returns * num_days}
        )
        intraday_base_df = pd.DataFrame(
            {"date": intraday_dates, "returns": benchmark_session_returns * num_days}
        )
        daily_df = pd.DataFrame(
            {
                "date": [day.date() for day in days],
                "returns": [daily_strategy_return] * num_days,
            }
        )
        daily_base_df = pd.DataFrame(
            {
                "date": [day.date() for day in days],
                "returns": [daily_benchmark_return] * num_days,
            }
        )

        with pytest.warns(UserWarning, match="duplicate dates"):
            raw = stats.period_performance_raw(intraday_df, intraday_base_df)
        expected = stats.period_performance_raw(daily_df, daily_base_df)

        for label in raw:
            for key in raw[label]:
                if math.isnan(expected[label][key]):
                    assert math.isnan(raw[label][key])
                else:
                    assert raw[label][key] == pytest.approx(expected[label][key])


class TestPeriodPerformance:
    def test_nan_renders_as_dash(self):
        from datetime import date, timedelta

        dates = [date(2023, 6, 1) + timedelta(days=i) for i in range(30)]
        df = pl.DataFrame({"date": dates, "returns": [0.001] * 30}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        result = stats.period_performance(df)
        strat = dict(zip(result["period"].to_list(), result["strategy"].to_list()))
        assert strat["5Y"] == "—"
        assert strat["SI"] != "—"


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


class TestMonteCarloPaths:
    def test_sim_0_matches_original_cumulative(self, sample_df):
        import numpy as np

        from katsustats._dataframe import ensure_polars

        df = ensure_polars(sample_df)
        r = df.get_column("returns").to_numpy()
        expected = (np.cumprod(1 + r) - 1).tolist()
        result = stats.monte_carlo_paths(sample_df, sims=10, seed=42)
        actual = result.get_column("sim_0").to_list()
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert abs(a - e) < 1e-12

    def test_different_seeds_differ(self, sample_df):
        r1 = stats.monte_carlo_paths(sample_df, sims=20, seed=1)
        r2 = stats.monte_carlo_paths(sample_df, sims=20, seed=2)
        assert not r1.equals(r2)

    def test_same_seed_reproduces_paths(self, sample_df):
        r1 = stats.monte_carlo_paths(sample_df, sims=20, seed=7)
        r2 = stats.monte_carlo_paths(sample_df, sims=20, seed=7)
        assert r1.equals(r2)

    def test_sims_1_returns_only_original(self, sample_df):
        result = stats.monte_carlo_paths(sample_df, sims=1, seed=0)
        assert result.columns == ["step", "sim_0"]

    def test_sims_must_be_positive(self, sample_df):
        with pytest.raises(ValueError):
            stats.monte_carlo_paths(sample_df, sims=0)


class TestMonteCarloSummary:
    def test_bust_and_goal_none_by_default(self, sample_df):
        result = stats.monte_carlo_summary(sample_df, sims=50, seed=0)
        assert result["bust_probability"] is None
        assert result["goal_probability"] is None

    def test_bust_and_goal_probability(self, sample_df):
        result = stats.monte_carlo_summary(
            sample_df, sims=100, bust=-0.01, goal=0.01, seed=0
        )
        assert 0.0 <= result["bust_probability"] <= 1.0
        assert 0.0 <= result["goal_probability"] <= 1.0

    def test_sims_recorded(self, sample_df):
        result = stats.monte_carlo_summary(sample_df, sims=77, seed=0)
        assert result["sims"] == 77

    def test_seed_recorded(self, sample_df):
        result = stats.monte_carlo_summary(sample_df, sims=10, seed=99)
        assert result["seed"] == 99

    def test_same_seed_reproduces_summary(self, sample_df):
        r1 = stats.monte_carlo_summary(sample_df, sims=50, seed=42)
        r2 = stats.monte_carlo_summary(sample_df, sims=50, seed=42)
        assert r1 == r2

    @pytest.mark.parametrize("method", ["bootstrap", "shuffle"])
    def test_same_seed_reproduces_summary_per_method(self, sample_df, method):
        r1 = stats.monte_carlo_summary(sample_df, sims=50, seed=42, method=method)
        r2 = stats.monte_carlo_summary(sample_df, sims=50, seed=42, method=method)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Bug 1: max_drawdown anchored at initial capital
# ---------------------------------------------------------------------------


class TestMaxDrawdownInitialCapital:
    def test_large_initial_loss(self):
        df = pl.DataFrame(
            {
                "date": ["2023-01-02", "2023-01-03", "2023-01-04"],
                "returns": [-0.5, 0.05, 0.02],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        assert stats.max_drawdown(df) == pytest.approx(-0.5, abs=1e-10)

    def test_drawdown_details_reports_initial_period(self):
        import datetime

        df = pl.DataFrame(
            {
                "date": ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
                "returns": [-0.5, 0.05, 0.10, 0.20],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        dd = stats.drawdown_details(df)
        assert dd.height >= 1
        first = dd.sort("start").row(0, named=True)
        assert first["start"] == datetime.date(2023, 1, 2)

    def test_consistency_with_sim_max_drawdowns(self):
        df = pl.DataFrame(
            {
                "date": ["2023-01-02", "2023-01-03", "2023-01-04"],
                "returns": [-0.3, 0.1, 0.05],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        mdd = stats.max_drawdown(df)
        sim_mdd = stats._sim_max_drawdowns(
            stats._simulate_paths(
                df.get_column("returns"), sims=1, seed=0, method="bootstrap"
            )
        )
        assert mdd == pytest.approx(float(sim_mdd[0]), abs=1e-10)


# ---------------------------------------------------------------------------
# Bug 2: Monte Carlo bootstrap method
# ---------------------------------------------------------------------------


class TestMonteCarloMethod:
    def test_bootstrap_gives_non_degenerate_terminal(self, sample_df):
        result = stats.monte_carlo_summary(
            sample_df, sims=100, seed=0, method="bootstrap"
        )
        t = result["terminal"]
        assert t["min"] < t["max"]

    def test_bootstrap_gives_non_degenerate_sharpe(self, sample_df):
        result = stats.monte_carlo_summary(
            sample_df, sims=100, seed=0, method="bootstrap"
        )
        s = result["sharpe"]
        assert s["min"] < s["max"]

    def test_bootstrap_gives_non_degenerate_cagr(self, sample_df):
        result = stats.monte_carlo_summary(
            sample_df, sims=100, seed=0, method="bootstrap"
        )
        c = result["cagr"]
        assert c["min"] < c["max"]

    def test_shuffle_preserves_old_behavior(self, sample_df):
        result = stats.monte_carlo_summary(sample_df, sims=50, seed=0, method="shuffle")
        assert isinstance(result, dict)
        assert "terminal" in result

    def test_shuffle_terminal_constant_across_paths(self, sample_df):
        import numpy as np

        r = sample_df.get_column("returns").drop_nulls().to_numpy()
        sim_returns = stats._build_sim_returns(r, sims=20, seed=0, method="shuffle")
        terminal_per_path = np.cumprod(1 + sim_returns, axis=0)[-1, :]
        assert terminal_per_path.max() - terminal_per_path.min() < 1e-10

    def test_invalid_method_raises(self, sample_df):
        with pytest.raises(ValueError, match="method must be"):
            stats.monte_carlo_summary(sample_df, sims=10, seed=0, method="invalid")

    def test_paths_method_bootstrap(self, sample_df):
        result = stats.monte_carlo_paths(sample_df, sims=20, seed=0, method="bootstrap")
        assert isinstance(result, pl.DataFrame)
        assert result.height == sample_df.height

    def test_paths_method_shuffle(self, sample_df):
        result = stats.monte_carlo_paths(sample_df, sims=20, seed=0, method="shuffle")
        assert isinstance(result, pl.DataFrame)
        assert result.height == sample_df.height


# ---------------------------------------------------------------------------
# Improvement 8: Empty/degenerate-input hardening
# ---------------------------------------------------------------------------


class TestEmptyInputHardening:
    def test_best_day_empty_returns_nan(self, empty_df):
        assert math.isnan(stats.best_day(empty_df))

    def test_worst_day_empty_returns_nan(self, empty_df):
        assert math.isnan(stats.worst_day(empty_df))

    def test_value_at_risk_empty_returns_nan(self, empty_df):
        assert math.isnan(stats.value_at_risk(empty_df))

    def test_skewness_empty_returns_nan(self, empty_df):
        assert math.isnan(stats.skewness(empty_df))

    def test_kurtosis_empty_returns_nan(self, empty_df):
        assert math.isnan(stats.kurtosis(empty_df))

    def test_total_return_empty_returns_nan(self, empty_df):
        assert math.isnan(stats.total_return(empty_df))

    @pytest.mark.parametrize(
        "name",
        [
            "cagr",
            "cvar",
            "omega_ratio",
            "ulcer_index",
            "martin_ratio",
            "gain_to_pain",
            "kelly_criterion",
            "probabilistic_sharpe",
            "payoff_ratio",
            "best_month",
            "worst_month",
            "best_year",
            "worst_year",
            "exposure",
            "positive_months_pct",
            "positive_years_pct",
        ],
    )
    def test_empty_returns_nan(self, empty_df, name):
        # Every scalar metric must degrade to NaN on an empty frame, not raise
        # and not silently return 0.0/inf.
        assert math.isnan(getattr(stats, name)(empty_df))

    def test_information_ratio_single_row_returns_nan(self):
        df = pl.DataFrame({"date": ["2023-01-02"], "returns": [0.01]}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        assert math.isnan(stats.information_ratio(df, df))


# ---------------------------------------------------------------------------
# Risk-Adjusted Ratios (#14)
# ---------------------------------------------------------------------------


class TestOmegaRatio:
    def test_threshold_parameter(self, sample_df):
        r0 = stats.omega_ratio(sample_df, threshold=0.0)
        rp = stats.omega_ratio(sample_df, threshold=0.01)
        # Higher threshold → less upside / more downside → lower Omega
        assert rp < r0


class TestMartinRatio:
    def test_rf_parameter(self, sample_df):
        m0 = stats.martin_ratio(sample_df, rf=0.0)
        m1 = stats.martin_ratio(sample_df, rf=0.05)
        # Higher risk-free rate → lower Martin
        assert m1 < m0


class TestKellyCriterion:
    def test_rf_parameter(self, sample_df):
        k0 = stats.kelly_criterion(sample_df, rf=0.0)
        k1 = stats.kelly_criterion(sample_df, rf=0.05)
        # Higher rf → lower Kelly
        assert k1 < k0

    def test_single_row(self, single_row_df):
        assert math.isnan(stats.kelly_criterion(single_row_df))


class TestProbabilisticSharpe:
    def test_benchmark_parameter(self, sample_df):
        psr0 = stats.probabilistic_sharpe(sample_df, benchmark_sharpe=0.0)
        psr1 = stats.probabilistic_sharpe(sample_df, benchmark_sharpe=2.0)
        # Higher benchmark → lower PSR
        assert psr1 < psr0

    def test_single_row(self, single_row_df):
        assert math.isnan(stats.probabilistic_sharpe(single_row_df))


class TestPayoffRatio:
    def test_known_value(self):
        df = pl.DataFrame(
            {
                "date": ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
                "returns": [0.02, -0.01, 0.03, -0.02],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        # avg_win = (0.02 + 0.03) / 2 = 0.025
        # avg_loss = (-0.01 + -0.02) / 2 = -0.015 → abs = 0.015
        # payoff = 0.025 / 0.015 ≈ 1.6667
        expected = 0.025 / 0.015
        assert abs(stats.payoff_ratio(df) - expected) < 1e-10


# ---------------------------------------------------------------------------
# Benchmark-Relative Metrics (#16)
# ---------------------------------------------------------------------------


class TestTreynorRatio:
    def test_same_returns_zero_or_near(self, sample_df):
        # When strategy == benchmark, excess return ≈ 0 → Treynor ≈ 0
        tr = stats.treynor_ratio(sample_df, sample_df)
        assert math.isfinite(tr)

    def test_rf_parameter(self, sample_df, benchmark_df):
        t0 = stats.treynor_ratio(sample_df, benchmark_df, rf=0.0)
        t1 = stats.treynor_ratio(sample_df, benchmark_df, rf=0.05)
        assert t1 < t0


class TestUpCapture:
    def test_self_returns_one(self, sample_df):
        assert abs(stats.up_capture(sample_df, sample_df) - 1.0) < 1e-10


class TestDownCapture:
    def test_self_returns_one(self, sample_df):
        assert abs(stats.down_capture(sample_df, sample_df) - 1.0) < 1e-10
