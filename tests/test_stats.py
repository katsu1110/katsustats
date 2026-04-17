"""Unit tests for katsustats.stats module."""

from __future__ import annotations

import math

import polars as pl

from katsustats import stats

# ---------------------------------------------------------------------------
# Scalar metrics — return type and basic sanity
# ---------------------------------------------------------------------------


class TestTotalReturn:
    def test_returns_float(self, sample_df):
        assert isinstance(stats.total_return(sample_df), float)

    def test_positive_when_mostly_gains(self, all_positive_df):
        assert stats.total_return(all_positive_df) > 0

    def test_negative_when_all_losses(self, all_negative_df):
        assert stats.total_return(all_negative_df) < 0

    def test_empty_df(self, empty_df):
        # empty product → 1.0 - 1 = 0.0
        assert stats.total_return(empty_df) == 0.0

    def test_known_value(self):
        # (1.1) * (0.9) - 1 = -0.01
        df = pl.DataFrame(
            {"date": ["2023-01-02", "2023-01-03"], "pnl": [0.1, -0.1]}
        ).with_columns(pl.col("date").cast(pl.Date))
        result = stats.total_return(df)
        assert abs(result - (-0.01)) < 1e-10


class TestCagr:
    def test_returns_float(self, sample_df):
        assert isinstance(stats.cagr(sample_df), float)

    def test_empty_df_returns_zero(self, empty_df):
        assert stats.cagr(empty_df) == 0.0

    def test_all_positive(self, all_positive_df):
        assert stats.cagr(all_positive_df) > 0


class TestVolatility:
    def test_returns_float(self, sample_df):
        assert isinstance(stats.volatility(sample_df), float)

    def test_all_positive_nonzero(self, all_positive_df):
        # Even all-positive returns have non-zero variance
        assert stats.volatility(all_positive_df) > 0

    def test_single_row_returns_nan(self, single_row_df):
        # std of 1 element (ddof=1) is undefined; should return nan not raise
        assert math.isnan(stats.volatility(single_row_df))


class TestSharpe:
    def test_returns_float(self, sample_df):
        assert isinstance(stats.sharpe(sample_df), float)

    def test_zero_std_returns_zero(self):
        # All identical returns → std=0 → sharpe=0
        df = pl.DataFrame(
            {"date": ["2023-01-02", "2023-01-03"], "pnl": [0.01, 0.01]}
        ).with_columns(pl.col("date").cast(pl.Date))
        assert stats.sharpe(df) == 0.0

    def test_rf_parameter(self, sample_df):
        s0 = stats.sharpe(sample_df, rf=0.0)
        s1 = stats.sharpe(sample_df, rf=0.05)
        # Higher risk-free rate → lower Sharpe (for positive excess returns)
        assert s0 != s1


class TestSortino:
    def test_returns_float(self, sample_df):
        result = stats.sortino(sample_df)
        assert isinstance(result, float)

    def test_all_positive_returns_inf(self, all_positive_df):
        assert stats.sortino(all_positive_df) == float("inf")

    def test_all_negative_negative_sortino(self, all_negative_df):
        assert stats.sortino(all_negative_df) < 0


class TestMaxDrawdown:
    def test_returns_float(self, sample_df):
        assert isinstance(stats.max_drawdown(sample_df), float)

    def test_is_negative_or_zero(self, sample_df):
        assert stats.max_drawdown(sample_df) <= 0

    def test_all_positive_returns_zero(self, all_positive_df):
        assert stats.max_drawdown(all_positive_df) == 0.0


class TestCalmar:
    def test_returns_float(self, sample_df):
        assert isinstance(stats.calmar(sample_df), float)

    def test_all_positive_returns_zero(self, all_positive_df):
        # No drawdown → calmar returns 0
        assert stats.calmar(all_positive_df) == 0.0


class TestWinRate:
    def test_returns_float(self, sample_df):
        assert isinstance(stats.win_rate(sample_df), float)

    def test_in_range(self, sample_df):
        wr = stats.win_rate(sample_df)
        assert 0.0 <= wr <= 1.0

    def test_all_positive(self, all_positive_df):
        assert stats.win_rate(all_positive_df) == 1.0

    def test_all_negative(self, all_negative_df):
        assert stats.win_rate(all_negative_df) == 0.0

    def test_empty_df(self, empty_df):
        assert stats.win_rate(empty_df) == 0.0


class TestProfitFactor:
    def test_returns_float(self, sample_df):
        result = stats.profit_factor(sample_df)
        assert isinstance(result, float)

    def test_all_positive_returns_inf(self, all_positive_df):
        assert stats.profit_factor(all_positive_df) == float("inf")

    def test_all_negative_returns_zero(self, all_negative_df):
        assert stats.profit_factor(all_negative_df) == 0.0

    def test_positive_with_mixed(self, sample_df):
        assert stats.profit_factor(sample_df) > 0


class TestBestWorstDay:
    def test_best_day_returns_float(self, sample_df):
        assert isinstance(stats.best_day(sample_df), float)

    def test_worst_day_returns_float(self, sample_df):
        assert isinstance(stats.worst_day(sample_df), float)

    def test_best_greater_than_worst(self, sample_df):
        assert stats.best_day(sample_df) > stats.worst_day(sample_df)

    def test_best_positive_in_all_positive(self, all_positive_df):
        assert stats.best_day(all_positive_df) > 0

    def test_worst_negative_in_all_negative(self, all_negative_df):
        assert stats.worst_day(all_negative_df) < 0


class TestAvgWinLoss:
    def test_avg_win_positive(self, sample_df):
        assert stats.avg_win(sample_df) > 0

    def test_avg_loss_negative(self, sample_df):
        assert stats.avg_loss(sample_df) < 0

    def test_avg_win_zero_when_all_negative(self, all_negative_df):
        assert stats.avg_win(all_negative_df) == 0.0

    def test_avg_loss_zero_when_all_positive(self, all_positive_df):
        assert stats.avg_loss(all_positive_df) == 0.0


class TestVaR:
    def test_returns_float(self, sample_df):
        assert isinstance(stats.value_at_risk(sample_df), float)

    def test_is_negative(self, sample_df):
        # VaR at 5% should be a loss
        assert stats.value_at_risk(sample_df) < 0

    def test_alpha_parameter(self, sample_df):
        var5 = stats.value_at_risk(sample_df, alpha=0.05)
        var10 = stats.value_at_risk(sample_df, alpha=0.10)
        # Higher alpha → less extreme quantile → higher (less negative) VaR
        assert var10 >= var5


class TestRecoveryFactor:
    def test_returns_float(self, sample_df):
        assert isinstance(stats.recovery_factor(sample_df), float)

    def test_no_drawdown_returns_zero(self, all_positive_df):
        assert stats.recovery_factor(all_positive_df) == 0.0


class TestSkewnessKurtosis:
    def test_skewness_returns_float(self, sample_df):
        assert isinstance(stats.skewness(sample_df), float)

    def test_kurtosis_returns_float(self, sample_df):
        assert isinstance(stats.kurtosis(sample_df), float)


# ---------------------------------------------------------------------------
# DataFrame-returning functions
# ---------------------------------------------------------------------------


class TestDrawdownDetails:
    def test_returns_dataframe(self, sample_df):
        assert isinstance(stats.drawdown_details(sample_df), pl.DataFrame)

    def test_schema(self, sample_df):
        dd = stats.drawdown_details(sample_df)
        assert set(dd.columns) == {"start", "trough", "recovery", "max_dd", "days"}

    def test_max_dd_negative(self, sample_df):
        dd = stats.drawdown_details(sample_df)
        if dd.height > 0:
            assert all(v <= 0 for v in dd.get_column("max_dd").to_list())

    def test_top_n_respected(self, sample_df):
        dd = stats.drawdown_details(sample_df, top_n=2)
        assert dd.height <= 2

    def test_no_drawdown_empty(self, all_positive_df):
        dd = stats.drawdown_details(all_positive_df)
        assert dd.height == 0


class TestDayOfWeekStats:
    def test_returns_dataframe(self, sample_df):
        assert isinstance(stats.day_of_week_stats(sample_df), pl.DataFrame)

    def test_schema(self, sample_df):
        dow = stats.day_of_week_stats(sample_df)
        assert "dow" in dow.columns
        assert "dow_name" in dow.columns
        assert "mean_return" in dow.columns
        assert "win_rate" in dow.columns
        assert "total_return" in dow.columns
        assert "count" in dow.columns

    def test_win_rate_in_range(self, sample_df):
        dow = stats.day_of_week_stats(sample_df)
        wr = dow.get_column("win_rate").to_list()
        assert all(0.0 <= v <= 1.0 for v in wr)

    def test_all_weekdays_represented(self, sample_df):
        # sample_df covers all 5 weekdays
        dow = stats.day_of_week_stats(sample_df)
        dow_values = sorted(dow.get_column("dow").to_list())
        assert dow_values == [1, 2, 3, 4, 5]


class TestRollingSharpe:
    def test_returns_dataframe(self, sample_df):
        result = stats.rolling_sharpe(sample_df, window=5)
        assert isinstance(result, pl.DataFrame)

    def test_schema(self, sample_df):
        result = stats.rolling_sharpe(sample_df, window=5)
        assert set(result.columns) == {"date", "rolling_sharpe"}

    def test_length_matches_input(self, sample_df):
        result = stats.rolling_sharpe(sample_df, window=5)
        assert result.height == sample_df.height

    def test_nan_before_window(self, sample_df):
        window = 5
        result = stats.rolling_sharpe(sample_df, window=window)
        vals = result.get_column("rolling_sharpe").to_list()
        # First `window` values should be NaN
        assert all(v is None or math.isnan(v) for v in vals[:window])


class TestRollingVolatility:
    def test_returns_dataframe(self, sample_df):
        result = stats.rolling_volatility(sample_df, window=5)
        assert isinstance(result, pl.DataFrame)

    def test_schema(self, sample_df):
        result = stats.rolling_volatility(sample_df, window=5)
        assert set(result.columns) == {"date", "rolling_vol"}

    def test_length_matches_input(self, sample_df):
        result = stats.rolling_volatility(sample_df, window=5)
        assert result.height == sample_df.height

    def test_values_positive_after_window(self, sample_df):
        window = 5
        result = stats.rolling_volatility(sample_df, window=window)
        vals = result.get_column("rolling_vol").to_list()
        valid = [v for v in vals[window:] if v is not None and not math.isnan(v)]
        assert all(v >= 0 for v in valid)


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------


class TestAlphaBeta:
    def test_returns_tuple(self, sample_df, benchmark_df):
        result = stats.alpha_beta(sample_df, benchmark_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_values_are_floats(self, sample_df, benchmark_df):
        alpha, beta = stats.alpha_beta(sample_df, benchmark_df)
        assert isinstance(alpha, float)
        assert isinstance(beta, float)

    def test_unequal_lengths(self, sample_df, benchmark_df):
        # Should not raise; truncates to min length
        shorter = benchmark_df.head(10)
        alpha, beta = stats.alpha_beta(sample_df, shorter)
        assert isinstance(alpha, float)


class TestCorrelation:
    def test_returns_float(self, sample_df, benchmark_df):
        assert isinstance(stats.correlation(sample_df, benchmark_df), float)

    def test_in_range(self, sample_df, benchmark_df):
        corr = stats.correlation(sample_df, benchmark_df)
        assert -1.0 <= corr <= 1.0

    def test_self_correlation_is_one(self, sample_df):
        assert abs(stats.correlation(sample_df, sample_df) - 1.0) < 1e-10


class TestInformationRatio:
    def test_returns_float(self, sample_df, benchmark_df):
        assert isinstance(stats.information_ratio(sample_df, benchmark_df), float)

    def test_same_returns_zero(self, sample_df):
        # IR is 0 when tracking error is 0
        assert stats.information_ratio(sample_df, sample_df) == 0.0


class TestExcessReturn:
    def test_returns_float(self, sample_df, benchmark_df):
        assert isinstance(stats.excess_return(sample_df, benchmark_df), float)

    def test_same_returns_zero(self, sample_df):
        assert abs(stats.excess_return(sample_df, sample_df)) < 1e-10


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


class TestSummaryMetrics:
    def test_returns_dataframe(self, sample_df):
        assert isinstance(stats.summary_metrics(sample_df), pl.DataFrame)

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
        # With benchmark adds Alpha, Beta, Correlation, IR, Excess Return
        assert with_bench.height == without.height + 5

    def test_all_values_are_strings(self, sample_df):
        result = stats.summary_metrics(sample_df)
        for val in result.get_column("strategy").to_list():
            assert isinstance(val, str)


class TestPandasInputs:
    def test_scalar_metrics_accept_pandas(self, sample_pandas_df):
        assert isinstance(stats.total_return(sample_pandas_df), float)
        assert isinstance(stats.max_drawdown(sample_pandas_df), float)

    def test_dataframe_metrics_accept_pandas(self, sample_pandas_df):
        assert isinstance(stats.drawdown_details(sample_pandas_df), pl.DataFrame)
        assert isinstance(
            stats.rolling_sharpe(sample_pandas_df, window=5), pl.DataFrame
        )

    def test_benchmark_metrics_accept_pandas(
        self, sample_pandas_df, benchmark_pandas_df
    ):
        result = stats.summary_metrics(sample_pandas_df, benchmark_pandas_df)
        assert "benchmark" in result.columns
