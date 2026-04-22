"""Unit tests for katsustats.plots module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import pytest
from matplotlib.figure import Figure

from katsustats import plots


@pytest.fixture(autouse=True)
def close_all_figures():
    """Close all matplotlib figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_cumulative_returns
# ---------------------------------------------------------------------------


class TestPlotCumulativeReturns:
    def test_returns_figure(self, sample_df):
        fig = plots.plot_cumulative_returns(sample_df)
        assert isinstance(fig, Figure)

    def test_with_benchmark(self, sample_df, benchmark_df):
        fig = plots.plot_cumulative_returns(sample_df, base_df=benchmark_df)
        assert isinstance(fig, Figure)

    def test_custom_figsize(self, sample_df):
        fig = plots.plot_cumulative_returns(sample_df, figsize=(8, 3))
        assert isinstance(fig, Figure)

    def test_accepts_pandas_inputs(self, sample_pandas_df, benchmark_pandas_df):
        fig = plots.plot_cumulative_returns(
            sample_pandas_df, base_df=benchmark_pandas_df
        )
        assert isinstance(fig, Figure)

    def test_offset_benchmark_dates_do_not_raise(self, sample_df):
        """Regression: misaligned benchmark dates handled via inner join."""
        offset_bench = pl.DataFrame(
            {
                "date": ["2023-01-03", "2023-01-04", "2023-01-05"],
                "pnl": [0.01, -0.01, 0.02],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        # Should not raise regardless of date overlap
        fig = plots.plot_cumulative_returns(sample_df, base_df=offset_bench)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_drawdown
# ---------------------------------------------------------------------------


class TestPlotDrawdown:
    def test_returns_figure(self, sample_df):
        fig = plots.plot_drawdown(sample_df)
        assert isinstance(fig, Figure)

    def test_all_positive_no_drawdown(self, all_positive_df):
        # Should not raise even when there's no drawdown
        fig = plots.plot_drawdown(all_positive_df)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_monthly_heatmap
# ---------------------------------------------------------------------------


class TestPlotMonthlyHeatmap:
    def test_returns_figure(self, sample_df):
        fig = plots.plot_monthly_heatmap(sample_df)
        assert isinstance(fig, Figure)

    def test_single_row(self, single_row_df):
        fig = plots.plot_monthly_heatmap(single_row_df)
        assert isinstance(fig, Figure)

    def test_empty_df_returns_figure(self, empty_df):
        # Should return a blank figure rather than raise on zero-size arrays
        fig = plots.plot_monthly_heatmap(empty_df)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_yearly_returns
# ---------------------------------------------------------------------------


class TestPlotYearlyReturns:
    def test_returns_figure(self, sample_df):
        fig = plots.plot_yearly_returns(sample_df)
        assert isinstance(fig, Figure)

    def test_with_benchmark(self, sample_df, benchmark_df):
        fig = plots.plot_yearly_returns(sample_df, base_df=benchmark_df)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_return_distribution
# ---------------------------------------------------------------------------


class TestPlotReturnDistribution:
    def test_returns_figure(self, sample_df):
        fig = plots.plot_return_distribution(sample_df)
        assert isinstance(fig, Figure)

    def test_with_benchmark(self, sample_df, benchmark_df):
        fig = plots.plot_return_distribution(sample_df, base_df=benchmark_df)
        assert isinstance(fig, Figure)

    def test_custom_bins(self, sample_df):
        fig = plots.plot_return_distribution(sample_df, bins=10)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_rolling_sharpe
# ---------------------------------------------------------------------------


class TestPlotRollingSharpe:
    def test_returns_figure(self, sample_df):
        fig = plots.plot_rolling_sharpe(sample_df, window=5)
        assert isinstance(fig, Figure)

    def test_with_benchmark(self, sample_df, benchmark_df):
        fig = plots.plot_rolling_sharpe(sample_df, base_df=benchmark_df, window=5)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_rolling_volatility
# ---------------------------------------------------------------------------


class TestPlotRollingVolatility:
    def test_returns_figure(self, sample_df):
        fig = plots.plot_rolling_volatility(sample_df, window=5)
        assert isinstance(fig, Figure)

    def test_with_benchmark(self, sample_df, benchmark_df):
        fig = plots.plot_rolling_volatility(sample_df, base_df=benchmark_df, window=5)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_drawdown_periods
# ---------------------------------------------------------------------------


class TestPlotDrawdownPeriods:
    def test_returns_figure(self, sample_df):
        fig = plots.plot_drawdown_periods(sample_df)
        assert isinstance(fig, Figure)

    def test_patch_count_matches_drawdowns(self, sample_df):
        from katsustats import stats

        dd = stats.drawdown_details(sample_df, top_n=5)
        fig = plots.plot_drawdown_periods(sample_df, top_n=5)
        ax = fig.axes[0]
        assert len(ax.patches) == dd.height

    def test_top_n_limits_shaded_regions(self, sample_df):
        top_n = 2
        fig = plots.plot_drawdown_periods(sample_df, top_n=top_n)
        ax = fig.axes[0]
        assert len(ax.patches) <= top_n

    def test_no_drawdown_no_patches(self, all_positive_df):
        fig = plots.plot_drawdown_periods(all_positive_df)
        ax = fig.axes[0]
        assert len(ax.patches) == 0

    def test_custom_figsize(self, sample_df):
        fig = plots.plot_drawdown_periods(sample_df, figsize=(8, 3))
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_dow_returns
# ---------------------------------------------------------------------------


class TestPlotDowReturns:
    def test_returns_figure(self, sample_df):
        fig = plots.plot_dow_returns(sample_df)
        assert isinstance(fig, Figure)

    def test_custom_figsize(self, sample_df):
        fig = plots.plot_dow_returns(sample_df, figsize=(8, 4))
        assert isinstance(fig, Figure)
