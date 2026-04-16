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
# plot_dow_returns
# ---------------------------------------------------------------------------


class TestPlotDowReturns:
    def test_returns_figure(self, sample_df):
        fig = plots.plot_dow_returns(sample_df)
        assert isinstance(fig, Figure)

    def test_custom_figsize(self, sample_df):
        fig = plots.plot_dow_returns(sample_df, figsize=(8, 4))
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_group_pnl
# ---------------------------------------------------------------------------


class TestPlotGroupPnl:
    def test_returns_figure(self, grouped_sample_df):
        fig = plots.plot_group_pnl(grouped_sample_df)
        assert isinstance(fig, Figure)

    def test_custom_group_column(self, grouped_sample_df):
        sector_df = grouped_sample_df.rename({"group": "sector"})
        fig = plots.plot_group_pnl(sector_df, group_col="sector")
        assert isinstance(fig, Figure)

    def test_non_string_group_values(self, grouped_sample_df):
        numeric_df = grouped_sample_df.with_columns(
            pl.when(pl.col("group") == "Tech")
            .then(1)
            .when(pl.col("group") == "Energy")
            .then(2)
            .otherwise(3)
            .alias("group_id")
        ).drop("group")
        fig = plots.plot_group_pnl(numeric_df, group_col="group_id")
        assert isinstance(fig, Figure)
