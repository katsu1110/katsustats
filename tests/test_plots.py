"""Unit tests for katsustats.plots module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from katsustats import plots


@pytest.fixture(autouse=True)
def close_all_figures():
    """Close all matplotlib figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


def _drawdown_fill_collections(fig: Figure) -> list[PolyCollection]:
    """Return PolyCollection artists added to the drawdown plot axes."""
    ax = fig.axes[0]
    return [
        collection
        for collection in ax.collections
        if isinstance(collection, PolyCollection)
    ]


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

    @pytest.mark.parametrize(
        ("fixture_name", "has_paths"),
        [("sample_df", True), ("all_positive_df", False)],
    )
    def test_fill_artist_is_added(self, fixture_name, has_paths, request):
        df = request.getfixturevalue(fixture_name)
        fig = plots.plot_drawdown(df)
        collections = _drawdown_fill_collections(fig)

        assert len(collections) == 1
        assert bool(collections[0].get_paths()) is has_paths


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
# plot_eoy_returns
# ---------------------------------------------------------------------------


class TestPlotEoyReturns:
    def test_returns_figure(self, sample_df):
        fig = plots.plot_eoy_returns(sample_df)
        assert isinstance(fig, Figure)

    def test_with_benchmark(self, sample_df, benchmark_df):
        fig = plots.plot_eoy_returns(sample_df, base_df=benchmark_df)
        assert isinstance(fig, Figure)

    def test_bar_count_equals_years(self):
        """Number of bars equals number of years in the data."""
        import datetime

        dates = [
            datetime.date(2021, 6, 1),
            datetime.date(2022, 6, 1),
            datetime.date(2023, 6, 1),
        ]
        df = pl.DataFrame({"date": dates, "pnl": [0.05, -0.03, 0.08]}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        fig = plots.plot_eoy_returns(df)
        ax = fig.axes[0]
        assert len(ax.patches) == 3  # 3 years → 3 bars

    def test_compounded_returns(self):
        """Bar values equal compounded annual returns, not arithmetic sum."""
        import datetime

        # Two rows in the same year: compounded = 1.1 * 1.1 - 1 = 0.21, not 0.20
        dates = [datetime.date(2023, 1, 2), datetime.date(2023, 6, 1)]
        df = pl.DataFrame({"date": dates, "pnl": [0.1, 0.1]}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        fig = plots.plot_eoy_returns(df)
        ax = fig.axes[0]
        bar_height = ax.patches[0].get_height()
        expected = 1.1 * 1.1 - 1  # 0.21
        assert abs(bar_height - expected) < 1e-9

    def test_with_benchmark_drops_non_overlapping_years(self):
        """Only intersecting years should be plotted when a benchmark is provided."""
        import datetime

        df = pl.DataFrame(
            {
                "date": [
                    datetime.date(2020, 6, 1),
                    datetime.date(2021, 6, 1),
                    datetime.date(2023, 6, 1),
                ],
                "pnl": [0.05, 0.02, 0.08],
            }
        ).with_columns(pl.col("date").cast(pl.Date))

        base_df = pl.DataFrame(
            {
                "date": [
                    datetime.date(2021, 6, 1),
                    datetime.date(2022, 6, 1),
                    datetime.date(2023, 6, 1),
                ],
                "pnl": [0.01, -0.02, 0.03],
            }
        ).with_columns(pl.col("date").cast(pl.Date))

        fig = plots.plot_eoy_returns(df, base_df=base_df)
        ax = fig.axes[0]

        tick_labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        assert tick_labels == ["2021", "2023"]
        assert "2020" not in tick_labels
        assert "2022" not in tick_labels
        assert len(ax.patches) == 4  # 2 intersecting years × 2 bar series


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


# ---------------------------------------------------------------------------
# plot_returns_vs_benchmark
# ---------------------------------------------------------------------------


class TestPlotReturnsVsBenchmark:
    def test_returns_figure(self, sample_df, benchmark_df):
        fig = plots.plot_returns_vs_benchmark(sample_df, benchmark_df)
        assert isinstance(fig, Figure)

    def test_custom_figsize(self, sample_df, benchmark_df):
        fig = plots.plot_returns_vs_benchmark(sample_df, benchmark_df, figsize=(6, 6))
        assert isinstance(fig, Figure)

    def test_contains_scatter_and_line(self, sample_df, benchmark_df):
        """Axes must contain a PathCollection (scatter) and at least one Line2D (regression)."""
        import matplotlib.collections as mcollections

        fig = plots.plot_returns_vs_benchmark(sample_df, benchmark_df)
        ax = fig.axes[0]
        collections = [
            c for c in ax.get_children() if isinstance(c, mcollections.PathCollection)
        ]
        lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 0]
        assert len(collections) >= 1, "Expected at least one scatter PathCollection"
        assert len(lines) >= 1, "Expected at least one regression Line2D"

    def test_accepts_pandas_inputs(self, sample_pandas_df, benchmark_pandas_df):
        fig = plots.plot_returns_vs_benchmark(sample_pandas_df, benchmark_pandas_df)
        assert isinstance(fig, Figure)
