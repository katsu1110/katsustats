"""Unit tests for katsustats.plots module."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import polars as pl
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from katsustats import plots
from katsustats.plots import _COLORS, _parse_window


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

    def test_accepts_pandas_indexed_df(
        self, sample_pandas_df_indexed, benchmark_pandas_df_indexed
    ):
        fig = plots.plot_cumulative_returns(
            sample_pandas_df_indexed, base_df=benchmark_pandas_df_indexed
        )
        assert isinstance(fig, Figure)

    def test_accepts_pandas_series(self, sample_pandas_series, benchmark_pandas_series):
        fig = plots.plot_cumulative_returns(
            sample_pandas_series, base_df=benchmark_pandas_series
        )
        assert isinstance(fig, Figure)

    def test_offset_benchmark_dates_do_not_raise(self, sample_df):
        """Regression: misaligned benchmark dates handled via inner join."""
        offset_bench = pl.DataFrame(
            {
                "date": ["2023-01-03", "2023-01-04", "2023-01-05"],
                "returns": [0.01, -0.01, 0.02],
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
        df = pl.DataFrame({"date": dates, "returns": [0.05, -0.03, 0.08]}).with_columns(
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
        df = pl.DataFrame({"date": dates, "returns": [0.1, 0.1]}).with_columns(
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
                "returns": [0.05, 0.02, 0.08],
            }
        ).with_columns(pl.col("date").cast(pl.Date))

        base_df = pl.DataFrame(
            {
                "date": [
                    datetime.date(2021, 6, 1),
                    datetime.date(2022, 6, 1),
                    datetime.date(2023, 6, 1),
                ],
                "returns": [0.01, -0.02, 0.03],
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

    def test_box_colors_reflect_median_sign(self):
        """Boxes with positive median are green; negative median are red."""
        import matplotlib.colors as mcolors

        dates = [f"2024-01-0{i}" for i in range(1, 6)]  # Mon–Fri
        # Make Mon/Tue strongly positive and Wed–Fri strongly negative
        returns = [0.05, 0.04, -0.05, -0.04, -0.03]
        df = pl.DataFrame({"date": dates, "returns": returns}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        fig = plots.plot_dow_returns(df)
        ax = fig.axes[0]
        boxes = [p for p in ax.patches if hasattr(p, "get_facecolor")]
        colors = [mcolors.to_hex(p.get_facecolor()[:3]) for p in boxes]

        pos = mcolors.to_hex(mcolors.to_rgb(_COLORS["positive"]))
        neg = mcolors.to_hex(mcolors.to_rgb(_COLORS["negative"]))
        # Mon and Tue should be positive color; Wed–Fri negative
        assert colors[0] == pos
        assert colors[1] == pos
        assert colors[2] == neg
        assert colors[3] == neg
        assert colors[4] == neg

    def test_twin_axis_present(self, sample_df):
        """Win-rate panel has a secondary y-axis for the total return overlay."""
        fig = plots.plot_dow_returns(sample_df)
        assert len(fig.axes) == 3  # box ax, win-rate ax, twin ax

    def test_legend_contains_both_series(self, sample_df):
        """Legend on the win-rate panel labels both Win Rate and Total Return."""
        fig = plots.plot_dow_returns(sample_df)
        win_rate_ax = fig.axes[1]
        legend = win_rate_ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert "Win Rate" in labels
        assert "Total Return" in labels

    def test_weekends_shown_when_present(self):
        """Boxes for Sat/Sun appear when the data includes weekend dates."""
        dates = [
            "2024-01-01",  # Mon
            "2024-01-02",  # Tue
            "2024-01-06",  # Sat
            "2024-01-07",  # Sun
        ]
        df = pl.DataFrame(
            {"date": dates, "returns": [0.01, -0.01, 0.02, -0.02]}
        ).with_columns(pl.col("date").cast(pl.Date))
        fig = plots.plot_dow_returns(df)
        ax = fig.axes[0]
        bar_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert "Sat" in bar_labels
        assert "Sun" in bar_labels


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


# ---------------------------------------------------------------------------
# plot_monte_carlo
# ---------------------------------------------------------------------------


class TestPlotMonteCarlo:
    def test_returns_figure(self, sample_df):
        fig = plots.plot_monte_carlo(sample_df, sims=50, seed=0)
        assert isinstance(fig, Figure)

    def test_has_one_axes(self, sample_df):
        fig = plots.plot_monte_carlo(sample_df, sims=50, seed=0)
        assert len(fig.axes) == 1

    def test_has_median_and_original_lines(self, sample_df):
        fig = plots.plot_monte_carlo(sample_df, sims=50, seed=0)
        ax = fig.axes[0]
        lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 0]
        assert len(lines) >= 2

    def test_subsamples_paths_above_200(self, sample_df):
        fig = plots.plot_monte_carlo(sample_df, sims=300, seed=0)
        assert isinstance(fig, Figure)

    def test_accepts_pandas_input(self, sample_pandas_df):
        fig = plots.plot_monte_carlo(sample_pandas_df, sims=20, seed=0)
        assert isinstance(fig, Figure)

    def test_custom_confidence_level(self, sample_df):
        fig = plots.plot_monte_carlo(sample_df, sims=50, seed=0, confidence_level=0.80)
        assert isinstance(fig, Figure)

    def test_custom_figsize(self, sample_df):
        fig = plots.plot_monte_carlo(sample_df, sims=20, seed=0, figsize=(8, 3))
        assert fig.get_size_inches()[0] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# plot_monte_carlo_distribution
# ---------------------------------------------------------------------------


class TestPlotMonteCarloDistribution:
    def test_returns_figure(self, sample_df):
        fig = plots.plot_monte_carlo_distribution(sample_df, sims=50, seed=0)
        assert isinstance(fig, Figure)

    def test_has_one_axes(self, sample_df):
        fig = plots.plot_monte_carlo_distribution(sample_df, sims=50, seed=0)
        assert len(fig.axes) == 1

    def test_has_vertical_percentile_lines(self, sample_df):
        fig = plots.plot_monte_carlo_distribution(sample_df, sims=50, seed=0)
        ax = fig.axes[0]
        vlines = [
            ln
            for ln in ax.get_lines()
            if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == ln.get_xdata()[1]
        ]
        assert len(vlines) >= 3

    def test_drawdown_values_are_negative(self, sample_df):
        fig = plots.plot_monte_carlo_distribution(sample_df, sims=50, seed=0)
        ax = fig.axes[0]
        vlines = [
            ln
            for ln in ax.get_lines()
            if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == ln.get_xdata()[1]
        ]
        for ln in vlines:
            assert ln.get_xdata()[0] <= 0

    def test_max_drawdowns_vary_across_sims(self, sample_df):
        import katsustats.stats as stats

        result = stats.monte_carlo_summary(sample_df, sims=100, seed=0)
        assert result["maxdd"]["std"] > 0

    def test_accepts_pandas_input(self, sample_pandas_df):
        fig = plots.plot_monte_carlo_distribution(sample_pandas_df, sims=20, seed=0)
        assert isinstance(fig, Figure)

    def test_custom_bins(self, sample_df):
        fig = plots.plot_monte_carlo_distribution(sample_df, sims=50, seed=0, bins=20)
        assert isinstance(fig, Figure)

    def test_custom_figsize(self, sample_df):
        fig = plots.plot_monte_carlo_distribution(
            sample_df, sims=20, seed=0, figsize=(8, 3)
        )
        assert fig.get_size_inches()[0] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# _parse_window
# ---------------------------------------------------------------------------


class TestParseWindow:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [("1W", 5), ("2W", 10), ("1M", 21), ("3M", 63)],
    )
    def test_string_specs(self, spec, expected):
        assert _parse_window(spec) == expected

    def test_bad_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Unrecognised window"):
            _parse_window("2Y")

    def test_int_passthrough(self):
        assert _parse_window(7) == 7

    def test_integer_string(self):
        assert _parse_window("10") == 10

    @pytest.mark.parametrize("spec", ["1D", "2Y", "6M"])
    def test_unrecognised_strings_raise_value_error(self, spec):
        with pytest.raises(ValueError, match="Unrecognised window"):
            _parse_window(spec)

    def test_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            _parse_window(0)

    def test_zero_string_raises_value_error(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            _parse_window("0")


# ---------------------------------------------------------------------------
# plot_snapshot
# ---------------------------------------------------------------------------


class TestPlotSnapshot:
    def test_returns_figure(self, sample_df):
        fig = plots.plot_snapshot(sample_df)
        assert isinstance(fig, Figure)

    def test_axes_count(self, sample_df):
        fig = plots.plot_snapshot(sample_df)
        assert len(fig.axes) == 6

    def test_custom_figsize(self, sample_df):
        fig = plots.plot_snapshot(sample_df, figsize=(8, 4))
        assert fig.get_size_inches()[0] == pytest.approx(8.0)

    def test_card_axes_no_ticks(self, sample_df):
        fig = plots.plot_snapshot(sample_df)
        for ax in fig.axes[:4]:
            assert len(ax.get_xticks()) == 0
            assert len(ax.get_yticks()) == 0

    def test_equity_curve_has_line(self, sample_df):
        fig = plots.plot_snapshot(sample_df, window=5)
        ax_curve = fig.axes[4]
        lines = [ln for ln in ax_curve.get_lines() if len(ln.get_xdata()) > 0]
        assert len(lines) >= 1

    def test_window_single_int_uses_two_curve_points(self, sample_df):
        # curve has n+1 points: baseline prepended at synthetic prior date
        fig = plots.plot_snapshot(sample_df, window=1)
        ax_curve = fig.axes[4]
        data_lines = [ln for ln in ax_curve.get_lines() if len(ln.get_xdata()) == 2]
        assert len(data_lines) >= 1

    def test_window_int_uses_n_rows(self, sample_df):
        # curve has n+1 points: baseline prepended at synthetic prior date
        fig = plots.plot_snapshot(sample_df, window=5)
        ax_curve = fig.axes[4]
        data_lines = [ln for ln in ax_curve.get_lines() if len(ln.get_xdata()) == 6]
        assert len(data_lines) >= 1

    @pytest.mark.parametrize("spec", ["1W", "2W", "1M", "3M"])
    def test_window_strings_all_return_figure(self, sample_df, spec):
        fig = plots.plot_snapshot(sample_df, window=spec)
        assert isinstance(fig, Figure)

    def test_positive_return_card_is_green(self, all_positive_df):
        fig = plots.plot_snapshot(all_positive_df)
        # Select the topmost patch (the card bg, not any shadow)
        face = mcolors.to_hex(fig.axes[0].patches[-1].get_facecolor()[:3])
        assert face == "#10b981"

    def test_negative_return_card_is_red(self, all_negative_df):
        fig = plots.plot_snapshot(all_negative_df)
        # Select the topmost patch (the card bg, not any shadow)
        face = mcolors.to_hex(fig.axes[0].patches[-1].get_facecolor()[:3])
        assert face == "#ef4444"

    def test_single_row_sharpe_shows_dash(self, single_row_df):
        fig = plots.plot_snapshot(single_row_df, window=1)
        sharpe_ax = fig.axes[1]
        texts = [t.get_text() for t in sharpe_ax.texts]
        assert any("—" in t for t in texts)

    def test_accepts_pandas_input(self, sample_pandas_df):
        fig = plots.plot_snapshot(sample_pandas_df)
        assert isinstance(fig, Figure)

    def test_suptitle_contains_title_and_window(self, sample_df):
        fig = plots.plot_snapshot(sample_df, title="MyStrat", window="1W")
        sup = fig._suptitle.get_text()
        assert "MyStrat" in sup
        assert "1W" in sup

    def test_daily_bars_present_for_short_window(self, sample_df):
        fig = plots.plot_snapshot(sample_df, window="1W")
        ax_curve = fig.axes[4]
        assert len(ax_curve.containers) >= 1

    def test_daily_bars_absent_for_long_window(self):
        import pandas as pd

        dates = pd.date_range("2023-01-01", periods=150)
        df = pd.DataFrame({"date": dates, "returns": [0.01] * 150})
        fig = plots.plot_snapshot(df, window=150)
        ax_curve = fig.axes[4]
        assert len(ax_curve.containers) == 0

    def test_daily_bars_colors(self, all_positive_df, all_negative_df):
        c_pos, c_neg = "#10b981", "#ef4444"
        fig_pos = plots.plot_snapshot(all_positive_df, window="1W")
        pos_colors = [p.get_facecolor() for p in fig_pos.axes[4].patches]
        assert all(mcolors.to_hex(c[:3]) == c_pos for c in pos_colors)

        fig_neg = plots.plot_snapshot(all_negative_df, window="1W")
        neg_colors = [p.get_facecolor() for p in fig_neg.axes[4].patches]
        assert all(mcolors.to_hex(c[:3]) == c_neg for c in neg_colors)

    # --- Dark theme tests ---

    def test_dark_theme_returns_figure(self, sample_df):
        fig = plots.plot_snapshot(sample_df, theme="dark")
        assert isinstance(fig, Figure)

    def test_dark_theme_background_color(self, sample_df):
        fig = plots.plot_snapshot(sample_df, theme="dark")
        assert mcolors.to_hex(fig.get_facecolor()[:3]) == "#0b0f19"

    def test_dark_theme_chart_axes_facecolor_is_dark(self, sample_df):
        # In dark mode both chart axes should NOT have a transparent background.
        # We verify they are visually distinct from the card axes (which are "none"/transparent).
        fig = plots.plot_snapshot(sample_df, theme="dark")
        card_axes_fc = fig.axes[
            0
        ].get_facecolor()  # card axes are set to "none" → (0,0,0,0)
        for ax in (fig.axes[4], fig.axes[5]):
            assert ax.get_facecolor() != card_axes_fc, (
                f"Chart ax facecolor {ax.get_facecolor()!r} should differ from transparent card ax"
            )

    def test_dark_theme_card_has_shadow_patch(self, sample_df):
        fig = plots.plot_snapshot(sample_df, theme="dark")
        # Dark mode adds a shadow patch before the card bg, so each card ax has >= 2 patches
        for ax in fig.axes[:4]:
            assert len(ax.patches) >= 2

    def test_dark_theme_glow_lines_present(self, sample_df):
        fig = plots.plot_snapshot(sample_df, theme="dark")
        ax_curve = fig.axes[4]
        # Dark mode plots 3 extra glow lines + 1 main line + axhline = at least 5
        assert len(ax_curve.get_lines()) >= 5


# Bug 1: plot_drawdown anchored at initial capital
# ---------------------------------------------------------------------------


class TestPlotDrawdownInitialCapital:
    def test_large_initial_loss_shows_drawdown(self):
        import datetime

        dates = [
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 3),
            datetime.date(2023, 1, 4),
            datetime.date(2023, 1, 5),
        ]
        df = pl.DataFrame(
            {"date": dates, "returns": [-0.5, 0.05, 0.10, 0.20]}
        ).with_columns(pl.col("date").cast(pl.Date))
        fig = plots.plot_drawdown(df)
        collections = _drawdown_fill_collections(fig)
        assert len(collections) == 1
        assert bool(collections[0].get_paths())


# ---------------------------------------------------------------------------
# Bug 5: plot_yearly_returns NaN for missing benchmark years
# ---------------------------------------------------------------------------


class TestPlotYearlyReturnsBenchmarkNaN:
    def test_missing_benchmark_year_not_zero(self):
        import datetime

        df = pl.DataFrame(
            {
                "date": [
                    datetime.date(2020, 6, 1),
                    datetime.date(2021, 6, 1),
                    datetime.date(2022, 6, 1),
                    datetime.date(2023, 6, 1),
                ],
                "returns": [0.05, 0.02, 0.03, 0.08],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        base_df = pl.DataFrame(
            {
                "date": [
                    datetime.date(2021, 6, 1),
                    datetime.date(2023, 6, 1),
                ],
                "returns": [0.01, 0.03],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        fig = plots.plot_yearly_returns(df, base_df=base_df)
        ax = fig.axes[0]
        bar_heights = [p.get_height() for p in ax.patches]
        import math

        nan_count = sum(1 for h in bar_heights if math.isnan(h))
        assert nan_count >= 1
