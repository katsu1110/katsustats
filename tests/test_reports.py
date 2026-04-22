"""Unit tests for katsustats.reports module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import pytest

from katsustats import reports, stats


@pytest.fixture(autouse=True)
def close_all_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# reports.full
# ---------------------------------------------------------------------------


class TestFull:
    def test_returns_dict_with_expected_keys(self, sample_df):
        result = reports.full(sample_df, show=False)
        assert set(result.keys()) == {
            "summary",
            "metrics",
            "drawdowns",
            "dow_stats",
            "figures",
        }

    def test_summary_is_raw_metrics_dict(self, sample_df):
        result = reports.full(sample_df, show=False)
        assert result["summary"] == stats.summary_metrics_raw(sample_df)

    def test_metrics_is_dataframe(self, sample_df):
        result = reports.full(sample_df, show=False)
        assert isinstance(result["metrics"], pl.DataFrame)

    def test_drawdowns_is_dataframe(self, sample_df):
        result = reports.full(sample_df, show=False)
        assert isinstance(result["drawdowns"], pl.DataFrame)

    def test_dow_stats_is_dataframe(self, sample_df):
        result = reports.full(sample_df, show=False)
        assert isinstance(result["dow_stats"], pl.DataFrame)

    def test_figures_is_dict_of_figures(self, sample_df):
        import matplotlib.figure

        result = reports.full(sample_df, show=False)
        figs = result["figures"]
        assert isinstance(figs, dict)
        assert len(figs) == 8
        for fig in figs.values():
            assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_benchmark(self, sample_df, benchmark_df):
        result = reports.full(sample_df, base_pnl=benchmark_df, show=False)
        assert "metrics" in result
        assert "alpha" in result["summary"]
        # Benchmark adds comparison rows
        assert result["metrics"].height > 17

    def test_invalid_input_missing_pnl_column(self):
        bad_df = pl.DataFrame({"date": ["2023-01-02"], "returns": [0.01]}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        with pytest.raises(AssertionError):
            reports.full(bad_df, show=False)

    def test_invalid_input_missing_date_column(self):
        bad_df = pl.DataFrame({"day": ["2023-01-02"], "pnl": [0.01]})
        with pytest.raises(AssertionError):
            reports.full(bad_df, show=False)

    def test_positional_rf_still_supported(self, sample_df):
        result = reports.full(sample_df, None, 0.04, show=False)
        assert isinstance(result["metrics"], pl.DataFrame)

    def test_duplicate_dates_raises(self, grouped_sample_pandas_df):
        with pytest.raises(AssertionError):
            reports.full(grouped_sample_pandas_df, show=False)

    def test_accepts_pandas_inputs(self, sample_pandas_df, benchmark_pandas_df):
        result = reports.full(
            sample_pandas_df, base_pnl=benchmark_pandas_df, show=False
        )
        assert isinstance(result["metrics"], pl.DataFrame)

    def test_verbose_false_suppresses_stdout(self, sample_df, capsys):
        reports.full(sample_df, show=False, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_true_prints_output(self, sample_df, capsys):
        reports.full(sample_df, show=False, verbose=True)
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# ---------------------------------------------------------------------------
# reports.html
# ---------------------------------------------------------------------------


class TestHtml:
    def test_returns_string(self, sample_df):
        result = reports.html(sample_df)
        assert isinstance(result, str)

    def test_is_valid_html(self, sample_df):
        result = reports.html(sample_df)
        assert result.startswith("<!DOCTYPE html>")
        assert "<html" in result
        assert "</html>" in result

    def test_contains_base64_charts(self, sample_df):
        result = reports.html(sample_df)
        assert "data:image/png;base64," in result

    def test_title_appears_in_output(self, sample_df):
        result = reports.html(sample_df, title="My Strategy")
        assert "My Strategy" in result

    def test_with_benchmark(self, sample_df, benchmark_df):
        result = reports.html(sample_df, base_pnl=benchmark_df)
        assert isinstance(result, str)
        assert "data:image/png;base64," in result

    def test_with_benchmark_includes_regime_analysis_section(
        self, sample_df, benchmark_df
    ):
        result = reports.html(sample_df, base_pnl=benchmark_df)
        assert "Regime Analysis" in result
        assert "bull_low_vol" in result
        assert result.index("Top Drawdowns") < result.index("Regime Analysis")

    def test_output_writes_file(self, sample_df, tmp_path):
        out_file = tmp_path / "report.html"
        result = reports.html(sample_df, output=str(out_file))
        assert out_file.exists()
        assert out_file.stat().st_size > 0
        assert out_file.read_text(encoding="utf-8") == result

    def test_output_none_does_not_create_file(self, sample_df, tmp_path):
        initial_files = set(tmp_path.iterdir())
        reports.html(sample_df, output=None)
        assert set(tmp_path.iterdir()) == initial_files

    def test_invalid_input_raises(self):
        bad_df = pl.DataFrame({"date": ["2023-01-02"], "returns": [0.01]}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        with pytest.raises(AssertionError):
            reports.html(bad_df)

    def test_positional_rf_still_supported(self, sample_df):
        result = reports.html(sample_df, None, 0.04)
        assert isinstance(result, str)

    def test_duplicate_dates_raises(self, grouped_sample_pandas_df):
        with pytest.raises(AssertionError):
            reports.html(grouped_sample_pandas_df)
