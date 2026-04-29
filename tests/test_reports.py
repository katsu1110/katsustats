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
        result = reports.full(sample_df, benchmark=benchmark_df, show=False)
        assert "metrics" in result
        assert "alpha" in result["summary"]
        # Benchmark adds comparison rows
        assert result["metrics"].height > 17

    def test_invalid_input_missing_returns_column(self):
        bad_df = pl.DataFrame({"date": ["2023-01-02"], "ret": [0.01]}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        with pytest.raises(AssertionError):
            reports.full(bad_df, show=False)

    def test_invalid_input_missing_date_column(self):
        bad_df = pl.DataFrame({"day": ["2023-01-02"], "returns": [0.01]})
        with pytest.raises(AssertionError):
            reports.full(bad_df, show=False)

    def test_positional_rf_still_supported(self, sample_df):
        result = reports.full(sample_df, None, 0.04, show=False)
        assert isinstance(result["metrics"], pl.DataFrame)

    def test_duplicate_dates_warns_and_aggregates(self):
        duplicate_dates_df = pl.DataFrame(
            {
                "date": ["2023-01-02", "2023-01-02", "2023-01-03"],
                "returns": [0.10, 0.20, -0.10],
            }
        ).with_columns(pl.col("date").cast(pl.Date))

        # 2023-01-02 => (1.10 * 1.20) - 1 = 0.32
        # across dates  => (1.32 * 0.90) - 1 = 0.188
        expected_total_return = (1.10 * 1.20) * 0.90 - 1.0

        with pytest.warns(UserWarning, match="duplicate dates"):
            result = reports.full(duplicate_dates_df, show=False)

        assert result["summary"]["total_return"] == pytest.approx(expected_total_return)

    def test_accepts_pandas_inputs(self, sample_pandas_df, benchmark_pandas_df):
        result = reports.full(
            sample_pandas_df, benchmark=benchmark_pandas_df, show=False
        )
        assert isinstance(result["metrics"], pl.DataFrame)

    def test_accepts_pandas_indexed_df(
        self, sample_pandas_df_indexed, benchmark_pandas_df_indexed
    ):
        result = reports.full(
            sample_pandas_df_indexed,
            benchmark=benchmark_pandas_df_indexed,
            show=False,
            verbose=False,
        )
        assert isinstance(result["metrics"], pl.DataFrame)

    def test_accepts_pandas_series(self, sample_pandas_series, benchmark_pandas_series):
        result = reports.full(
            sample_pandas_series,
            benchmark=benchmark_pandas_series,
            show=False,
            verbose=False,
        )
        assert isinstance(result["metrics"], pl.DataFrame)

    def test_datetime_date_column_accepted(self, sample_df):
        dt_df = sample_df.with_columns(pl.col("date").cast(pl.Datetime))
        result = reports.full(dt_df, show=False, verbose=False)
        assert isinstance(result["drawdowns"], pl.DataFrame)

    def test_pandas_datetime_date_column_accepted(self):
        from datetime import datetime, timedelta

        import pandas as pd

        start = datetime(2024, 1, 1)
        dates = [start + timedelta(days=i) for i in range(20)]
        returns = [0.01, -0.005, 0.008, -0.012, 0.003] * 4
        df = pd.DataFrame({"date": dates, "returns": returns})

        result = reports.full(df, show=False, verbose=False)

        assert isinstance(result["drawdowns"], pl.DataFrame)

    def test_verbose_false_suppresses_stdout(self, sample_df, capsys):
        reports.full(sample_df, show=False, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_true_prints_output(self, sample_df, capsys):
        reports.full(sample_df, show=False, verbose=True)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_stdout_no_raw_floats_in_drawdown_table(self, sample_df, capsys):
        reports.full(sample_df, show=False, verbose=True)
        out = capsys.readouterr().out
        # raw nanosecond timestamps (1.7e18) or unformatted floats must not appear
        assert "e+" not in out
        assert "e-" not in out

    def test_stdout_pct_columns_formatted_in_dow_table(self, sample_df, capsys):
        reports.full(sample_df, show=False, verbose=True)
        out = capsys.readouterr().out
        # Day-of-Week section must contain formatted percentages
        assert "%" in out


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
        result = reports.html(sample_df, benchmark=benchmark_df)
        assert isinstance(result, str)
        assert "data:image/png;base64," in result

    def test_includes_period_performance_section(self, sample_df):
        result = reports.html(sample_df)
        assert "Period Performance" in result
        assert "MTD" in result
        assert "SI" in result
        assert result.index("Period Performance") < result.index("Top Drawdowns")

    def test_with_benchmark_includes_regime_analysis_section(self):
        from datetime import date, timedelta

        import numpy as np

        # 600 daily rows — long enough for trend_window=200 and
        # vol_window=60 to populate at least one regime.
        rng = np.random.default_rng(0)
        n = 600
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        strat = pl.DataFrame(
            {"date": dates, "returns": rng.normal(0.0008, 0.012, n).tolist()}
        ).with_columns(pl.col("date").cast(pl.Date))
        bench = pl.DataFrame(
            {"date": dates, "returns": rng.normal(0.0004, 0.009, n).tolist()}
        ).with_columns(pl.col("date").cast(pl.Date))

        result = reports.html(strat, benchmark=bench)
        assert "Regime Analysis" in result
        assert any(
            label in result
            for label in (
                "bull_low_vol",
                "bull_high_vol",
                "bear_low_vol",
                "bear_high_vol",
            )
        )
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
        bad_df = pl.DataFrame({"date": ["2023-01-02"], "ret": [0.01]}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        with pytest.raises(AssertionError):
            reports.html(bad_df)

    def test_positional_rf_still_supported(self, sample_df):
        result = reports.html(sample_df, None, 0.04)
        assert isinstance(result, str)

    def test_duplicate_dates_warns_and_aggregates(self):
        duplicate_dates_df = pl.DataFrame(
            {
                "date": ["2023-01-02", "2023-01-02", "2023-01-03"],
                "returns": [0.10, 0.20, -0.10],
            }
        ).with_columns(pl.col("date").cast(pl.Date))

        with pytest.warns(UserWarning, match="duplicate dates"):
            result = reports.html(duplicate_dates_df)

        # HTML should reflect the compounded series (2 rows, not 3)
        assert isinstance(result, str)
        assert result == reports.html(
            pl.DataFrame(
                {"date": ["2023-01-02", "2023-01-03"], "returns": [0.32, -0.10]}
            ).with_columns(pl.col("date").cast(pl.Date))
        )
