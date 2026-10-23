"""Unit tests for katsustats.reports module."""

from __future__ import annotations

import json

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
    def test_positional_rf_still_supported(self, sample_df):
        # Guards the positional order (returns, benchmark, rf, ...): a reorder
        # would silently apply 0.04 as something else instead of raising.
        positional = reports.full(sample_df, None, 0.04, show=False)
        keyword = reports.full(sample_df, benchmark=None, rf=0.04, show=False)
        assert positional["summary"] == keyword["summary"]

    def test_with_benchmark(self, sample_df, benchmark_df):
        result = reports.full(sample_df, benchmark=benchmark_df, show=False)
        assert "metrics" in result
        assert "alpha" in result["summary"]
        # Benchmark adds comparison rows
        assert result["metrics"].height > 17

    def test_verbose_false_suppresses_stdout(self, sample_df, capsys):
        reports.full(sample_df, show=False, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_datetime_date_column_accepted(self, sample_df):
        dt_df = sample_df.with_columns(pl.col("date").cast(pl.Datetime))
        result = reports.full(dt_df, show=False, verbose=False)
        assert isinstance(result["drawdowns"], pl.DataFrame)

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

    def test_positional_rf_still_supported(self, sample_df):
        # Guards the positional order (returns, benchmark, rf, ...).
        positional = reports.html(sample_df, None, 0.04)
        keyword = reports.html(sample_df, benchmark=None, rf=0.04)
        assert positional == keyword

    def test_with_benchmark(self, sample_df, benchmark_df):
        result = reports.html(sample_df, benchmark=benchmark_df)
        assert isinstance(result, str)
        assert "data:image/png;base64," in result

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


# ---------------------------------------------------------------------------
# reports.json
# ---------------------------------------------------------------------------


class TestJson:
    def test_returns_json_string(self, sample_df):
        result = reports.json(sample_df)
        payload = json.loads(result)
        assert isinstance(payload, dict)

    def test_with_benchmark_includes_benchmark_and_comparison(
        self, sample_df, benchmark_df
    ):
        payload = json.loads(reports.json(sample_df, benchmark=benchmark_df))
        assert payload["benchmark"] is not None
        assert payload["comparison"] is not None
        assert "alpha" in payload["comparison"]

    def test_with_benchmark_gaps_aligns_period_performance(
        self, sample_df, benchmark_df
    ):
        gap_benchmark = benchmark_df.slice(5, benchmark_df.height - 5)

        payload = json.loads(reports.json(sample_df, benchmark=gap_benchmark))
        aligned = stats.period_performance_raw(sample_df, gap_benchmark)

        assert payload["strategy"]["period_performance"]["SI"][
            "strategy"
        ] == pytest.approx(aligned["SI"]["strategy"])
        assert payload["benchmark"]["period_performance"]["SI"][
            "benchmark"
        ] == pytest.approx(aligned["SI"]["benchmark"])

    def test_without_benchmark_uses_null_sections(self, sample_df):
        payload = json.loads(reports.json(sample_df))
        assert payload["benchmark"] is None
        assert payload["comparison"] is None
        assert payload["regime_analysis"] == []

    def test_output_writes_json_file(self, sample_df, tmp_path):
        out_file = tmp_path / "report.json"
        result = reports.json(sample_df, output=str(out_file))
        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8") == result
        assert json.loads(result)["metadata"]["title"] == "Strategy"

    def test_dates_are_iso_strings(self, sample_df):
        payload = json.loads(reports.json(sample_df))
        assert payload["metadata"]["start_date"] == "2023-01-02"
        if payload["drawdowns"]:
            assert isinstance(payload["drawdowns"][0]["start"], str)

    def test_mc_seed_reproduces_identical_json(self, sample_df):
        j1 = reports.json(sample_df, monte_carlo=True, mc_sims=50, mc_seed=42)
        j2 = reports.json(sample_df, monte_carlo=True, mc_sims=50, mc_seed=42)
        assert j1 == j2


# ---------------------------------------------------------------------------
# reports.markdown
# ---------------------------------------------------------------------------


class TestMarkdown:
    def test_with_regime_data_includes_regime_section(self):
        from datetime import date, timedelta

        import numpy as np

        rng = np.random.default_rng(0)
        n = 600
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        strat = pl.DataFrame(
            {"date": dates, "returns": rng.normal(0.0008, 0.012, n).tolist()}
        ).with_columns(pl.col("date").cast(pl.Date))
        bench = pl.DataFrame(
            {"date": dates, "returns": rng.normal(0.0004, 0.009, n).tolist()}
        ).with_columns(pl.col("date").cast(pl.Date))

        result = reports.markdown(strat, benchmark=bench)
        assert "## Regime Analysis" in result
        assert "bull_low_vol" in result or "bear_low_vol" in result

    def test_output_writes_markdown_file(self, sample_df, tmp_path):
        out_file = tmp_path / "report.md"
        result = reports.markdown(sample_df, output=str(out_file))
        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8") == result


# ---------------------------------------------------------------------------
# Monte Carlo integration tests
# ---------------------------------------------------------------------------


class TestMonteCarloReports:
    def test_full_returns_monte_carlo_summary(self, sample_df):
        result = reports.full(
            sample_df, show=False, monte_carlo=True, mc_sims=50, mc_seed=0
        )
        mc = result["monte_carlo"]
        assert isinstance(mc, dict)
        assert "terminal" in mc
        assert "maxdd" in mc
        assert "sharpe" in mc
        assert "cagr" in mc

    def test_full_monte_carlo_false_returns_none(self, sample_df):
        result = reports.full(sample_df, show=False, monte_carlo=False)
        assert result["monte_carlo"] is None

    def test_full_monte_carlo_false_omits_figures(self, sample_df):
        result = reports.full(sample_df, show=False, monte_carlo=False)
        figs = result["figures"]
        assert "monte_carlo" not in figs
        assert "monte_carlo_distribution" not in figs

    def test_full_monte_carlo_true_includes_figures(self, sample_df):
        result = reports.full(
            sample_df, show=False, monte_carlo=True, mc_sims=20, mc_seed=0
        )
        figs = result["figures"]
        assert "monte_carlo" in figs
        assert "monte_carlo_distribution" in figs

    def test_html_contains_monte_carlo_section(self, sample_df):
        result = reports.html(sample_df, monte_carlo=True, mc_sims=50, mc_seed=0)
        assert "Monte Carlo Projection" in result

    def test_html_no_monte_carlo_omits_section(self, sample_df):
        result = reports.html(sample_df, monte_carlo=False)
        assert "Monte Carlo Projection" not in result

    def test_json_includes_monte_carlo_key(self, sample_df):
        payload = json.loads(
            reports.json(sample_df, monte_carlo=True, mc_sims=50, mc_seed=0)
        )
        assert "monte_carlo" in payload
        assert payload["monte_carlo"] is not None
        assert "terminal" in payload["monte_carlo"]

    def test_json_no_monte_carlo_is_null(self, sample_df):
        payload = json.loads(reports.json(sample_df, monte_carlo=False))
        assert payload["monte_carlo"] is None

    def test_json_bust_goal_probs_present_when_set(self, sample_df):
        payload = json.loads(
            reports.json(
                sample_df,
                monte_carlo=True,
                mc_sims=50,
                mc_bust=-0.01,
                mc_goal=0.01,
                mc_seed=0,
            )
        )
        mc = payload["monte_carlo"]
        assert mc["bust_probability"] is not None
        assert mc["goal_probability"] is not None
        assert 0.0 <= mc["bust_probability"] <= 1.0
        assert 0.0 <= mc["goal_probability"] <= 1.0

    def test_markdown_includes_monte_carlo_section(self, sample_df):
        result = reports.markdown(sample_df, monte_carlo=True, mc_sims=50, mc_seed=0)
        assert "## Monte Carlo" in result

    def test_markdown_no_monte_carlo_omits_section(self, sample_df):
        result = reports.markdown(sample_df, monte_carlo=False)
        assert "## Monte Carlo" not in result


def test_df_to_html_table_escapes_xss():
    """Test that HTML characters are escaped to prevent XSS."""
    df = pl.DataFrame(
        {"<script>alert('xss')</script>": ["<img src=x onerror=alert(1)>", "safe_val"]}
    )
    html_output = reports._df_to_html_table(df)

    assert "<script>" not in html_output
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html_output
    assert "<img src=x onerror=alert(1)>" not in html_output
    assert "&lt;img src=x onerror=alert(1)&gt;" in html_output
    assert "safe_val" in html_output


class TestHtmlTitleEscaping:
    def test_script_tag_in_title_is_escaped(self, sample_df):
        result = reports.html(sample_df, title="<script>alert('xss')</script>")
        assert "<script>alert" not in result
        assert "&lt;script&gt;" in result

    def test_ampersand_in_title_is_escaped(self, sample_df):
        result = reports.html(sample_df, title="Alpha & Beta")
        assert "Alpha &amp; Beta" in result
