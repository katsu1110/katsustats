"""Integration tests for the katsustats CLI."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import polars as pl
import pytest

from katsustats.__main__ import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATES = [datetime.date(2023, 1, 2) + datetime.timedelta(days=i) for i in range(20)]
_RETURNS = [
    0.01,
    -0.005,
    0.008,
    -0.012,
    0.003,
    0.015,
    -0.007,
    0.002,
    -0.004,
    0.009,
    -0.011,
    0.006,
    0.013,
    -0.002,
    0.004,
    0.007,
    -0.008,
    0.001,
    -0.003,
    0.011,
]


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "returns.csv"
    pl.DataFrame({"date": _DATES, "returns": _RETURNS}).write_csv(p)
    return p


@pytest.fixture
def parquet_file(tmp_path: Path) -> Path:
    p = tmp_path / "returns.parquet"
    pl.DataFrame({"date": _DATES, "returns": _RETURNS}).write_parquet(p)
    return p


@pytest.fixture
def csv_file_custom_cols(tmp_path: Path) -> Path:
    p = tmp_path / "returns_custom.csv"
    pl.DataFrame({"day": _DATES, "pnl": _RETURNS}).write_csv(p)
    return p


@pytest.fixture
def benchmark_csv(tmp_path: Path) -> Path:
    p = tmp_path / "benchmark.csv"
    bench = [r * 0.6 for r in _RETURNS]
    pl.DataFrame({"date": _DATES, "returns": bench}).write_csv(p)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReportCommand:
    def test_csv_produces_html(self, csv_file: Path, tmp_path: Path, monkeypatch):
        out = tmp_path / "out.html"
        monkeypatch.setattr(
            "sys.argv", ["katsustats", "report", str(csv_file), "-o", str(out)]
        )
        main()
        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_parquet_produces_html(
        self, parquet_file: Path, tmp_path: Path, monkeypatch
    ):
        out = tmp_path / "out.html"
        monkeypatch.setattr(
            "sys.argv", ["katsustats", "report", str(parquet_file), "-o", str(out)]
        )
        main()
        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_default_output_path(self, csv_file: Path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["katsustats", "report", str(csv_file)])
        main()
        expected = csv_file.with_suffix(".html")
        assert expected.exists()
        assert expected.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_json_output_format(self, csv_file: Path, tmp_path: Path, monkeypatch):
        out = tmp_path / "out.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "katsustats",
                "report",
                str(csv_file),
                "--format",
                "json",
                "-o",
                str(out),
            ],
        )
        main()
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["metadata"]["has_benchmark"] is False

    def test_json_default_output_path(self, csv_file: Path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["katsustats", "report", str(csv_file), "--format", "json"],
        )
        main()
        expected = csv_file.with_suffix(".json")
        assert expected.exists()
        payload = json.loads(expected.read_text(encoding="utf-8"))
        assert payload["metadata"]["title"] == "Strategy"

    def test_markdown_output_format(self, csv_file: Path, tmp_path: Path, monkeypatch):
        out = tmp_path / "out.md"
        monkeypatch.setattr(
            "sys.argv",
            [
                "katsustats",
                "report",
                str(csv_file),
                "--format",
                "markdown",
                "-o",
                str(out),
            ],
        )
        main()
        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("# Strategy Backtest Summary")

    def test_markdown_default_output_path(self, csv_file: Path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["katsustats", "report", str(csv_file), "--format", "markdown"],
        )
        main()
        expected = csv_file.with_suffix(".md")
        assert expected.exists()
        assert expected.read_text(encoding="utf-8").startswith(
            "# Strategy Backtest Summary"
        )

    def test_custom_column_names(
        self, csv_file_custom_cols: Path, tmp_path: Path, monkeypatch
    ):
        out = tmp_path / "out.html"
        monkeypatch.setattr(
            "sys.argv",
            [
                "katsustats",
                "report",
                str(csv_file_custom_cols),
                "--date-col",
                "day",
                "--returns-col",
                "pnl",
                "-o",
                str(out),
            ],
        )
        main()
        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_with_benchmark(
        self, csv_file: Path, benchmark_csv: Path, tmp_path: Path, monkeypatch
    ):
        out = tmp_path / "out.html"
        monkeypatch.setattr(
            "sys.argv",
            [
                "katsustats",
                "report",
                str(csv_file),
                "--benchmark",
                str(benchmark_csv),
                "-o",
                str(out),
            ],
        )
        main()
        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_json_with_benchmark(
        self, csv_file: Path, benchmark_csv: Path, tmp_path: Path, monkeypatch
    ):
        out = tmp_path / "out.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "katsustats",
                "report",
                str(csv_file),
                "--benchmark",
                str(benchmark_csv),
                "--format",
                "json",
                "-o",
                str(out),
            ],
        )
        main()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["benchmark"] is not None
        assert payload["comparison"] is not None

    def test_markdown_with_benchmark(
        self, csv_file: Path, benchmark_csv: Path, tmp_path: Path, monkeypatch
    ):
        out = tmp_path / "out.md"
        monkeypatch.setattr(
            "sys.argv",
            [
                "katsustats",
                "report",
                str(csv_file),
                "--benchmark",
                str(benchmark_csv),
                "--format",
                "markdown",
                "-o",
                str(out),
            ],
        )
        main()
        text = out.read_text(encoding="utf-8")
        assert "| Metric | Strategy | Benchmark |" in text

    def test_custom_title_appears_in_output(
        self, csv_file: Path, tmp_path: Path, monkeypatch
    ):
        out = tmp_path / "out.html"
        monkeypatch.setattr(
            "sys.argv",
            [
                "katsustats",
                "report",
                str(csv_file),
                "--title",
                "My Alpha Strategy",
                "-o",
                str(out),
            ],
        )
        main()
        assert "My Alpha Strategy" in out.read_text(encoding="utf-8")

    def test_unsupported_extension_exits(self, tmp_path: Path, monkeypatch):
        bad = tmp_path / "data.xlsx"
        bad.write_text("")
        monkeypatch.setattr(
            "sys.argv",
            ["katsustats", "report", str(bad), "-o", str(tmp_path / "out.html")],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert "not supported" in str(exc_info.value)
        assert ".xlsx" in str(exc_info.value)

    def test_no_extension_exits_with_clear_message(self, tmp_path: Path, monkeypatch):
        bad = tmp_path / "data"
        bad.write_text("")
        monkeypatch.setattr(
            "sys.argv",
            ["katsustats", "report", str(bad), "-o", str(tmp_path / "out.html")],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert "no extension" in str(exc_info.value)

    def test_missing_file_exits_with_clear_message(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "katsustats",
                "report",
                str(tmp_path / "nonexistent.csv"),
                "-o",
                str(tmp_path / "out.html"),
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert "file not found" in str(exc_info.value)

    def test_invalid_csv_parsing_failure(self, tmp_path: Path, monkeypatch):
        bad = tmp_path / "malformed.csv"
        # Create a CSV with ragged lines that polars read_csv will reject
        bad.write_text(
            "date,returns\n2020-01-01\n2020-01-02,0.01,0.02\n", encoding="utf-8"
        )

        monkeypatch.setattr(
            "sys.argv",
            ["katsustats", "report", str(bad), "-o", str(tmp_path / "out.html")],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert "Failed to parse file" in str(exc_info.value)

    def test_missing_column_exits_with_clear_message(
        self, csv_file: Path, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(
            "sys.argv",
            [
                "katsustats",
                "report",
                str(csv_file),
                "--returns-col",
                "pnl",
                "-o",
                str(tmp_path / "out.html"),
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert "pnl" in str(exc_info.value)
        assert "--returns-col" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Monte Carlo CLI flags
# ---------------------------------------------------------------------------


class TestMonteCarloCliFlags:
    def test_monte_carlo_off_by_default(
        self, csv_file: Path, tmp_path: Path, monkeypatch
    ):
        out = tmp_path / "out.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "katsustats",
                "report",
                str(csv_file),
                "--format",
                "json",
                "-o",
                str(out),
            ],
        )
        main()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["monte_carlo"] is None

    def test_mc_sims_and_seed_produce_deterministic_output(
        self, csv_file: Path, tmp_path: Path, monkeypatch
    ):
        out1 = tmp_path / "out1.json"
        out2 = tmp_path / "out2.json"
        args_base = [
            "katsustats",
            "report",
            str(csv_file),
            "--format",
            "json",
            "--mc-sims",
            "50",
            "--mc-seed",
            "99",
        ]
        monkeypatch.setattr("sys.argv", args_base + ["-o", str(out1)])
        main()
        monkeypatch.setattr("sys.argv", args_base + ["-o", str(out2)])
        main()
        j1 = json.loads(out1.read_text(encoding="utf-8"))
        j2 = json.loads(out2.read_text(encoding="utf-8"))
        assert j1["monte_carlo"] == j2["monte_carlo"]

    def test_mc_bust_and_goal_flags(self, csv_file: Path, tmp_path: Path, monkeypatch):
        out = tmp_path / "out.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "katsustats",
                "report",
                str(csv_file),
                "--format",
                "json",
                "--monte-carlo",
                "--mc-sims",
                "50",
                "--mc-seed",
                "0",
                "--mc-bust",
                "-0.05",
                "--mc-goal",
                "0.05",
                "-o",
                str(out),
            ],
        )
        main()
        payload = json.loads(out.read_text(encoding="utf-8"))
        mc = payload["monte_carlo"]
        assert mc["bust_probability"] is not None
        assert mc["goal_probability"] is not None


class TestPeriodsAndMcMethodCliFlags:
    def test_periods_flag(self, csv_file: Path, tmp_path: Path, monkeypatch):
        out = tmp_path / "out.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "katsustats",
                "report",
                str(csv_file),
                "--format",
                "json",
                "--periods",
                "365",
                "-o",
                str(out),
            ],
        )
        main()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["metadata"]["periods"] == 365

    def test_mc_method_shuffle(self, csv_file: Path, tmp_path: Path, monkeypatch):
        out = tmp_path / "out.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "katsustats",
                "report",
                str(csv_file),
                "--format",
                "json",
                "--monte-carlo",
                "--mc-sims",
                "50",
                "--mc-seed",
                "0",
                "--mc-method",
                "shuffle",
                "-o",
                str(out),
            ],
        )
        main()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["monte_carlo"] is not None

    def test_mc_method_default_bootstrap(
        self, csv_file: Path, tmp_path: Path, monkeypatch
    ):
        out = tmp_path / "out.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "katsustats",
                "report",
                str(csv_file),
                "--format",
                "json",
                "--monte-carlo",
                "--mc-sims",
                "50",
                "--mc-seed",
                "0",
                "-o",
                str(out),
            ],
        )
        main()
        payload = json.loads(out.read_text(encoding="utf-8"))
        mc = payload["monte_carlo"]
        assert mc["terminal"]["min"] < mc["terminal"]["max"]


# ---------------------------------------------------------------------------
# snapshot subcommand
# ---------------------------------------------------------------------------


class TestSnapshotCommand:
    def test_creates_png(self, csv_file: Path, tmp_path: Path, monkeypatch):
        out = tmp_path / "snap.png"
        monkeypatch.setattr(
            "sys.argv",
            ["katsustats", "snapshot", str(csv_file), "-o", str(out)],
        )
        main()
        assert out.exists()
        assert out.read_bytes()[:4] == b"\x89PNG"

    def test_default_output_path(self, csv_file: Path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["katsustats", "snapshot", str(csv_file)],
        )
        main()
        expected = csv_file.parent / (csv_file.stem + "_snapshot.png")
        assert expected.exists()

    def test_custom_window_string(self, csv_file: Path, tmp_path: Path, monkeypatch):
        out = tmp_path / "snap.png"
        monkeypatch.setattr(
            "sys.argv",
            ["katsustats", "snapshot", str(csv_file), "--window", "1M", "-o", str(out)],
        )
        main()
        assert out.exists()

    def test_integer_window(self, csv_file: Path, tmp_path: Path, monkeypatch):
        out = tmp_path / "snap.png"
        monkeypatch.setattr(
            "sys.argv",
            ["katsustats", "snapshot", str(csv_file), "--window", "5", "-o", str(out)],
        )
        main()
        assert out.exists()

    def test_bad_window_exits(self, csv_file: Path, tmp_path: Path, monkeypatch):
        out = tmp_path / "snap.png"
        monkeypatch.setattr(
            "sys.argv",
            ["katsustats", "snapshot", str(csv_file), "--window", "2Y", "-o", str(out)],
        )
        with pytest.raises(SystemExit):
            main()

    def test_custom_title(self, csv_file: Path, tmp_path: Path, monkeypatch):
        out = tmp_path / "snap.png"
        monkeypatch.setattr(
            "sys.argv",
            [
                "katsustats",
                "snapshot",
                str(csv_file),
                "--title",
                "MyAlpha",
                "-o",
                str(out),
            ],
        )
        main()
        assert out.exists()

    def test_parquet_input(self, parquet_file: Path, tmp_path: Path, monkeypatch):
        out = tmp_path / "snap.png"
        monkeypatch.setattr(
            "sys.argv",
            ["katsustats", "snapshot", str(parquet_file), "-o", str(out)],
        )
        main()
        assert out.exists()

    def test_missing_file_exits(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["katsustats", "snapshot", str(tmp_path / "nope.csv")],
        )
        with pytest.raises(SystemExit, match="file not found"):
            main()
