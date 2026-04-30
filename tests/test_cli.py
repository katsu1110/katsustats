"""Integration tests for the katsustats CLI."""

from __future__ import annotations

import datetime
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
