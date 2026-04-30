"""CLI entry point: ``katsustats report <file> [options]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from katsustats import reports


def _load(path: str, date_col: str, returns_col: str) -> pl.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        df = pl.read_parquet(p)
    elif suffix == ".csv":
        df = pl.read_csv(p, try_parse_dates=True)
    else:
        sys.exit(f"Unsupported file format '{suffix}'. Use .csv or .parquet.")

    rename: dict[str, str] = {}
    if date_col != "date":
        rename[date_col] = "date"
    if returns_col != "returns":
        rename[returns_col] = "returns"
    if rename:
        df = df.rename(rename)

    return df


def _cmd_report(args: argparse.Namespace) -> None:
    df = _load(args.file, args.date_col, args.returns_col)

    benchmark = None
    if args.benchmark:
        benchmark = _load(
            args.benchmark,
            args.benchmark_date_col,
            args.benchmark_returns_col,
        )

    output = args.output or str(Path(args.file).with_suffix(".html"))
    reports.html(
        df,
        benchmark=benchmark,
        rf=args.rf,
        title=args.title,
        output=output,
    )
    print(f"Report written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="katsustats",
        description="Generate backtest tearsheet reports from return series files.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    p_report = sub.add_parser(
        "report",
        help="Generate a self-contained HTML tearsheet from a CSV or Parquet file.",
    )
    p_report.add_argument("file", help="Path to a .csv or .parquet returns file.")
    p_report.add_argument(
        "--title", default="Strategy", help="Report title (default: Strategy)."
    )
    p_report.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HTML file path (default: <file>.html).",
    )
    p_report.add_argument(
        "--date-col",
        default="date",
        dest="date_col",
        help="Name of the date column (default: date).",
    )
    p_report.add_argument(
        "--returns-col",
        default="returns",
        dest="returns_col",
        help="Name of the returns column (default: returns).",
    )
    p_report.add_argument(
        "--rf",
        type=float,
        default=0.0,
        help="Annualized risk-free rate (default: 0.0).",
    )
    p_report.add_argument(
        "--benchmark",
        default=None,
        help="Path to a benchmark .csv or .parquet file.",
    )
    p_report.add_argument(
        "--benchmark-date-col",
        default="date",
        dest="benchmark_date_col",
        help="Benchmark date column name (default: date).",
    )
    p_report.add_argument(
        "--benchmark-returns-col",
        default="returns",
        dest="benchmark_returns_col",
        help="Benchmark returns column name (default: returns).",
    )
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
