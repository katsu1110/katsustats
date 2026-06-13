# CLAUDE.md

## Project overview

katsustats is a Python library and CLI for generating backtest reports from financial return series. It takes a Polars DataFrame with `["date", "returns"]` columns (or a pandas DataFrame/Series) and produces summary metrics, drawdown analysis, matplotlib charts, and self-contained HTML, JSON, or Markdown reports.

## Architecture

Functional design in `src/katsustats/`:

- `stats.py` — Pure metric computation (Sharpe, CAGR, drawdowns, etc.)
- `plots.py` — Matplotlib chart generation (13 chart types)
- `reports.py` — Orchestration: `full()` for dict output, `html()` for self-contained HTML, `json()` for structured JSON, `markdown()` for Markdown tables
- `_dataframe.py` — Input normalisation: `ensure_polars()` converts pandas DataFrames, pandas Series (with DatetimeIndex), and Polars DataFrames to the canonical `["date", "returns"]` Polars frame
- `__main__.py` — CLI entry point (`katsustats report`); reads CSV/Parquet and dispatches to `reports.html()`, `reports.json()`, or `reports.markdown()` via `--format`

No classes. All public functions accept a Polars DataFrame, pandas DataFrame, or pandas Series with `date` and `returns` data.

## Setup

```bash
uv sync --dev   # includes pytest + ruff
```

- Python 3.13 (`.python-version`), requires >=3.9
- Build backend: hatchling (src layout)
- Dependencies: polars, numpy, matplotlib
- Dev dependencies: pytest, ruff, pandas, pre-commit

## Build

```bash
uv build
```

## Tests and linting

```bash
uv run ruff check src/ tests/          # lint
uv run ruff format --check src/ tests/ # format check
uv run pytest tests/ -v                # tests across stats/plots/reports/cli
```

CI runs on every PR and push to main (`.github/workflows/ci.yml`). All tests and ruff checks must pass before merging.

## Code conventions

- `from __future__ import annotations` in all modules
- `py.typed` marker for PEP 561
- snake_case functions, `_` prefix for private helpers
- Short docstrings on all public functions
- Section separators: `# ---...---` comment banners
- Input validation via `assert` statements in library code; `sys.exit()` with user-facing messages in the CLI
- pandas inputs (DataFrame with `date` column, DatetimeIndex DataFrame, or Series) are normalised in `_dataframe.py:ensure_polars()` — no changes needed in stats/plots/reports
- Plot styling centralized in `_COLORS` dict with `_apply_style()` / `_add_title()` helpers
- Purely functional — no OOP
- Ruff: line-length=88, target py39, select E/W/F/I/UP, ignore E501

## Slash Commands (Claude Code)

- `/publish` — cut a new PyPI release (bumps version, commits, creates GitHub release that triggers the publish workflow)
