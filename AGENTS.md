# AGENTS.md

This file provides instructions and context for AI agents working on the `katsustats` repository.

## Context & Resources
- **Project Name**: katsustats
- **Description**: A Python library and CLI for generating financial backtest reports (HTML, JSON, Markdown). Takes a Polars DataFrame with `["date", "returns"]` columns (or pandas) and produces summary metrics, drawdown analysis, matplotlib charts, and self-contained reports.
- **Core Files**:
  - `README.md`: General overview and usage for humans.
  - `CONTRIBUTING.md`: Contribution guidelines.

## Core Principles & Architecture
- **Functional Design**: Purely functional approach in `src/katsustats/`. Avoid classes unless absolutely necessary for internal typing.
- **Data Normalization**: All public functions expect `date` and `returns` columns. Use `src/katsustats/_dataframe.py:ensure_polars()` to handle varied inputs (pandas DataFrames, pandas Series with DatetimeIndex, Polars).
- **No Side Effects**: Library code should avoid global state or unexpected side effects.
- **Modules**:
  - `stats.py`: Pure metric computation (Sharpe, CAGR, drawdowns, etc.).
  - `plots.py`: Matplotlib chart generation (18 chart types); styling centralized via `_COLORS`, `_apply_style()`, `_add_title()`.
  - `reports.py`: Orchestration — `full()` (dict), `html()` (self-contained HTML), `json()` (structured JSON), `markdown()` (Markdown tables).
  - `_constants.py`: Shared numeric constants (trading days per year, etc.) used across stats and plots.
  - `__main__.py`: CLI entry point (`katsustats report`); reads CSV/Parquet and dispatches via `--format`.

## Setup & Build
```bash
uv sync --dev   # includes pytest + ruff (+ pandas for input normalization)
uv build
```
- Python 3.13 (`.python-version`), requires >=3.9
- Build backend: hatchling (src layout)
- Dependencies: polars, numpy, matplotlib
- Dev dependencies: pytest, ruff, pandas, pre-commit

## CI
`.github/workflows/ci.yml` runs on every PR and push to main. All tests and ruff checks must pass before merging.

## Common Agent Tasks
- **Testing**: Run `uv run pytest tests/ -v`.
- **Linting**: Run `uv run ruff check src/ tests/`.
- **Formatting**: Run `uv run ruff format src/ tests/` (CI uses `--check`).
- **Adding Metrics**: Implement in `stats.py`, then expose via `reports.py` (both JSON and Markdown formats).
- **Adding Plots**: Implement in `plots.py`, ensure they use `_apply_style()`, and update `reports.html()`.
- **CLI sanity check**: `uv run katsustats report --help`.

## Code Conventions
- `from __future__ import annotations` in all modules; `py.typed` marker for PEP 561
- snake_case functions, `_` prefix for private helpers; short docstrings on all public functions
- Section separators: `# ---...---` comment banners
- Input validation via `raise ValueError` in library code; `sys.exit()` with user-facing messages in the CLI
- pandas inputs are normalized in `_dataframe.py:ensure_polars()` — no changes needed in stats/plots/reports
- Purely functional — no OOP
- Ruff: line-length=88, target py39, select E/W/F/I/UP, ignore E501

## Agent Skills
Skills are located in `.claude/skills/`:
- `publish` — Publish a new katsustats release to PyPI.
