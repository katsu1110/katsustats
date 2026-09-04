# AGENTS.md

Instructions and context for AI agents working on the `katsustats` repository.

## Context & Resources
- **Project Name**: katsustats
- **Description**: A Python library and CLI for generating financial backtest reports (HTML, JSON, Markdown). Takes a Polars DataFrame with `["date", "returns"]` columns (or pandas) and produces summary metrics, drawdown analysis, matplotlib charts, and self-contained reports.
- **Core Files**:
  - `README.md`: Overview and quickstart for humans.
  - `docs/`: Feature docs (`reports.md`, `metrics.md`, `monte-carlo.md`, `snapshot.md`, `cli.md`, `data-format.md`).
  - `examples/`: Runnable usage (`quickstart.py`, `with_benchmark.py`, `html_report.py`).
  - `CONTRIBUTING.md`: Contribution guidelines.

## Core Principles & Architecture
- **Functional Design**: Pure functions in `src/katsustats/`; no classes except for internal typing, no global state, no side effects.
- **Data Normalization**: All public functions expect `date` and `returns` columns. Use `src/katsustats/_dataframe.py:ensure_polars()` for varied inputs (pandas DataFrames, pandas Series with DatetimeIndex, Polars). Exception: `utils.to_returns()` takes price frames (no `returns` col) and converts without `ensure_polars()`.
- **Modules**:
  - `stats.py`: Pure metric computation (70+ functions: Sharpe, CAGR, drawdowns, rolling metrics, Monte Carlo, quantstats-parity extras).
  - `plots.py`: Matplotlib chart generation (21 `plot_*` functions); styling centralized via `_COLORS`, `_apply_style()`, `_add_title()`.
  - `reports.py`: Orchestration — `full()` (dict + figures), `basic()` (slim variant), `metrics(mode="basic"|"full")` (table only), `html()` / `json()` / `markdown()` (self-contained outputs).
  - `utils.py`: Prices/returns/log-returns conversions, period aggregation, monthly pivot, `download_returns()` (requires optional yfinance).
  - `_constants.py`: Column-name constants (`COL_DATE`, `COL_RETURNS`) shared across modules.
  - `__main__.py`: CLI entry point (`katsustats report` with `--format html|json|markdown`, plus `katsustats snapshot`); reads CSV/Parquet.

## Setup & Build
```bash
uv sync --dev   # includes pytest + ruff (+ pandas for input normalization)
uv build
```
- Python 3.13 (`.python-version`), requires >=3.9
- Build backend: hatchling (src layout)
- Dependencies: polars, numpy, matplotlib
- Dev dependencies: pytest, ruff, pandas, pre-commit
- Optional: yfinance (only for `utils.download_returns()`)

## CI
`.github/workflows/ci.yml` runs on every PR and push to main. All tests and ruff checks must pass before merging.

## Common Agent Tasks
- **Testing**: Run `uv run pytest tests/ -v`.
- **Linting**: Run `uv run ruff check src/ tests/`.
- **Formatting**: Run `uv run ruff format src/ tests/` (CI uses `--check`).
- **Adding Metrics**: Implement in `stats.py`, register display specs (`_SUMMARY_METRIC_SPECS` or `_FULL_EXTRA_METRICS` in `reports.py`); `summary_metrics_raw()` + JSON/Markdown pick them up automatically.
- **Adding Plots**: Implement in `plots.py` using `_apply_style()`; wire into `reports.full()` figures dict and the HTML builder.
- **CLI sanity check**: `uv run katsustats report --help`.

## Code Conventions
- `from __future__ import annotations` in all modules; `py.typed` marker for PEP 561
- snake_case functions, `_` prefix for private helpers; short docstrings on all public functions
- Section separators: `# ---...---` comment banners
- Input validation via `assert` or `raise ValueError` in library code; `sys.exit()` with user-facing messages in the CLI
- Ruff: line-length=88, target py39, select E/W/F/I/UP, ignore E501

## Agent Skills
Skills are located in `.claude/skills/`:
- `publish` — Publish a new katsustats release to PyPI.
