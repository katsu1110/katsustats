# CLAUDE.md

## Project overview

katsustats is a Python library for generating backtest reports from financial return series. It takes a Polars DataFrame with `["date", "returns"]` columns and produces summary metrics, drawdown analysis, matplotlib charts, and self-contained HTML reports.

## Architecture

Three-module functional design in `src/katsustats/`:

- `stats.py` — Pure metric computation (Sharpe, CAGR, drawdowns, etc.)
- `plots.py` — Matplotlib chart generation (8 chart types)
- `reports.py` — Orchestration: `full()` for dict output, `html()` for HTML report

No classes. All functions expect a Polars DataFrame with `date: pl.Date` and `returns: pl.Float64` columns.

## Setup

```bash
uv sync --dev   # includes pytest + ruff
```

- Python 3.13 (`.python-version`), requires >=3.9
- Build backend: hatchling (src layout)
- Dependencies: polars, numpy, matplotlib
- Dev dependencies: pytest, ruff

## Build

```bash
uv build
```

## Tests and linting

```bash
uv run ruff check src/ tests/         # lint
uv run ruff format --check src/ tests/ # format check
uv run pytest tests/ -v               # 113 tests across stats/plots/reports
```

CI runs on every PR and push to main (`.github/workflows/ci.yml`). All tests and ruff checks must pass before merging.

## Code conventions

- `from __future__ import annotations` in all modules
- `py.typed` marker for PEP 561
- snake_case functions, `_` prefix for private helpers
- Short docstrings on all public functions
- Section separators: `# ---...---` comment banners
- Input validation via `assert` statements
- Plot styling centralized in `_COLORS` dict with `_apply_style()` / `_add_title()` helpers
- Purely functional — no OOP
- Ruff: line-length=88, target py39, select E/W/F/I/UP, ignore E501
