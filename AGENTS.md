# AGENTS.md

This file provides instructions and context for AI agents working on the `katsustats` repository.

## Context & Resources
- **Project Name**: katsustats
- **Description**: A Python library and CLI for generating financial backtest reports (HTML, JSON, Markdown).
- **Core Files**:
  - `README.md`: General overview and usage for humans.
  - `CLAUDE.md`: Tech stack, architecture, and development workflows.
  - `CONTRIBUTING.md`: Contribution guidelines.

## Core Principles & Architecture
- **Functional Design**: Purely functional approach in `src/katsustats/`. Avoid classes unless absolutely necessary for internal typing.
- **Data Normalization**: All public functions expect `date` and `returns` columns. Use `src/katsustats/_dataframe.py:ensure_polars()` to handle varied inputs (pandas, Series, etc.).
- **No Side Effects**: Library code should avoid global state or unexpected side effects.
- **Plot Styling**: Centralized via `_COLORS`, `_apply_style()`, and `_add_title()` in `plots.py`.
- **Report Formats**:
  - `stats.py`: Raw metrics.
  - `plots.py`: Matplotlib charts.
  - `reports.py`: Orchestration for HTML, JSON, and Markdown.

## Common Agent Tasks
- **Testing**: Run `uv run pytest tests/ -v`.
- **Linting**: Run `uv run ruff check src/ tests/`.
- **Formatting**: Run `uv run ruff format src/ tests/`.
- **Adding Metrics**: Implement in `stats.py`, then expose via `reports.py` (both JSON and Markdown formats).
- **Adding Plots**: Implement in `plots.py`, ensure they use `_apply_style()`, and update `reports.html()`.

## Development Commands
- `uv sync --dev` - Install dependencies.
- `uv build` - Build the package.
- `uv run katsustats report --help` - Verify CLI functionality.
