# Contributing

Contributions are welcome. Please open an issue before submitting non-trivial changes.

## Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (package manager)

## Setup

```bash
git clone https://github.com/katsu1110/katsustats.git
cd katsustats
uv sync --dev
```

## Running tests

```bash
uv run pytest tests/ -v
```

## Linting and formatting

```bash
uv run ruff check src/ tests/ examples/
uv run ruff format src/ tests/ examples/
```

CI enforces both checks on every pull request.

## Code conventions

- All modules start with `from __future__ import annotations`
- Prefer functions over introducing new public classes; typing-only/internal classes (e.g. `Protocol`s) are allowed where appropriate
- snake_case function names; `_` prefix for private helpers
- Public functions should have clear docstrings; use one-line or multi-line docstrings as appropriate
- Validate inputs explicitly and follow existing patterns; both `assert` and `raise ValueError` are used in the codebase where appropriate
- Plot styling goes through the shared `_apply_style()` / `_add_title()` helpers in `plots.py`
- Keep examples self-contained — no helper imports between example scripts

## Pull requests

- Keep PRs focused on a single concern
- Update `CHANGELOG.md` under `[Unreleased]` if your change affects public behaviour
- All existing tests must pass; add tests for new public functions
