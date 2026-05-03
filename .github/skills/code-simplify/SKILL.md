---
name: 'code-simplify'
description: 'Refactor recently changed code for clarity, consistency, and maintainability without changing behavior. Use when asked to simplify code, clean up a recent implementation, or run /code-simplify after making changes.'
license: 'Apache-2.0'
---

# Code Simplify

Refine recent changes in this repository so they are easier to read, easier to maintain, and fully behavior-preserving.

Use this skill when the user wants a high-quality cleanup pass after implementing a change.

## When to Use This Skill

Use this skill when:
- The user asks to simplify or refactor recently changed code
- The user wants a cleanup pass after a feature or bug fix
- The user invokes `/code-simplify`
- A recent implementation works but could be clearer or more consistent with the repository style

## Requirements

- Preserve exact behavior, outputs, public APIs, and tests unless the user explicitly asks for a behavioral change
- Focus on files changed in the current branch, PR, or session unless the user asks for a broader refactor
- Follow repository conventions from `CLAUDE.md`, especially the functional style, snake_case naming, short public docstrings, and existing validation/error-handling patterns
- Prefer explicit, readable Python over clever or overly compressed code
- Keep changes surgical and avoid unrelated edits
- Run the existing repository validation commands after making refinements

## Step-by-Step Workflow

### 1. Identify the active refactor scope

Determine which files or hunks were recently changed by checking the current diff, recent commits, or the files touched in the current task. If the scope is unclear, default to the recent changes instead of the whole repository.

### 2. Inspect the changed code before editing

Look for simplification opportunities that improve maintainability without changing behavior, such as:
- Reducing unnecessary branching or nesting
- Removing duplication or dead intermediate abstractions
- Choosing clearer names where it does not affect external APIs
- Grouping closely related logic more cleanly
- Reusing existing helpers and patterns already present in the repository

### 3. Apply repository-specific simplifications

Refactor toward the existing katsustats style:
- Keep the code functional and straightforward
- Match existing module organization and section-banner conventions
- Preserve public signatures unless the user asked for an API change
- Add or keep short public docstrings when needed for consistency
- Do not introduce classes, frameworks, or new dependencies just to simplify code

### 4. Verify nothing changed semantically

After simplifying, run validation that actually covers the files you changed.

- If the simplification touched Python code under `src/` or `tests/`, run the repository checks already used in CI:
  - `uv run ruff check src/ tests/`
  - `uv run ruff format --check src/ tests/`
  - `uv run pytest tests/ -v`
- If the simplification touched other files (for example docs, workflows, or other skills), run any relevant repository validation for those files if available.
- Do not claim the refactor was fully verified if your commands did not cover every changed file; instead, say exactly which checks you ran and which edited files had no automated validation.

### 5. Summarize only meaningful refinements

Report the important simplifications that improve readability, consistency, or maintainability. Do not pad the summary with trivial formatting details.

## Gotchas

- Do not broaden the scope from "recent changes" to unrelated cleanup unless the user explicitly asks
- Do not optimize for fewer lines if it makes the code harder to follow
- Do not remove useful structure or helper boundaries that make the code easier to understand
- Do not change validation, numeric behavior, or report output unless that is the requested task
- Do not add comments that explain obvious code; prefer clearer code instead

## References

- `CLAUDE.md`
- `.github/workflows/ci.yml`
