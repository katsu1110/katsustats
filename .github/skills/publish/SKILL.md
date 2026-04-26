---
name: publish
description: Publish a new katsustats release to PyPI. Use this when asked to cut a release, bump the package version, create GitHub release notes, or publish katsustats to PyPI.
license: Apache-2.0
---

# Publish katsustats to PyPI

Use this skill when the user wants to publish a new katsustats release.

This repository is a Python package named `katsustats`.

Important repository facts:
- The package version is defined in `src/katsustats/__init__.py` as `__version__`.
- `pyproject.toml` uses dynamic versioning via `[tool.hatch.version] path = "src/katsustats/__init__.py"`.
- The release workflow is defined in `.github/workflows/publish.yml`.
- Publishing to PyPI happens automatically when a GitHub release is published.
- The workflow builds with `uv build` and publishes with trusted publishing.
- The default branch is `main`.

Follow this process exactly.

## 1. Determine the new version

1. Read the current version from `src/katsustats/__init__.py`.
2. If the user explicitly provided a version, use it.
3. Otherwise, inspect commits since the latest release tag and choose the bump level:
   - **patch** (`x.y.Z`): bug fixes only, internal cleanups, no new public API.
   - **minor** (`x.Y.0`): new public functions, new features, backward-compatible enhancements.
   - **major** (`X.0.0`): breaking public API changes.
4. Base the rationale on actual commits, not assumptions.
5. If there is no prior tag, compare against the full history and explain that no previous release tag exists.

## 2. Confirm before making changes

Before editing files, committing, pushing, or creating a release:
- State the current version, proposed new version, and bump rationale in one sentence.
- Ask the user to confirm.
- Stop and wait for confirmation.

Never perform the publish steps without explicit confirmation.

## 3. Bump the version

After confirmation:
1. Update `src/katsustats/__init__.py` so `__version__` matches the new version.
2. Keep the change minimal.

## 4. Commit and push

After the version bump:
1. Stage `src/katsustats/__init__.py`.
2. Commit with this exact message:
   - `chore: bump version to <new_version>`
3. Push to `origin main`.

Before committing, check whether the working tree contains unrelated changes. If it does, warn the user and avoid including unrelated files unless they explicitly ask for that.

## 5. Build release notes

1. Find the latest release tag.
2. Collect commits since that tag.
3. Exclude the version-bump commit from the release notes.
4. Group the notes into concise sections such as:
   - Features
   - Fixes
   - Docs
   - Maintenance
5. Keep the notes short and user-facing.
6. If there are no commits in a section, omit that section.

## 6. Create the GitHub release

Create a GitHub release with:
- Tag: `v<new_version>`
- Title: `v<new_version>`
- Target: `main`
- Notes: the release notes you prepared

This release triggers `.github/workflows/publish.yml`, which builds and uploads the package to PyPI.

## 7. Finish clearly

After the release is created:
- Return the GitHub release URL.
- Remind the user to watch the Actions tab and confirm the publish workflow succeeds.
- If release creation fails, report the failure briefly and include the next actionable step.

## Operational guidance

- Prefer reading the repository state directly instead of guessing.
- Use git tags and commit history to justify the version bump.
- Do not include the version-bump commit itself in the release notes.
- Do not change any files other than the version file unless the user asks.
- Keep responses short and action-oriented.

## Example prompts

- `Use the /publish skill to cut the next PyPI release for katsustats.`
- `Use the /publish skill and release version 0.3.0.`
- `Publish katsustats to PyPI.`
