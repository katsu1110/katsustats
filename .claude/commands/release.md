# /release — Publish a new katsustats release to PyPI

Cut a new release for this project. Follow every step in order.

## Step 1 — Determine the new version

Read the current version from `src/katsustats/__init__.py` (`__version__`).
Decide the bump level based on the commits since the last release tag:
- **patch** (`x.y.Z`): bug fixes only, no new public API
- **minor** (`x.Y.0`): new public functions/classes, new features, backward-compatible
- **major** (`X.0.0`): breaking changes to the public API

If the user specified a version (e.g. `/release 0.3.0`), use that instead.

## Step 2 — Confirm with the user

State the current version, the proposed new version, and the bump rationale in one sentence.
Then ask the user to confirm before proceeding. Wait for confirmation.

## Step 3 — Bump the version

Edit `src/katsustats/__init__.py`: update `__version__ = "x.y.z"` to the new version.

## Step 4 — Commit and push

```
git add src/katsustats/__init__.py
git commit -m "chore: bump version to <new_version>"
git push origin main
```

## Step 5 — Build release notes

Collect commits since the last release tag:
```
git log <last_tag>..HEAD --oneline
```

Group them into sections (Features, Fixes, etc.) and write concise release notes.
Do not include the version-bump commit itself.

## Step 6 — Create the GitHub Release

```
gh release create v<new_version> \
  --title "v<new_version>" \
  --notes "<release notes>" \
  --target main
```

This triggers the publish workflow (`.github/workflows/publish.yml`) which builds
and uploads the package to PyPI automatically.

## Step 7 — Confirm

Print the GitHub release URL and remind the user to watch the Actions tab to confirm
the PyPI publish workflow completes successfully.
