# CLI

Generate a report directly from a CSV or Parquet file — no script needed.

```bash
# HTML tearsheet (default)
katsustats report trades.csv -o report.html

# Structured JSON for AI agents / downstream tooling
katsustats report trades.csv --format json -o report.json

# Markdown summary for humans and agents
katsustats report trades.csv --format markdown -o report.md

# Custom column names, benchmark, and title
katsustats report trades.csv --date-col day --returns-col pnl \
  --benchmark benchmark.csv --title "My Strategy" -o report.html

# Compact single-image performance card
katsustats snapshot trades.csv --window 3M --title "My Strategy" -o snapshot.png
katsustats snapshot trades.csv --theme dark -o snapshot_dark.png
```

If `-o` is omitted the report is written alongside the input file (e.g.
`trades.html`, `trades.json`, `trades.md`). Run `katsustats report --help`
for all options (`--periods 365` for crypto data, `--rf`, `--monte-carlo`,
`--mc-sims`, `--mc-seed`, `--mc-method`).
