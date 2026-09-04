# Reports

Three self-contained formats, from Python or the [CLI](cli.md):

```python
katsustats.reports.html(returns, benchmark=benchmark, title="My Strategy", output="report.html")
katsustats.reports.json(returns, benchmark=benchmark, title="My Strategy", output="report.json")
katsustats.reports.markdown(returns, benchmark=benchmark, title="My Strategy", output="report.md")

# Or get the string directly
html_str = katsustats.reports.html(returns, title="My Strategy")
```

| format | function | CLI | example |
|--------|----------|-----|---------|
| HTML | `reports.html()` | `katsustats report trades.csv -o report.html` | [btc_eth_report.html](../examples/reports/btc_eth_report.html) |
| JSON | `reports.json()` | `katsustats report trades.csv --format json -o report.json` | [btc_eth_report.json](../examples/reports/btc_eth_report.json) |
| Markdown | `reports.markdown()` | `katsustats report trades.csv --format markdown -o report.md` | [btc_eth_report.md](../examples/reports/btc_eth_report.md) |

All formats include headline metrics, performance and period-performance
tables, top drawdowns, and day-of-week statistics; a benchmark adds regime
analysis.

## Console + figures

`reports.full(returns, benchmark=..., rf=..., periods=..., show=...,
monte_carlo=...)` prints the metrics tables and returns a dict:

| key | type | description |
|-----|------|-------------|
| `summary` | `dict[str, float]` | Raw numeric summary values |
| `metrics` | `pl.DataFrame` | Summary metrics table |
| `drawdowns` | `pl.DataFrame` | Top-5 drawdown periods |
| `dow_stats` | `pl.DataFrame` | Day-of-week statistics |
| `figures` | `dict[str, Figure]` | All matplotlib figures (12 base, +2 with benchmark, +2 with Monte Carlo) |
| `monte_carlo` | `dict \| None` | Simulation summary when `monte_carlo=True` |

`reports.basic(...)` is the lightweight variant (7 core metrics, 3 charts).
`reports.metrics(returns, mode="basic" | "full")` returns just the table —
`"full"` appends the quantstats-parity metrics (expected return, CPC index,
outlier ratios, …).
