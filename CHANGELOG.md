# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Breaking changes
- **Column rename**: the required DataFrame column is now `"returns"` (was `"pnl"`).
  Update your DataFrames: `df.rename({"pnl": "returns"})`.
- **Parameter rename**: `reports.full()` and `reports.html()` now accept
  `returns` (was `pnl`) and `benchmark` (was `base_pnl`).

## [0.1.0] — 2024-01-01

### Added
- `katsustats.reports.full()` — full backtest report with metrics, drawdown analysis, and 8 charts
- `katsustats.reports.html()` — self-contained HTML report with embedded charts
- `katsustats.stats` — 40+ pure metric functions (Sharpe, Sortino, CAGR, Calmar, VaR, CVaR, tail ratio, common sense ratio, risk of ruin, drawdown details, rolling metrics, alpha/beta, regime stats, and more)
- `katsustats.plots` — 11 matplotlib chart functions (cumulative returns, drawdown, monthly heatmap, yearly/end-of-year bar charts, return distribution, rolling Sharpe/volatility, day-of-week analysis, benchmark overlay)
- Benchmark support via `base_pnl` parameter — adds Alpha, Beta, Correlation, Information Ratio, Excess Return to summary table
- Pandas input support — DataFrames are converted to Polars automatically
- `py.typed` marker for PEP 561 type-checking support
- Apache-2.0 license

[Unreleased]: https://github.com/katsu1110/katsustats/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/katsu1110/katsustats/releases/tag/v0.1.0
