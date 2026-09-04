# Metrics

| metric | description |
|--------|-------------|
| Total Return | Compounded return over the full period |
| CAGR | Compound Annual Growth Rate |
| Sharpe Ratio | Annualized risk-adjusted return |
| Sortino Ratio | Sharpe using only downside deviation |
| Max Drawdown | Largest peak-to-trough decline |
| Calmar Ratio | CAGR / \|Max Drawdown\| |
| Volatility (ann.) | Annualized standard deviation |
| Win Rate | % of days with positive returns |
| Profit Factor | Gross profit / gross loss |
| Best / Worst Day | Largest single-day gain / loss |
| Avg Win / Avg Loss | Mean return on winning / losing days |
| Daily VaR (95%) | 5th-percentile daily return |
| CVaR (95%) | Mean return in the worst 5% tail |
| Recovery Factor | Total return / \|Max Drawdown\| |
| Skewness / Kurtosis | Distribution shape statistics |
| Best / Worst Month | Largest / smallest monthly return |
| Best / Worst Year | Largest / smallest yearly return |
| Positive Months / Years | Share of profitable months / years |
| Omega / Martin / Gain-to-Pain | Tail- and pain-aware ratios |
| Ulcer Index | Root-mean-square of running drawdowns |
| Kelly Criterion | Half-Kelly allocation fraction |
| Probabilistic Sharpe | Bailey–López de Prado significance |
| Payoff Ratio | Average win / \|average loss\| |

With a benchmark: **Alpha**, **Beta**, **Correlation**,
**Information Ratio**, **Excess Return**, **Treynor Ratio**, **R-Squared**,
**Up/Down Capture**.

Quantstats-parity extras (`reports.metrics(mode="full")`): **Expected Return**
/ Geometric Mean / GHPR, **Adjusted Sortino**, **Risk-Return Ratio**, **RAR**,
**CPC Index**, **Outlier Win/Loss Ratios**, **Profit Ratio**, **Win-Loss
Ratio**, **Implied Volatility**, plus rolling greeks and drawdown-series
helpers in `katsustats.stats`.

`katsustats.stats` exposes 70+ functions and `katsustats.plots` exposes 21
chart functions — all documented in their docstrings. See
[`examples/`](../examples/) for runnable usage.
