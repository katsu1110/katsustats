# Katsustats

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/katsustats.svg)](https://pypi.org/project/katsustats/)
[![Downloads](https://img.shields.io/pypi/dm/katsustats.svg)](https://pypi.org/project/katsustats/)
[![CI](https://github.com/katsu1110/katsustats/actions/workflows/ci.yml/badge.svg)](https://github.com/katsu1110/katsustats/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Sponsor](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/katsu1110)

Polars-powered backtest analytics and reporting, inspired by [quantstats](https://github.com/ranaroussi/quantstats).

Pass a DataFrame with `date` and `returns` — get 70+ metrics, 21 charts, and a self-contained HTML report.

![HTML Report Preview](https://raw.githubusercontent.com/katsu1110/katsustats/main/img/html_report.png)

[**View a full BTC vs ETH backtest report**](https://htmlpreview.github.io/?https://github.com/katsu1110/katsustats/blob/main/examples/reports/btc_eth_report.html)

## Quickstart

```bash
pip install katsustats yfinance
```

```python
from katsustats import reports, utils

returns = utils.download_returns("AAPL")  # ["date", "returns"]
reports.html(returns, output="report.html")
```

Pandas, quantstats-style `DatetimeIndex` inputs, and price frames (via
`utils.to_returns`) all work — see [Data format](docs/data-format.md).
Runnable examples: [`examples/quickstart.py`](examples/quickstart.py),
[`examples/with_benchmark.py`](examples/with_benchmark.py).

## Install

```bash
pip install katsustats
```

<details>
<summary>Other install paths</summary>

```bash
uv add katsustats            # uv project
pipx install katsustats      # CLI-only use
uv tool install katsustats   # CLI-only via uv
```

Pre-built binaries (no Python needed) are on the
[Releases page](https://github.com/katsu1110/katsustats/releases):

```bash
chmod +x katsustats-linux-x86_64
./katsustats-linux-x86_64 report trades.csv -o report.html
```

</details>

Try it online: [Google Colab](https://colab.research.google.com/drive/1PnbZdvZboEtV8F8gjrF3oTrQ3IzC5CdT?usp=sharing) ·
[Kaggle](https://www.kaggle.com/code/code1110/katsustats-quickstart)

## Features

- **Reports** — self-contained offline HTML, agent-friendly JSON, readable Markdown ([docs](docs/reports.md))
- **Benchmarks** — Alpha, Beta, capture ratios, regime analysis with one argument
- **Monte Carlo** — bootstrap/shuffle luck analysis with bust/goal probabilities ([docs](docs/monte-carlo.md))
- **Snapshots** — dark/light single-image performance cards for sharing ([docs](docs/snapshot.md))
- **CLI + binary** — CSV/Parquet to report with no script ([docs](docs/cli.md))
- **70+ metrics, 21 charts** — functional `stats` / `plots` API ([docs](docs/metrics.md))

```bash
katsustats report trades.csv -o report.html
katsustats report trades.csv --format json -o report.json
katsustats snapshot trades.csv --window 3M --theme dark -o snapshot.png
```

## katsustats vs quantstats

| | katsustats | quantstats |
|---|---|---|
| Engine | Polars-first (pandas accepted) | pandas-only |
| No-script CLI / binary | ✅ | ❌ |
| JSON + Markdown reports (agent-friendly) | ✅ | HTML only |
| Snapshot cards (dark/light) | ✅ | light only |
| Monte Carlo methods | bootstrap + shuffle | bootstrap |
| Rolling greeks, outlier/CPC/RAR metrics | ✅ | partial |

## Docs

- [Data format](docs/data-format.md) · [Reports](docs/reports.md) · [CLI](docs/cli.md)
- [Metrics](docs/metrics.md) · [Monte Carlo](docs/monte-carlo.md) · [Snapshots](docs/snapshot.md)
