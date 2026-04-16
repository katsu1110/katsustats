"""
katsustats.plots — Matplotlib chart functions for backtest visualization.

All plot functions accept Polars DataFrames with ["date", "pnl"] columns
and return matplotlib Figure objects for inline notebook display.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from . import stats

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_COLORS = {
    "strategy": "#2196F3",  # Material Blue
    "benchmark": "#FF9800",  # Material Orange
    "positive": "#4CAF50",  # Material Green
    "negative": "#F44336",  # Material Red
    "neutral": "#9E9E9E",  # Material Grey
    "fill": "#BBDEFB",  # Light Blue
    "grid": "#E0E0E0",
    "text": "#212121",
    "text_secondary": "#757575",
    "bg": "#FAFAFA",
}

_DOW_COLORS = ["#E53935", "#FB8C00", "#FDD835", "#43A047", "#1E88E5"]  # Mon-Fri


def _apply_style(ax, fig):
    """Apply a clean, modern style to axes."""
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["left"].set_color(_COLORS["grid"])
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_color(_COLORS["grid"])
    ax.grid(axis="y", color=_COLORS["grid"], linewidth=0.5, alpha=0.7)
    ax.tick_params(colors=_COLORS["text_secondary"], labelsize=9)


def _add_title(ax, fig, title: str, subtitle: str = ""):
    """Add title and subtitle."""
    fig.suptitle(title, fontweight="bold", fontsize=13, color=_COLORS["text"], y=0.97)
    if subtitle:
        ax.set_title(subtitle, fontsize=9, color=_COLORS["text_secondary"], pad=8)


def _pct_formatter(x, _):
    return f"{x:.0%}" if abs(x) < 1 else f"{x:.1%}"


def _empty_plot(message: str, figsize: tuple[float, float], title: str) -> Figure:
    """Return a blank figure with a centered message."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title, fontweight="bold", fontsize=13)
    fig.set_facecolor("white")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot: Cumulative Returns
# ---------------------------------------------------------------------------


def plot_cumulative_returns(
    df: pl.DataFrame,
    base_df: pl.DataFrame | None = None,
    figsize: tuple = (12, 5),
) -> Figure:
    """Cumulative return curve for strategy vs benchmark."""
    r = stats._to_returns(df)
    cumval = stats._cumulative(r)
    dates = df.get_column("date").to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    ax.fill_between(dates, 0, cumval.to_numpy(), alpha=0.15, color=_COLORS["strategy"])
    ax.plot(
        dates, cumval.to_numpy(), lw=1.8, color=_COLORS["strategy"], label="Strategy"
    )

    if base_df is not None:
        br = stats._to_returns(base_df)
        bcum = stats._cumulative(br)
        ax.plot(
            base_df.get_column("date").to_numpy(),
            bcum.to_numpy(),
            lw=1.4,
            color=_COLORS["benchmark"],
            label="Benchmark",
            alpha=0.85,
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.axhline(0, color=_COLORS["neutral"], lw=0.8, ls="--")
    ax.legend(fontsize=9, frameon=False)
    _add_title(ax, fig, "Cumulative Returns")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot: Drawdown (Underwater)
# ---------------------------------------------------------------------------


def plot_drawdown(df: pl.DataFrame, figsize: tuple = (12, 4)) -> Figure:
    """Underwater chart showing drawdown periods."""
    r = stats._to_returns(df)
    cumval = stats._cumulative_value(r)
    running_max = cumval.cum_max()
    dd = ((cumval - running_max) / running_max).to_numpy()
    dates = df.get_column("date").to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    ax.fill_between(dates, dd, 0, alpha=0.45, color=_COLORS["negative"])
    ax.plot(dates, dd, lw=0.8, color=_COLORS["negative"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.set_ylim(top=0)
    _add_title(ax, fig, "Drawdown")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot: Monthly Heatmap
# ---------------------------------------------------------------------------


def plot_monthly_heatmap(df: pl.DataFrame, figsize: tuple = (12, 5)) -> Figure:
    """Month × Year return heatmap."""
    monthly = (
        df.with_columns(
            pl.col("date").cast(pl.Date).dt.year().alias("year"),
            pl.col("date").cast(pl.Date).dt.month().alias("month"),
        )
        .group_by(["year", "month"])
        .agg(((pl.col("pnl") + 1).product() - 1).alias("ret"))
        .sort(["year", "month"])
    )

    years = sorted(monthly.get_column("year").unique().to_list())

    if not years:
        return _empty_plot("No data", figsize, "Monthly Returns")
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    # Build 2D array
    data = np.full((len(years), 12), np.nan)
    for row in monthly.iter_rows(named=True):
        yi = years.index(row["year"])
        mi = row["month"] - 1
        data[yi, mi] = row["ret"]

    fig, ax = plt.subplots(figsize=figsize)

    # Custom diverging colormap
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    if vmax == 0:
        vmax = 0.01
    im = ax.imshow(
        data,
        cmap="RdYlGn",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names, fontsize=9)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years], fontsize=9)

    # Annotate cells
    for i in range(len(years)):
        for j in range(12):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else _COLORS["text"]
                ax.text(
                    j,
                    i,
                    f"{val:.1%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    fontweight="bold",
                )

    ax.set_title(
        "Monthly Returns", fontweight="bold", fontsize=13, color=_COLORS["text"], pad=12
    )
    fig.colorbar(
        im, ax=ax, format=mticker.FuncFormatter(_pct_formatter), shrink=0.8, pad=0.02
    )
    fig.set_facecolor("white")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot: Group Cumulative PnL
# ---------------------------------------------------------------------------


def plot_group_pnl(
    df: pl.DataFrame,
    group_col: str = "group",
    figsize: tuple = (12, 5),
) -> Figure:
    """
    Line chart of cumulative PnL by group.

    Args:
        df: Polars DataFrame with ["date", "pnl", group_col] columns. Raw rows
            are aggregated by date/group before plotting, and missing
            date/group combinations are treated as zero PnL.
        group_col: Group column name (default "group").
        figsize: Figure size.

    Returns:
        matplotlib Figure containing cumulative group PnL lines.
    """
    assert "date" in df.columns, "df must have a 'date' column"
    assert "pnl" in df.columns, "df must have a 'pnl' column"
    assert group_col in df.columns, f"df must have a '{group_col}' column"

    grouped = (
        df.group_by(["date", group_col])
        .agg(pl.col("pnl").sum().alias("pnl"))
        .sort(["date", group_col])
    )

    group_names = grouped.get_column(group_col).unique().sort().to_list()
    if not group_names:
        return _empty_plot("No group data", figsize, "Group-level PnL")

    wide = (
        grouped.pivot(
            index="date", on=group_col, values="pnl", aggregate_function="sum"
        )
        .sort("date")
        .fill_null(0.0)
    )
    cum = wide.with_columns(pl.exclude("date").cum_sum())
    dates = cum.get_column("date").to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    cmap = plt.get_cmap("tab20")
    for idx, name in enumerate(group_names):
        column_name = str(name)
        ax.plot(
            dates,
            cum.get_column(column_name).to_numpy(),
            lw=1.6,
            label=column_name,
            color=cmap(idx % 20),
        )

    ax.axhline(0, color=_COLORS["neutral"], lw=0.8, ls="--")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.legend(fontsize=9, frameon=False, ncol=min(4, max(1, len(group_names))))
    _add_title(ax, fig, "Group-level PnL", "Cumulative sum by group")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot: Yearly Returns
# ---------------------------------------------------------------------------


def plot_yearly_returns(
    df: pl.DataFrame,
    base_df: pl.DataFrame | None = None,
    figsize: tuple = (12, 5),
) -> Figure:
    """Bar chart of annual returns."""

    def _yearly(d: pl.DataFrame) -> pl.DataFrame:
        return (
            d.with_columns(pl.col("date").cast(pl.Date).dt.year().alias("year"))
            .group_by("year")
            .agg(((pl.col("pnl") + 1).product() - 1).alias("ret"))
            .sort("year")
        )

    strat_y = _yearly(df)
    years = strat_y.get_column("year").to_numpy()
    strat_vals = strat_y.get_column("ret").to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    x = np.arange(len(years))
    width = 0.35 if base_df is not None else 0.6

    ax.bar(
        x, strat_vals, width, label="Strategy", color=_COLORS["strategy"], alpha=0.85
    )

    if base_df is not None:
        bench_y = _yearly(base_df)
        # Align by year
        bench_dict = dict(
            zip(
                bench_y.get_column("year").to_list(),
                bench_y.get_column("ret").to_list(),
            )
        )
        bench_vals = np.array([bench_dict.get(y, 0.0) for y in years])
        ax.bar(
            x + width,
            bench_vals,
            width,
            label="Benchmark",
            color=_COLORS["benchmark"],
            alpha=0.85,
        )

    ax.set_xticks(x + (width / 2 if base_df is not None else 0))
    ax.set_xticklabels([str(y) for y in years], fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.axhline(0, color=_COLORS["neutral"], lw=0.8, ls="--")
    ax.legend(fontsize=9, frameon=False)
    _add_title(ax, fig, "Yearly Returns")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot: Return Distribution
# ---------------------------------------------------------------------------


def plot_return_distribution(
    df: pl.DataFrame,
    base_df: pl.DataFrame | None = None,
    bins: int = 50,
    figsize: tuple = (12, 5),
) -> Figure:
    """Histogram of daily returns."""
    r = stats._to_returns(df).to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    ax.hist(
        r,
        bins=bins,
        alpha=0.7,
        color=_COLORS["strategy"],
        label="Strategy",
        edgecolor="white",
        linewidth=0.5,
        density=True,
    )

    if base_df is not None:
        br = stats._to_returns(base_df).to_numpy()
        ax.hist(
            br,
            bins=bins,
            alpha=0.5,
            color=_COLORS["benchmark"],
            label="Benchmark",
            edgecolor="white",
            linewidth=0.5,
            density=True,
        )

    ax.axvline(
        float(np.mean(r)),
        color=_COLORS["negative"],
        ls="--",
        lw=1.2,
        label=f"Mean ({np.mean(r):.4f})",
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.legend(fontsize=9, frameon=False)
    _add_title(ax, fig, "Daily Return Distribution")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot: Rolling Sharpe
# ---------------------------------------------------------------------------


def plot_rolling_sharpe(
    df: pl.DataFrame,
    base_df: pl.DataFrame | None = None,
    window: int = 126,
    figsize: tuple = (12, 4),
) -> Figure:
    """Rolling Sharpe ratio over a given window."""
    roll = stats.rolling_sharpe(df, window)
    dates = roll.get_column("date").to_numpy()
    vals = roll.get_column("rolling_sharpe").to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    ax.plot(dates, vals, lw=1.4, color=_COLORS["strategy"], label="Strategy")

    if base_df is not None:
        broll = stats.rolling_sharpe(base_df, window)
        ax.plot(
            broll.get_column("date").to_numpy(),
            broll.get_column("rolling_sharpe").to_numpy(),
            lw=1.2,
            color=_COLORS["benchmark"],
            label="Benchmark",
            alpha=0.8,
        )

    ax.axhline(0, color=_COLORS["neutral"], lw=0.8, ls="--")
    ax.legend(fontsize=9, frameon=False)
    _add_title(ax, fig, f"Rolling Sharpe ({window}d)")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot: Rolling Volatility
# ---------------------------------------------------------------------------


def plot_rolling_volatility(
    df: pl.DataFrame,
    base_df: pl.DataFrame | None = None,
    window: int = 126,
    figsize: tuple = (12, 4),
) -> Figure:
    """Rolling annualized volatility over a given window."""
    roll = stats.rolling_volatility(df, window)
    dates = roll.get_column("date").to_numpy()
    vals = roll.get_column("rolling_vol").to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    ax.fill_between(dates, 0, vals, alpha=0.2, color=_COLORS["strategy"])
    ax.plot(dates, vals, lw=1.4, color=_COLORS["strategy"], label="Strategy")

    if base_df is not None:
        broll = stats.rolling_volatility(base_df, window)
        ax.plot(
            broll.get_column("date").to_numpy(),
            broll.get_column("rolling_vol").to_numpy(),
            lw=1.2,
            color=_COLORS["benchmark"],
            label="Benchmark",
            alpha=0.8,
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.legend(fontsize=9, frameon=False)
    _add_title(ax, fig, f"Rolling Volatility ({window}d)")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot: Day-of-Week Returns (New Feature)
# ---------------------------------------------------------------------------


def plot_dow_returns(df: pl.DataFrame, figsize: tuple = (10, 5)) -> Figure:
    """Day-of-week bar chart showing mean return and win rate."""
    dow_df = stats.day_of_week_stats(df)
    # Filter to weekdays (1-5)
    dow_df = dow_df.filter(pl.col("dow") <= 5)

    names = dow_df.get_column("dow_name").to_list()
    mean_ret = dow_df.get_column("mean_return").to_numpy()
    wr = dow_df.get_column("win_rate").to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Mean Return ---
    _apply_style(ax1, fig)
    colors = [_COLORS["positive"] if v >= 0 else _COLORS["negative"] for v in mean_ret]
    bars1 = ax1.bar(
        names, mean_ret, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5
    )
    ax1.axhline(0, color=_COLORS["neutral"], lw=0.8, ls="--")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax1.set_title(
        "Mean Return by Day", fontweight="bold", fontsize=11, color=_COLORS["text"]
    )

    # Add value labels
    for bar, val in zip(bars1, mean_ret):
        y_offset = bar.get_height() * 0.1
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"{val:.3%}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8,
            color=_COLORS["text_secondary"],
        )

    # --- Win Rate ---
    _apply_style(ax2, fig)
    bars2 = ax2.bar(
        names,
        wr,
        color=_DOW_COLORS[: len(names)],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.axhline(0.5, color=_COLORS["neutral"], lw=0.8, ls="--")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax2.set_ylim(0, 1)
    ax2.set_title(
        "Win Rate by Day", fontweight="bold", fontsize=11, color=_COLORS["text"]
    )

    for bar, val in zip(bars2, wr):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=_COLORS["text_secondary"],
        )

    fig.set_facecolor("white")
    fig.suptitle(
        "Day-of-Week Analysis",
        fontweight="bold",
        fontsize=13,
        color=_COLORS["text"],
        y=1.02,
    )
    fig.tight_layout()
    return fig
