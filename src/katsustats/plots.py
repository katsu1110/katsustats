"""
katsustats.plots — Matplotlib chart functions for backtest visualization.

All plot functions accept Polars or pandas DataFrames with ["date", "returns"]
columns and return matplotlib Figure objects for inline notebook display.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from . import stats
from ._constants import COL_DATE, COL_RETURNS
from ._dataframe import DataFrameLike, ensure_polars

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_COLORS = {
    "strategy": "#2563eb",  # matches HTML report accent blue
    "benchmark": "#f97316",  # muted orange, easier on the eye than Material Orange
    "positive": "#4CAF50",  # Material Green
    "negative": "#F44336",  # Material Red
    "neutral": "#9E9E9E",  # Material Grey
    "fill": "#dbeafe",  # soft blue tint matching strategy colour
    "grid": "#E0E0E0",
    "text": "#212121",
    "text_secondary": "#757575",
    "bg": "#FAFAFA",
}

_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "katsustats_diverging",
    [_COLORS["negative"], "#ffffff", _COLORS["positive"]],
)

# Prefer system sans-serif fonts (Helvetica/Arial) over matplotlib's DejaVu Sans
# so chart typography matches the HTML report's sans-serif stack.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Helvetica",
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
]


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
    a = abs(x)
    if a >= 0.1:
        return f"{x:.0%}"
    if a >= 0.01:
        return f"{x:.1%}"
    return f"{x:.2%}"


def _align_to_common_dates(
    df: pl.DataFrame, base_df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Inner-join df and base_df on date, returning aligned (df, base_df) pair.

    The sort ensures chronological order since joins do not preserve row order.
    """
    joined = df.join(
        base_df.rename({COL_RETURNS: "_base_returns"}), on=COL_DATE, how="inner"
    ).sort(COL_DATE)
    return joined.select(["date", "returns"]), joined.select(
        [pl.col(COL_DATE), pl.col("_base_returns").alias(COL_RETURNS)]
    )


def _returns_by_day_of_week(df: pl.DataFrame, dow_order: list[int]) -> list[np.ndarray]:
    """Return daily return arrays ordered by ISO weekday number."""
    dow_df = df.with_columns(pl.col(COL_DATE).cast(pl.Date).dt.weekday().alias("dow"))
    return [
        dow_df.filter(pl.col("dow") == dow).get_column(COL_RETURNS).to_numpy()
        for dow in dow_order
    ]


def _color_boxplot_by_median(bp: dict) -> None:
    """Color boxplot patches green/red based on the median line."""
    for patch, median_line in zip(bp["boxes"], bp["medians"]):
        median = median_line.get_ydata()[0]
        patch.set_facecolor(_COLORS["positive"] if median >= 0 else _COLORS["negative"])
        patch.set_alpha(0.7)


# ---------------------------------------------------------------------------
# Plot: Cumulative Returns
# ---------------------------------------------------------------------------


def plot_cumulative_returns(
    df: DataFrameLike,
    base_df: DataFrameLike | None = None,
    figsize: tuple = (12, 5),
) -> Figure:
    """Cumulative return curve for strategy vs benchmark."""
    df = ensure_polars(df)
    if base_df is not None:
        base_df = ensure_polars(base_df, name="base_df")
        df, base_df = _align_to_common_dates(df, base_df)
    r = stats._to_returns(df)
    cumval = stats._cumulative(r)
    dates = df.get_column(COL_DATE).to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    ax.plot(
        dates, cumval.to_numpy(), lw=1.8, color=_COLORS["strategy"], label="Strategy"
    )

    if base_df is not None:
        br = stats._to_returns(base_df)
        bcum = stats._cumulative(br)
        ax.plot(
            base_df.get_column(COL_DATE).to_numpy(),
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


def plot_drawdown(df: DataFrameLike, figsize: tuple = (12, 4)) -> Figure:
    """Underwater chart showing drawdown periods."""
    df = ensure_polars(df)
    r = stats._to_returns(df)
    cumval = stats._cumulative_value(r)
    running_max = cumval.cum_max().clip(lower_bound=1.0)
    dd = ((cumval - running_max) / running_max).to_numpy()
    dates = df.get_column(COL_DATE).to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    ax.fill_between(
        dates,
        dd,
        0,
        where=dd < 0,
        color=_COLORS["negative"],
        alpha=0.2,
        interpolate=True,
    )
    ax.plot(dates, dd, lw=0.8, color=_COLORS["negative"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.set_ylim(top=0)
    _add_title(ax, fig, "Drawdown")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot: Drawdown Periods
# ---------------------------------------------------------------------------


def plot_drawdown_periods(
    df: DataFrameLike, top_n: int = 5, figsize: tuple = (12, 4)
) -> Figure:
    """Cumulative return chart with top-N drawdown periods shaded."""
    df = ensure_polars(df)
    r = stats._to_returns(df)
    cumval = stats._cumulative_value(r).to_numpy()
    dates = df.get_column(COL_DATE).to_numpy()
    dd_details = stats.drawdown_details(df, top_n=top_n)

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    ax.plot(dates, cumval, lw=1.2, color=_COLORS["strategy"])

    for row in dd_details.iter_rows(named=True):
        start = row["start"]
        recovery = row["recovery"]
        end = recovery if recovery is not None else dates[-1]
        ax.axvspan(start, end, alpha=0.2, color=_COLORS["negative"], lw=0)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    _add_title(ax, fig, f"Worst {top_n} Drawdown Periods")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot: Monthly Heatmap
# ---------------------------------------------------------------------------


def plot_monthly_heatmap(df: DataFrameLike, figsize: tuple = (12, 5)) -> Figure:
    """Month × Year return heatmap."""
    df = ensure_polars(df)
    monthly = (
        df.with_columns(
            pl.col(COL_DATE).cast(pl.Date).dt.year().alias("year"),
            pl.col(COL_DATE).cast(pl.Date).dt.month().alias("month"),
        )
        .group_by(["year", "month"])
        .agg(((pl.col(COL_RETURNS) + 1).product() - 1).alias("ret"))
        .sort(["year", "month"])
    )

    years = sorted(monthly.get_column("year").unique().to_list())

    if not years:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Monthly Returns", fontweight="bold", fontsize=13)
        fig.set_facecolor("white")
        fig.tight_layout()
        return fig
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
        cmap=_HEATMAP_CMAP,
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
                color = "white" if abs(val) > vmax * 0.5 else _COLORS["text"]
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
# Plot: Yearly Returns
# ---------------------------------------------------------------------------


def plot_yearly_returns(
    df: DataFrameLike,
    base_df: DataFrameLike | None = None,
    figsize: tuple = (12, 5),
) -> Figure:
    """Bar chart of annual returns."""
    df = ensure_polars(df)
    if base_df is not None:
        base_df = ensure_polars(base_df, name="base_df")

    def _yearly(d: pl.DataFrame) -> pl.DataFrame:
        return (
            d.with_columns(pl.col(COL_DATE).cast(pl.Date).dt.year().alias("year"))
            .group_by("year")
            .agg(((pl.col(COL_RETURNS) + 1).product() - 1).alias("ret"))
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
        bench_vals = np.array([bench_dict.get(y, float("nan")) for y in years])
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
# Plot: EOY (End-of-Year) Compounded Returns
# ---------------------------------------------------------------------------


def plot_eoy_returns(
    df: DataFrameLike,
    base_df: DataFrameLike | None = None,
    figsize: tuple = (12, 5),
) -> Figure:
    """Bar chart of compounded end-of-year returns."""
    df = ensure_polars(df)
    if base_df is not None:
        base_df = ensure_polars(base_df, name="base_df")

    def _eoy(d: pl.DataFrame) -> pl.DataFrame:
        return (
            d.with_columns(pl.col(COL_DATE).cast(pl.Date).dt.year().alias("year"))
            .group_by("year")
            .agg(((pl.col(COL_RETURNS) + 1).product() - 1).alias("ret"))
            .sort("year")
        )

    strat_y = _eoy(df)

    if base_df is not None:
        bench_y = _eoy(base_df)
        # Align on common years via inner join
        aligned_y = strat_y.join(
            bench_y.rename({"ret": "bench_ret"}), on="year", how="inner"
        ).sort("year")
        strat_y = aligned_y.select(["year", "ret"])
        bench_y = aligned_y.select(["year", "bench_ret"]).rename({"bench_ret": "ret"})

    years = strat_y.get_column("year").to_numpy()
    strat_vals = strat_y.get_column("ret").to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    x = np.arange(len(years))
    width = 0.35 if base_df is not None else 0.6

    strat_colors = [
        _COLORS["positive"] if v >= 0 else _COLORS["negative"] for v in strat_vals
    ]
    ax.bar(x, strat_vals, width, label="Strategy", color=strat_colors, alpha=0.85)

    if base_df is not None:
        bench_vals = bench_y.get_column("ret").to_numpy()
        bench_colors = [
            _COLORS["positive"] if v >= 0 else _COLORS["negative"] for v in bench_vals
        ]
        ax.bar(
            x + width,
            bench_vals,
            width,
            label="Benchmark",
            color=bench_colors,
            alpha=0.6,
        )

    ax.set_xticks(x + (width / 2 if base_df is not None else 0))
    ax.set_xticklabels([str(y) for y in years], fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.axhline(0, color=_COLORS["neutral"], lw=1.2, ls="-")
    ax.legend(fontsize=9, frameon=False)
    _add_title(ax, fig, "EOY Returns")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot: Return Distribution
# ---------------------------------------------------------------------------


def plot_return_distribution(
    df: DataFrameLike,
    base_df: DataFrameLike | None = None,
    bins: int = 50,
    figsize: tuple = (12, 5),
) -> Figure:
    """Histogram of daily returns."""
    df = ensure_polars(df)
    if base_df is not None:
        base_df = ensure_polars(base_df, name="base_df")
        df, base_df = _align_to_common_dates(df, base_df)
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
    df: DataFrameLike,
    base_df: DataFrameLike | None = None,
    window: int = 126,
    figsize: tuple = (12, 4),
) -> Figure:
    """Rolling Sharpe ratio over a given window."""
    df = ensure_polars(df)
    if base_df is not None:
        base_df = ensure_polars(base_df, name="base_df")
        df, base_df = _align_to_common_dates(df, base_df)
    roll = stats.rolling_sharpe(df, window)
    dates = roll.get_column(COL_DATE).to_numpy()
    vals = roll.get_column("rolling_sharpe").to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    ax.plot(dates, vals, lw=1.4, color=_COLORS["strategy"], label="Strategy")

    if base_df is not None:
        broll = stats.rolling_sharpe(base_df, window)
        ax.plot(
            broll.get_column(COL_DATE).to_numpy(),
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
    df: DataFrameLike,
    base_df: DataFrameLike | None = None,
    window: int = 126,
    figsize: tuple = (12, 4),
) -> Figure:
    """Rolling annualized volatility over a given window."""
    df = ensure_polars(df)
    if base_df is not None:
        base_df = ensure_polars(base_df, name="base_df")
        df, base_df = _align_to_common_dates(df, base_df)
    roll = stats.rolling_volatility(df, window)
    dates = roll.get_column(COL_DATE).to_numpy()
    vals = roll.get_column("rolling_vol").to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    ax.plot(dates, vals, lw=1.4, color=_COLORS["strategy"], label="Strategy")

    if base_df is not None:
        broll = stats.rolling_volatility(base_df, window)
        ax.plot(
            broll.get_column(COL_DATE).to_numpy(),
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
# Plot: Returns vs Benchmark (Scatter + Regression)
# ---------------------------------------------------------------------------


def plot_returns_vs_benchmark(
    df: DataFrameLike,
    base_df: DataFrameLike,
    figsize: tuple = (7, 7),
) -> Figure:
    """Scatter plot of strategy daily returns (y) vs benchmark daily returns (x) with regression line."""
    df = ensure_polars(df)
    base_df = ensure_polars(base_df, name="base_df")
    df, base_df = _align_to_common_dates(df, base_df)

    r = stats._to_returns(df).to_numpy()
    b = stats._to_returns(base_df).to_numpy()

    if len(b) < 2 or len(r) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(ax, fig)
        ax.text(
            0.5,
            0.5,
            "No overlapping dates",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color=_COLORS["text_secondary"],
        )
        ax.set_xticks([])
        ax.set_yticks([])
        _add_title(ax, fig, "Returns vs Benchmark")
        fig.tight_layout()
        return fig
    # OLS regression: y = beta*x + alpha_daily
    beta, alpha_daily = np.polyfit(b, r, 1)
    beta = float(beta)
    alpha_daily = float(alpha_daily)

    # R²
    mean_r = float(np.mean(r))
    r_pred = beta * b + alpha_daily
    ss_res = float(np.sum((r - r_pred) ** 2))
    ss_tot = float(np.sum((r - mean_r) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    # Scatter points
    ax.scatter(b, r, color=_COLORS["neutral"], alpha=0.6, s=18, linewidths=0)

    # Regression line
    x_line = np.array([b.min(), b.max()])
    ax.plot(x_line, beta * x_line + alpha_daily, color=_COLORS["strategy"], lw=1.8)

    # Zero reference lines
    ax.axhline(0, color=_COLORS["neutral"], lw=0.8, ls="--")
    ax.axvline(0, color=_COLORS["neutral"], lw=0.8, ls="--")

    # Annotation
    sign = "+" if alpha_daily >= 0 else ""
    annotation = (
        f"α = {sign}{alpha_daily:.4%}/day\nβ = {beta:.2f}\nR² = {r_squared:.2f}"
    )
    ax.text(
        0.05,
        0.95,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=_COLORS["text"],
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.7,
            "edgecolor": _COLORS["grid"],
        },
    )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.set_xlabel("Benchmark Return", fontsize=10, color=_COLORS["text_secondary"])
    ax.set_ylabel("Strategy Return", fontsize=10, color=_COLORS["text_secondary"])
    _add_title(ax, fig, "Returns vs Benchmark")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot: Day-of-Week Returns (New Feature)
# ---------------------------------------------------------------------------


def plot_dow_returns(df: DataFrameLike, figsize: tuple = (10, 5)) -> Figure:
    """Day-of-week return distribution (box plot), win rate, and total return overlay."""
    df = ensure_polars(df)
    dow_df = stats.day_of_week_stats(df)

    names = dow_df.get_column("dow_name").to_list()
    dow_order = dow_df.get_column("dow").to_list()
    box_data = _returns_by_day_of_week(df, dow_order)
    win_rates = dow_df.get_column("win_rate").to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Return Distribution ---
    _apply_style(ax1, fig)
    bp = ax1.boxplot(
        box_data,
        tick_labels=names,
        patch_artist=True,
        boxprops=dict(linewidth=0.8),
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=_COLORS["neutral"], linewidth=0.8),
        capprops=dict(color=_COLORS["neutral"], linewidth=0.8),
        flierprops=dict(
            marker="o",
            markerfacecolor=_COLORS["neutral"],
            markersize=3,
            linestyle="none",
            alpha=0.5,
        ),
    )
    _color_boxplot_by_median(bp)
    ax1.axhline(0, color=_COLORS["neutral"], lw=0.8, ls="--")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax1.set_title(
        "Return Distribution by Day",
        fontweight="bold",
        fontsize=11,
        color=_COLORS["text"],
    )

    _apply_style(ax2, fig)
    bars2 = ax2.bar(
        names,
        win_rates,
        color=_COLORS["strategy"],
        alpha=0.65,
        edgecolor="white",
        linewidth=0.5,
        label="Win Rate",
    )
    ax2.axhline(0.5, color=_COLORS["neutral"], lw=0.8, ls="--")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax2.set_ylim(0, 1.1)
    ax2.set_title(
        "Win Rate & Total Return by Day",
        fontweight="bold",
        fontsize=11,
        color=_COLORS["text"],
    )

    for bar, val in zip(bars2, win_rates):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=_COLORS["text_secondary"],
        )

    total_returns = dow_df.get_column("total_return").to_numpy()
    ax2r = ax2.twinx()
    ax2r.plot(
        names,
        total_returns,
        color=_COLORS["positive"],
        marker="o",
        markersize=5,
        linewidth=1.8,
        label="Total Return",
        zorder=3,
    )
    ax2r.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax2r.tick_params(axis="y", labelcolor=_COLORS["positive"], labelsize=8)
    ax2r.set_ylabel("Total Return", color=_COLORS["positive"], fontsize=9)
    ax2r.spines["right"].set_color(_COLORS["positive"])
    ax2r.spines["right"].set_linewidth(0.8)
    for spine in ("top", "left", "bottom"):
        ax2r.spines[spine].set_visible(False)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        fontsize=7,
        framealpha=0.7,
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


# ---------------------------------------------------------------------------
# Plot: Monte Carlo Projection
# ---------------------------------------------------------------------------


def plot_monte_carlo(
    df: DataFrameLike,
    sims: int = 1000,
    seed: int | None = None,
    confidence_level: float = 0.95,
    figsize: tuple = (12, 5),
    _paths_df: pl.DataFrame | None = None,
    method: str = "bootstrap",
) -> Figure:
    """Fan chart of Monte Carlo simulated paths with a confidence band.

    Plots up to 200 individual paths as faint lines, a filled confidence
    band, the median path, and the original (unshuffled) path.
    """
    paths_df = (
        _paths_df
        if _paths_df is not None
        else stats.monte_carlo_paths(df, sims=sims, seed=seed, method=method)
    )
    steps = paths_df.get_column("step").to_numpy()
    sim_cols = [c for c in paths_df.columns if c.startswith("sim_")]
    paths_matrix = paths_df.select(sim_cols).to_numpy()  # (n_steps, sims)

    alpha = (1.0 - confidence_level) / 2.0
    lower = np.percentile(paths_matrix, alpha * 100, axis=1)
    upper = np.percentile(paths_matrix, (1.0 - alpha) * 100, axis=1)
    median = np.percentile(paths_matrix, 50, axis=1)
    original = paths_matrix[:, 0]

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    # Individual paths — subsample to keep the chart readable (cap at 200 lines)
    n_sims = paths_matrix.shape[1]
    stride = max(1, (n_sims + 199) // 200)
    subsampled = paths_matrix[:, ::stride]
    for i in range(subsampled.shape[1]):
        ax.plot(
            steps,
            subsampled[:, i],
            lw=0.3,
            color=_COLORS["neutral"],
            alpha=0.25,
        )

    ax.fill_between(
        steps,
        lower,
        upper,
        color=_COLORS["strategy"],
        alpha=0.15,
        label=f"{confidence_level:.0%} CI",
    )
    ax.plot(steps, median, lw=1.8, color=_COLORS["strategy"], label="Median")
    ax.plot(
        steps,
        original,
        lw=1.6,
        color=_COLORS["benchmark"],
        ls="--",
        label="Original",
    )
    ax.axhline(0, color=_COLORS["neutral"], lw=0.8, ls="--")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.legend(fontsize=9, frameon=False)
    _add_title(ax, fig, f"Monte Carlo Projection ({sims:,} sims)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot: Monte Carlo Max Drawdown Distribution
# ---------------------------------------------------------------------------


def plot_monte_carlo_distribution(
    df: DataFrameLike,
    sims: int = 1000,
    seed: int | None = None,
    bins: int = 50,
    figsize: tuple = (12, 5),
    _paths_df: pl.DataFrame | None = None,
    method: str = "bootstrap",
) -> Figure:
    """Histogram of max drawdowns across Monte Carlo simulation paths."""
    paths_df = (
        _paths_df
        if _paths_df is not None
        else stats.monte_carlo_paths(df, sims=sims, seed=seed, method=method)
    )
    sim_cols = [c for c in paths_df.columns if c.startswith("sim_")]
    cum_paths = paths_df.select(sim_cols).to_numpy()
    max_drawdowns = stats._sim_max_drawdowns(cum_paths)

    original_mdd = float(max_drawdowns[0])
    p5, p50, p95 = (float(v) for v in np.percentile(max_drawdowns, [5, 50, 95]))

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(ax, fig)

    ax.hist(
        max_drawdowns,
        bins=bins,
        color=_COLORS["negative"],
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(
        p5,
        color=_COLORS["negative"],
        lw=1.4,
        ls="--",
        label=f"5th pct ({p5:.1%})",
    )
    ax.axvline(p50, color=_COLORS["strategy"], lw=1.6, label=f"Median ({p50:.1%})")
    ax.axvline(
        p95,
        color=_COLORS["positive"],
        lw=1.4,
        ls="--",
        label=f"95th pct ({p95:.1%})",
    )
    ax.axvline(
        original_mdd,
        color=_COLORS["benchmark"],
        lw=1.6,
        ls="-.",
        label=f"Original ({original_mdd:.1%})",
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.legend(fontsize=9, frameon=False)
    _add_title(ax, fig, f"Max Drawdown Distribution ({sims:,} sims)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

_WINDOW_MAP: dict[str, int] = {
    "1W": 5,
    "2W": 10,
    "1M": 21,
    "3M": 63,
}


def _parse_window(window: str | int) -> int:
    """Convert a window spec to a trailing row count."""
    if isinstance(window, int):
        if window < 1:
            raise ValueError("window must be >= 1")
        return window
    if isinstance(window, str):
        upper = window.upper()
        if upper in _WINDOW_MAP:
            return _WINDOW_MAP[upper]
        try:
            n = int(upper)
        except ValueError:
            raise ValueError(
                f"Unrecognised window {window!r}. "
                f"Use one of {list(_WINDOW_MAP)} or an integer."
            )
        if n < 1:
            raise ValueError("window must be >= 1")
        return n
    raise TypeError(f"window must be str or int, got {type(window).__name__}")


def _draw_metric_card(
    ax,
    value_str: str,
    label: str,
    bg_color: str,
    text_color: str = "white",
    shadow: bool = False,
) -> None:
    """Draw a metric tile: colored background, bold value, small label."""
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    if shadow:
        shadow_rect = FancyBboxPatch(
            (0.02, -0.05),
            1,
            1,
            boxstyle="round,pad=0,rounding_size=0.20",
            ec="none",
            fc="black",
            alpha=0.3,
            transform=ax.transAxes,
            zorder=0,
        )
        ax.add_patch(shadow_rect)

    rect = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0,rounding_size=0.20",
        ec="none",
        fc=bg_color,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(rect)

    ax.text(
        0.5,
        0.62,
        value_str,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=text_color,
        zorder=2,
    )
    ax.text(
        0.5,
        0.24,
        label.upper(),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=text_color,
        alpha=0.90,
        zorder=2,
    )


# ---------------------------------------------------------------------------
# Plot: Snapshot (compact performance card)
# ---------------------------------------------------------------------------


def plot_snapshot(
    df: DataFrameLike,
    window: str | int = "1W",
    title: str = "Strategy",
    figsize: tuple = (10, 8),
    theme: str = "light",
) -> Figure:
    """Compact performance card: equity curve, underwater drawdown, and 4 metric tiles for the given window."""
    df = ensure_polars(df)
    n_rows = _parse_window(window)
    df_window = df.tail(n_rows)
    if df_window.height == 0:
        raise ValueError(
            "input DataFrame is empty; plot_snapshot requires at least 1 row"
        )

    ret_val = stats.total_return(df_window)
    mdd_val = stats.max_drawdown(df_window)
    wr_val = stats.win_rate(df_window)
    sharpe_val = stats.sharpe(df_window)

    ret_str = f"{ret_val:.2%}"
    sharpe_str = "—" if df_window.height < 2 else f"{sharpe_val:.2f}"
    mdd_str = f"{mdd_val:.2%}"
    wr_str = f"{wr_val:.2%}"

    is_dark = theme.lower() == "dark"
    bg_fig = "#0B0F19" if is_dark else "white"
    bg_ax = "#0B0F19" if is_dark else "white"

    fig = plt.figure(figsize=figsize, facecolor=bg_fig)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 2.0, 1.1], hspace=0.25, wspace=0.10)
    ax_cards = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_curve = fig.add_subplot(gs[1, :])
    ax_dd = fig.add_subplot(gs[2, :], sharex=ax_curve)

    # Theme colors
    c_pos = "#00FFA3" if is_dark else "#10B981"  # Emerald/Neon Green
    c_neg = "#FF3366" if is_dark else "#EF4444"  # Red/Neon Red
    c_line = "white" if is_dark else "#374151"
    c_grid = "#1F2937" if is_dark else "#F3F4F6"
    c_border = "#374151" if is_dark else "#E5E7EB"
    c_neutral_line = "#4B5563" if is_dark else "#9CA3AF"
    c_label = "#9CA3AF" if is_dark else "#4B5563"
    c_title = "white" if is_dark else "#111827"
    text_sec = "#9CA3AF" if is_dark else _COLORS["text_secondary"]

    # Card colors
    if is_dark:
        c_pos_card, text_pos = "#064E3B", c_pos  # Dark green bg, neon green text
        c_neg_card, text_neg = "#7F1D1D", c_neg  # Dark red bg, neon red text
        c_neu_card, text_neu = "#1F2937", "white"
    else:
        c_pos_card, text_pos = "#10B981", "white"
        c_neg_card, text_neg = "#EF4444", "white"
        c_neu_card, text_neu = "#6B7280", "white"

    def _card_colors_sharpe(v: float | None) -> tuple[str, str]:
        if v is None or (isinstance(v, float) and v != v):
            return c_neu_card, text_neu
        if v > 1.0:
            return c_pos_card, text_pos
        if v < 0.5:
            return c_neg_card, text_neg
        return c_neu_card, text_neu

    def _card_colors_mdd(v: float) -> tuple[str, str]:
        if v > -0.10:
            return c_pos_card, text_pos
        if v < -0.20:
            return c_neg_card, text_neg
        return c_neu_card, text_neu

    def _card_colors_wr(v: float) -> tuple[str, str]:
        if v > 0.55:
            return c_pos_card, text_pos
        if v < 0.45:
            return c_neg_card, text_neg
        return c_neu_card, text_neu

    def _card_colors_ret(v: float) -> tuple[str, str]:
        if v >= 0:
            return c_pos_card, text_pos
        return c_neg_card, text_neg

    card_specs = [
        (ret_str, "Return", *_card_colors_ret(ret_val)),
        (sharpe_str, "Sharpe", *_card_colors_sharpe(sharpe_val)),
        (mdd_str, "Max DD", *_card_colors_mdd(mdd_val)),
        (wr_str, "Win Rate", *_card_colors_wr(wr_val)),
    ]
    for ax, (val_s, lbl, bg, txt) in zip(ax_cards, card_specs):
        _draw_metric_card(ax, val_s, lbl, bg, text_color=txt, shadow=is_dark)

    r = stats._to_returns(df_window)
    dates_raw = df_window.get_column("date").to_numpy()

    # Cumulative return (fraction)
    cumret_raw = stats._cumulative(r).to_numpy()
    cumret = np.concatenate([[0.0], cumret_raw])
    dates = np.concatenate([[dates_raw[0] - np.timedelta64(1, "D")], dates_raw])

    # Drawdown
    cumval = stats._cumulative_value(r).to_numpy()
    running_max = np.maximum.accumulate(cumval)
    cumval_full = np.concatenate([[1.0], cumval])
    running_max_full = np.concatenate([[1.0], running_max])
    dd_full = (cumval_full / running_max_full) - 1.0

    for ax in (ax_curve, ax_dd):
        ax.set_facecolor(bg_ax)
        ax.grid(color=c_grid, linewidth=1.0, axis="y", zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(c_border)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.tick_params(colors=text_sec, labelsize=9)

    ax_curve.tick_params(labelbottom=False)

    if n_rows <= 126:
        r_vals = r.to_numpy()
        bar_colors = [c_pos if v >= 0 else c_neg for v in r_vals]
        ax_curve.bar(
            dates_raw,
            r_vals,
            color=bar_colors,
            alpha=0.8 if is_dark else 0.6,
            width=0.6,
            edgecolor="none",
            zorder=1,
            label="Daily Return",
        )

    # Equity curve fills and line
    ax_curve.fill_between(
        dates,
        cumret,
        0,
        where=cumret >= 0,
        color=c_pos,
        alpha=0.15,
        interpolate=True,
        zorder=1,
    )
    ax_curve.fill_between(
        dates,
        cumret,
        0,
        where=cumret < 0,
        color=c_neg,
        alpha=0.15,
        interpolate=True,
        zorder=1,
    )

    # Glow effect for dark mode
    if is_dark:
        for lw, a in [(6, 0.1), (4, 0.2), (2, 0.5)]:
            ax_curve.plot(dates, cumret, lw=lw, color=c_line, alpha=a, zorder=2)

    marker = "o" if n_rows <= 40 else ""
    ax_curve.plot(
        dates,
        cumret,
        lw=2.2,
        color=c_line,
        marker=marker,
        markersize=3,
        markerfacecolor=bg_ax if is_dark else "white",
        markeredgewidth=1.5,
        zorder=3,
        label="Cumulative Return",
    )
    ax_curve.axhline(0, color=c_neutral_line, lw=1.2, ls="--", zorder=2)
    ax_curve.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax_curve.set_ylabel("Cum. Return", fontsize=9, color=c_label)

    # Drawdown fill
    ax_dd.fill_between(
        dates, dd_full, 0, color=c_neg, alpha=0.3 if is_dark else 0.3, zorder=1
    )
    ax_dd.plot(dates, dd_full, color=c_neg, lw=1.5 if is_dark else 1.0, zorder=2)
    ax_dd.axhline(0, color=c_neutral_line, lw=1.2, ls="--", zorder=2)
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax_dd.set_ylabel("Drawdown", fontsize=9, color=c_label)

    # Add a clean legend to the top chart
    leg = ax_curve.legend(
        loc="upper left",
        frameon=True,
        facecolor=bg_ax,
        edgecolor=c_border,
        fontsize=9,
        labelcolor=text_sec if is_dark else _COLORS["text"],
    )
    if is_dark:
        leg.get_frame().set_alpha(0.8)

    fig.autofmt_xdate(rotation=45)

    date_start = dates_raw[0]
    date_end = dates_raw[-1]
    window_label = (
        window
        if (isinstance(window, str) and window.upper() in _WINDOW_MAP)
        else f"{n_rows}d"
    )

    # Use generic hyphen instead of missing glyph \N{RIGHTWARDS ARROW}
    fig.suptitle(
        f"{title}  ·  {window_label}  ({date_start} to {date_end})",
        fontsize=12,
        fontweight="bold",
        color=c_title,
        y=1.02,
    )
    fig.tight_layout()
    return fig
