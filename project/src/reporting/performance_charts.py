"""Institutional-quality performance charts for README and tear sheets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

_STYLE = {
    "figure.facecolor": "#FAFAF8",
    "axes.facecolor": "#FAFAF8",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "text.color": "#1A1A1A",
    "grid.color": "#DDDDDD",
    "grid.alpha": 0.6,
    "font.size": 10,
}


def _calendar_returns(daily_returns: pd.Series) -> pd.Series:
    """Reindex to business days; missing days = 0 return (no position)."""
    if daily_returns.empty:
        return daily_returns
    idx = pd.date_range(daily_returns.index.min(), daily_returns.index.max(), freq="B")
    return daily_returns.reindex(idx).fillna(0.0)


def _underwater_series(daily_returns: pd.Series) -> pd.Series:
    r = _calendar_returns(daily_returns)
    equity = (1 + r).cumprod()
    return equity / equity.cummax() - 1.0


def plot_equity_curve(
    daily_returns: pd.Series,
    output_path: Path,
    benchmark_returns: pd.Series | None = None,
) -> None:
    """Growth-of-$1 equity curve on log scale."""
    plt.rcParams.update(_STYLE)
    fig, ax = plt.subplots(figsize=(11, 4.5))

    equity = (1 + _calendar_returns(daily_returns)).cumprod()
    ax.plot(equity.index, equity.values, color="#1B3A5C", linewidth=1.4, label="Strategy")

    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(daily_returns.index).fillna(0.0)
        bench_equity = (1 + bench).cumprod()
        ax.plot(
            bench_equity.index,
            bench_equity.values,
            color="#9AA5B1",
            linewidth=1.0,
            linestyle="--",
            label="SPY",
        )

    ax.set_yscale("log")
    ax.set_title("Cumulative Return (Growth of $1, Log Scale)", fontsize=12, fontweight="600")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, axis="y")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(
    daily_returns: pd.Series,
    output_path: Path,
    resample_rule: str = "ME",
) -> dict[str, float]:
    """Underwater drawdown with continuous calendar series and monthly line overlay."""
    plt.rcParams.update(_STYLE)
    dd_daily = _underwater_series(daily_returns)
    dd_line = dd_daily.resample(resample_rule).min()

    max_dd = float(dd_daily.min())
    max_dd_date = dd_daily.idxmin()

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(
        dd_daily.index,
        dd_daily.values * 100,
        0,
        color="#C44E52",
        alpha=0.22,
        linewidth=0,
        interpolate=True,
    )
    ax.plot(dd_line.index, dd_line.values * 100, color="#8B2E32", linewidth=1.5)

    ax.axhline(0, color="#333333", linewidth=0.6)
    ax.set_title("Drawdown from Peak (Monthly)", fontsize=12, fontweight="600")
    ax.set_ylabel("Drawdown (%)")
    y_floor = min(dd_daily.min() * 100 * 1.12, -5)
    ax.set_ylim(y_floor, 2)
    ax.grid(True, axis="y")

    ax.annotate(
        f"Max: {max_dd * 100:.1f}% ({max_dd_date.strftime('%Y-%m')})",
        xy=(max_dd_date, max_dd * 100),
        xytext=(12, -18),
        textcoords="offset points",
        fontsize=9,
        color="#8B2E32",
        arrowprops=dict(arrowstyle="->", color="#8B2E32", lw=0.8),
    )

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"max_drawdown": max_dd, "max_drawdown_date": str(max_dd_date.date())}


def save_performance_charts(
    daily_returns: pd.Series,
    output_dir: Path,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_equity_curve(daily_returns, output_dir / "equity_curve.png", benchmark_returns)
    meta = plot_drawdown(daily_returns, output_dir / "drawdown.png")
    return meta
