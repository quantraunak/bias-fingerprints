"""Factor Information Coefficient (IC) analysis.

For each factor, compute the daily cross-sectional Spearman rank correlation
between the factor value and next-day stock returns.  Produces:
  - factor_ic.csv        – daily IC time series per factor
  - factor_ic_summary.json – mean IC, IC std, IC IR, pct positive
  - factor_ic.png        – IC time series plot
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.labels.forward_returns import compute_forward_returns


def _spearman_ic_series(
    factor: pd.DataFrame,
    fwd_returns: pd.DataFrame,
) -> pd.Series:
    """Cross-sectional Spearman IC for each date.

    Parameters
    ----------
    factor : wide DataFrame (date × ticker) of factor values
    fwd_returns : wide DataFrame (date × ticker) of next-day returns

    Returns
    -------
    pd.Series indexed by date with the Spearman rank correlation.
    """
    common_dates = factor.index.intersection(fwd_returns.index)
    common_tickers = factor.columns.intersection(fwd_returns.columns)
    factor = factor.loc[common_dates, common_tickers]
    fwd_returns = fwd_returns.loc[common_dates, common_tickers]

    ics = {}
    for dt in common_dates:
        f_row = factor.loc[dt].dropna()
        r_row = fwd_returns.loc[dt].dropna()
        shared = f_row.index.intersection(r_row.index)
        if len(shared) < 5:
            continue
        ic = f_row[shared].rank().corr(r_row[shared].rank())
        if not np.isnan(ic):
            ics[dt] = ic

    return pd.Series(ics, dtype=float)


def compute_factor_ic(
    factors: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    horizon_days: int = 21,
) -> tuple[pd.DataFrame, dict]:
    """Spearman IC vs h-day forward returns (same horizon as model labels)."""
    fwd = compute_forward_returns(prices, horizon_days)

    ic_series: dict[str, pd.Series] = {}
    for name, fdf in factors.items():
        ic_series[name] = _spearman_ic_series(fdf, fwd)

    ic_df = pd.DataFrame(ic_series).sort_index().dropna(how="all")

    ic_summary: dict[str, dict] = {}
    for name in ic_df.columns:
        s = ic_df[name].dropna()
        ic_summary[name] = {
            "mean_ic": float(s.mean()) if len(s) else 0.0,
            "ic_std": float(s.std()) if len(s) else 0.0,
            "ic_ir": float(s.mean() / s.std()) if len(s) and s.std() != 0 else 0.0,
            "pct_positive": float((s > 0).mean()) if len(s) else 0.0,
            "n_days": int(len(s)),
        }

    return ic_df, ic_summary


def save_factor_ic(
    ic_df: pd.DataFrame,
    ic_summary: dict,
    output_dir: Path,
) -> None:
    """Persist IC CSV, summary JSON, and time-series plot."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ic_df.to_csv(output_dir / "factor_ic.csv")
    (output_dir / "factor_ic_summary.json").write_text(
        json.dumps(ic_summary, indent=2)
    )

    # --- plot ---
    rolling_window = 63  # ~3-month rolling mean for readability
    fig, axes = plt.subplots(
        len(ic_df.columns), 1,
        figsize=(12, 3 * len(ic_df.columns)),
        sharex=True,
        squeeze=False,
    )
    for ax_row, name in zip(axes, ic_df.columns):
        ax = ax_row[0]
        s = ic_df[name].dropna()
        ax.bar(s.index, s.values, width=1, color="steelblue", alpha=0.25, label="daily IC")
        if len(s) >= rolling_window:
            rolling = s.rolling(rolling_window).mean()
            ax.plot(rolling.index, rolling.values, color="navy", linewidth=1.2, label=f"{rolling_window}d mean")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("IC")
        ax.set_title(f"{name}  (mean={ic_summary[name]['mean_ic']:.4f}, IR={ic_summary[name]['ic_ir']:.3f})")
        ax.legend(loc="upper right", fontsize=7)

    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(output_dir / "factor_ic.png", dpi=120)
    plt.close(fig)

