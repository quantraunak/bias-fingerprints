"""Signal evaluation, run before any portfolio is constructed.

If the information coefficient is not there, no optimiser will rescue it, and a
backtest that looks acceptable anyway is telling you about leverage and luck
rather than about the signal. So the IC table is the gate: it is produced first,
and it uses forward returns directly, with no weighting, costs or constraints in
between to launder a weak result.

The headline number is the IC information ratio -- mean IC over its standard
deviation, annualised by the number of independent observations per year. A
long/short book's Sharpe is bounded by roughly IC * sqrt(breadth), so ICIR is
the honest early read on whether the strategy can work at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def information_coefficient(scores: pd.Series, forward: pd.Series, method: str = "spearman") -> pd.Series:
    """Per-date cross-sectional correlation between score and realised forward return."""
    frame = pd.DataFrame({"score": scores, "forward": forward}).dropna()
    return (
        frame.groupby(level="date")
        .apply(lambda block: block["score"].corr(block["forward"], method=method) if len(block) > 10 else np.nan)
        .dropna()
        .rename("ic")
    )


def summarize(ic: pd.Series, horizon_days: int) -> dict:
    """Mean IC, its information ratio, and a t-statistic that respects overlap.

    Successive ICs of an h-day forward return overlap unless they are sampled at
    least h days apart, so the naive t-statistic overstates significance. The
    deflator is driven by the *observed* spacing of the IC series rather than
    assumed: scored daily, 150 dates are worth about 7 independent draws; scored
    at monthly rebalances, the same 150 dates are worth 150.
    """
    if ic.empty:
        return {"n": 0}
    spacing = _median_spacing_days(ic.index)
    independent = max(len(ic) * min(spacing / horizon_days, 1.0), 1.0)
    mean, std = float(ic.mean()), float(ic.std())
    ir = mean / std if std > 0 else 0.0
    return {
        "n_dates": int(len(ic)),
        "n_independent": round(independent, 1),
        "mean_ic": round(mean, 5),
        "ic_std": round(std, 5),
        "icir": round(ir, 4),
        "icir_annual": round(ir * np.sqrt(TRADING_DAYS / horizon_days), 4),
        "t_stat": round(mean / (std / np.sqrt(independent)), 3) if std > 0 else 0.0,
        "pct_positive": round(float((ic > 0).mean()), 4),
        "ic_autocorr_1": round(float(ic.autocorr(1)), 4) if len(ic) > 2 else np.nan,
    }


def _median_spacing_days(index: pd.DatetimeIndex) -> float:
    """Median trading days between consecutive scoring dates."""
    if len(index) < 2:
        return 1.0
    gaps = pd.Series(index).diff().dt.days.dropna()
    return max(float(gaps.median()) * (252 / 365.25), 1.0)


def factor_table(features: pd.DataFrame, forward: pd.Series, horizon_days: int) -> pd.DataFrame:
    """One row per factor: is this thing predictive on its own, and how reliably?"""
    rows = {}
    for name in features.columns:
        ic = information_coefficient(features[name], forward)
        rows[name] = summarize(ic, horizon_days)
    return pd.DataFrame(rows).T.sort_values("icir", ascending=False)


def decay(scores: pd.Series, prices: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """How long the signal survives. A signal that decays inside the rebalance
    interval cannot be traded at that frequency without paying for it twice."""
    from src.model.target import forward_returns

    rows = {}
    for horizon in horizons:
        forward = forward_returns(prices, horizon).stack(future_stack=True)
        forward.index.names = ["date", "ticker"]
        rows[horizon] = summarize(information_coefficient(scores, forward), horizon)
    return pd.DataFrame(rows).T.rename_axis("horizon_days")


def quantile_returns(scores: pd.Series, forward: pd.Series, n_quantiles: int = 10) -> pd.DataFrame:
    """Mean forward return by score quantile, plus the top-minus-bottom spread.

    A monotone profile is much stronger evidence than a wide spread alone: two
    extreme buckets can separate by accident, ten in order cannot.
    """
    frame = pd.DataFrame({"score": scores, "forward": forward}).dropna()
    frame["bucket"] = frame.groupby(level="date")["score"].transform(
        lambda values: pd.qcut(values.rank(method="first"), n_quantiles, labels=False, duplicates="drop")
    )
    by_bucket = frame.groupby("bucket")["forward"].agg(["mean", "std", "count"])
    by_bucket["mean_bps"] = (by_bucket["mean"] * 1e4).round(1)

    per_date = frame.groupby([frame.index.get_level_values("date"), "bucket"])["forward"].mean().unstack()
    spread = (per_date[per_date.columns.max()] - per_date[per_date.columns.min()]).dropna()
    by_bucket.attrs["spread"] = {
        "mean_bps": round(float(spread.mean() * 1e4), 2),
        "t_stat": round(float(spread.mean() / spread.std() * np.sqrt(len(spread))), 2) if spread.std() > 0 else 0.0,
        "pct_positive": round(float((spread > 0).mean()), 4),
    }
    return by_bucket


def turnover(scores: pd.Series, quantile: float = 0.1) -> float:
    """Fraction of the selected long book that changes between rebalances."""
    selections = []
    for _, block in scores.groupby(level="date"):
        n = max(int(len(block) * quantile), 1)
        selections.append(set(block.nlargest(n).index.get_level_values("ticker")))
    if len(selections) < 2:
        return 0.0
    changes = [
        1.0 - len(a & b) / max(len(a), 1) for a, b in zip(selections[:-1], selections[1:])
    ]
    return round(float(np.mean(changes)), 4)
