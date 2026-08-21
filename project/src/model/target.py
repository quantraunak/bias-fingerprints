"""The prediction target.

The old pipeline regressed on the raw 21-day forward return. Measured on this
panel, **28% of that variable's variance is the common market move on the date**
-- a term identical for every name in the cross-section, so it cannot change the
ranking the strategy actually trades, yet an L2 loss still spends that share of
its budget trying to fit it. On top of that, the raw return's cross-sectional
dispersion swings by a factor of three between calm and stressed months, so the
loss is implicitly weighted towards crisis dates.

Ranking within each date fixes both at once: the market component is removed by
construction, every date contributes equally, and outliers cannot dominate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(prices: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Return from t to t+h, in trading days. Rows near the end become NaN."""
    return prices.shift(-horizon_days) / prices - 1.0


def cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank within each date, mapped to [-0.5, 0.5]."""
    return frame.rank(axis=1, pct=True) - 0.5


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(frame.mean(axis=1), axis=0).div(std, axis=0)


def build(prices: pd.DataFrame, horizon_days: int, kind: str = "rank") -> pd.Series:
    """Stacked (date, ticker) target series."""
    raw = forward_returns(prices, horizon_days)
    transforms = {
        "rank": cross_sectional_rank,
        "zscore": cross_sectional_zscore,
        "raw": lambda frame: frame,
    }
    if kind not in transforms:
        raise ValueError(f"Unknown target '{kind}'; expected one of {sorted(transforms)}")
    target = transforms[kind](raw).stack(future_stack=True).rename("target")
    target.index.names = ["date", "ticker"]
    return target.dropna()


def to_grades(target: pd.Series, n_grades: int = 10) -> pd.Series:
    """Discretise the target into per-date grades for a learning-to-rank objective."""
    grades = target.groupby(level="date").transform(
        lambda values: pd.qcut(values.rank(method="first"), n_grades, labels=False, duplicates="drop")
    )
    return grades.fillna(0).astype(int).rename("grade")
