"""Trading costs.

One-way cost is charged on the notional actually traded, so a rebalance that
moves a name from +2% to +2.5% is charged on the 0.5%, not on the position.
"""

from __future__ import annotations

import pandas as pd


def turnover(previous: pd.Series, target: pd.Series) -> float:
    """One-way turnover: total absolute change in weights."""
    combined = previous.reindex(previous.index.union(target.index)).fillna(0.0)
    proposed = target.reindex(combined.index).fillna(0.0)
    return float((proposed - combined).abs().sum())


def cost(previous: pd.Series, target: pd.Series, commission_bps: float, slippage_bps: float) -> float:
    return turnover(previous, target) * (commission_bps + slippage_bps) / 1e4
