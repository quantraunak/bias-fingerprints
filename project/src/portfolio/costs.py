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


TRADING_DAYS = 252


def financing(short_notional: float, annual_bps: float) -> float:
    """Daily stock-borrow cost, charged on the short side only.

    Trading cost scales with turnover and is therefore neutral to leverage:
    doubling gross doubles both the cost and the P&L it is charged against.
    Borrow is not. It accrues every day the book is held and scales with the
    short book, so it is what decides whether running market-neutral at 5x gross
    is genuinely cheaper per unit of alpha than at 1.5x. A construction sweep
    that raises leverage without charging carry measures an accounting artifact.

    Charged on short notional rather than on gross. In a dollar-neutral book the
    long side is funded by the short proceeds, so billing the full gross at a
    funding rate double-counts: it charges the longs for cash the shorts already
    supplied. The residual cost is the borrow on the shares themselves, roughly
    25-50bp annually for general-collateral large caps and far higher for
    hard-to-borrow names -- which is a reason to treat this as a floor rather
    than an estimate, since a small-cap short book does not borrow at GC.
    """
    return short_notional * (annual_bps / 1e4) / TRADING_DAYS
