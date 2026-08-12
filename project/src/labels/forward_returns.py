from __future__ import annotations

import pandas as pd


def compute_forward_returns(prices: pd.DataFrame, horizon_days: int = 21) -> pd.DataFrame:
    """Trading-day forward return: P(t+h) / P(t) - 1."""
    return prices.shift(-horizon_days) / prices - 1.0


def last_label_date(
    trading_index: pd.DatetimeIndex,
    as_of: pd.Timestamp,
    horizon_days: int,
) -> pd.Timestamp | None:
    """Last date t where the h-day forward return uses only prices on or before as_of."""
    valid = trading_index[trading_index <= as_of]
    if len(valid) <= horizon_days:
        return None
    return valid[-(horizon_days + 1)]
