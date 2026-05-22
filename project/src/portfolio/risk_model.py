"""Covariance estimation and volatility targeting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def shrink_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage estimator for the covariance matrix."""
    clean = returns.dropna(axis=0, how="any")
    if clean.shape[0] < clean.shape[1] + 1:
        # Fallback: diagonal with shrinkage toward equal variance
        var = returns.var().fillna(returns.var().mean())
        return pd.DataFrame(np.diag(var.values), index=returns.columns, columns=returns.columns)

    lw = LedoitWolf().fit(clean.values)
    return pd.DataFrame(lw.covariance_, index=clean.columns, columns=clean.columns)


def compute_vol_scaled_leverage(
    port_returns: pd.Series,
    base_leverage: float,
    target_vol: float,
    lookback: int = 63,
    min_leverage: float = 0.5,
    max_leverage: float = 3.0,
) -> float:
    """Scale gross leverage so realized portfolio vol tracks target_vol.

    Uses trailing `lookback` days of portfolio returns to estimate
    annualized realized volatility, then adjusts leverage proportionally.
    """
    if port_returns is None or len(port_returns) < lookback:
        return base_leverage

    recent = port_returns.iloc[-lookback:]
    realized_vol = float(recent.std() * np.sqrt(252))

    if realized_vol <= 0 or np.isnan(realized_vol):
        return base_leverage

    scaled = base_leverage * (target_vol / realized_vol)
    return float(np.clip(scaled, min_leverage, max_leverage))
