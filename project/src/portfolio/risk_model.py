"""Covariance estimation, volatility targeting, and drawdown governance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def shrink_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage estimator for the covariance matrix."""
    clean = returns.dropna(axis=0, how="any")
    if clean.shape[0] < clean.shape[1] + 1:
        var = returns.var().fillna(returns.var().mean())
        return pd.DataFrame(np.diag(var.values), index=returns.columns, columns=returns.columns)

    lw = LedoitWolf().fit(clean.values)
    return pd.DataFrame(lw.covariance_, index=clean.columns, columns=clean.columns)


def _ewma_annualized_vol(returns: pd.Series, span: int) -> float:
    if returns.empty or len(returns) < max(20, span // 2):
        return 0.0
    vol = returns.ewm(span=span, adjust=False).std().iloc[-1]
    if pd.isna(vol) or vol <= 0:
        return 0.0
    return float(vol * np.sqrt(252))


def compute_vol_scaled_leverage(
    port_returns: pd.Series,
    base_leverage: float,
    target_vol: float,
    lookback: int = 63,
    min_leverage: float = 0.5,
    max_leverage: float = 2.0,
    ewma_span: int | None = None,
) -> float:
    """Scale gross leverage so EWMA realized vol tracks target_vol."""
    if port_returns is None or len(port_returns) < lookback:
        return base_leverage

    recent = port_returns.iloc[-lookback:]
    span = ewma_span or lookback
    realized_vol = _ewma_annualized_vol(recent, span)

    if realized_vol <= 0:
        return base_leverage

    scaled = base_leverage * (target_vol / realized_vol)
    return float(np.clip(scaled, min_leverage, max_leverage))


def compute_drawdown_scale(
    port_returns: pd.Series,
    dd_threshold: float = 0.08,
    min_scale: float = 0.35,
) -> float:
    """De-gross when underwater: 1.0 at peak, ramps down to min_scale past threshold."""
    if port_returns.empty:
        return 1.0

    equity = (1 + port_returns).cumprod()
    current_dd = float(equity.iloc[-1] / equity.cummax().iloc[-1] - 1.0)

    if current_dd >= -dd_threshold:
        return 1.0

    # Linear de-risk between threshold and 2× threshold depth
    depth = abs(current_dd) - dd_threshold
    span = dd_threshold
    reduction = min(1.0, depth / span)
    return max(min_scale, 1.0 - reduction * (1.0 - min_scale))


def compute_effective_leverage(
    port_returns: pd.Series,
    base_leverage: float,
    target_vol: float | None,
    vol_lookback: int,
    min_leverage: float,
    max_leverage: float,
    ewma_span: int | None,
    dd_threshold: float | None,
    dd_min_scale: float,
) -> float:
    """Vol targeting × drawdown governor."""
    lev = base_leverage
    if target_vol is not None:
        lev = compute_vol_scaled_leverage(
            port_returns,
            base_leverage,
            target_vol,
            lookback=vol_lookback,
            min_leverage=min_leverage,
            max_leverage=max_leverage,
            ewma_span=ewma_span,
        )

    if dd_threshold is not None and not port_returns.empty:
        lev *= compute_drawdown_scale(port_returns, dd_threshold, dd_min_scale)

    return float(np.clip(lev, min_leverage, max_leverage))
