"""Covariance, betas, and how much risk to take.

Sample covariance on 250 days and 90 names is badly conditioned -- the smallest
eigenvalues are mostly estimation error, and an optimiser will happily load up
on them. Ledoit-Wolf shrinkage pulls the matrix towards a scaled identity by an
analytically chosen amount, which is the standard fix and costs nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def shrunk_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf covariance, **annualised**.

    The units matter more than they look. Daily covariance entries are order
    1e-4, so `w'Sigma w` for a levered book is order 1e-5 -- against an alpha
    term of order 1, the risk penalty is four orders of magnitude too small to
    bind and the optimiser degenerates into "buy the box constraint on whichever
    names have the most extreme score". Annualising puts risk and return in the
    same units and makes `risk_aversion` mean what a textbook says it means.
    """
    clean = returns.dropna(axis=1, how="all").fillna(0.0)
    estimate = LedoitWolf().fit(clean.to_numpy())
    annual = estimate.covariance_ * TRADING_DAYS
    return pd.DataFrame(annual, index=clean.columns, columns=clean.columns)


def betas(returns: pd.DataFrame, market: pd.Series) -> pd.Series:
    """Market betas for the whole block in one pass."""
    aligned = returns.join(market.rename("_market"), how="inner").dropna(how="all")
    market_column = aligned.pop("_market")
    variance = market_column.var()
    if not variance > 0:
        raise ValueError("Cannot estimate betas: market variance is zero.")
    return aligned.apply(lambda column: column.cov(market_column)).div(variance)


def drawdown_scale(
    equity_returns: pd.Series,
    threshold: float,
    min_scale: float,
) -> float:
    """Multiplier on the risk budget while the book is in drawdown.

    Volatility targeting proper now lives in the optimiser as an explicit
    constraint, so the only job left here is the de-grossing overlay: cut risk
    as losses deepen, restore it as they recover. Uses realised history only.
    """
    if len(equity_returns) < 20:
        return 1.0

    equity = (1.0 + equity_returns).cumprod()
    drawdown = float(equity.iloc[-1] / equity.cummax().iloc[-1] - 1.0)
    if drawdown >= -threshold:
        return 1.0

    excess = min(abs(drawdown) - threshold, threshold)
    return float(np.clip(1.0 - (1.0 - min_scale) * (excess / threshold), min_scale, 1.0))
