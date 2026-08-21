"""Price and volume factors.

Every factor is signed so that **higher means predicted-higher return**, in the
direction the published anomaly runs. That makes the information coefficients
directly comparable across the book and makes a negative IC mean "this anomaly
did not work in this sample" rather than "I flipped a sign somewhere".

Horizons are in trading days throughout, via `lag_trading_days`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.transforms import lag_trading_days

MONTH = 21
YEAR = 252


# ------------------------------------------------------------------ momentum


def mom_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """Jegadeesh-Titman momentum: 12-month return skipping the most recent month.

    The skip matters -- the last month carries short-term reversal, which is the
    opposite sign, and including it dilutes the signal.
    """
    return lag_trading_days(prices, MONTH) / lag_trading_days(prices, YEAR) - 1.0


def mom_6_1(prices: pd.DataFrame) -> pd.DataFrame:
    return lag_trading_days(prices, MONTH) / lag_trading_days(prices, 6 * MONTH) - 1.0


def mom_1m_reversal(prices: pd.DataFrame) -> pd.DataFrame:
    """Short-term reversal (Jegadeesh 1990): last month's winners underperform."""
    return -(prices / lag_trading_days(prices, MONTH) - 1.0)


def mom_residual_12_1(returns: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """Residual momentum (Blitz, Huij, Martens 2011).

    Momentum measured on market-residual returns rather than total returns.
    Stripping the market component removes most of momentum's time-varying beta,
    which is what drives its crashes, and historically raises its Sharpe.
    """
    beta = _rolling_beta(returns, market, YEAR)
    residual = returns.sub(beta.mul(market, axis=0))
    cumulative = residual.rolling(YEAR - MONTH, min_periods=YEAR // 2).sum()
    return lag_trading_days(cumulative, MONTH)


# ------------------------------------------------------------------ risk/vol


def vol_60d(returns: pd.DataFrame) -> pd.DataFrame:
    """Low-volatility anomaly: negated so low vol scores high."""
    return -returns.rolling(60, min_periods=40).std()


def idio_vol_60d(returns: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """Idiosyncratic volatility puzzle (Ang, Hodrick, Xing, Zhang 2006), negated."""
    beta = _rolling_beta(returns, market, 60)
    residual = returns.sub(beta.mul(market, axis=0))
    return -residual.rolling(60, min_periods=40).std()


def beta_252d(returns: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """Betting-against-beta (Frazzini-Pedersen 2014): low beta scores high."""
    return -_rolling_beta(returns, market, YEAR)


def max_ret_1m(returns: pd.DataFrame) -> pd.DataFrame:
    """MAX effect (Bali, Cakici, Whitelaw 2011): lottery-like stocks underperform."""
    return -returns.rolling(MONTH, min_periods=15).max()


def trend_200d(prices: pd.DataFrame) -> pd.DataFrame:
    return prices / prices.rolling(200, min_periods=120).mean() - 1.0


# -------------------------------------------------------------------- volume


def turnover_1m(volume: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Share turnover, negated: high-turnover names underperform (Datar et al. 1998)."""
    turnover = volume.div(shares.replace(0.0, np.nan))
    return -turnover.rolling(MONTH, min_periods=15).mean()


def amihud_illiquidity(returns: pd.DataFrame, dollar_volume: pd.DataFrame) -> pd.DataFrame:
    """Amihud (2002) illiquidity: price impact per dollar traded, as a premium."""
    impact = returns.abs().div(dollar_volume.replace(0.0, np.nan))
    return np.log1p(impact.rolling(MONTH, min_periods=15).mean())


def liquidity_shock(dollar_volume: pd.DataFrame) -> pd.DataFrame:
    """Recent volume relative to its own year: a *change*, not a level.

    The old `liquidity` factor was log dollar volume outright, whose monthly
    cross-sectional rank autocorrelation was 0.96 -- a near-static size ranking
    that consumed more model capacity than any other feature while carrying the
    most negative IC in the book. Dividing by the name's own trailing average
    turns a size proxy into an attention/flow signal.
    """
    short = dollar_volume.rolling(MONTH, min_periods=15).mean()
    long = dollar_volume.rolling(YEAR, min_periods=120).mean()
    return np.log(short.div(long.replace(0.0, np.nan)))


# ------------------------------------------------------------------- helpers


def _rolling_beta(returns: pd.DataFrame, market: pd.Series, window: int) -> pd.DataFrame:
    """Rolling market beta, computed once for the whole panel."""
    market = market.reindex(returns.index)
    min_periods = window // 2
    market_var = market.rolling(window, min_periods=min_periods).var()
    covariance = returns.rolling(window, min_periods=min_periods).cov(market)
    return covariance.div(market_var.replace(0.0, np.nan), axis=0)
