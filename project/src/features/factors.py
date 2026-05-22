from __future__ import annotations

import numpy as np
import pandas as pd


def _shift_by_calendar_days(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices index must be a DatetimeIndex")
    target_dates = prices.index - pd.Timedelta(days=days)
    shifted = prices.reindex(target_dates, method="ffill")
    shifted.index = prices.index
    return shifted


def _momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    price_t_minus_21 = _shift_by_calendar_days(prices, 21)
    price_t_minus_252 = _shift_by_calendar_days(prices, 252)
    return price_t_minus_21 / price_t_minus_252 - 1.0


def _momentum_n_1(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    price_t_minus_21 = _shift_by_calendar_days(prices, 21)
    price_t_minus_n = _shift_by_calendar_days(prices, lookback_days)
    return price_t_minus_21 / price_t_minus_n - 1.0


def _short_term_reversal(prices: pd.DataFrame) -> pd.DataFrame:
    """5-day reversal: negative short-term return (mean reversion signal)."""
    price_now = _shift_by_calendar_days(prices, 0)
    price_5d = _shift_by_calendar_days(prices, 5)
    return -(price_now / price_5d - 1.0)


def _idiosyncratic_vol(returns: pd.DataFrame, market_returns: pd.Series) -> pd.DataFrame:
    """Rolling 60-day residual vol after removing market beta.

    Lower idio vol is a stronger signal (defensive tilt), so we negate.
    """
    window = 60
    idio_vol = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for start in range(0, len(returns) - window + 1):
        end = start + window
        chunk = returns.iloc[start:end]
        mkt = market_returns.iloc[start:end]
        date = returns.index[end - 1]

        mkt_clean = mkt.dropna()
        mkt_var = mkt_clean.var()
        if mkt_var == 0 or np.isnan(mkt_var):
            continue

        for col in returns.columns:
            s = chunk[col].dropna()
            common = s.index.intersection(mkt_clean.index)
            if len(common) < window // 2:
                continue
            beta = s.loc[common].cov(mkt_clean.loc[common]) / mkt_var
            resid = s.loc[common] - beta * mkt_clean.loc[common]
            idio_vol.loc[date, col] = resid.std()

    return -idio_vol.astype(float)


def _volume_momentum(volumes: pd.DataFrame) -> pd.DataFrame:
    """Ratio of 20-day avg volume to 60-day avg volume (unusual activity)."""
    short_avg = volumes.rolling(20).mean()
    long_avg = volumes.rolling(60).mean()
    return (short_avg / long_avg.replace(0, np.nan)) - 1.0


def compute_factors(prices: pd.DataFrame, volumes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    prices = prices.sort_index()
    returns = prices.pct_change(fill_method=None)

    mom_12_1 = _momentum_12_1(prices)
    mom_6_1 = _momentum_n_1(prices, 126)
    mom_3_1 = _momentum_n_1(prices, 63)

    reversal = _short_term_reversal(prices)

    low_vol = -returns.rolling(252).std()

    market_ret = returns.get("SPY", returns.mean(axis=1))
    idio_vol = _idiosyncratic_vol(returns, market_ret)

    trend = prices / prices.rolling(200).mean() - 1.0

    dollar_volume = prices * volumes
    liquidity = np.log1p(dollar_volume.rolling(20).mean())

    vol_mom = _volume_momentum(volumes)

    rolling_mean = returns.rolling(252).mean()
    rolling_std = returns.rolling(252).std()
    quality_proxy = rolling_mean / rolling_std.replace(0.0, np.nan)

    return {
        "mom_12_1": mom_12_1,
        "mom_6_1": mom_6_1,
        "mom_3_1": mom_3_1,
        "reversal": reversal,
        "low_vol": low_vol,
        "idio_vol": idio_vol,
        "trend": trend,
        "liquidity": liquidity,
        "vol_momentum": vol_mom,
        "quality_proxy": quality_proxy,
    }

