"""Performance statistics.

One correction worth naming. The previous implementation annualised with
`252 / len(returns)`, which is only the growth rate if the return series covers
every trading day. It did not -- 29% of days were missing because skipped
rebalances dropped their P&L entirely -- so the reported 5.9% CAGR was really
4.1% over the elapsed period. Here the horizon is measured in calendar time
between the first and last observation, which is right whether or not the series
has gaps, and `continuity` reports whether it has any.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def summarize(returns: pd.Series, market: pd.Series | None = None) -> dict:
    returns = returns.dropna().sort_index()
    if returns.empty:
        return {"n_days": 0}

    equity = (1.0 + returns).cumprod()
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    total_growth = float(equity.iloc[-1])

    volatility = float(returns.std()) * np.sqrt(TRADING_DAYS)
    downside = returns[returns < 0].std()
    drawdown = float((equity / equity.cummax() - 1.0).min())

    stats = {
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "years": round(years, 2),
        "n_days": int(len(returns)),
        "total_return": round(total_growth - 1.0, 4),
        "cagr": round(total_growth ** (1.0 / years) - 1.0, 4) if years > 0 else np.nan,
        "annual_vol": round(volatility, 4),
        "sharpe": round(float(returns.mean()) / float(returns.std()) * np.sqrt(TRADING_DAYS), 3)
        if returns.std() > 0
        else np.nan,
        "sortino": round(float(returns.mean()) / float(downside) * np.sqrt(TRADING_DAYS), 3)
        if downside and downside > 0
        else np.nan,
        "max_drawdown": round(drawdown, 4),
        "calmar": round((total_growth ** (1.0 / years) - 1.0) / abs(drawdown), 3)
        if years > 0 and drawdown < 0
        else np.nan,
        "hit_rate": round(float((returns > 0).mean()), 4),
        "skew": round(float(returns.skew()), 3),
        "worst_day": round(float(returns.min()), 4),
    }
    stats.update(continuity(returns))
    if market is not None:
        stats.update(_market_exposure(returns, market))
    return stats


def continuity(returns: pd.Series) -> dict:
    """Does the return series actually cover every trading day it claims to?

    A backtest with holes in it can post a respectable Sharpe purely because the
    bad stretches are absent. This is cheap to check and was the single largest
    error in the previous engine, so it is reported alongside the headline.
    """
    expected = pd.bdate_range(returns.index[0], returns.index[-1])
    covered = len(returns) / len(expected)
    return {
        "business_days_in_span": int(len(expected)),
        "coverage": round(float(covered), 4),
    }


def _market_exposure(returns: pd.Series, market: pd.Series) -> dict:
    aligned = pd.DataFrame({"strategy": returns, "market": market}).dropna()
    if len(aligned) < 60:
        return {}
    variance = aligned["market"].var()
    beta = aligned["strategy"].cov(aligned["market"]) / variance if variance > 0 else np.nan
    alpha_daily = aligned["strategy"].mean() - beta * aligned["market"].mean()
    correlation = aligned["strategy"].corr(aligned["market"])
    return {
        "beta_vs_market": round(float(beta), 4),
        "alpha_annual": round(float(alpha_daily) * TRADING_DAYS, 4),
        "corr_vs_market": round(float(correlation), 4),
    }


def cost_stress(gross_returns: pd.Series, turnovers: pd.Series, stress_bps: tuple[float, ...]) -> dict:
    """Re-charge the same trades at higher slippage. Cheap and very informative:
    a strategy whose Sharpe collapses by 20bps is a cost story, not an alpha one."""
    out = {}
    for bps in stress_bps:
        charged = gross_returns.copy()
        charges = turnovers * bps / 1e4
        charged.loc[charges.index] = charged.loc[charges.index] - charges
        out[f"sharpe_at_{int(bps)}bps"] = round(
            float(charged.mean()) / float(charged.std()) * np.sqrt(TRADING_DAYS), 3
        )
    return out
