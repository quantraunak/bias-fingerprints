"""Fundamental factors, built only from figures that had already been filed.

Inputs arrive as daily date x ticker frames that were forward-filled from each
statement's **filing** date, so a factor never uses a number before the market
could have read it. Year-over-year comparisons shift by 252 trading days, which
compares what was known today with what was known a year ago -- not two period
ends, which would quietly reintroduce look-ahead through revisions.

Signs follow the published direction: higher score means predicted-higher return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

YEAR = 252


def _safe_divide(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    return numerator.div(denominator.replace(0.0, np.nan))


def _positive_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Negative book equity or sales make a ratio meaningless rather than extreme."""
    return frame.where(frame > 0.0)


# --------------------------------------------------------------------- value


def book_to_market(equity: pd.DataFrame, market_cap: pd.DataFrame) -> pd.DataFrame:
    """Classic value (Fama-French 1992). Negative book equity is excluded."""
    return _safe_divide(_positive_only(equity), market_cap)


def earnings_yield(net_income: pd.DataFrame, market_cap: pd.DataFrame) -> pd.DataFrame:
    return _safe_divide(net_income, market_cap)


def cash_flow_yield(operating_cash_flow: pd.DataFrame, market_cap: pd.DataFrame) -> pd.DataFrame:
    """Cash-flow-to-price. Harder to manipulate than earnings, so usually cleaner."""
    return _safe_divide(operating_cash_flow, market_cap)


def sales_to_price(revenue: pd.DataFrame, market_cap: pd.DataFrame) -> pd.DataFrame:
    return _safe_divide(_positive_only(revenue), market_cap)


# ------------------------------------------------------------- profitability


def gross_profitability(gross_profit: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    """Novy-Marx (2013): gross profit over assets, the cleanest profitability measure."""
    return _safe_divide(gross_profit, _positive_only(assets))


def roe(net_income: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    return _safe_divide(net_income, _positive_only(equity))


def operating_margin(operating_income: pd.DataFrame, revenue: pd.DataFrame) -> pd.DataFrame:
    return _safe_divide(operating_income, _positive_only(revenue))


# ---------------------------------------------------------------- investment


def asset_growth(assets: pd.DataFrame) -> pd.DataFrame:
    """Cooper, Gulen, Schill (2008): firms that grow the balance sheet underperform."""
    return -(_safe_divide(assets, _positive_only(assets.shift(YEAR))) - 1.0)


def accruals(net_income: pd.DataFrame, operating_cash_flow: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    """Sloan (1996): earnings that are not backed by cash reverse. Negated."""
    return -_safe_divide(net_income - operating_cash_flow, _positive_only(assets))


def net_share_issuance(shares: pd.DataFrame) -> pd.DataFrame:
    """Pontiff-Woodgate (2008): issuers underperform, repurchasers outperform."""
    return -(_safe_divide(shares, _positive_only(shares.shift(YEAR))) - 1.0)


def market_cap(prices: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Price times the share count disclosed on the most recent filing's cover page."""
    return prices.mul(shares.reindex_like(prices))
