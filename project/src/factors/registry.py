"""One name -> one factor. Adding a factor means adding a line here.

Each entry is a callable that takes the panel and returns a raw date x ticker
frame. Standardisation is applied uniformly afterwards, so individual factors
only express their economics and never worry about scaling.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.data.panel import Panel
from src.factors import fundamental as fund
from src.factors import price as px
from src.factors.transforms import standardize

Factor = Callable[[Panel], pd.DataFrame]

REGISTRY: dict[str, Factor] = {
    # price / volume
    "mom_12_1": lambda p: px.mom_12_1(p.prices),
    "mom_6_1": lambda p: px.mom_6_1(p.prices),
    "mom_1m_reversal": lambda p: px.mom_1m_reversal(p.prices),
    "mom_residual_12_1": lambda p: px.mom_residual_12_1(p.returns, p.market),
    "vol_60d": lambda p: px.vol_60d(p.returns),
    "idio_vol_60d": lambda p: px.idio_vol_60d(p.returns, p.market),
    "beta_252d": lambda p: px.beta_252d(p.returns, p.market),
    "max_ret_1m": lambda p: px.max_ret_1m(p.returns),
    "turnover_1m": lambda p: px.turnover_1m(p.volume, p.fundamentals["shares"]),
    "amihud_illiquidity": lambda p: px.amihud_illiquidity(p.returns, p.dollar_volume),
    "liquidity_shock": lambda p: px.liquidity_shock(p.dollar_volume),
    "trend_200d": lambda p: px.trend_200d(p.prices),
    # fundamental, point-in-time by filing date
    "book_to_market": lambda p: fund.book_to_market(p.fundamentals["equity"], p.market_cap),
    "earnings_yield": lambda p: fund.earnings_yield(p.fundamentals["net_income"], p.market_cap),
    "cash_flow_yield": lambda p: fund.cash_flow_yield(p.fundamentals["operating_cash_flow"], p.market_cap),
    "sales_to_price": lambda p: fund.sales_to_price(p.fundamentals["revenue"], p.market_cap),
    "gross_profitability": lambda p: fund.gross_profitability(p.fundamentals["gross_profit"], p.fundamentals["assets"]),
    "roe": lambda p: fund.roe(p.fundamentals["net_income"], p.fundamentals["equity"]),
    "operating_margin": lambda p: fund.operating_margin(p.fundamentals["operating_income"], p.fundamentals["revenue"]),
    "asset_growth": lambda p: fund.asset_growth(p.fundamentals["assets"]),
    "accruals": lambda p: fund.accruals(
        p.fundamentals["net_income"], p.fundamentals["operating_cash_flow"], p.fundamentals["assets"]
    ),
    "net_share_issuance": lambda p: fund.net_share_issuance(p.fundamentals["shares"]),
}


def compute_raw(panel: Panel, names: list[str]) -> dict[str, pd.DataFrame]:
    """Raw factor values, masked to the tradable universe."""
    unknown = set(names) - set(REGISTRY)
    if unknown:
        raise KeyError(f"Unknown factors: {sorted(unknown)}")
    return {name: REGISTRY[name](panel).where(panel.tradable) for name in names}


def risk_exposures(panel: Panel) -> dict[str, pd.DataFrame]:
    """Exposures a factor can be residualised against, when `risk_neutral` is set.

    Off by default, on evidence rather than taste. The argument for it is sound:
    59% of the raw decile spread here is a market beta tilt, and a book paid for
    beta is being paid for exposure it did not intend to take. But residualising
    every factor against beta and size costs most of the signal -- out-of-sample
    IC falls from 0.0164 to 0.0062 -- because at this horizon the value edge
    substantially *is* a size effect, and orthogonalising it away removes the
    alpha along with the exposure. Beta is therefore controlled where it is
    cheaper to control: as a constraint in the optimiser.
    """
    return {
        "beta": -px.beta_252d(panel.returns, panel.market),  # un-negate: raw beta
        "size": np.log(panel.market_cap.where(panel.market_cap > 0)),
    }


def build_matrix(
    panel: Panel, names: list[str], sector_neutral: bool = True, risk_neutral: bool = False
) -> pd.DataFrame:
    """Standardised (date, ticker) x factor matrix ready for the model.

    Rows are kept when *any* factor is present rather than all of them: a name
    with no filed fundamentals still carries usable price information, and
    LightGBM splits on missing values natively. Dropping those rows would have
    thrown away the whole pre-XBRL era and every recent listing.
    """
    industries = panel.industries if sector_neutral else None
    exposures = risk_exposures(panel) if risk_neutral else None
    raw = compute_raw(panel, names)
    columns = [
        standardize(frame, industries, exposures).stack(future_stack=True).rename(name)
        for name, frame in raw.items()
    ]
    matrix = pd.concat(columns, axis=1)
    matrix.index.names = ["date", "ticker"]
    return matrix.dropna(how="all")
