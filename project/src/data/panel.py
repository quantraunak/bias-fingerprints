"""Assemble prices, fundamentals, industries and membership into one panel.

Everything downstream reads from `Panel`, so there is exactly one place where
the universe is defined and one place where the tradability screens live.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.data import fundamentals, prices, sectors, universe

FUNDAMENTAL_ITEMS = [
    "assets", "equity", "shares", "net_income", "revenue",
    "gross_profit", "operating_income", "operating_cash_flow",
]


@dataclass
class Panel:
    prices: pd.DataFrame
    volume: pd.DataFrame
    returns: pd.DataFrame
    market: pd.Series
    dollar_volume: pd.DataFrame
    fundamentals: dict[str, pd.DataFrame]
    market_cap: pd.DataFrame
    membership: pd.DataFrame
    industries: pd.Series
    tradable: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.prices.index

    def coverage(self) -> pd.DataFrame:
        """Per-date name counts: index members, names with prices, names tradable."""
        return pd.DataFrame(
            {
                "in_index": self.membership.sum(axis=1),
                "with_price": self.prices.notna().sum(axis=1),
                "tradable": self.tradable.sum(axis=1),
                "with_fundamentals": self.fundamentals["assets"].notna().sum(axis=1),
            }
        )


def build(config: Config) -> Panel:
    spells = universe.load_spells()
    benchmark = config.data.benchmark.upper()
    # The benchmark is never an index constituent, so it has to be asked for explicitly.
    tickers = universe.tickers_in_window(spells, config.data.start, config.data.end) + [benchmark]

    price_frame, volume = prices.load_panel(tickers, config.data.start, config.data.end)
    if benchmark not in price_frame.columns:
        raise ValueError(f"Benchmark {benchmark} has no price history in the cache.")

    returns = price_frame.pct_change(fill_method=None)
    market = returns[benchmark]

    # The benchmark is a data input, not an investable name.
    available = [t for t in price_frame.columns if t != benchmark]
    price_frame = price_frame[available]
    volume = volume[available]
    returns = returns[available]

    dollar_volume = price_frame.mul(volume)
    membership = universe.membership_matrix(spells, price_frame.index).reindex(
        columns=available, fill_value=False
    )

    facts = fundamentals.load_facts(available)
    fundamental_panel = fundamentals.as_of_panel(facts, price_frame.index)
    wide = {
        item: fundamental_panel[item].unstack("ticker").reindex(columns=available)
        if item in fundamental_panel.columns
        else pd.DataFrame(np.nan, index=price_frame.index, columns=available)
        for item in FUNDAMENTAL_ITEMS
    }

    market_cap = price_frame.mul(wide["shares"])
    industries = sectors.industry_map(available, config.data.sec_user_agent)

    return Panel(
        prices=price_frame,
        volume=volume,
        returns=returns,
        market=market,
        dollar_volume=dollar_volume,
        fundamentals=wide,
        market_cap=market_cap,
        membership=membership,
        industries=industries,
        tradable=_tradable(price_frame, dollar_volume, membership, config),
    )


def _tradable(
    price_frame: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    membership: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Names we would actually have been able to trade, evaluated each day.

    Screens use only trailing information, so a name that becomes illiquid drops
    out on the day that becomes visible rather than for the whole sample.
    """
    adv = dollar_volume.rolling(21, min_periods=10).mean()
    history = price_frame.notna().cumsum() >= config.data.min_history_days
    return (
        membership
        & price_frame.notna()
        & (price_frame >= config.data.min_price)
        & (adv >= config.data.min_dollar_volume)
        & history
    )
