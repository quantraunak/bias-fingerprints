"""Export the point-in-time layer as a redistributable dataset.

SEC EDGAR XBRL is public domain and the membership spells are reconstructed from
a public wiki history, so both can be shared. Prices are not included: they come
from Yahoo via yfinance, whose terms do not permit redistribution. Users fetch
their own with `make data`.

What makes this worth downloading rather than rebuilding is the third column set:
every fact carries the date it was actually filed, restatements are resolved to
the earliest filing, cumulative cash-flow spans are differenced to true quarters,
and rows whose filing date is an XBRL-adoption artifact are flagged rather than
silently trusted.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import Config
from src.data import fundamentals, universe

OUT = Path("dist")
# The XBRL mandate phased in over 2009-2011. A filing that reports a period
# ending long before it was filed is backfilled history, not a late filer: the
# figure was already public through a pre-XBRL paper filing this feed cannot see.
BACKFILL_LAG_DAYS = 180


def main() -> None:
    config = Config.load("configs/default.yaml")
    spells = universe.load_spells()
    tickers = universe.tickers_in_window(spells, config.data.start, config.data.end)

    facts = fundamentals.load_facts(tickers).copy()
    facts["lag_days"] = (facts["filed"] - facts["period_end"]).dt.days
    facts["backfilled"] = facts["lag_days"] > BACKFILL_LAG_DAYS
    # Cover-page share counts are tagged with the date the cover was dated, which
    # can fall after the fiscal period end. Harmless -- the panel keys on `filed`,
    # so the figure still becomes visible when it was filed -- but a handful are
    # mis-tagged outright, so the rows are flagged rather than quietly shipped.
    facts["cover_page_dated"] = facts["lag_days"] < 0
    facts = facts.sort_values(["ticker", "item", "period_end", "filed"])
    facts = facts[
        ["ticker", "item", "kind", "period_end", "filed", "lag_days",
         "backfilled", "cover_page_dated", "value"]
    ]

    OUT.mkdir(exist_ok=True)
    facts.to_csv(OUT / "pit_fundamentals.csv.gz", index=False, compression="gzip")
    spells.to_csv(OUT / "sp500_membership_spells.csv", index=False)

    usable = facts[~facts.backfilled & ~facts.cover_page_dated]
    print(f"pit_fundamentals.csv.gz   {len(facts):,} facts · {facts.ticker.nunique()} issuers")
    print(f"   flagged as backfilled  {facts.backfilled.sum():,} ({facts.backfilled.mean():.1%})")
    print(f"   usable filing lag      median {usable.lag_days.median():.0f}d · "
          f"p90 {usable.lag_days.quantile(0.9):.0f}d")
    print(f"   flagged cover-page     {facts.cover_page_dated.sum():,} ({facts.cover_page_dated.mean():.2%})")
    print(f"   period coverage        {usable.period_end.min():%Y-%m} to {usable.period_end.max():%Y-%m}")
    print(f"sp500_membership_spells.csv   {len(spells):,} spells · {spells.ticker.nunique()} tickers")


if __name__ == "__main__":
    main()
