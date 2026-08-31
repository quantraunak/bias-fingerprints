"""Does correct point-in-time dating change what a factor appears to earn?

Every factor in the registry, computed three times over the same prices, the
same tradability screen and the same forward returns. The only thing that moves
is the date on which a fundamental figure becomes visible:

  PIT     the truth: visible the day the filing landed
  LAG45   common practice: assume every company files 45 days after quarter end.
          Right on average, wrong for the filings that arrive later than that.
  NAIVE   the bug: visible the instant the quarter closed, which is what a
          careless join on period end gives you.

Eleven of the twenty-two factors read no filed figure at all. They are carried
through anyway, as controls. If a price-only factor's IC moves when the filing
calendar moves, the difference is coming from the harness rather than from
look-ahead, and the whole table is void.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import Config
from src.data import fundamentals, panel as panel_module
from src.data.panel import FUNDAMENTAL_ITEMS, Panel
from src.eval import ic as ic_module
from src.factors import registry
from src.factors.transforms import standardize
from src.model.target import forward_returns

CONVENTIONS = [("filed", "PIT"), ("lag45", "LAG45"), ("period_end", "NAIVE")]
HORIZON = 21

# The factors that read a filed figure: the ten fundamentals, plus turnover,
# which divides volume by shares outstanding and so inherits the filing calendar
# through its denominator. It is the only price factor that does, and it is easy
# to miss -- which is the point of checking this rather than assuming it.
EXPECTED_SENSITIVE = {
    "book_to_market", "earnings_yield", "cash_flow_yield", "sales_to_price",
    "gross_profitability", "roe", "operating_margin", "asset_growth",
    "accruals", "net_share_issuance", "turnover_1m",
}


def redate(facts: pd.DataFrame, stamp: str) -> pd.DataFrame:
    """The same facts, moved to the date each convention claims they were visible."""
    if stamp == "filed":
        return facts
    out = facts.copy()
    if stamp == "period_end":
        out["filed"] = out["period_end"]
    elif stamp == "lag45":
        out["filed"] = out["period_end"] + pd.Timedelta(days=45)
    else:
        raise KeyError(f"Unknown dating convention: {stamp}")
    return out


def variant(base: Panel, facts: pd.DataFrame, stamp: str) -> Panel:
    """`base` with its fundamentals re-dated. Prices, screens and universe untouched.

    `tradable` is deliberately not rebuilt. It depends only on price, volume and
    index membership, so holding it fixed keeps the cross-section identical
    across conventions and leaves the dating as the only thing that differs.
    """
    tickers = list(base.prices.columns)
    wide_frame = fundamentals.as_of_panel(redate(facts, stamp), base.dates)
    wide = {
        item: wide_frame[item].unstack("ticker").reindex(columns=tickers)
        if item in wide_frame.columns
        else pd.DataFrame(np.nan, index=base.dates, columns=tickers)
        for item in FUNDAMENTAL_ITEMS
    }
    return replace(base, fundamentals=wide, market_cap=base.prices.mul(wide["shares"]))


def score_all(panel: Panel, names: list[str], forward: pd.Series, oos_start: str) -> dict[str, dict]:
    """Out-of-sample IC summary for every factor under one dating convention."""
    out = {}
    for name, frame in registry.compute_raw(panel, names).items():
        scores = standardize(frame, panel.industries).stack(future_stack=True)
        scores.index.names = ["date", "ticker"]
        oos = scores.index.get_level_values("date") >= oos_start
        out[name] = ic_module.summarize(
            ic_module.information_coefficient(scores[oos], forward), HORIZON
        )
    return out


config = Config.load("configs/default.yaml")
base = panel_module.build(config)
facts = fundamentals.load_facts(list(base.prices.columns))
names = list(registry.REGISTRY)

forward = forward_returns(base.prices.where(base.tradable), HORIZON).stack(future_stack=True)
forward.index.names = ["date", "ticker"]

print(f"scoring {len(names)} factors under {len(CONVENTIONS)} dating conventions ...", flush=True)
results: dict[str, dict[str, dict]] = {}
for stamp, label in CONVENTIONS:
    for name, summary in score_all(variant(base, facts, stamp), names, forward, config.oos_start).items():
        results.setdefault(name, {})[label] = summary
    print(f"  {label} done", flush=True)

table = pd.DataFrame(
    [
        {
            "factor": name,
            "pit_ic": r["PIT"]["mean_ic"],
            "lag45_ic": r["LAG45"]["mean_ic"],
            "naive_ic": r["NAIVE"]["mean_ic"],
            "pit_t": r["PIT"]["t_stat"],
            "lag45_t": r["LAG45"]["t_stat"],
            "naive_t": r["NAIVE"]["t_stat"],
        }
        for name, r in results.items()
    ]
).set_index("factor")

# Sensitivity is measured, not declared, so a factor that quietly picks up a
# filed input later shows up here rather than hiding among the controls.
moved = (table[["lag45_ic", "naive_ic"]].sub(table.pit_ic, axis=0).abs() > 1e-12).any(axis=1)
measured_sensitive = set(table.index[moved])

header = f"{'factor':<22}{'PIT IC':>9}{'LAG45':>9}{'NAIVE':>9}   {'PIT t':>7}{'LAG45 t':>8}{'NAIVE t':>8}"


def block(title: str, subset: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(header)
    for name, r in subset.iterrows():
        print(
            f"{name:<22}{r.pit_ic:>9.5f}{r.lag45_ic:>9.5f}{r.naive_ic:>9.5f}   "
            f"{r.pit_t:>7.2f}{r.lag45_t:>8.2f}{r.naive_t:>8.2f}"
        )


block(f"Reads a filed figure ({moved.sum()} factors)", table[moved])
block(f"Price and volume only -- controls ({(~moved).sum()} factors)", table[~moved])

if measured_sensitive != EXPECTED_SENSITIVE:
    print("\nWARNING: dating sensitivity is not where it was expected.")
    print(f"  moved but should not have: {sorted(measured_sensitive - EXPECTED_SENSITIVE)}")
    print(f"  should have moved but did not: {sorted(EXPECTED_SENSITIVE - measured_sensitive)}")
else:
    drift = table[~moved][["lag45_ic", "naive_ic"]].sub(table.pit_ic, axis=0).abs().max().max()
    print(f"\nControls held: {(~moved).sum()} price-only factors, max |IC drift| {drift:.1e}.")

# Summarised over the affected factors only. Averaging in eleven controls that
# cannot move by construction would halve the reported inflation for free.
affected = table[moved]
print()
for label, ic_col, t_col in [
    ("fixed 45-day lag", "lag45_ic", "lag45_t"),
    ("period-end join", "naive_ic", "naive_t"),
]:
    inflation = affected[ic_col] - affected.pit_ic
    crossed = affected.index[(affected.pit_t < 2) & (affected[t_col] >= 2)]
    print(label)
    print(
        f"   mean IC inflation vs PIT        {inflation.mean():+.5f}"
        f"   ({inflation.mean() / affected.pit_ic.abs().mean():.0%} of true IC)"
    )
    print(f"   factors that look better        {(inflation > 0).sum()} of {len(affected)}")
    print(f"   factors crossing t=2 spuriously {len(crossed)} of {len(affected)}")
    if len(crossed):
        print(f"      {', '.join(crossed)}")

table.to_csv("reports/pit_vs_naive.csv")
print("\nwrote reports/pit_vs_naive.csv")
