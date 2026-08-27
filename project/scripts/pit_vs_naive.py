"""Does correct point-in-time dating change what a factor appears to earn?

Two versions of every fundamental factor:
  PIT    - each figure visible only from the day it was actually FILED
  NAIVE  - each figure visible from the day the QUARTER ENDED (the common
           shortcut, and what you get if you join Compustat carelessly)

Everything else is identical. The difference is pure look-ahead.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import Config
from src.data import panel as panel_module, fundamentals
from src.factors import fundamental as ff
from src.factors.transforms import standardize
from src.eval import ic as ic_module
from src.model.target import forward_returns

config = Config.load("configs/default.yaml")
panel = panel_module.build(config)
dates = panel.prices.index
tickers = list(panel.prices.columns)

facts = fundamentals.load_facts(tickers)

def as_of(stamp: str) -> dict[str, pd.DataFrame]:
    """Fundamental panel under one dating convention.

    filed      - the truth: visible the day the filing landed
    period_end - the worst case: visible the instant the quarter closed
    lag45      - common practice: assume every company files 45 days after
                 quarter end. Right on average, wrong for the ~18% of
                 filings that arrive later than that.
    """
    f = facts.copy()
    if stamp == "period_end":
        f["filed"] = f["period_end"]
    elif stamp == "lag45":
        f["filed"] = f["period_end"] + pd.Timedelta(days=45)
    wide = fundamentals.as_of_panel(f, dates)
    return {
        item: wide[item].unstack("ticker").reindex(columns=tickers)
        if item in wide.columns
        else pd.DataFrame(np.nan, index=dates, columns=tickers)
        for item in ["assets", "equity", "shares", "net_income", "revenue",
                     "gross_profit", "operating_income", "operating_cash_flow"]
    }

def factor_set(w: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    mcap = panel.prices.mul(w["shares"])
    return {
        "book_to_market":      ff.book_to_market(w["equity"], mcap),
        "earnings_yield":      ff.earnings_yield(w["net_income"], mcap),
        "cash_flow_yield":     ff.cash_flow_yield(w["operating_cash_flow"], mcap),
        "sales_to_price":      ff.sales_to_price(w["revenue"], mcap),
        "gross_profitability": ff.gross_profitability(w["gross_profit"], w["assets"]),
        "roe":                 ff.roe(w["net_income"], w["equity"]),
        "operating_margin":    ff.operating_margin(w["operating_income"], w["revenue"]),
        "asset_growth":        ff.asset_growth(w["assets"]),
        "accruals":            ff.accruals(w["net_income"], w["operating_cash_flow"], w["assets"]),
        "net_share_issuance":  ff.net_share_issuance(w["shares"]),
    }

fwd = forward_returns(panel.prices.where(panel.tradable), 21).stack(future_stack=True)
fwd.index.names = ["date", "ticker"]

print("building both panels ...", flush=True)
results = {}
for stamp, label in [("filed", "PIT"), ("lag45", "LAG45"), ("period_end", "NAIVE")]:
    w = as_of(stamp)
    facs = factor_set(w)
    for name, frame in facs.items():
        std = standardize(frame.where(panel.tradable), panel.industries)
        s = std.stack(future_stack=True)
        s.index.names = ["date", "ticker"]
        oos = s.index.get_level_values("date") >= config.oos_start
        summ = ic_module.summarize(ic_module.information_coefficient(s[oos], fwd), 21)
        results.setdefault(name, {})[label] = summ
    print(f"  {label} done", flush=True)

hdr = f"{'factor':<22}{'PIT IC':>9}{'LAG45':>9}{'NAIVE':>9}   {'PIT t':>7}{'LAG45 t':>8}{'NAIVE t':>8}"
print("\n" + hdr)
rows = []
for name, r in results.items():
    p_, l_, n_ = r["PIT"], r["LAG45"], r["NAIVE"]
    rows.append((name, p_["mean_ic"], l_["mean_ic"], n_["mean_ic"],
                 p_["t_stat"], l_["t_stat"], n_["t_stat"]))
    print(f"{name:<22}{p_['mean_ic']:>9.5f}{l_['mean_ic']:>9.5f}{n_['mean_ic']:>9.5f}   "
          f"{p_['t_stat']:>7.2f}{l_['t_stat']:>8.2f}{n_['t_stat']:>8.2f}")

d = pd.DataFrame(rows, columns=["factor","pit_ic","lag45_ic","naive_ic","pit_t","lag45_t","naive_t"])
print()
for lbl, ic, t in [("fixed 45-day lag", "lag45_ic", "lag45_t"), ("period-end join", "naive_ic", "naive_t")]:
    infl = d[ic] - d.pit_ic
    crossed = ((d.pit_t < 2) & (d[t] >= 2)).sum()
    print(f"{lbl}")
    print(f"   mean IC inflation vs PIT      {infl.mean():+.5f}"
          f"   ({infl.abs().mean()/d.pit_ic.abs().mean():.0%} of true IC)")
    print(f"   factors that look better      {(infl > 0).sum()} of {len(d)}")
    print(f"   factors crossing t=2 spuriously {crossed} of {len(d)}")
d.to_csv("reports/pit_vs_naive.csv", index=False)
