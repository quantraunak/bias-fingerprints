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
    """Build the fundamental panel keyed on `stamp` ('filed' or 'period_end')."""
    f = facts.copy()
    if stamp == "period_end":
        f["filed"] = f["period_end"]          # pretend it was public immediately
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
for stamp, label in [("filed", "PIT"), ("period_end", "NAIVE")]:
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

print(f"\n{'factor':<22}{'PIT IC':>10}{'NAIVE IC':>11}{'inflation':>11}{'PIT t':>8}{'NAIVE t':>9}")
rows = []
for name, r in results.items():
    p, n = r["PIT"], r["NAIVE"]
    infl = n["mean_ic"] - p["mean_ic"]
    rows.append((name, p["mean_ic"], n["mean_ic"], infl, p["t_stat"], n["t_stat"]))
    print(f"{name:<22}{p['mean_ic']:>10.5f}{n['mean_ic']:>11.5f}{infl:>+11.5f}"
          f"{p['t_stat']:>8.2f}{n['t_stat']:>9.2f}")

d = pd.DataFrame(rows, columns=["factor", "pit_ic", "naive_ic", "inflation", "pit_t", "naive_t"])
print(f"\nmean inflation from ignoring filing dates: {d.inflation.mean():+.5f}")
print(f"factors where naive looks better          : {(d.inflation > 0).sum()} of {len(d)}")
print(f"mean |inflation| relative to |PIT IC|     : {(d.inflation.abs().mean() / d.pit_ic.abs().mean()):.1%}")
d.to_csv("reports/pit_vs_naive.csv", index=False)
