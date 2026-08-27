"""Day-one test: are late filers systematically different companies?

If filing lag is essentially random, a fixed-lag convention introduces noise
and the study is a footnote. If lag correlates with size, profitability or
distress, then look-ahead bias concentrates in exactly the factors people
care about, and the convention manufactures alpha where it hurts most.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import Config
from src.data import panel as panel_module, fundamentals

config = Config.load("configs/default.yaml")
panel = panel_module.build(config)

facts = fundamentals.load_facts([c for c in panel.prices.columns])
facts = facts[facts["kind"].isin(["stock", "flow"])].copy()
facts["lag"] = (facts["filed"] - facts["period_end"]).dt.days
facts = facts[(facts["lag"] >= 0) & (facts["lag"] <= 400)]

# One lag per company-quarter: when the whole quarter became public.
quarter = (
    facts.groupby(["ticker", "period_end"])["lag"].max().rename("lag").reset_index()
)
quarter = quarter[quarter["period_end"] >= "2010-01-01"]
print(f"company-quarters: {len(quarter):,}   tickers: {quarter.ticker.nunique()}")
print(f"lag: median {quarter.lag.median():.0f}d   >45d {100*(quarter.lag>45).mean():.1f}%"
      f"   >90d {100*(quarter.lag>90).mean():.1f}%\n")

# Characteristics as of the period end, using only what was known by then.
mcap = panel.market_cap
equity = panel.fundamentals["equity"]
assets = panel.fundamentals["assets"]
income = panel.fundamentals["net_income"]
ret12 = panel.prices / panel.prices.shift(252) - 1.0
vol = panel.returns.rolling(252, min_periods=120).std() * np.sqrt(252)

def lookup(frame: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """Value of `frame` for each (ticker, period_end), forward-filled."""
    ff = frame.ffill()
    idx = ff.index.searchsorted(rows["period_end"].values, side="right") - 1
    ok = (idx >= 0) & rows["ticker"].isin(ff.columns).values
    out = np.full(len(rows), np.nan)
    cols = {c: i for i, c in enumerate(ff.columns)}
    arr = ff.to_numpy()
    for n, (good, r, t) in enumerate(zip(ok, idx, rows["ticker"].values)):
        if good:
            out[n] = arr[r, cols[t]]
    return out

q = quarter.copy()
q["mcap"] = lookup(mcap, q)
q["equity"] = lookup(equity, q)
q["assets"] = lookup(assets, q)
q["income"] = lookup(income, q)
q["ret12"] = lookup(ret12, q)
q["vol"] = lookup(vol, q)

q["log_size"] = np.log(q["mcap"].where(q["mcap"] > 0))
q["book_to_market"] = q["equity"].where(q["equity"] > 0) / q["mcap"]
q["roa"] = q["income"] / q["assets"].where(q["assets"] > 0)
q = q.replace([np.inf, -np.inf], np.nan)

q["late"] = q["lag"] > 45

print("LATE FILERS (>45 days) vs ON-TIME, median characteristics\n")
print(f"  {'characteristic':<22}{'on-time':>12}{'late':>12}{'difference':>13}")
rows = [
    ("log market cap", "log_size"),
    ("book-to-market", "book_to_market"),
    ("return on assets", "roa"),
    ("past 12m return", "ret12"),
    ("annualised vol", "vol"),
]
for label, col in rows:
    a = q.loc[~q["late"], col].median()
    b = q.loc[q["late"], col].median()
    print(f"  {label:<22}{a:>12.4f}{b:>12.4f}{b - a:>13.4f}")

print("\nRANK CORRELATION of filing lag with each characteristic")
for label, col in rows:
    sub = q[["lag", col]].dropna()
    r = sub["lag"].corr(sub[col], method="spearman")
    print(f"  {label:<22}{r:>+8.4f}   n={len(sub):,}")

print("\nLAG DECILES (10 = slowest filers)")
q["decile"] = pd.qcut(q["lag"].rank(method="first"), 10, labels=False) + 1
tbl = q.groupby("decile").agg(
    lag_days=("lag", "median"),
    log_size=("log_size", "median"),
    book_to_market=("book_to_market", "median"),
    roa=("roa", "median"),
    n=("lag", "size"),
)
print(tbl.round(4).to_string())
