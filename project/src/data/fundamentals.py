"""Point-in-time fundamentals from SEC EDGAR XBRL company facts.

Every fact carries the date it was **filed**, which is what makes this usable
for backtesting: a factor built from a quarter ending 31 March is only visible
once the 10-Q lands in May. Joining on period end instead of filing date is the
classic look-ahead in fundamental research, and it is worth several points of
Sharpe on its own.

Two details that matter and are easy to get wrong:

* **Restatements.** A period can be reported many times as it is revised. We
  keep the *earliest* filing for each period, because that is the number the
  market actually saw.
* **The missing fourth quarter.** Companies file three 10-Qs and a 10-K; the
  10-K reports the full year, not Q4. Taking "quarterly" facts at face value
  silently drops a quarter of the flow history, so Q4 is derived as
  annual minus the three quarters inside it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import RAW

FACTS_DIR = RAW / "fundamentals"
CIK_MAP_PATH = RAW / "fundamentals" / "_cik_map.json"
CIK_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
RATE_LIMIT_SECONDS = 0.12  # SEC asks for <= 10 requests/second

# Balance-sheet items: a level at a point in time.
# Tags are listed best-first and *merged*, not chosen: filers switch tags over
# time, so taking only the first one that exists truncates the history.
STOCK_TAGS = {
    "assets": [("us-gaap", "Assets")],
    "equity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ],
    "current_assets": [("us-gaap", "AssetsCurrent")],
    "current_liabilities": [("us-gaap", "LiabilitiesCurrent")],
    "shares": [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesIssued"),
    ],
}

# Flow items: measured over a period, so they need duration handling and TTM.
FLOW_TAGS = {
    "net_income": [("us-gaap", "NetIncomeLoss")],
    "revenue": [
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "Revenues"),
        ("us-gaap", "SalesRevenueNet"),
    ],
    "gross_profit": [("us-gaap", "GrossProfit")],
    "operating_income": [("us-gaap", "OperatingIncomeLoss")],
    "operating_cash_flow": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities")],
}

FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
QUARTER_DAYS = (75, 115)


# --------------------------------------------------------------------- CIKs


def load_cik_map(user_agent: str) -> dict[str, int]:
    """Ticker -> CIK. Covers current filers; delisted names may be absent."""
    if not CIK_MAP_PATH.exists():
        CIK_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(CIK_URL, headers={"User-Agent": user_agent}, timeout=60)
        response.raise_for_status()
        CIK_MAP_PATH.write_bytes(response.content)
    raw = json.loads(CIK_MAP_PATH.read_text())
    return {
        row["ticker"].upper().replace(".", "-"): int(row["cik_str"])
        for row in raw.values()
    }


# ------------------------------------------------------------------ download


def fetch(tickers: list[str], user_agent: str, force: bool = False) -> dict:
    """Download company facts and cache the extracted long frame per ticker."""
    FACTS_DIR.mkdir(parents=True, exist_ok=True)
    cik_map = load_cik_map(user_agent)
    headers = {"User-Agent": user_agent}

    unmapped, failed, done = [], [], 0
    for ticker in tickers:
        path = FACTS_DIR / f"{ticker}.parquet"
        if path.exists() and not force:
            done += 1
            continue
        cik = cik_map.get(ticker)
        if cik is None:
            unmapped.append(ticker)
            continue

        time.sleep(RATE_LIMIT_SECONDS)
        response = requests.get(FACTS_URL.format(cik=cik), headers=headers, timeout=60)
        if response.status_code != 200:
            failed.append(ticker)
            continue

        frame = extract_facts(response.json())
        if frame.empty:
            failed.append(ticker)
            continue
        frame.to_parquet(path)
        done += 1

    return {
        "requested": len(tickers),
        "cached": done,
        "no_cik": sorted(unmapped),
        "failed": sorted(failed),
    }


# ------------------------------------------------------------------- extract


def extract_facts(payload: dict) -> pd.DataFrame:
    """Flatten company facts to columns [item, period_end, filed, value, kind]."""
    facts_by_taxonomy = payload.get("facts", {})
    rows = []

    for item, candidates in STOCK_TAGS.items():
        rows.extend(_stock_rows(item, _merge_tags(facts_by_taxonomy, candidates)))

    for item, candidates in FLOW_TAGS.items():
        rows.extend(_flow_rows(item, _merge_tags(facts_by_taxonomy, candidates)))

    if not rows:
        return pd.DataFrame(columns=["item", "period_end", "filed", "value", "kind"])

    frame = pd.DataFrame(rows)
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    frame["filed"] = pd.to_datetime(frame["filed"])
    # Earliest filing wins: that is the number the market saw first.
    frame = frame.sort_values("filed").drop_duplicates(["item", "period_end"], keep="first")
    return frame.sort_values(["item", "period_end"]).reset_index(drop=True)


def _merge_tags(facts_by_taxonomy: dict, candidates: list[tuple[str, str]]) -> list[dict]:
    """Union the candidate tags, preferring earlier ones where periods collide.

    Filers migrate between tags -- `SalesRevenueNet` gave way to
    `RevenueFromContractWithCustomerExcludingAssessedTax` when ASC 606 landed in
    2018 -- so selecting a single tag silently truncates history at the switch.
    """
    merged: dict[tuple, dict] = {}
    for rank, (taxonomy, tag) in enumerate(candidates):
        block = facts_by_taxonomy.get(taxonomy, {}).get(tag)
        if block is None:
            continue
        units = block["units"]
        unit = "USD" if "USD" in units else next(iter(units))
        for fact in units[unit]:
            key = (fact.get("start"), fact["end"])
            if key not in merged or rank < merged[key]["_rank"]:
                merged[key] = {**fact, "_rank": rank}
    return list(merged.values())


def _stock_rows(item: str, facts: list[dict]) -> list[dict]:
    return [
        {
            "item": item,
            "period_end": fact["end"],
            "filed": fact["filed"],
            "value": float(fact["val"]),
            "kind": "stock",
        }
        for fact in facts
        if fact.get("form") in FORMS and "start" not in fact
    ]


def _flow_rows(item: str, facts: list[dict]) -> list[dict]:
    """Discrete quarterly flows, recovered from whatever periods the filer reported.

    Cash-flow and income statements are usually filed year-to-date, so a filer
    publishes 3-, 6-, 9- and 12-month spans rather than four quarters, and the
    10-K reports the year instead of Q4. Both are the same problem: a long
    period that shares its start with a shorter one. Subtracting the shorter
    from the longer leaves the missing quarter, and repeating that to a fixed
    point recovers the full quarterly series.
    """
    periods: dict[tuple, dict] = {}
    for fact in facts:
        if fact.get("form") not in FORMS or "start" not in fact:
            continue
        start, end = pd.Timestamp(fact["start"]), pd.Timestamp(fact["end"])
        key = (start, end)
        filed = pd.Timestamp(fact["filed"])
        if key not in periods or filed < periods[key]["filed"]:
            periods[key] = {"start": start, "end": end, "value": float(fact["val"]), "filed": filed}

    for _ in range(6):  # four quarters plus slack; converges well inside this
        added = False
        for (start, end), long in list(periods.items()):
            if (end - start).days <= QUARTER_DAYS[1]:
                continue
            inner = [p for (s, e), p in periods.items() if s == start and e < end]
            if not inner:
                continue
            best = max(inner, key=lambda p: p["end"])
            key = (best["end"], end)
            if key in periods:
                continue
            periods[key] = {
                "start": best["end"],
                "end": end,
                "value": long["value"] - best["value"],
                # Knowable only once both filings are public.
                "filed": max(long["filed"], best["filed"]),
            }
            added = True
        if not added:
            break

    return [
        {
            "item": item,
            "period_end": period["end"],
            "filed": period["filed"],
            "value": period["value"],
            "kind": "flow",
        }
        for period in periods.values()
        if QUARTER_DAYS[0] <= (period["end"] - period["start"]).days <= QUARTER_DAYS[1]
    ]


# ------------------------------------------------------------- PIT assembly


def load_facts(tickers: list[str]) -> pd.DataFrame:
    """Long frame of every cached fact, with a ticker column."""
    frames = []
    for ticker in tickers:
        path = FACTS_DIR / f"{ticker}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame["ticker"] = ticker
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["item", "period_end", "filed", "value", "kind", "ticker"])
    return pd.concat(frames, ignore_index=True)


def as_of_panel(facts: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Wide MultiIndex (date, ticker) panel of what was known on each date.

    Flows are trailing-twelve-month sums of the four most recent quarters that
    had been filed; stocks are the latest filed level. Both are forward-filled
    from the filing date only, never from the period end.
    """
    facts = facts.sort_values("filed")
    out = []

    for item, group in facts.groupby("item", sort=False):
        kind = group["kind"].iloc[0]
        series = _ttm(group) if kind == "flow" else group[["ticker", "filed", "value"]]
        wide = (
            series.pivot_table(index="filed", columns="ticker", values="value", aggfunc="last")
            .sort_index()
            .reindex(dates.union(series["filed"].unique()))
            .ffill()
            .reindex(dates)
        )
        out.append(wide.stack(future_stack=True).rename(item))

    panel = pd.concat(out, axis=1)
    panel.index.names = ["date", "ticker"]
    return panel


def _ttm(group: pd.DataFrame) -> pd.DataFrame:
    """Trailing four filed quarters, stamped with the filing date of the last one."""
    group = group.sort_values(["ticker", "period_end"])
    group = group.drop_duplicates(["ticker", "period_end"], keep="first")
    group["value"] = (
        group.groupby("ticker")["value"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
    )
    # A rolling sum over period ends is only known once the last quarter is filed,
    # and filings can arrive out of period order, so take a running max of `filed`.
    group["filed"] = group.groupby("ticker")["filed"].cummax()
    return group.dropna(subset=["value"])[["ticker", "filed", "value"]]
