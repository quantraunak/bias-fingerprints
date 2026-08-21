"""Industry classification from SEC SIC codes, mapped to Fama-French 12.

The previous pipeline pulled GICS sectors from yfinance, which only knows about
names that still trade. A point-in-time universe is full of companies that do
not, so half the cross-section would have fallen into an "Unknown" bucket and
sector-neutralisation would have been neutralising against noise.

SEC keeps the SIC code of every filer it has ever had, so coverage matches the
universe. The FF12 grouping is Ken French's standard partition of 4-digit SIC
into twelve industries -- coarse enough that each bucket has enough names to
demean against, and the convention the literature uses.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import requests

from src.config import RAW
from src.data.fundamentals import load_cik_map

SIC_PATH = RAW / "metadata" / "sic_codes.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
RATE_LIMIT_SECONDS = 0.12

# (upper bound, industry). Ranges are searched in order; first match wins.
FF12 = [
    ((100, 999), "NoDur"), ((2000, 2399), "NoDur"), ((2700, 2749), "NoDur"),
    ((2770, 2799), "NoDur"), ((3100, 3199), "NoDur"), ((3940, 3989), "NoDur"),
    ((2500, 2519), "Durbl"), ((2590, 2599), "Durbl"), ((3630, 3659), "Durbl"),
    ((3710, 3711), "Durbl"), ((3714, 3714), "Durbl"), ((3716, 3716), "Durbl"),
    ((3750, 3751), "Durbl"), ((3792, 3792), "Durbl"), ((3900, 3939), "Durbl"),
    ((3990, 3999), "Durbl"),
    ((2520, 2589), "Manuf"), ((2600, 2699), "Manuf"), ((2750, 2769), "Manuf"),
    ((3000, 3099), "Manuf"), ((3200, 3569), "Manuf"), ((3580, 3629), "Manuf"),
    ((3700, 3709), "Manuf"), ((3712, 3713), "Manuf"), ((3715, 3715), "Manuf"),
    ((3717, 3749), "Manuf"), ((3752, 3791), "Manuf"), ((3793, 3799), "Manuf"),
    ((3830, 3839), "Manuf"), ((3860, 3899), "Manuf"),
    ((1200, 1399), "Enrgy"), ((2900, 2999), "Enrgy"),
    ((2800, 2829), "Chems"), ((2840, 2899), "Chems"),
    ((3570, 3579), "BusEq"), ((3660, 3692), "BusEq"), ((3694, 3699), "BusEq"),
    ((3810, 3829), "BusEq"), ((7370, 7379), "BusEq"),
    ((4800, 4899), "Telcm"),
    ((4900, 4949), "Utils"),
    ((5000, 5999), "Shops"), ((7200, 7299), "Shops"), ((7600, 7699), "Shops"),
    ((2830, 2839), "Hlth"), ((3693, 3693), "Hlth"), ((3840, 3859), "Hlth"),
    ((8000, 8099), "Hlth"),
    ((6000, 6999), "Money"),
]


def sic_to_industry(sic: int | float | None) -> str:
    if sic is None or pd.isna(sic):
        return "Other"
    code = int(sic)
    for (low, high), industry in FF12:
        if low <= code <= high:
            return industry
    return "Other"


def fetch_sic_codes(tickers: list[str], user_agent: str) -> dict[str, int]:
    """Look up each ticker's SIC code from its SEC submissions record."""
    cached = json.loads(SIC_PATH.read_text()) if SIC_PATH.exists() else {}
    cik_map = load_cik_map(user_agent)
    headers = {"User-Agent": user_agent}

    for ticker in tickers:
        if ticker in cached:
            continue
        cik = cik_map.get(ticker)
        if cik is None:
            cached[ticker] = None
            continue
        time.sleep(RATE_LIMIT_SECONDS)
        response = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=headers, timeout=60)
        cached[ticker] = int(response.json().get("sic") or 0) if response.status_code == 200 else None

    SIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIC_PATH.write_text(json.dumps(cached, indent=0, sort_keys=True))
    return cached


def industry_map(tickers: list[str], user_agent: str) -> pd.Series:
    codes = fetch_sic_codes(tickers, user_agent)
    return pd.Series({t: sic_to_industry(codes.get(t)) for t in tickers}, name="industry")
