"""Candidate universe: industries where a few OEMs structurally dominate demand.

The pilot showed named counterparties appear where customer concentration is
material, not across the large-cap cross-section. Turning that into a universe
needs care, because the obvious move -- listing the firms already known to name
big customers -- selects on the outcome and would guarantee a positive result.

So the universe is defined by **SIC industry**, which is assigned by the filer at
registration and is knowable before any filing is read. Contract manufacturers,
component makers and aerospace and automotive suppliers sell into a handful of
OEMs by the structure of their industry, not because of anything measured here.
Whether any individual firm actually names a customer is left to the data.

Membership is then point-in-time in the same way the index universe is: a firm
is in the cross-section on a date if it had filed by then and still had a price.
"""

from __future__ import annotations

import json
import re
import time

import pandas as pd
import requests

from src.config import RAW

CANDIDATES_PATH = RAW / "filings" / "_supplier_universe.json"
BROWSE_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&SIC={sic}"
    "&type=10-K&dateb=&owner=include&count=100&start={start}&output=atom"
)
RATE_LIMIT_SECONDS = 0.15

# Industries whose customer base is a small number of large OEMs.
SIC_CODES = {
    "3559": "special industry machinery (semiconductor equipment)",
    "3661": "telephone and telegraph apparatus",
    "3663": "radio and tv broadcasting and communications equipment",
    "3672": "printed circuit boards",
    "3674": "semiconductors and related devices",
    "3675": "electronic capacitors",
    "3677": "electronic coils and transformers",
    "3678": "electronic connectors",
    "3679": "electronic components",
    "3714": "motor vehicle parts and accessories",
    "3728": "aircraft parts and auxiliary equipment",
    "3812": "search, detection, navigation and guidance systems",
    "3825": "instruments for measuring and testing electricity",
    "3827": "laboratory analytical instruments",
}


def _fetch_ciks(sic: str, user_agent: str, max_pages: int = 6) -> list[str]:
    """CIKs filing 10-Ks under one SIC code, following pagination."""
    found: list[str] = []
    for page in range(max_pages):
        time.sleep(RATE_LIMIT_SECONDS)
        try:
            response = requests.get(
                BROWSE_URL.format(sic=sic, start=page * 100),
                headers={"User-Agent": user_agent},
                timeout=60,
            )
        except requests.exceptions.RequestException:
            break
        if response.status_code != 200:
            break
        page_ciks = list(dict.fromkeys(re.findall(r"CIK=(\d{7,10})", response.text)))
        new = [c for c in page_ciks if c not in found]
        found.extend(new)
        if len(page_ciks) < 100:
            break
    return found


def build_candidates(user_agent: str, force: bool = False) -> pd.DataFrame:
    """Enumerate the candidate universe and map it onto tradable tickers."""
    if CANDIDATES_PATH.exists() and not force:
        return pd.DataFrame(json.loads(CANDIDATES_PATH.read_text()))

    time.sleep(RATE_LIMIT_SECONDS)
    listing = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": user_agent},
        timeout=60,
    ).json()
    by_cik: dict[int, dict] = {}
    for entry in listing.values():
        cik = int(entry["cik_str"])
        ticker = entry["ticker"].upper()
        # Prefer the shortest ticker per CIK: the common line over its classes.
        if cik not in by_cik or len(ticker) < len(by_cik[cik]["ticker"]):
            by_cik[cik] = {"ticker": ticker, "title": entry["title"]}

    rows = []
    for sic, description in SIC_CODES.items():
        ciks = _fetch_ciks(sic, user_agent)
        matched = 0
        for raw in ciks:
            cik = int(raw)
            entry = by_cik.get(cik)
            if entry is None:  # deregistered or never listed
                continue
            rows.append({"cik": cik, "ticker": entry["ticker"], "title": entry["title"],
                         "sic": sic, "industry": description})
            matched += 1
        print(f"  SIC {sic}  {len(ciks):>4} filers  {matched:>4} listed  {description}")

    frame = pd.DataFrame(rows).drop_duplicates("cik").reset_index(drop=True)
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES_PATH.write_text(json.dumps(frame.to_dict("records"), indent=1))
    return frame
