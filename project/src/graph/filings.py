"""Fetch 10-K documents from EDGAR and cache their text.

The XBRL company-facts API used by `src/data/fundamentals.py` returns numbers
only. Economic links -- who a company sells to, who it buys from -- live in the
prose of Item 1, which means fetching the filing document itself.

Two things make this point-in-time safe:

* **Filing date, not period end.** Every document is stored under the date EDGAR
  received it. A relationship disclosed in a 10-K covering fiscal 2015 is not
  usable until the filing lands, typically 50-90 days later.
* **The document as filed.** EDGAR keeps the original submission, so no later
  amendment silently rewrites what the market could see at the time. 10-K/A
  amendments are fetched as separate documents with their own filing dates
  rather than folded into the original.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import RAW
from src.data.fundamentals import load_cik_map

FILINGS_DIR = RAW / "filings"
INDEX_PATH = FILINGS_DIR / "_index.parquet"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
RATE_LIMIT_SECONDS = 0.12  # SEC asks for <= 10 requests/second
FORMS = {"10-K", "10-K/A"}


@dataclass(frozen=True)
class Filing:
    ticker: str
    cik: int
    accession: str
    form: str
    filed: pd.Timestamp
    period: pd.Timestamp | None
    document: str

    @property
    def path(self) -> Path:
        return FILINGS_DIR / self.ticker / f"{self.accession}.txt"


def list_filings(ticker: str, cik: int, user_agent: str, start: str, end: str) -> list[Filing]:
    """Every 10-K EDGAR holds for one issuer, from the submissions index.

    `recent` covers roughly the last thousand filings; older ones live in
    paginated overflow files. Large-cap issuers with long histories exceed the
    recent window, so the overflow pages have to be followed or the early years
    silently disappear.
    """
    time.sleep(RATE_LIMIT_SECONDS)
    response = requests.get(
        SUBMISSIONS_URL.format(cik=cik), headers={"User-Agent": user_agent}, timeout=60
    )
    if response.status_code != 200:
        return []
    payload = response.json()

    blocks = [payload.get("filings", {}).get("recent", {})]
    for extra in payload.get("filings", {}).get("files", []):
        time.sleep(RATE_LIMIT_SECONDS)
        page = requests.get(
            f"https://data.sec.gov/submissions/{extra['name']}",
            headers={"User-Agent": user_agent},
            timeout=60,
        )
        if page.status_code == 200:
            blocks.append(page.json())

    out: list[Filing] = []
    for block in blocks:
        forms = block.get("form", [])
        for i, form in enumerate(forms):
            if form not in FORMS:
                continue
            filed = pd.Timestamp(block["filingDate"][i])
            if not (pd.Timestamp(start) <= filed <= pd.Timestamp(end)):
                continue
            period_raw = block.get("reportDate", [None] * len(forms))[i]
            out.append(
                Filing(
                    ticker=ticker,
                    cik=cik,
                    accession=block["accessionNumber"][i].replace("-", ""),
                    form=form,
                    filed=filed,
                    period=pd.Timestamp(period_raw) if period_raw else None,
                    document=block["primaryDocument"][i],
                )
            )
    return sorted(out, key=lambda f: f.filed)


def fetch_text(filing: Filing, user_agent: str, force: bool = False) -> str | None:
    """Download one filing and cache its plain text. Returns None on failure."""
    if filing.path.exists() and not force:
        return filing.path.read_text(encoding="utf-8", errors="ignore")

    time.sleep(RATE_LIMIT_SECONDS)
    url = ARCHIVE_URL.format(cik=filing.cik, accession=filing.accession, document=filing.document)
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=120)
    if response.status_code != 200:
        return None

    text = to_text(response.text)
    if len(text) < 5000:  # a stub or an exhibit rather than the filing body
        return None

    filing.path.parent.mkdir(parents=True, exist_ok=True)
    filing.path.write_text(text, encoding="utf-8")
    return text


HEADING_TABLE_CHARS = 200


def to_text(html: str) -> str:
    """Strip a filing to readable text, keeping paragraph structure.

    Data tables carry most of a 10-K's bulk and almost none of its relationship
    language, and they shred into meaningless token soup when flattened, so they
    are dropped rather than passed to a model that has to ignore them.

    *Short* tables are kept, because a good number of filers lay their section
    headings out in one-row tables -- Caterpillar's "Item 1. Business" is a table
    cell, not a paragraph. Dropping every table erased the headings for those
    filers and left the section splitter with nothing to anchor on.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for table in soup.find_all("table"):
        if len(table.get_text(" ", strip=True)) > HEADING_TABLE_CHARS:
            table.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def build_index(
    tickers: list[str], user_agent: str, start: str, end: str, force: bool = False
) -> pd.DataFrame:
    """Download every 10-K for `tickers` and return the index of what landed."""
    cik_map = load_cik_map(user_agent)
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    for ticker in tickers:
        cik = cik_map.get(ticker)
        if cik is None:
            missing.append(ticker)
            continue
        for filing in list_filings(ticker, cik, user_agent, start, end):
            text = fetch_text(filing, user_agent, force=force)
            if text is None:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "accession": filing.accession,
                    "form": filing.form,
                    "filed": filing.filed,
                    "period": filing.period,
                    "chars": len(text),
                    "path": str(filing.path),
                }
            )

    index = pd.DataFrame(rows)
    if not index.empty:
        index = index.sort_values(["ticker", "filed"]).reset_index(drop=True)
        index.to_parquet(INDEX_PATH)
    index.attrs["no_cik"] = sorted(missing)
    return index


def load_index() -> pd.DataFrame:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"No filing index at {INDEX_PATH}; run scripts.build_graph --stage filings")
    return pd.read_parquet(INDEX_PATH)
