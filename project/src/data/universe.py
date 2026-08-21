"""Point-in-time S&P 500 membership.

The old pipeline used a snapshot of *current* index membership for the whole
history, so every company that was ever deleted -- acquired, bankrupted,
demoted -- was invisible. That biases every result upward.

Here membership is a set of (ticker, start, end) spells, so a name enters the
cross-section on the day it joined the index and leaves on the day it left.
Tickers with more than one spell are handled: several names have been added,
removed and re-added.

Source: github.com/fja05680/sp500, which reconstructs membership from the
revision history of the Wikipedia constituents page. It is the best free
approximation of a CRSP/Norgate point-in-time file. `validate_against_live`
diffs the final date against the live Wikipedia table so drift is visible
rather than assumed away.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.config import RAW

SPELLS_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SPELLS_PATH = RAW / "universe" / "sp500_spells.csv"


def download_spells(path: Path = SPELLS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(SPELLS_URL, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def load_spells(path: Path = SPELLS_PATH) -> pd.DataFrame:
    """(ticker, start, end) membership spells. `end` is NaT while still a member."""
    if not path.exists():
        download_spells(path)
    spells = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    spells = spells.rename(columns={"start_date": "start", "end_date": "end"})
    spells["ticker"] = spells["ticker"].str.upper().str.replace(".", "-", regex=False)
    return spells.sort_values(["ticker", "start"]).reset_index(drop=True)


def tickers_in_window(spells: pd.DataFrame, start: str, end: str) -> list[str]:
    """Every ticker that was an index member at any point in [start, end]."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    overlaps = (spells["start"] <= hi) & (spells["end"].isna() | (spells["end"] >= lo))
    return sorted(spells.loc[overlaps, "ticker"].unique())


def membership_matrix(spells: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean date x ticker frame: True where the name was in the index that day."""
    tickers = sorted(spells["ticker"].unique())
    matrix = pd.DataFrame(False, index=dates, columns=tickers)
    for ticker, start, end in spells[["ticker", "start", "end"]].itertuples(index=False):
        hi = dates.max() if pd.isna(end) else end
        matrix.loc[(dates >= start) & (dates <= hi), ticker] = True
    return matrix


def active_on(membership: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    """Index members as of `date`, using the last known row at or before it."""
    rows = membership.loc[:date]
    if rows.empty:
        return []
    last = rows.iloc[-1]
    return last.index[last].tolist()


def validate_against_live(membership: pd.DataFrame) -> dict:
    """Diff the final membership row against the live Wikipedia constituents table."""
    tables = pd.read_html(requests.get(WIKI_URL, timeout=60, headers={"User-Agent": "research"}).text)
    live = set(tables[0]["Symbol"].str.upper().str.replace(".", "-", regex=False))
    reconstructed = set(active_on(membership, membership.index.max()))
    return {
        "n_live": len(live),
        "n_reconstructed": len(reconstructed),
        "missing_from_reconstruction": sorted(live - reconstructed),
        "extra_in_reconstruction": sorted(reconstructed - live),
    }
