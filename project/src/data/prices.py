"""Daily split/dividend-adjusted prices and volume, cached one parquet per ticker.

The cache is keyed on ticker, not on the universe, so adding names later does
not invalidate what is already downloaded. Delisted tickers are expected to
fail; `fetch` records them instead of raising, because a point-in-time universe
necessarily asks for names that no longer trade, and the coverage report is
part of the result rather than an error condition.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import RAW

PRICE_DIR = RAW / "prices"
BATCH = 40


def _cache_path(ticker: str) -> Path:
    return PRICE_DIR / f"{ticker}.parquet"


def cached_tickers() -> set[str]:
    return {p.stem for p in PRICE_DIR.glob("*.parquet")}


def fetch(tickers: list[str], start: str, end: str, force: bool = False) -> dict:
    """Download missing tickers into the cache. Returns a coverage summary."""
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    have = set() if force else cached_tickers()
    todo = [t for t in tickers if t not in have]
    failed: list[str] = []

    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        frame = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )
        for ticker in batch:
            panel = _extract(frame, ticker)
            if panel is None or panel["adj_close"].notna().sum() < 60:
                failed.append(ticker)
                continue
            panel.to_parquet(_cache_path(ticker))

    return {
        "requested": len(tickers),
        "downloaded": len(todo) - len(failed),
        "already_cached": len(tickers) - len(todo),
        "failed": sorted(failed),
    }


def _extract(frame: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Pull one ticker out of a yfinance multi- or single-ticker frame."""
    if isinstance(frame.columns, pd.MultiIndex):
        if ("Close", ticker) not in frame.columns:
            return None
        close = frame[("Close", ticker)]
        volume = frame[("Volume", ticker)]
    else:
        close, volume = frame["Close"], frame["Volume"]

    panel = pd.DataFrame({"adj_close": close, "volume": volume}).dropna(how="all")
    if panel.empty:
        return None
    panel.index = pd.to_datetime(panel.index).tz_localize(None)
    panel.index.name = "date"
    return panel


def load_panel(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble cached tickers into aligned (prices, volumes) date x ticker frames."""
    closes, volumes = {}, {}
    for ticker in tickers:
        path = _cache_path(ticker)
        if not path.exists():
            continue
        panel = pd.read_parquet(path)
        closes[ticker] = panel["adj_close"]
        volumes[ticker] = panel["volume"]

    prices = pd.DataFrame(closes).sort_index().loc[start:end]
    volume = pd.DataFrame(volumes).sort_index().reindex(prices.index)
    prices.index.name = volume.index.name = "date"
    return prices, volume
