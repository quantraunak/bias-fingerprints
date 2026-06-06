from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

_CACHE_NAME = "sectors.json"
_VALID_SOURCES = frozenset({"yfinance"})


def _fetch_sector_yfinance(ticker: str, retries: int = 3) -> str:
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector")
            if sector:
                return str(sector)
            return "Unknown"
        except Exception as exc:
            last_err = exc
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"yfinance sector lookup failed for {ticker}: {last_err}") from last_err


def load_sector_map(tickers: list[str], cache_dir: Path, source: str = "yfinance") -> pd.Series:
    if source not in _VALID_SOURCES:
        raise ValueError(f"universe.sector_source must be one of {sorted(_VALID_SOURCES)}, got {source!r}")

    cache_path = cache_dir / "metadata" / _CACHE_NAME
    cached: dict[str, str] = {}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())

    missing = [t for t in tickers if t not in cached]
    if missing:
        for ticker in missing:
            try:
                cached[ticker] = _fetch_sector_yfinance(ticker)
            except RuntimeError:
                cached[ticker] = "Unknown"
            time.sleep(0.15)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached, indent=2, sort_keys=True))

    return pd.Series({t: cached.get(t, "Unknown") for t in tickers})
