from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.types import FetchResult

_VALID_SOURCES = frozenset({"yfinance", "stooq"})


@dataclass(frozen=True)
class FetchConfig:
    source: str
    start: str
    end: str
    force_refresh: bool = False
    min_history_days: int = 252
    min_tickers: int = 80
    min_success_ratio: float = 0.75
    retries: int = 3


@dataclass(frozen=True)
class PriceData:
    prices: pd.DataFrame
    volumes: pd.DataFrame
    fetch_result: FetchResult


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / "prices" / f"{ticker}.parquet"


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "prices" / "_manifest.json"


def _load_manifest(cache_dir: Path) -> dict:
    p = _manifest_path(cache_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def _save_manifest(cache_dir: Path, manifest: dict) -> None:
    p = _manifest_path(cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _assert_cache_source(cache_dir: Path, cfg: FetchConfig) -> None:
    manifest = _load_manifest(cache_dir)
    if not manifest or cfg.force_refresh:
        return
    sources = {meta.get("source") for meta in manifest.values() if meta.get("source")}
    if sources and sources != {cfg.source}:
        raise RuntimeError(
            f"Price cache sources {sorted(sources)} do not match data.price_source={cfg.source!r}. "
            "Set data.force_refresh: true or delete project/data/raw/prices/."
        )


def _load_cached(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("empty cache file")
    return df


def _download_yfinance_batch(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    out: dict[str, pd.DataFrame] = {}
    if len(tickers) == 1:
        t = tickers[0]
        frame = pd.DataFrame({"adj_close": raw["Close"], "volume": raw.get("Volume", 0.0)})
        frame.index = pd.to_datetime(frame.index)
        frame.index.name = "date"
        return {t: frame.dropna(subset=["adj_close"])}

    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        sub = raw[t]
        frame = pd.DataFrame({"adj_close": sub["Close"], "volume": sub.get("Volume", 0.0)})
        frame.index = pd.to_datetime(frame.index)
        frame.index.name = "date"
        if not frame.empty:
            out[t] = frame.dropna(subset=["adj_close"])
    return out


def _download_stooq(ticker: str, start: str, end: str, retries: int) -> pd.DataFrame:
    from pandas_datareader import data as pdr

    symbols = [f"{ticker}.US"]
    if "-" in ticker:
        symbols.append(f"{ticker.replace('-', '.')}.US")

    last_err: Exception | None = None
    for sym in symbols:
        for attempt in range(retries):
            try:
                raw = pdr.DataReader(sym, "stooq", start, end).sort_index()
                if raw is None or raw.empty:
                    continue
                out = pd.DataFrame(
                    {"adj_close": raw["Close"], "volume": raw.get("Volume", 0.0)},
                    index=pd.to_datetime(raw.index),
                )
                out.index.name = "date"
                return out.dropna(subset=["adj_close"])
            except Exception as exc:
                last_err = exc
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"stooq failed for {ticker}: {last_err}") from last_err


def get_price_data(tickers: Iterable[str], cfg: FetchConfig, cache_dir: Path) -> PriceData:
    if cfg.source not in _VALID_SOURCES:
        raise ValueError(f"data.price_source must be one of {sorted(_VALID_SOURCES)}, got {cfg.source!r}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "prices").mkdir(parents=True, exist_ok=True)
    tickers = sorted({t.upper().strip() for t in tickers if t})

    _assert_cache_source(cache_dir, cfg)
    manifest = _load_manifest(cache_dir)
    frames: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}
    to_fetch: list[str] = []

    for ticker in tickers:
        path = _cache_path(cache_dir, ticker)
        use_cache = (
            path.exists()
            and not cfg.force_refresh
            and manifest.get(ticker, {}).get("source") == cfg.source
        )
        if use_cache:
            frames[ticker] = _load_cached(path)
            continue
        to_fetch.append(ticker)

    if to_fetch and cfg.source == "yfinance":
        for i in range(0, len(to_fetch), 50):
            batch = to_fetch[i : i + 50]
            batch_frames = _download_yfinance_batch(batch, cfg.start, cfg.end)
            for t, df in batch_frames.items():
                df.to_parquet(_cache_path(cache_dir, t))
                manifest[t] = {
                    "source": cfg.source,
                    "start": cfg.start,
                    "end": cfg.end,
                    "rows": int(len(df)),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                frames[t] = df
            for t in batch:
                if t not in batch_frames:
                    failed[t] = "no data returned"

    elif to_fetch:
        for ticker in to_fetch:
            try:
                df = _download_stooq(ticker, cfg.start, cfg.end, cfg.retries)
                df.to_parquet(_cache_path(cache_dir, ticker))
                manifest[ticker] = {
                    "source": cfg.source,
                    "start": cfg.start,
                    "end": cfg.end,
                    "rows": int(len(df)),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                frames[ticker] = df
            except Exception as exc:
                failed[ticker] = str(exc)

    _save_manifest(cache_dir, manifest)

    if not frames:
        raise RuntimeError("No price data loaded.")

    prices = pd.concat([f["adj_close"].rename(t) for t, f in frames.items()], axis=1).sort_index()
    volumes = pd.concat([f["volume"].rename(t) for t, f in frames.items()], axis=1).sort_index()

    counts = prices.count()
    short = counts[counts < cfg.min_history_days].index.tolist()
    if short:
        prices = prices.drop(columns=short)
        volumes = volumes.drop(columns=[c for c in short if c in volumes.columns])

    n_loaded = prices.shape[1]
    n_requested = len(tickers)
    ratio = n_loaded / n_requested if n_requested else 0.0

    if n_loaded < cfg.min_tickers:
        raise RuntimeError(
            f"Only {n_loaded}/{n_requested} tickers passed filters (min_tickers={cfg.min_tickers}). "
            f"Failed={len(failed)}, dropped_short_history={len(short)}."
        )
    if ratio < cfg.min_success_ratio:
        raise RuntimeError(
            f"Success ratio {ratio:.1%} below min_success_ratio={cfg.min_success_ratio:.0%}. "
            f"price_source={cfg.source}, failed={len(failed)}."
        )

    return PriceData(
        prices=prices,
        volumes=volumes,
        fetch_result=FetchResult(
            n_requested=n_requested,
            n_loaded=n_loaded,
            n_failed=len(failed),
            n_dropped_short_history=len(short),
            price_source=cfg.source,
            failed_tickers=failed,
        ),
    )
