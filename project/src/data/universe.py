from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.types import UniverseLoadResult

_VALID_SOURCES = frozenset({"file_snapshot", "file_pit", "sp500_wikipedia_snapshot"})


def _read_snapshot_csv(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError(f"{path}: snapshot universe must have column 'ticker'")
    return sorted({str(t).upper().strip() for t in df["ticker"] if pd.notna(t) and str(t).strip()})


def _read_pit_csv(path: Path) -> tuple[list[str], pd.DataFrame]:
    df = pd.read_csv(path)
    for col in ("date", "ticker"):
        if col not in df.columns:
            raise ValueError(f"{path}: PIT universe must have columns 'date' and 'ticker'")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return sorted(df["ticker"].unique().tolist()), df[["date", "ticker"]]


def _sp500_from_wikipedia() -> list[str]:
    import io

    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, timeout=30, headers={"User-Agent": "r1000-ls-strategy/1.0"}).text
    table = pd.read_html(io.StringIO(html))[0]
    return sorted(
        table["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper().unique().tolist()
    )


def load_universe(source: str, path: Path | None, benchmark: str = "SPY") -> UniverseLoadResult:
    """Load universe from exactly one configured source. No silent fallbacks."""
    if source not in _VALID_SOURCES:
        raise ValueError(f"universe.source must be one of {sorted(_VALID_SOURCES)}, got {source!r}")

    warnings: list[str] = []
    is_pit = False
    resolved_path: str | None = None
    membership: pd.DataFrame | None = None

    if source == "file_snapshot":
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"universe.source=file_snapshot requires existing universe.path; missing: {path}"
            )
        tickers = _read_snapshot_csv(path)
        resolved_path = str(path)
        warnings.append("Snapshot universe: membership is fixed; not point-in-time.")

    elif source == "file_pit":
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"universe.source=file_pit requires existing universe.path; missing: {path}"
            )
        tickers, membership = _read_pit_csv(path)
        resolved_path = str(path)
        is_pit = True

    else:  # sp500_wikipedia_snapshot
        tickers = _sp500_from_wikipedia()
        warnings.append(
            "sp500_wikipedia_snapshot: current S&P 500 constituents only; survivorship-biased vs history."
        )
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"ticker": tickers}).to_csv(path, index=False)
            resolved_path = str(path)
            warnings.append(f"Wrote snapshot to {path} for reproducibility.")

    bench = benchmark.upper()
    if bench not in tickers:
        tickers = sorted(set(tickers) | {bench})

    if len(tickers) < 20:
        raise ValueError(f"Universe has only {len(tickers)} tickers after adding benchmark.")

    return UniverseLoadResult(
        tickers=tickers,
        source=source,
        path=resolved_path,
        is_point_in_time=is_pit,
        warnings=tuple(warnings),
        membership=membership,
    )


def active_tickers_on(membership: pd.DataFrame, dt: pd.Timestamp) -> set[str]:
    """Tickers in the PIT membership panel on rebalance date dt."""
    on = membership.loc[membership["date"] == dt, "ticker"]
    if not on.empty:
        return set(on)
    prior = membership[membership["date"] <= dt]
    if prior.empty:
        return set()
    last = prior["date"].max()
    return set(prior.loc[prior["date"] == last, "ticker"])
