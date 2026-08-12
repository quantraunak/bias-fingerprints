from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class UniverseLoadResult:
    tickers: list[str]
    source: str
    path: str | None
    is_point_in_time: bool
    warnings: tuple[str, ...] = ()
    membership: pd.DataFrame | None = None  # columns: date, ticker (PIT only)


@dataclass
class FetchResult:
    n_requested: int
    n_loaded: int
    n_failed: int
    n_dropped_short_history: int
    price_source: str
    failed_tickers: dict[str, str] = field(default_factory=dict)
