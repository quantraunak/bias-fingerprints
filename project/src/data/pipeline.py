from __future__ import annotations

from pathlib import Path

from src.data.fetch_prices import FetchConfig, get_price_data
from src.data.sectors import load_sector_map
from src.data.types import UniverseLoadResult
from src.data.universe import load_universe
from src.data.validate import validate_price_panel


def load_market_data(project_root: Path, config: dict) -> dict:
    """Single entry point: universe -> prices -> validation -> sector map."""
    data_cfg = config["data"]
    uni_cfg = config["universe"]

    uni_path = project_root / uni_cfg["path"] if uni_cfg.get("path") else None
    universe: UniverseLoadResult = load_universe(
        source=uni_cfg["source"],
        path=uni_path,
        benchmark=uni_cfg.get("benchmark", "SPY"),
    )

    fetch_cfg = FetchConfig(
        source=data_cfg["price_source"],
        start=data_cfg["start"],
        end=data_cfg["end"],
        force_refresh=data_cfg.get("force_refresh", False),
        min_history_days=data_cfg.get("min_history_days", 252),
        min_tickers=data_cfg.get("min_tickers", 100),
        min_success_ratio=data_cfg.get("min_success_ratio", 0.80),
        retries=data_cfg.get("retries", 3),
    )

    cache_dir = project_root / "data" / "raw"
    price_data = get_price_data(universe.tickers, fetch_cfg, cache_dir)

    prices, volumes, panel_stats = validate_price_panel(
        price_data.prices,
        price_data.volumes,
        benchmark=uni_cfg.get("benchmark", "SPY"),
        max_missing_pct=data_cfg.get("max_missing_pct", 0.05),
    )

    sectors = load_sector_map(
        list(price_data.prices.columns),
        cache_dir,
        source=uni_cfg.get("sector_source", "yfinance"),
    )

    return {
        "prices": prices,
        "volumes": volumes,
        "sectors": sectors,
        "universe": universe,
        "fetch": price_data.fetch_result,
        "panel_stats": panel_stats,
    }
