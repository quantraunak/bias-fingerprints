from __future__ import annotations

import pandas as pd


def validate_price_panel(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    benchmark: str,
    max_missing_pct: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate loaded market data. Raises on hard failures."""
    if prices.empty:
        raise ValueError("Price panel is empty.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)

    if prices.index.duplicated().any():
        raise ValueError("Price index contains duplicate dates.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price index is not sorted ascending.")

    bench = benchmark.upper()
    if bench not in prices.columns:
        raise ValueError(f"Benchmark {bench} missing from price panel.")

    missing_pct = prices.isna().mean()
    bad = missing_pct[missing_pct > max_missing_pct].index.tolist()
    if bad:
        import warnings

        warnings.warn(
            f"Dropping {len(bad)} tickers with >{max_missing_pct:.0%} missing bars: {bad[:10]}"
            + ("..." if len(bad) > 10 else ""),
            stacklevel=2,
        )
        prices = prices.drop(columns=bad)
        volumes = volumes.reindex(columns=prices.columns)

    if (prices <= 0).any().any():
        raise ValueError("Non-positive prices detected.")

    vol_aligned = volumes.reindex(prices.index).reindex(columns=prices.columns)
    coverage = prices.count().mean() / len(prices)

    stats = {
        "n_tickers": int(prices.shape[1]),
        "n_days": int(prices.shape[0]),
        "date_start": str(prices.index.min().date()),
        "date_end": str(prices.index.max().date()),
        "avg_coverage": float(coverage),
        "benchmark": bench,
        "dropped_high_missing": bad,
    }
    return prices, vol_aligned, stats
