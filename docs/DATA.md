# Data contract

## Universe (`universe.source`)

| Source | File | PIT? | Use |
|--------|------|------|-----|
| `file_snapshot` | `universe.path` with column `ticker` | No | Fixed membership list |
| `file_pit` | `universe.path` with `date`, `ticker` | Yes | Membership filtered at each rebalance |
| `sp500_wikipedia_snapshot` | optional write to `universe.path` | No | Current S&P 500 from Wikipedia |

No silent fallbacks. Missing files or invalid sources fail the run.

Build snapshots:

```bash
python project/scripts/build_universe.py --source sp500_wikipedia_snapshot --path project/data/raw/universe_sp500.csv
python project/scripts/universe_from_price_cache.py --out project/data/raw/universe.csv
```

## Prices (`data.price_source`)

| Source | API | Notes |
|--------|-----|-------|
| `yfinance` | Yahoo Finance via `yfinance` | Adjusted close, batched download |
| `stooq` | Stooq via `pandas_datareader` | Symbol `TICKER.US` |

One source per run. Cache: `project/data/raw/prices/{TICKER}.parquet` with `_manifest.json` recording `source`, `start`, `end`.

Switching `price_source` without `force_refresh: true` raises if cache source mismatches.

## Sectors (`universe.sector_source`)

| Source | API |
|--------|-----|
| `yfinance` | `Ticker.info['sector']`, cached in `data/raw/metadata/sectors.json` |

## Validation

`validate_price_panel` checks monotonic dates, benchmark presence, positive prices, and drops tickers with >5% missing bars (logged via `warnings.warn`).

## Label purge (no lookahead)

Training labels are 21-day forward returns. At rebalance date `dt`, training rows are restricted to feature dates `t ≤ last_label_date(dt)`, where `last_label_date` is the last date whose forward return uses only prices on or before `dt`. See `src/labels/forward_returns.py`.

## Out-of-sample

`research.oos_start`: reported PnL and metrics include only rebalance periods ending on or after this date. Walk-forward training still uses history before each rebalance.
