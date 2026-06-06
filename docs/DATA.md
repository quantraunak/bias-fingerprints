# Data contract

## Universe (`universe.source`)

| Source | File | PIT? | Use |
|--------|------|------|-----|
| `file_snapshot` | `universe.path` with column `ticker` | No | Default; fixed membership list |
| `file_pit` | `universe.path` with `date`, `ticker` | Yes | Historical membership panel |
| `sp500_wikipedia_snapshot` | optional write to `universe.path` | No | Fetches current S&P 500 from Wikipedia |

No silent fallbacks. If `file_snapshot` is configured and the file is missing, the run fails.

Build a snapshot file:

```bash
python project/scripts/build_universe.py --source sp500_wikipedia_snapshot --path data/raw/universe_sp500.csv
```

Fast local runs aligned to existing price cache (explicit subset, not an automatic fallback):

```bash
python project/scripts/universe_from_price_cache.py --out data/raw/universe.csv
```

Point `universe.path` at `universe_sp500.csv` for full S&P coverage once prices are fetched.

## Prices (`data.price_source`)

| Source | API | Notes |
|--------|-----|-------|
| `yfinance` | Yahoo Finance via `yfinance` | Adjusted close, `auto_adjust=True` |
| `stooq` | Stooq via `pandas_datareader` | Symbol format `TICKER.US` |

One source per run. Cache: `project/data/raw/prices/{TICKER}.parquet` with manifest `project/data/raw/prices/_manifest.json` recording `source`, `start`, `end`, `rows`.

## Sectors (`universe.sector_source`)

| Source | API |
|--------|-----|
| `yfinance` | `Ticker.info['sector']`, cached in `data/raw/metadata/sectors.json` |

## Validation

After load, `validate_price_panel` checks:

- Monotonic dates, no duplicates
- Benchmark present
- Max 5% missing bars per ticker
- Strictly positive prices

## Out-of-sample

`research.oos_start`: portfolio PnL and reported metrics use rebalance periods ending on or after this date only. Training still uses walk-forward history before each rebalance date.
