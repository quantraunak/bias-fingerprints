#!/usr/bin/env python3
"""Write universe.csv from cached price manifest (explicit subset, not a fallback)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/raw/prices/_manifest.json")
    parser.add_argument("--out", default="data/raw/universe.csv")
    parser.add_argument("--benchmark", default="SPY")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    tickers = sorted({t.upper() for t in manifest if not t.startswith("_")})
    bench = args.benchmark.upper()
    if bench not in tickers:
        tickers.append(bench)

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("ticker\n" + "\n".join(tickers) + "\n")
    print(f"wrote {len(tickers)} tickers to {out}")


if __name__ == "__main__":
    main()
