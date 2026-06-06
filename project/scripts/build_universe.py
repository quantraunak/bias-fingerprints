#!/usr/bin/env python3
"""Build universe.csv from an explicit source. No fallbacks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.universe import load_universe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["sp500_wikipedia_snapshot", "file_snapshot"],
        default="sp500_wikipedia_snapshot",
    )
    parser.add_argument("--path", default="data/raw/universe.csv")
    args = parser.parse_args()

    out = PROJECT_ROOT / args.path
    result = load_universe(source=args.source, path=out if args.source == "file_snapshot" else out)

    print(f"source={result.source}")
    print(f"path={result.path}")
    print(f"tickers={len(result.tickers)}")
    for w in result.warnings:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
