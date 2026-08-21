"""Download everything the research pipeline needs and report coverage.

Idempotent: each stage skips what is already cached, so this can be re-run to
extend the sample or fill gaps.

    python -m scripts.build_data --stage universe
    python -m scripts.build_data --stage prices
    python -m scripts.build_data --stage fundamentals
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW, Config  # noqa: E402
from src.data import fundamentals, prices, universe  # noqa: E402

COVERAGE = RAW / "coverage.json"


def _record(stage: str, payload: dict) -> None:
    existing = json.loads(COVERAGE.read_text()) if COVERAGE.exists() else {}
    existing[stage] = payload
    COVERAGE.parent.mkdir(parents=True, exist_ok=True)
    COVERAGE.write_text(json.dumps(existing, indent=2, default=str))
    print(json.dumps({stage: payload}, indent=2, default=str)[:2000])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage", required=True, choices=["universe", "prices", "fundamentals"])
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    config = Config.load(args.config)
    spells = universe.load_spells()
    tickers = universe.tickers_in_window(spells, config.data.start, config.data.end)

    if args.stage == "universe":
        _record(
            "universe",
            {
                "n_spells": len(spells),
                "n_tickers_in_window": len(tickers),
                "n_currently_active": int(spells["end"].isna().sum()),
                "window": [config.data.start, config.data.end],
            },
        )
        return

    if args.stage == "prices":
        with_benchmark = tickers + [config.data.benchmark.upper()]
        _record("prices", prices.fetch(with_benchmark, config.data.start, config.data.end, args.force))
        return

    _record("fundamentals", fundamentals.fetch(tickers, config.data.sec_user_agent, args.force))


if __name__ == "__main__":
    main()
