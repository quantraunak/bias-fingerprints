"""Fit the walk-forward once per seed and cache the scores.

Portfolio construction experiments do not change the signal, so refitting the
model for each one wastes most of the runtime. This writes
`data/processed/scores_seed<N>.parquet`; `scripts/sweep_portfolio.py` reads them.

    python -m scripts.cache_scores --seeds 0 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import PROCESSED, Config  # noqa: E402
from src.model import walkforward  # noqa: E402
from scripts.research import build_dataset  # noqa: E402


def scores_path(seed: int) -> Path:
    return PROCESSED / f"scores_seed{seed}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--refit-months", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = Config.load(args.config)
    features, target, _ = build_dataset(config)
    print(f"features {features.shape}")

    for seed in args.seeds:
        path = scores_path(seed)
        if path.exists() and not args.force:
            print(f"seed {seed}: cached")
            continue

        seeded = dataclasses.replace(config, model=dataclasses.replace(config.model, seed=seed))
        start = time.time()
        walk = walkforward.run(features, target, seeded, refit_months=args.refit_months)
        if walk.scores.empty:
            raise RuntimeError(f"seed {seed}: walk-forward produced no scores")

        walk.scores.to_frame().to_parquet(path)
        dates = walk.scores.index.get_level_values("date")
        print(
            f"seed {seed}: {len(walk.scores):,} scores over {dates.nunique()} dates "
            f"in {time.time() - start:.0f}s -> {path.name}"
        )


if __name__ == "__main__":
    main()
