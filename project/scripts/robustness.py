"""How much of the headline number is the signal, and how much is the seed?

Re-ordering the factor list -- which changes nothing economically, only
LightGBM's column sampling and tie-breaks -- moved the backtest Sharpe from 0.07
to 0.29. That is not a result, that is a warning, and the right response is to
measure the dispersion rather than to report whichever draw looks best.

Each run re-fits the whole walk-forward under a different model seed and
re-prices the book. The spread of the outcomes is the honest error bar.

    python -m scripts.robustness --seeds 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import Config  # noqa: E402
from src.eval import backtest, ic as ic_module, metrics  # noqa: E402
import dataclasses  # noqa: E402

from src.model import target as target_module, walkforward  # noqa: E402
from scripts.research import build_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()

    config = Config.load(args.config)
    features, target, panel = build_dataset(config)
    forward = target_module.forward_returns(
        panel.prices.where(panel.tradable), config.label.horizon_days
    ).stack(future_stack=True)
    forward.index.names = ["date", "ticker"]

    rows = []
    for seed in range(args.seeds):
        variant = dataclasses.replace(config, model=dataclasses.replace(config.model, seed=seed))
        walk = walkforward.run(features, target, variant, refit_months=12)
        result = backtest.run(panel, walk.scores, variant)
        performance = metrics.summarize(result.net_returns, panel.market)
        summary = ic_module.summarize(
            ic_module.information_coefficient(walk.scores, forward), config.label.horizon_days
        )
        rows.append(
            {
                "seed": seed,
                "ic": summary["mean_ic"],
                "icir": summary["icir"],
                "sharpe": performance["sharpe"],
                "cagr": performance["cagr"],
                "max_dd": performance["max_drawdown"],
            }
        )
        print(f"  seed {seed}: IC {rows[-1]['ic']:.5f}  Sharpe {rows[-1]['sharpe']:+.3f}", flush=True)

    table = pd.DataFrame(rows).set_index("seed")
    print("\n" + table.to_string())
    print("\nacross seeds")
    for column in ["ic", "sharpe", "cagr"]:
        values = table[column]
        print(
            f"  {column:<8} mean {values.mean():+.4f}   sd {values.std():.4f}   "
            f"min {values.min():+.4f}   max {values.max():+.4f}"
        )
    sharpe = table["sharpe"]
    print(
        f"\n  Sharpe is {sharpe.mean():.3f} +/- {sharpe.std():.3f} across identical-economics seeds; "
        f"{'includes' if sharpe.min() < 0 < sharpe.max() else 'excludes'} zero."
    )


if __name__ == "__main__":
    main()
