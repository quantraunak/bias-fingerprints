"""Compare candidate signal constructions on identical out-of-sample dates.

Includes a deliberately naive baseline: the equal-weighted average of the
standardised factors, with no learning at all. If a gradient-boosted ensemble
cannot beat the average of its own inputs, that is the finding, and it is a more
useful one than a tuned number.

    python -m scripts.compare_models --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import Config  # noqa: E402
from src.eval import ic as ic_module  # noqa: E402
from src.model import target as target_module  # noqa: E402
from src.model import walkforward  # noqa: E402
from scripts.research import build_dataset  # noqa: E402

CANDIDATES = ["ridge", "lgbm", "lgbm_rank", "ensemble"]


def equal_weight_baseline(features: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Average of the standardised factors, restricted to the scoring dates."""
    subset = features.loc[features.index.get_level_values("date").isin(dates)]
    return subset.mean(axis=1, skipna=True).rename("score")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--refit-months", type=int, default=12)
    args = parser.parse_args()

    config = Config.load(args.config)
    features, target, panel = build_dataset(config)

    forward = target_module.forward_returns(
        panel.prices.where(panel.tradable), config.label.horizon_days
    ).stack(future_stack=True)
    forward.index.names = ["date", "ticker"]

    trading_days = pd.DatetimeIndex(sorted(features.index.get_level_values("date").unique()))
    rebalance_dates = walkforward.month_end_trading_days(trading_days)
    rebalance_dates = rebalance_dates[rebalance_dates >= pd.Timestamp(config.oos_start)]

    rows = {}
    baseline = equal_weight_baseline(features, rebalance_dates)
    rows["equal_weight_factors"] = _score(baseline, forward, config)

    for kind in CANDIDATES:
        variant = dataclasses.replace(config, model=dataclasses.replace(config.model, kind=kind))
        result = walkforward.run(features, target, variant, refit_months=args.refit_months)
        rows[kind] = _score(result.scores, forward, config)
        print(f"  {kind:<22} done", flush=True)

    table = pd.DataFrame(rows).T
    print("\nout-of-sample signal quality, identical dates\n")
    print(table[["mean_ic", "icir", "t_stat", "pct_positive", "spread_bps", "spread_t", "n_dates"]].to_string())


def _score(scores: pd.Series, forward: pd.Series, config: Config) -> dict:
    ic = ic_module.information_coefficient(scores, forward)
    summary = ic_module.summarize(ic, config.label.horizon_days)
    quantiles = ic_module.quantile_returns(scores, forward)
    summary["spread_bps"] = quantiles.attrs["spread"]["mean_bps"]
    summary["spread_t"] = quantiles.attrs["spread"]["t_stat"]
    return summary


if __name__ == "__main__":
    main()
