"""End-to-end run: data -> factors -> signal -> portfolio -> report.

    python -m scripts.run_backtest --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import Config  # noqa: E402
from src.eval import backtest, ic as ic_module, metrics, report  # noqa: E402
from src.model import target as target_module, walkforward  # noqa: E402
from scripts.research import build_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--refit-months", type=int, default=12)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    config = Config.load(args.config)
    features, target, panel = build_dataset(config, args.rebuild)

    forward = target_module.forward_returns(
        panel.prices.where(panel.tradable), config.label.horizon_days
    ).stack(future_stack=True)
    forward.index.names = ["date", "ticker"]

    walk = walkforward.run(features, target, config, refit_months=args.refit_months)
    if walk.scores.empty:
        raise RuntimeError("Walk-forward produced no scores; check the out-of-sample window.")

    signal_ic = ic_module.information_coefficient(walk.scores, forward)
    signal = ic_module.summarize(signal_ic, config.label.horizon_days)
    quantiles = ic_module.quantile_returns(walk.scores, forward)
    signal["decile_spread_bps"] = quantiles.attrs["spread"]["mean_bps"]
    signal["decile_spread_t"] = quantiles.attrs["spread"]["t_stat"]
    signal["selection_turnover"] = ic_module.turnover(walk.scores, config.portfolio.quantile)

    oos = features.index.get_level_values("date") >= config.oos_start
    factor_table = ic_module.factor_table(features[oos], forward, config.label.horizon_days)

    result = backtest.run(panel, walk.scores, config)
    performance = metrics.summarize(result.net_returns, panel.market)
    performance.update(
        metrics.cost_stress(
            result.gross_returns,
            result.rebalances["turnover"] if not result.rebalances.empty else pd.Series(dtype=float),
            config.costs.stress_bps,
        )
    )
    performance["avg_turnover_per_rebalance"] = (
        round(float(result.rebalances["turnover"].mean()), 4) if not result.rebalances.empty else None
    )
    performance["avg_gross_leverage"] = (
        round(float(result.rebalances["gross"].mean()), 4) if not result.rebalances.empty else None
    )
    performance["avg_names"] = (
        round(float(result.rebalances["n_names"].mean()), 1) if not result.rebalances.empty else None
    )
    performance["rebalances_held_over"] = len(result.skipped)

    directory = report.new_run_directory()
    report.write(
        directory, config, panel.coverage(), factor_table, signal, quantiles, performance, result,
        walk.importances(),
    )
    print(report.summary_text(performance, signal, factor_table))
    print(f"\nwritten to {directory}")


if __name__ == "__main__":
    main()
