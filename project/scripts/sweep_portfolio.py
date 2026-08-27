"""Compare portfolio constructions on one fixed signal.

    python -m scripts.sweep_portfolio --seed 7

The signal is held constant -- cached by `scripts.cache_scores` -- so every
difference below is attributable to construction rather than to the model. That
separation is the point: a backtest Sharpe conflates the quality of a forecast
with how much of it survives implementation, and the transfer coefficient
reported here is what tells the two apart.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import Config  # noqa: E402
from src.eval import backtest, metrics  # noqa: E402
from scripts.cache_scores import scores_path  # noqa: E402
from scripts.research import build_dataset  # noqa: E402

VARIANTS = {
    "A baseline (deciles, gross 1.5)": {
        "selection": "deciles", "risk_in_objective": False, "industry_tolerance": -1.0,
    },
    "B full cross-section, gross 1.5": {
        "selection": "full", "risk_in_objective": True, "industry_tolerance": -1.0,
    },
    "C full, gross 3.0": {
        "selection": "full", "risk_in_objective": True, "industry_tolerance": -1.0,
        "gross_leverage": 3.0,
    },
    "D full, gross 5.0": {
        "selection": "full", "risk_in_objective": True, "industry_tolerance": -1.0,
        "gross_leverage": 5.0,
    },
    "E full, gross 5.0, industry neutral": {
        "selection": "full", "risk_in_objective": True, "industry_tolerance": 0.02,
        "gross_leverage": 5.0,
    },
    "F full, gross 5.0, ind neutral, no dd throttle": {
        "selection": "full", "risk_in_objective": True, "industry_tolerance": 0.02,
        "gross_leverage": 5.0, "dd_threshold": 1.0,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-cross-section", type=int, default=300)
    parser.add_argument("--bisection-steps", type=int, default=12)
    parser.add_argument("--only", nargs="*", default=None, help="variant key prefixes to run")
    args = parser.parse_args()

    config = Config.load(args.config)
    path = scores_path(args.seed)
    if not path.exists():
        raise FileNotFoundError(f"No cached scores at {path}; run scripts.cache_scores first.")
    scores = pd.read_parquet(path)["score"]
    _features, _target, panel = build_dataset(config)

    rows = []
    for name, overrides in VARIANTS.items():
        if args.only and not any(name.startswith(k) for k in args.only):
            continue
        portfolio = dataclasses.replace(
            config.portfolio, max_cross_section=args.max_cross_section, **overrides
        )
        variant = dataclasses.replace(config, portfolio=portfolio)

        began = time.time()
        result = backtest.run(panel, scores, variant)
        summary = metrics.summarize(result.net_returns, panel.market)
        rebalances = result.rebalances

        rows.append({
            "variant": name,
            "sharpe": summary["sharpe"],
            "cagr": summary["cagr"],
            "vol": summary["annual_vol"],
            "maxdd": summary["max_drawdown"],
            "beta": summary["beta_vs_market"],
            "names": round(float(rebalances["n_names"].mean()), 0) if not rebalances.empty else None,
            "turnover": round(float(rebalances["turnover"].mean()), 3) if not rebalances.empty else None,
            "tc": round(float(rebalances["transfer_coefficient"].mean()), 3)
                  if "transfer_coefficient" in rebalances else None,
            "minutes": round((time.time() - began) / 60, 1),
        })
        print(f"  {name:45} sharpe={rows[-1]['sharpe']:+.3f} "
              f"tc={rows[-1]['tc']} names={rows[-1]['names']} ({rows[-1]['minutes']}m)", flush=True)

    frame = pd.DataFrame(rows)
    print("\n" + frame.to_string(index=False))


if __name__ == "__main__":
    main()
