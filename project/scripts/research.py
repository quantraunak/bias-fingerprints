"""Signal research: build the factor matrix and measure it, before any trading.

This is the gate. If the factor table and the model's out-of-sample IC are not
there, the backtest is not worth running -- a portfolio optimiser cannot create
information that the signal does not contain, it can only lever what is there.

    python -m scripts.research --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import PROCESSED, Config  # noqa: E402
from src.data import panel as panel_module  # noqa: E402
from src.eval import ic as ic_module  # noqa: E402
from src.factors import registry  # noqa: E402
from src.model import target as target_module  # noqa: E402

FEATURES_PATH = PROCESSED / "features.parquet"
TARGET_PATH = PROCESSED / "target.parquet"


def build_dataset(config: Config, rebuild: bool = False) -> tuple[pd.DataFrame, pd.Series, object]:
    panel = panel_module.build(config)
    if FEATURES_PATH.exists() and not rebuild:
        features = pd.read_parquet(FEATURES_PATH)
        target = pd.read_parquet(TARGET_PATH)["target"]
    else:
        features = registry.build_matrix(panel, config.factors)
        target = target_module.build(
            panel.prices.where(panel.tradable), config.label.horizon_days, config.label.target
        )
        PROCESSED.mkdir(parents=True, exist_ok=True)
        features.to_parquet(FEATURES_PATH)
        target.to_frame().to_parquet(TARGET_PATH)
    return features, target, panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    config = Config.load(args.config)
    features, target, panel = build_dataset(config, args.rebuild)

    raw_forward = target_module.forward_returns(
        panel.prices.where(panel.tradable), config.label.horizon_days
    ).stack(future_stack=True)
    raw_forward.index.names = ["date", "ticker"]

    oos = features.index.get_level_values("date") >= config.oos_start
    print(f"features {features.shape}  target {target.shape}")
    print(f"dates {features.index.get_level_values('date').min().date()} "
          f"-> {features.index.get_level_values('date').max().date()}")
    print(f"names per date (median): "
          f"{features.groupby(level='date').size().median():.0f}\n")

    print("per-factor information coefficient, out-of-sample "
          f"(from {config.oos_start}, {config.label.horizon_days}d forward)\n")
    table = ic_module.factor_table(features[oos], raw_forward, config.label.horizon_days)
    print(table[["mean_ic", "icir", "icir_annual", "t_stat", "pct_positive", "n_dates"]].to_string())


if __name__ == "__main__":
    main()
