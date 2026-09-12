"""Cache the per-date IC series for each panel variant. Input to gates one and two.

    python3 scripts/gate_ic_series.py

A signature is a vector of mean-IC shifts across factors. The mean is taken over
dates, so the per-date IC series is the sufficient statistic: subperiod splits are
slices of it and the bootstrap resamples it. Computing it once turns gates one
and two from panel rebuilds into arithmetic.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.config import Config, PROCESSED
from src.data import fundamentals, panel as panel_module, universe
from src.data.panel import FUNDAMENTAL_ITEMS, Panel
from src.eval import ic as ic_module
from src.factors import registry
from src.factors.transforms import standardize
from src.model.target import forward_returns

from survivorship import survivor_panel
from pit_vs_naive import redate, variant

HORIZON = 21
OUT = PROCESSED / "gate_ic_series.parquet"


def ic_series(panel: Panel, names: list[str], oos_start: str) -> pd.DataFrame:
    """Per-date IC for every factor, against this panel's own forward returns."""
    forward = forward_returns(panel.prices.where(panel.tradable), HORIZON).stack(future_stack=True)
    forward.index.names = ["date", "ticker"]
    out = {}
    for name, frame in registry.compute_raw(panel, names).items():
        scores = standardize(frame, panel.industries).stack(future_stack=True)
        scores.index.names = ["date", "ticker"]
        oos = scores.index.get_level_values("date") >= oos_start
        out[name] = ic_module.information_coefficient(scores[oos], forward)
    return pd.DataFrame(out)


def main() -> None:
    config = Config.load("configs/default.yaml")
    names = list(registry.REGISTRY)
    base = panel_module.build(config)
    facts = fundamentals.load_facts(list(base.prices.columns))

    panels = {
        "PIT": base,
        "SURVIVOR": survivor_panel(base, config),
        "NAIVE": variant(base, facts, "period_end"),
    }

    frames = []
    for label, panel in panels.items():
        print(f"scoring {label} ...", flush=True)
        frame = ic_series(panel, names, config.oos_start)
        frame.columns = pd.MultiIndex.from_product([[label], frame.columns],
                                                   names=["panel", "factor"])
        frames.append(frame)
    joined = pd.concat(frames, axis=1)
    joined.to_parquet(OUT)
    print(f"\n{joined.shape[0]} dates x {joined.shape[1]} panel-factor series -> {OUT}")
    print("panels:", sorted({c[0] for c in joined.columns}))


if __name__ == "__main__":
    main()
