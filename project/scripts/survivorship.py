"""What does it cost to use today's index membership for the whole history?

The companion to `pit_vs_naive.py`. That script holds the universe fixed and
moves the filing calendar; this one holds the filing calendar fixed and moves
the universe:

  PIT        a name enters the cross-section the day it joined the index and
             leaves the day it left, from reconstructed membership spells
  SURVIVOR   the bug: today's constituents, applied to the whole history, which
             is what you get from a current-members snapshot and a bulk price
             download

Unlike the dating bias, this one has no control group. Survivorship touches
every factor, because it changes which names exist rather than what is known
about them -- and it changes the forward returns too, since the companies that
were acquired, delisted or demoted are exactly the ones removed.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import Config
from src.data import panel as panel_module, universe
from src.data.panel import Panel
from src.eval import ic as ic_module
from src.factors import registry
from src.factors.transforms import standardize
from src.model.target import forward_returns

HORIZON = 21


def survivor_panel(base: Panel, config: Config) -> Panel:
    """`base` restricted to names that are still index members on the last date.

    Membership is rebuilt, then the tradability screen is recomputed through the
    same function the real panel uses, so the only difference between the two
    panels is which names are eligible -- not how eligibility is decided.
    """
    final = base.membership.index.max()
    current = universe.active_on(base.membership, final)
    survivor = pd.DataFrame(False, index=base.membership.index, columns=base.membership.columns)
    survivor[[c for c in current if c in survivor.columns]] = True
    return replace(
        base,
        membership=survivor,
        tradable=panel_module._tradable(base.prices, base.dollar_volume, survivor, config),
    )


def score(panel: Panel, names: list[str], oos_start: str) -> dict[str, dict]:
    """IC summary for every factor, against forward returns from this panel's own universe.

    The forward returns are rebuilt per panel on purpose. A survivorship-biased
    study does not merely rank a smaller cross-section, it also never sees the
    returns of the names that left, and that second effect is most of the damage.
    """
    forward = forward_returns(panel.prices.where(panel.tradable), HORIZON).stack(future_stack=True)
    forward.index.names = ["date", "ticker"]
    out = {}
    for name, frame in registry.compute_raw(panel, names).items():
        scores = standardize(frame, panel.industries).stack(future_stack=True)
        scores.index.names = ["date", "ticker"]
        oos = scores.index.get_level_values("date") >= oos_start
        out[name] = ic_module.summarize(
            ic_module.information_coefficient(scores[oos], forward), HORIZON
        )
    return out


config = Config.load("configs/default.yaml")
base = panel_module.build(config)
survivor = survivor_panel(base, config)
names = list(registry.REGISTRY)

pit_names = base.membership.any().sum()
sur_names = survivor.membership.any().sum()
print(f"universe:  point-in-time {pit_names} names ever a member")
print(f"           survivor     {sur_names} names still a member on {base.dates.max():%Y-%m-%d}")
print(f"           dropped      {pit_names - sur_names} deleted names the biased panel never sees\n")

coverage = pd.DataFrame(
    {"pit": base.tradable.sum(axis=1), "survivor": survivor.tradable.sum(axis=1)}
).groupby(base.dates.year).mean()
coverage["pct_of_pit"] = 100 * coverage.survivor / coverage.pit
print("Tradable names per day, by year")
print(coverage.round(1).to_string())

print("\nscoring both universes ...", flush=True)
results = {"PIT": score(base, names, config.oos_start), "SURVIVOR": score(survivor, names, config.oos_start)}
print("  done")

table = pd.DataFrame(
    [
        {
            "factor": name,
            "pit_ic": results["PIT"][name]["mean_ic"],
            "survivor_ic": results["SURVIVOR"][name]["mean_ic"],
            "pit_t": results["PIT"][name]["t_stat"],
            "survivor_t": results["SURVIVOR"][name]["t_stat"],
        }
        for name in names
    ]
).set_index("factor")
table["ic_change"] = table.survivor_ic - table.pit_ic

print(f"\n{'factor':<22}{'PIT IC':>10}{'SURV IC':>10}{'change':>10}   {'PIT t':>7}{'SURV t':>8}")
for name, r in table.iterrows():
    print(
        f"{name:<22}{r.pit_ic:>10.5f}{r.survivor_ic:>10.5f}{r.ic_change:>+10.5f}   "
        f"{r.pit_t:>7.2f}{r.survivor_t:>8.2f}"
    )

crossed = table.index[(table.pit_t < 2) & (table.survivor_t >= 2)]
vanished = table.index[(table.pit_t >= 2) & (table.survivor_t < 2)]
print(
    f"\nmean IC change                  {table.ic_change.mean():+.5f}"
    f"   ({table.ic_change.mean() / table.pit_ic.abs().mean():+.0%} of true IC)"
)
print(f"factors that look better        {(table.ic_change > 0).sum()} of {len(table)}")
print(f"factors crossing t=2 spuriously {len(crossed)} of {len(table)}")
if len(crossed):
    print(f"   {', '.join(crossed)}")
if len(vanished):
    print(f"factors that lose significance  {len(vanished)}: {', '.join(vanished)}")

table.to_csv("reports/survivorship.csv")
print("\nwrote reports/survivorship.csv")
