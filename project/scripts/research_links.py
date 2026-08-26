"""Test the link-lagged signal against its pre-registered falsification criteria.

    python -m scripts.research_links --config configs/default.yaml

Runs the whole chain on whatever has been extracted so far: resolve claims to
tickers, assemble the point-in-time edge panel, report graph coverage, then
measure IC against the same forward-return label the factor pipeline uses. The
placebo -- a degree-preserving edge shuffle -- runs beside every test, because
the number that decides the hypothesis is the gap between them, not the level.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROCESSED, Config  # noqa: E402
from src.eval import ic as ic_module  # noqa: E402
from src.graph import build, resolve, signal as signal_module  # noqa: E402
from src.model import target as target_module, walkforward  # noqa: E402
from scripts.research import build_dataset  # noqa: E402

CLAIMS_PATH = PROCESSED / "link_claims.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--placebo-seeds", type=int, default=3)
    args = parser.parse_args()

    config = Config.load(args.config)
    claims = pd.read_parquet(CLAIMS_PATH)
    print(f"claims: {len(claims):,} from {claims['accession'].nunique():,} filings")

    # ---------------------------------------------------------------- resolve
    reference = resolve.load_reference(config.data.sec_user_agent)
    resolved = resolve.resolve_frame(claims, reference)
    resolved = resolved.rename(columns={"ticker": "source_ticker"})
    stats = resolve.coverage(resolved)
    print(f"resolution: {stats['claims_resolved']:,}/{stats['claims']:,} claims "
          f"({100*stats['claim_match_rate']:.0f}%), "
          f"{stats['distinct_resolved']:,}/{stats['distinct_names']:,} distinct names")
    print(f"  by tier: {stats['by_tier']}")

    # ------------------------------------------------------------------ graph
    edges = build.edges(resolved)
    if edges.empty:
        print("\nNo directed edges survived resolution. Stopping.")
        return
    print(f"\ngraph: {build.summarize(edges)}")

    # ------------------------------------------------------- panel and labels
    features, target, panel = build_dataset(config)
    returns = panel.returns
    trading_days = returns.index
    dates = walkforward.month_end_trading_days(trading_days)
    dates = dates[dates >= pd.Timestamp(config.oos_start)]

    forward = target_module.forward_returns(
        panel.prices.where(panel.tradable), config.label.horizon_days
    ).stack(future_stack=True)
    forward.index.names = ["date", "ticker"]

    # --------------------------------------------------------------- coverage
    for order in args.orders:
        sig = signal_module.link_signal(edges, returns, dates, order=order)
        if sig.empty:
            print(f"\norder {order}: no signal produced")
            continue
        cover = signal_module.coverage(sig, panel.tradable, dates)
        print(f"\norder {order}: {len(sig):,} observations over "
              f"{sig.index.get_level_values('date').nunique()} dates")
        print(f"  coverage of tradable cross-section: "
              f"mean {100*cover['share'].mean():.1f}%  median {100*cover['share'].median():.1f}%")
        print(f"  names per date: mean {cover['covered'].mean():.0f}")

        real = _score(sig, forward, config)
        print(f"  IC {real['mean_ic']:+.4f}  t={real['t_stat']:+.2f}  "
              f"ICIR {real['icir']:+.3f}  n={real['n_dates']}")

        placebo_ics = []
        for seed in range(args.placebo_seeds):
            shuffled = signal_module.placebo(edges, seed=seed)
            fake = signal_module.link_signal(shuffled, returns, dates, order=order)
            if fake.empty:
                continue
            placebo_ics.append(_score(fake, forward, config)["mean_ic"])
        if placebo_ics:
            mean, sd = float(np.mean(placebo_ics)), float(np.std(placebo_ics))
            gap = (real["mean_ic"] - mean) / sd if sd > 1e-12 else float("nan")
            print(f"  placebo IC {mean:+.4f} +/- {sd:.4f} over {len(placebo_ics)} shuffles")
            print(f"  real minus placebo: {real['mean_ic']-mean:+.4f}  ({gap:+.1f} placebo sd)")
            verdict = "PASSES" if (real["t_stat"] > 2.0 and gap > 1.0) else "FAILS"
            print(f"  -> {verdict} the pre-registered bar (t>2.0 and >1sd over placebo)")


def _score(sig: pd.Series, forward: pd.Series, config: Config) -> dict:
    series = ic_module.information_coefficient(sig, forward)
    summary = ic_module.summarize(series, config.label.horizon_days)
    summary["n_dates"] = int(series.notna().sum())
    return summary


if __name__ == "__main__":
    main()
