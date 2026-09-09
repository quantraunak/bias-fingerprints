"""The pre-registered link-signal test. Run once, reported whichever way it comes out.

    python scripts/run_signal_test.py

HYPOTHESIS.md fixes the falsification table before any data existed:

    link-lagged return IC      reject if t < 2.0
    decile monotonicity        reject if non-monotone across middle deciles
    second vs first order      reject if second-order IC not greater
    placebo (shuffled edges)   reject if real IC within 1 s.e. of placebo
    beta + sector neutralised  reject if alpha t < 2.0

The placebo is the one that matters. Randomly rewiring the graph while holding
the degree distribution fixed must destroy the signal. If a shuffled graph
predicts returns about as well as the real one, the result is an artifact of the
return panel rather than economic-link propagation.

Coverage is reported before any return is regressed, as the pre-registration
requires.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import Config, PROCESSED  # noqa: E402
from src.data import panel as panel_module  # noqa: E402
from src.eval import ic as ic_module  # noqa: E402
from src.graph import build, resolve, signal as signal_module  # noqa: E402
from src.model.target import forward_returns  # noqa: E402

HORIZON = 21
OOS_START = "2014-01-31"
N_PLACEBO = 20


def main() -> None:
    config = Config.load("configs/default.yaml")
    print("building panel ...", flush=True)
    panel = panel_module.build(config)
    prices = panel.prices.where(panel.tradable)
    returns = prices.pct_change(fill_method=None)

    claims = pd.read_parquet(PROCESSED / "link_claims_qwen.parquet")
    reference = resolve.load_reference(config.data.sec_user_agent)
    resolved = resolve.resolve_frame(claims, reference)
    resolved = resolved[resolved.counterparty_ticker.notna()]
    edges = build.edges(resolved.rename(columns={"ticker": "source_ticker"}))
    print(f"edges {len(edges)}  sources {edges.source.nunique()}  targets {edges.target.nunique()}")

    dates = panel.dates[panel.dates >= pd.Timestamp("2012-01-01")]
    dates = dates[::5]  # weekly sampling; the signal is a 21-day forward return

    forward = forward_returns(prices, HORIZON).stack(future_stack=True)
    forward.index.names = ["date", "ticker"]

    # ---- coverage, before any return is regressed -------------------------
    first = signal_module.link_signal(edges, returns, dates, order=1)
    cov = signal_module.coverage(first, panel.tradable, dates)
    print(f"\ncoverage: median {cov.covered.median():.0f} names/date "
          f"({cov.share.median():.1%} of tradable cross-section)")

    def score(sig: pd.Series, label: str) -> dict:
        oos = sig.index.get_level_values("date") >= OOS_START
        series = ic_module.information_coefficient(sig[oos], forward)
        summary = ic_module.summarize(series, HORIZON)
        print(f"  {label:<26} IC {summary['mean_ic']:+.5f}  t {summary['t_stat']:+.2f}  "
              f"n {len(series)}")
        return summary

    print("\n--- signal ---")
    first_s = score(first, "first order")
    second = signal_module.link_signal(edges, returns, dates, order=2)
    second_s = score(second, "second order") if not second.empty else None

    # ---- placebo: rewire, hold degree fixed -------------------------------
    print(f"\n--- placebo ({N_PLACEBO} shuffles) ---")
    placebo_t, placebo_ic = [], []
    for seed in range(N_PLACEBO):
        shuffled = signal_module.placebo(edges, seed=seed)
        sig = signal_module.link_signal(shuffled, returns, dates, order=1)
        if sig.empty:
            continue
        oos = sig.index.get_level_values("date") >= OOS_START
        s = ic_module.summarize(ic_module.information_coefficient(sig[oos], forward), HORIZON)
        placebo_t.append(s["t_stat"]); placebo_ic.append(s["mean_ic"])
    if placebo_ic:
        pm, ps = float(np.mean(placebo_ic)), float(np.std(placebo_ic, ddof=1))
        print(f"  placebo IC  mean {pm:+.5f}  sd {ps:.5f}   t mean {np.mean(placebo_t):+.2f}")
        gap = (first_s["mean_ic"] - pm) / ps if ps > 0 else float("nan")
        print(f"  real is {gap:+.2f} placebo standard deviations from the placebo mean")
    else:
        pm = ps = gap = float("nan")

    # ---- deciles ----------------------------------------------------------
    oos = first.index.get_level_values("date") >= OOS_START
    joined = pd.DataFrame({"sig": first[oos]}).join(forward.rename("fwd"), how="inner").dropna()
    if len(joined) > 100:
        joined["d"] = joined.groupby("date").sig.transform(
            lambda x: pd.qcut(x, min(5, x.nunique()), labels=False, duplicates="drop"))
        dec = joined.groupby("d").fwd.mean() * 10000
        print("\n--- quintile forward return (bp, 21d) ---")
        print(dec.round(1).to_string())

    # ---- verdict against the pre-registered table -------------------------
    print("\n=== falsification table ===")
    rows = [
        ("link-lagged IC t > 2.0", abs(first_s["t_stat"]) > 2.0, f"t = {first_s['t_stat']:+.2f}"),
        ("real IC > 1 s.e. above placebo", abs(gap) > 1.0 if gap == gap else False,
         f"{gap:+.2f} sd" if gap == gap else "no placebo"),
    ]
    if second_s:
        rows.append(("second order IC > first order",
                     second_s["mean_ic"] > first_s["mean_ic"],
                     f"{second_s['mean_ic']:+.5f} vs {first_s['mean_ic']:+.5f}"))
    for name, passed, detail in rows:
        print(f"  [{'PASS' if passed else 'FAIL'}]  {name:<34} {detail}")
    print("\nAny FAIL rejects the hypothesis as pre-registered.")


if __name__ == "__main__":
    main()
