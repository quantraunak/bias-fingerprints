"""Link-lagged returns: the signal the pre-registration commits to testing.

For firm `i` on date `d`, the signal is the weighted average return of `i`'s
counterparties over the trailing month, as the graph stood on `d`. The claim is
that this predicts `i`'s own return over the following month, because attention
does not travel along links no database publishes.

Everything here is point-in-time by construction: edges are filtered to those
already filed by `d`, and counterparty returns are taken strictly before `d`.
The two look-ahead routes into a signal like this are using an edge before its
filing date and using a counterparty return from the prediction window itself,
and both are closed at the point of computation rather than checked afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.graph import build

LOOKBACK_DAYS = 21


def _window_returns(returns: pd.DataFrame, as_of: pd.Timestamp, lookback: int) -> pd.Series:
    """Compounded return per name over the `lookback` days ending at `as_of`.

    `as_of` is inclusive: the rebalance happens at that close, so that day's
    return is known. Anything after it is the prediction window.
    """
    history = returns.loc[:as_of].tail(lookback)
    if history.empty:
        return pd.Series(dtype=float)
    return (1.0 + history).prod(min_count=1) - 1.0


def link_signal(
    edge_frame: pd.DataFrame,
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    order: int = 1,
    lookback: int = LOOKBACK_DAYS,
    min_links: int = 1,
) -> pd.Series:
    """Weighted counterparty return per (date, ticker).

    `order=1` uses direct counterparties, `order=2` their counterparties. Names
    with fewer than `min_links` resolvable counterparties on a date get no
    value rather than a zero, so "no graph coverage" stays distinguishable from
    "counterparties were flat".
    """
    records = {}
    for as_of in dates:
        first = build.neighbours(edge_frame, as_of)
        adjacency = first if order == 1 else build.second_order(first)
        if not adjacency:
            continue

        window = _window_returns(returns, as_of, lookback)
        if window.empty:
            continue

        values = {}
        for source, links in adjacency.items():
            numerator = denominator = 0.0
            used = 0
            for target, weight in links:
                value = window.get(target)
                if value is None or not np.isfinite(value):
                    continue
                numerator += weight * value
                denominator += weight
                used += 1
            if used >= min_links and denominator > 0:
                values[source] = numerator / denominator

        if values:
            records[as_of] = pd.Series(values)

    if not records:
        return pd.Series(dtype=float, name="link_signal")

    out = pd.concat(records, names=["date", "ticker"]).rename("link_signal")
    return out.dropna()


def coverage(signal: pd.Series, tradable: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """How much of the tradable cross-section the graph reaches, per date.

    Reported before any return is regressed on the signal. A signal covering a
    tenth of the universe is not comparable to one covering all of it, and the
    difference has to be visible up front rather than discovered in the IC.
    """
    rows = []
    for as_of in dates:
        if as_of not in tradable.index:
            continue
        universe = tradable.loc[as_of]
        universe = set(universe[universe].index)
        if as_of in signal.index.get_level_values("date"):
            covered = set(signal.xs(as_of, level="date").index) & universe
        else:
            covered = set()
        rows.append({
            "date": as_of,
            "tradable": len(universe),
            "covered": len(covered),
            "share": round(len(covered) / len(universe), 4) if universe else 0.0,
        })
    return pd.DataFrame(rows).set_index("date")


def placebo(edge_frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Degree-preserving edge shuffle: the null the hypothesis must beat.

    Targets are permuted within each date-interval bucket, so every source keeps
    its out-degree and every target keeps its in-degree while the pairing is
    destroyed. If the real graph does not beat this, the measured effect is
    cross-sectional autocorrelation or sector clustering wearing a graph, and
    the economic story is not doing any work.
    """
    rng = np.random.default_rng(seed)
    shuffled = edge_frame.copy()

    for _, block in shuffled.groupby(["valid_from", "relation"]):
        if len(block) < 2:
            continue
        targets = block["target"].to_numpy().copy()
        rng.shuffle(targets)
        shuffled.loc[block.index, "target"] = targets

    # A shuffle can create self-loops; drop them so the placebo is not handed a
    # firm's own lagged return, which would make it beat the real graph.
    return shuffled[shuffled["source"] != shuffled["target"]].reset_index(drop=True)
