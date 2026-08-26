"""Assemble extracted claims into a point-in-time edge panel.

A claim is "issuer i's 10-K, filed on date d, named counterparty j". An *edge*
is that claim with a validity interval attached, so the graph can be asked what
it looked like on any past date without leaking the future.

The interval rule follows how 10-Ks actually work. Each annual filing restates
the issuer's material relationships, so a relationship disclosed in filing `k`
is treated as live from `filed_k` until the issuer's *next* filing lands. If the
next filing names it again the interval extends; if it does not, the edge
lapses. That handles ended relationships without needing to detect an ending,
which filings rarely state outright. An issuer's final filing gets a fixed
`max_age` tail rather than living forever.
"""

from __future__ import annotations

import pandas as pd

MAX_AGE_DAYS = 550  # a filing's claims outlive it by ~18 months at most

# Only these carry a directional return-propagation hypothesis. Competitor links
# are extracted and kept in the claims table for later study, but a competitor's
# good month has no signed prediction for the filer, so they are not edges here.
DIRECTED = {"customer", "supplier", "partner"}


def edges(resolved: pd.DataFrame, max_age_days: int = MAX_AGE_DAYS) -> pd.DataFrame:
    """Claims -> edges with [valid_from, valid_to) intervals.

    Returns columns: source, target, relation, valid_from, valid_to, weight,
    confidence. `source` is the filing issuer, `target` the counterparty.
    """
    usable = resolved[
        resolved["counterparty_ticker"].notna()
        & resolved["relation"].isin(DIRECTED)
    ].copy()
    if usable.empty:
        return pd.DataFrame(
            columns=["source", "target", "relation", "valid_from", "valid_to", "weight", "confidence"]
        )

    usable["filed"] = pd.to_datetime(usable["filed"])
    # A counterparty that resolves to the filer itself is a parsing artifact --
    # a subsidiary or a self-reference -- and a self-loop would feed the firm's
    # own lagged return back in as a predictor of itself.
    usable = usable[usable["source_ticker"] != usable["counterparty_ticker"]]

    # Next filing date per issuer bounds every edge disclosed in this one.
    filings = (
        usable[["source_ticker", "filed"]]
        .drop_duplicates()
        .sort_values(["source_ticker", "filed"])
    )
    filings["next_filed"] = filings.groupby("source_ticker")["filed"].shift(-1)
    usable = usable.merge(filings, on=["source_ticker", "filed"], how="left")

    cap = usable["filed"] + pd.Timedelta(days=max_age_days)
    usable["valid_from"] = usable["filed"]
    usable["valid_to"] = usable["next_filed"].fillna(cap).clip(upper=cap)

    usable["weight"] = _weight(usable)

    out = usable.rename(columns={"source_ticker": "source", "counterparty_ticker": "target"})
    out = out[["source", "target", "relation", "valid_from", "valid_to", "weight", "confidence"]]
    return out.sort_values(["valid_from", "source", "target"]).reset_index(drop=True)


def _weight(frame: pd.DataFrame) -> pd.Series:
    """Edge weight: disclosed revenue share where stated, else a confidence prior.

    Revenue share is the only quantitative measure of link strength a filing
    offers, and it is disclosed for a minority of edges. Filling the rest with a
    confidence-graded constant keeps the two on one scale without inventing
    precision that the text does not support.
    """
    stated = pd.to_numeric(frame["revenue_pct"], errors="coerce") / 100.0
    prior = frame["confidence"].map({"high": 0.10, "medium": 0.06, "low": 0.03}).fillna(0.03)
    return stated.where(stated.between(0.001, 1.0), prior)


def active(edge_frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Edges visible on `as_of`. Half-open interval: [valid_from, valid_to)."""
    mask = (edge_frame["valid_from"] <= as_of) & (edge_frame["valid_to"] > as_of)
    return edge_frame[mask]


def neighbours(edge_frame: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, list[tuple[str, float]]]:
    """source -> [(target, weight)] for one date."""
    out: dict[str, list[tuple[str, float]]] = {}
    for row in active(edge_frame, as_of).itertuples():
        out.setdefault(row.source, []).append((row.target, row.weight))
    return out


def second_order(first: dict[str, list[tuple[str, float]]]) -> dict[str, list[tuple[str, float]]]:
    """Neighbours of neighbours, excluding self and anything already first-order.

    This is the set the hypothesis says should carry the most information: no
    vendor product surfaces a customer's customer, so inattention has nowhere to
    be corrected from.
    """
    out: dict[str, list[tuple[str, float]]] = {}
    for source, direct in first.items():
        direct_set = {target for target, _ in direct}
        reached: dict[str, float] = {}
        for target, w1 in direct:
            for indirect, w2 in first.get(target, []):
                if indirect == source or indirect in direct_set:
                    continue
                reached[indirect] = reached.get(indirect, 0.0) + w1 * w2
        if reached:
            out[source] = sorted(reached.items())
    return out


def summarize(edge_frame: pd.DataFrame) -> dict:
    if edge_frame.empty:
        return {"edges": 0}
    span = pd.date_range(edge_frame["valid_from"].min(), edge_frame["valid_to"].max(), freq="YE")
    per_date = [len(active(edge_frame, d)) for d in span] or [0]
    return {
        "edges": len(edge_frame),
        "sources": edge_frame["source"].nunique(),
        "targets": edge_frame["target"].nunique(),
        "by_relation": edge_frame["relation"].value_counts().to_dict(),
        "median_active_per_year_end": int(pd.Series(per_date).median()),
        "weight_from_disclosed_pct": int((edge_frame["weight"] > 0.101).sum()),
    }
