"""Fit competing mechanisms to the cardinality curve and select between them.

    python scripts/fit_curves.py

Three hypotheses about why extraction might miss items, each a statement about
mechanism rather than a description of the data, and each making a different
prediction for recovered-versus-planted:

    independent   R(n) = p * n                 every item is recovered with the
                                               same probability regardless of
                                               how many there are. Straight line
                                               through the origin.

    saturating    R(n) = k * (1 - e^(-n/k))    the model has a capacity of about
                                               k items. Rises, then flattens.
                                               Recall per item falls as 1/n.

    dilution      R(n) = n * p0 * e^(-l*n)     per-item recall decays with the
                                               number of competing items, so the
                                               total can rise and then FALL.

They are distinguishable, which is the reason for measuring many cardinalities
rather than two buckets. Selection is by AICc -- corrected for small samples,
since a curve with eleven cardinalities is a small sample -- with the
independent model as the null: it is nested in neither of the others but is the
hypothesis that nothing cardinality-specific is happening, and it wins whenever
the extra parameter does not earn its keep.

`k` is the number that matters downstream. If saturation wins, a graph built by
LLM extraction has its degree distribution truncated near k, hub nodes are
clipped to look ordinary, and every centrality measure computed on it is wrong
in a knowable direction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

from src.config import PROCESSED  # noqa: E402

DATA = PROCESSED / "cardinality.parquet"


def independent(n, p):
    return p * n


def saturating(n, k):
    # k*(1 - exp(-n/k)) -> n for n << k, -> k for n >> k. One parameter, so it
    # is compared against `independent` on equal terms.
    return k * (1.0 - np.exp(-n / np.maximum(k, 1e-9)))


def dilution(n, p0, lam):
    return n * p0 * np.exp(-lam * n)


MODELS = {
    "independent": (independent, [0.9], ([0.0], [1.0])),
    "saturating": (saturating, [10.0], ([0.5], [1e4])),
    "dilution": (dilution, [0.9, 0.01], ([0.0, 0.0], [1.0, 1.0])),
}


def aicc(residuals: np.ndarray, n_params: int) -> float:
    """AIC with the small-sample correction. Gaussian errors, so AIC = n ln(RSS/n)."""
    n = len(residuals)
    rss = float(np.sum(residuals ** 2))
    if rss <= 0:
        rss = 1e-12
    aic = n * np.log(rss / n) + 2 * n_params
    if n - n_params - 1 <= 0:
        return float("inf")
    return aic + (2 * n_params * (n_params + 1)) / (n - n_params - 1)


def fit(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit every model to one (model, spread) group's raw replicate points."""
    x = frame.n_planted.to_numpy(float)
    y = frame.n_recovered.to_numpy(float)

    rows = []
    for name, (func, guess, bounds) in MODELS.items():
        try:
            params, _ = curve_fit(func, x, y, p0=guess, bounds=bounds, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        residuals = y - func(x, *params)
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        rows.append({
            "form": name,
            "params": ", ".join(f"{p:.3f}" for p in params),
            "aicc": round(aicc(residuals, len(params)), 2),
            "r2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else float("nan"),
            "rmse": round(float(np.sqrt(ss_res / len(residuals))), 3),
            "_k": params[0] if name == "saturating" else None,
        })
    out = pd.DataFrame(rows).sort_values("aicc").reset_index(drop=True)
    out["delta_aicc"] = (out.aicc - out.aicc.min()).round(2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    frame = frame[frame.n_planted > 0]  # R(0)=0 for every model; it carries no information

    for (model, spread), group in frame.groupby(["model", "spread"]):
        print(f"\n=== {model}  spread={spread}  "
              f"({len(group)} points, {group.n_planted.nunique()} cardinalities) ===")
        summary = (group.groupby("n_planted")
                   .agg(mean_recovered=("n_recovered", "mean"),
                        sd=("n_recovered", "std"), reps=("n_recovered", "size")))
        summary["recall"] = (summary.mean_recovered / summary.index.to_series()).round(3)
        print(summary.round(2).to_string())

        table = fit(group)
        if table.empty:
            print("  no model converged")
            continue
        print("\n" + table.drop(columns="_k").to_string(index=False))

        best = table.iloc[0]
        print(f"\n  best: {best.form}  (delta AICc to next: {table.delta_aicc.iloc[1]:.2f})")
        if best.form == "saturating":
            k = float(best._k)
            print(f"  capacity k = {k:.1f} items")
            print(f"  -> a graph built this way truncates degree near {k:.0f}: a node with")
            print(f"     30 true counterparties is recovered as roughly "
                  f"{saturating(30, k):.0f}, so hubs are clipped toward the mean.")
        elif best.form == "independent":
            print("  no cardinality effect: per-item recall is flat, and the")
            print("  bucket-average observation that motivated this was an artifact.")
        else:
            print("  per-item recall decays with competing items; total recovery")
            print("  eventually falls as documents name more counterparties.")

    print("\nAICc differences below ~2 are not a selection; report both.")


if __name__ == "__main__":
    main()
