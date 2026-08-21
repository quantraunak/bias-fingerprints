"""Out-of-sample signal generation.

The model is refitted on a schedule and predicts every rebalance date in
between, which is how a desk actually operates: nobody retrains from scratch
every month, and pretending otherwise both flatters the backtest and makes it
unreasonably slow.

Every fit obeys the same rule -- training data stops `horizon + embargo` trading
days before the date being predicted -- so no fitted model has ever seen a label
whose window overlaps the day it is scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Config
from src.model import estimators, splits


@dataclass
class WalkForward:
    scores: pd.Series
    fits: list[dict] = field(default_factory=list)

    def importances(self) -> pd.DataFrame:
        rows = {fit["fitted_on"]: fit["importances"] for fit in self.fits if fit.get("importances") is not None}
        return pd.DataFrame(rows).T if rows else pd.DataFrame()


def month_end_trading_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last **actual trading day** of each month.

    The previous engine used `prices.resample("ME").last().index`, which returns
    calendar month-ends. Whenever the 31st fell on a weekend the rebalance date
    was not in the price index at all, the cross-section came back empty, and the
    engine skipped the month -- including its profit and loss. 55 of 180
    month-ends were not trading days, and 29% of the backtest simply had no
    returns. Taking the last observed index entry per month cannot do that.
    """
    series = pd.Series(index, index=index)
    return pd.DatetimeIndex(series.groupby([index.year, index.month]).last().to_numpy())


def run(features: pd.DataFrame, target: pd.Series, config: Config, refit_months: int = 12) -> WalkForward:
    trading_days = pd.DatetimeIndex(sorted(features.index.get_level_values("date").unique()))
    rebalance_dates = month_end_trading_days(trading_days)
    rebalance_dates = rebalance_dates[rebalance_dates >= pd.Timestamp(config.oos_start)]

    aligned = features.join(target, how="inner")
    labelled_features = aligned[features.columns]
    labelled_target = aligned["target"]

    scores, fits = [], []
    model, fitted_on = None, None

    for rebalance_date in rebalance_dates:
        needs_refit = fitted_on is None or (rebalance_date - fitted_on).days >= refit_months * 30
        if needs_refit:
            window = splits.training_window(
                rebalance_date,
                trading_days,
                config.label.horizon_days,
                config.model.embargo_days,
                config.model.train_years,
            )
            if window is None:
                continue
            fit = _fit(labelled_features, labelled_target, window, config)
            if fit is None:
                continue
            model, fitted_on = fit["model"], rebalance_date
            fits.append({k: v for k, v in fit.items() if k != "model"} | {"fitted_on": rebalance_date})

        if model is None:
            continue
        today = features.loc[features.index.get_level_values("date") == rebalance_date]
        if today.empty:
            continue
        scores.append(model.predict(today))

    combined = pd.concat(scores).rename("score") if scores else pd.Series(dtype=float, name="score")
    return WalkForward(scores=smooth(combined, config.model.signal_smoothing), fits=fits)


def smooth(scores: pd.Series, months: int) -> pd.Series:
    """Average each name's score over the trailing `months` rebalances.

    The factors carrying most of the information here are fundamental and change
    only when a filing lands, but the fitted score still moved enough to churn
    159% of the book every month -- about 1.1% a year in costs against a gross
    edge of the same order. Averaging the score is the standard way to separate
    the signal's genuine speed from the model's month-to-month noise, and it
    costs almost none of the information coefficient because the underlying
    factors barely move.
    """
    if months <= 1 or scores.empty:
        return scores
    wide = scores.unstack("ticker").sort_index()
    averaged = wide.rolling(months, min_periods=1).mean()
    out = averaged.stack(future_stack=True).rename("score")
    out.index.names = ["date", "ticker"]
    return out.dropna()


def _fit(features: pd.DataFrame, target: pd.Series, window: splits.TrainingWindow, config: Config) -> dict | None:
    dates = features.index.get_level_values("date")
    in_window = (dates >= window.train_start) & (dates <= window.train_end)
    if in_window.sum() < 5000:
        return None

    X, y = features[in_window], target[in_window]
    keep = splits.thin(X.index, config.model.sample_every_n_days)
    X, y = X[keep], y[keep]

    weights = splits.uniqueness_weights(
        pd.Series(X.index.get_level_values("date")), config.label.horizon_days
    )

    model = estimators.build(config.model)
    model.fit(X, y, weights)

    return {
        "model": model,
        "train_start": window.train_start,
        "train_end": window.train_end,
        "n_rows": int(len(X)),
        "n_dates": int(X.index.get_level_values("date").nunique()),
        "importances": getattr(model, "importances_", None),
        "blend": getattr(model, "weights_", None),
    }
