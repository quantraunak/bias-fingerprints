"""Training-set construction for overlapping labels.

A 21-day forward label observed every day overlaps its twenty neighbours. Three
things follow, and the old pipeline handled none of them:

1. **Purging.** A training label that straddles the prediction date has already
   seen the future. Training data must stop `horizon` days before the date being
   predicted, not on it.
2. **Embargo.** Serial correlation leaks a little further still, so a further
   `embargo` days are dropped after the purge boundary (Lopez de Prado, 2018).
3. **Effective sample size.** Nominally 300k rows; because of the overlap the
   independent count is closer to 15k. Thinning the sample and weighting each
   observation by how much of its label window it owns stops the model, and more
   importantly the early-stopping rule, from believing it has twenty times the
   evidence it really has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainingWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # inclusive, already purged and embargoed
    predict_on: pd.Timestamp


def training_window(
    predict_on: pd.Timestamp,
    trading_days: pd.DatetimeIndex,
    horizon_days: int,
    embargo_days: int,
    train_years: int,
) -> TrainingWindow | None:
    """Last usable training date, given the label horizon and the embargo."""
    history = trading_days[trading_days <= predict_on]
    cutoff = horizon_days + embargo_days
    if len(history) <= cutoff:
        return None
    train_end = history[-(cutoff + 1)]
    train_start = train_end - pd.DateOffset(years=train_years)
    return TrainingWindow(train_start, train_end, predict_on)


def thin(index: pd.MultiIndex, every_n_days: int) -> np.ndarray:
    """Boolean mask keeping every Nth trading date, most recent date always kept."""
    dates = index.get_level_values("date")
    unique = pd.DatetimeIndex(sorted(dates.unique()))
    keep = set(unique[::-1][::every_n_days])
    return dates.isin(keep)


def uniqueness_weights(dates: pd.Series, horizon_days: int) -> np.ndarray:
    """Average uniqueness of each label window (Lopez de Prado, 2018, ch. 4).

    Two labels sampled a day apart share twenty-twentyoneths of their window and
    should not count as two independent observations. Each sample is weighted by
    the mean of 1/concurrency over the days its label spans, so a densely
    sampled stretch contributes no more total evidence than a sparse one.
    """
    unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
    position = pd.Series(np.arange(len(unique_dates)), index=unique_dates)
    start = position.reindex(dates).to_numpy()
    end = np.minimum(start + horizon_days, len(unique_dates))

    concurrency = np.zeros(len(unique_dates) + 1)
    np.add.at(concurrency, start, 1)
    np.add.at(concurrency, end, -1)
    concurrency = np.cumsum(concurrency)[: len(unique_dates)]

    inverse = np.divide(1.0, concurrency, out=np.zeros_like(concurrency), where=concurrency > 0)
    cumulative = np.concatenate([[0.0], np.cumsum(inverse)])
    span = np.maximum(end - start, 1)
    weights = (cumulative[end] - cumulative[start]) / span
    return weights / weights.mean()
