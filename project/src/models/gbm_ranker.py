"""LightGBM cross-sectional ranker with purged walk-forward CV."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GBMConfig:
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 50
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    n_cv_splits: int = 3
    embargo_days: int = 21


@dataclass
class GBMRanker:
    config: GBMConfig = field(default_factory=GBMConfig)
    random_state: int = 7
    model_: Any = field(default=None, repr=False)
    feature_importance_: pd.Series | None = field(default=None, repr=False)

    def _base_params(self) -> dict:
        return {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "min_child_samples": self.config.min_child_samples,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GBMRanker:
        import lightgbm as lgb

        params = self._base_params()
        n_est = params.pop("n_estimators")

        splits = self._purged_cv_splits(X, y)
        if not splits:
            self.model_ = lgb.LGBMRegressor(**self._base_params()).fit(X.values, y.values)
            self.feature_importance_ = pd.Series(
                self.model_.feature_importances_, index=X.columns, name="importance"
            )
            return self

        best_n = self._tune_n_estimators(X, y, splits, params, n_est)
        final_params = self._base_params()
        final_params["n_estimators"] = best_n
        self.model_ = lgb.LGBMRegressor(**final_params).fit(X.values, y.values)
        self.feature_importance_ = pd.Series(
            self.model_.feature_importances_, index=X.columns, name="importance"
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model_.predict(X.values)
        return pd.Series(preds, index=X.index, name="prediction")

    def _purged_cv_splits(
        self, X: pd.DataFrame, y: pd.Series
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Time-series purged CV: split by date, embargo between train/test."""
        if not isinstance(X.index, pd.MultiIndex):
            return []

        dates = X.index.get_level_values(0).unique().sort_values()
        n_dates = len(dates)
        if n_dates < self.config.n_cv_splits + 1:
            return []

        fold_size = n_dates // (self.config.n_cv_splits + 1)
        splits = []

        for k in range(self.config.n_cv_splits):
            test_start = fold_size * (k + 1)
            test_end = min(test_start + fold_size, n_dates)
            test_dates = dates[test_start:test_end]
            if len(test_dates) == 0:
                continue

            embargo_cutoff = test_dates.min() - pd.Timedelta(days=self.config.embargo_days)
            train_dates = dates[dates <= embargo_cutoff]
            if len(train_dates) < fold_size:
                continue

            all_dates = X.index.get_level_values(0)
            train_idx = np.where(all_dates.isin(train_dates))[0]
            test_idx = np.where(all_dates.isin(test_dates))[0]
            splits.append((train_idx, test_idx))

        return splits

    @staticmethod
    def _tune_n_estimators(
        X: pd.DataFrame,
        y: pd.Series,
        splits: list[tuple[np.ndarray, np.ndarray]],
        params: dict,
        max_n: int,
    ) -> int:
        """Early-stop on purged CV to find optimal n_estimators."""
        import lightgbm as lgb

        best_n_per_fold = []
        for train_idx, test_idx in splits:
            X_tr, y_tr = X.values[train_idx], y.values[train_idx]
            X_val, y_val = X.values[test_idx], y.values[test_idx]

            model = lgb.LGBMRegressor(n_estimators=max_n, **params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
            best_n_per_fold.append(model.best_iteration_ or max_n)

        return max(20, int(np.median(best_n_per_fold)))
