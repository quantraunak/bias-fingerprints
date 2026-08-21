"""Models that map the factor matrix to a cross-sectional score.

Three estimators, one interface. Which one is used is a config choice, and the
ensemble weights them by the information coefficient they actually achieved on
held-out folds rather than by assertion.

A note on the ranking objective. LightGBM's `lambdarank` optimises NDCG, which
is deliberately top-heavy: it cares about getting the best items right and
discounts everything below. That is the correct loss for search results and the
wrong one for a market-neutral book, which earns as much from the short tail as
the long. `rank_xendcg` is less top-heavy, and plain regression on the
rank-transformed target is symmetric by construction. All three are available so
the choice can be settled by measured IC instead of fashion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import ModelConfig


class Estimator(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series, weights: np.ndarray | None) -> "Estimator": ...
    def predict(self, X: pd.DataFrame) -> pd.Series: ...


# ------------------------------------------------------------------ LightGBM


@dataclass
class GradientBoosted:
    config: ModelConfig
    objective: str = "regression"  # or "lambdarank" / "rank_xendcg"
    n_grades: int = 10
    seed: int = 7  # threaded from config so robustness runs can vary it
    model_: object = field(default=None, repr=False)
    importances_: pd.Series | None = field(default=None, repr=False)

    def _params(self) -> dict:
        return {
            "n_estimators": self.config.n_estimators,
            "learning_rate": self.config.learning_rate,
            "num_leaves": self.config.num_leaves,
            "min_child_samples": self.config.min_child_samples,
            "colsample_bytree": self.config.feature_fraction,
            "subsample": self.config.bagging_fraction,
            "subsample_freq": 1,
            "reg_lambda": self.config.lambda_l2,
            "random_state": self.seed,
            "n_jobs": -1,
            "verbose": -1,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series, weights: np.ndarray | None = None) -> "GradientBoosted":
        import lightgbm as lgb

        if self.objective == "regression":
            self.model_ = lgb.LGBMRegressor(**self._params())
            self.model_.fit(X.values, y.values, sample_weight=weights)
        else:
            from src.model.target import to_grades

            order = X.index.get_level_values("date").argsort(kind="stable")
            X, y = X.iloc[order], y.iloc[order]
            weights = None if weights is None else weights[order]
            grades = to_grades(y, self.n_grades)
            group = y.groupby(level="date", sort=False).size().to_numpy()

            self.model_ = lgb.LGBMRanker(
                objective=self.objective,
                label_gain=list(range(self.n_grades)),
                **self._params(),
            )
            self.model_.fit(X.values, grades.values, group=group, sample_weight=weights)

        self.importances_ = pd.Series(self.model_.feature_importances_, index=X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model_.predict(X.values), index=X.index, name="score")


# --------------------------------------------------------------------- ridge


@dataclass
class RidgeLinear:
    config: ModelConfig
    models_: dict = field(default_factory=dict, repr=False)
    alpha_: float | None = None
    coefficients_: pd.Series | None = field(default=None, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series, weights: np.ndarray | None = None) -> "RidgeLinear":
        """Ridge over the full factor set, alpha chosen on the last fifth of the window.

        Heavy shrinkage on many weakly informative predictors beats selecting a
        few (Kelly, Malamud, Zhou 2024): the ridge keeps every factor and lets
        the penalty decide how much of each to believe.
        """
        filled = X.fillna(0.0)
        dates = X.index.get_level_values("date")
        split = dates.unique()[int(len(dates.unique()) * 0.8)]
        fit_mask, validate_mask = dates < split, dates >= split

        best_alpha, best_ic = self.config.ridge_alphas[0], -np.inf
        for alpha in self.config.ridge_alphas:
            model = Ridge(alpha=alpha).fit(
                filled[fit_mask], y[fit_mask], sample_weight=None if weights is None else weights[fit_mask]
            )
            prediction = pd.Series(model.predict(filled[validate_mask]), index=X.index[validate_mask])
            ic = _mean_rank_ic(prediction, y[validate_mask])
            if ic > best_ic:
                best_alpha, best_ic = alpha, ic

        self.alpha_ = best_alpha
        final = Ridge(alpha=best_alpha).fit(filled, y, sample_weight=weights)
        self.models_ = {"ridge": final}
        self.coefficients_ = pd.Series(final.coef_, index=X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        prediction = self.models_["ridge"].predict(X.fillna(0.0))
        return pd.Series(prediction, index=X.index, name="score")


# ------------------------------------------------------------------ ensemble


@dataclass
class Ensemble:
    members: dict[str, Estimator]
    weights_: dict[str, float] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series, weights: np.ndarray | None = None) -> "Ensemble":
        """Fit each member, then weight by validation IC on the tail of the window."""
        dates = X.index.get_level_values("date")
        unique = dates.unique()
        split = unique[int(len(unique) * 0.8)]
        fit_mask, validate_mask = dates < split, dates >= split

        scores = {}
        for name, member in self.members.items():
            member.fit(
                X[fit_mask], y[fit_mask], None if weights is None else weights[fit_mask]
            )
            scores[name] = max(_mean_rank_ic(member.predict(X[validate_mask]), y[validate_mask]), 0.0)

        total = sum(scores.values())
        self.weights_ = (
            {name: score / total for name, score in scores.items()}
            if total > 0
            else {name: 1.0 / len(self.members) for name in self.members}
        )

        for name, member in self.members.items():
            member.fit(X, y, weights)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        blended = sum(
            self.weights_[name] * _rank_within_date(member.predict(X))
            for name, member in self.members.items()
        )
        return blended.rename("score")


# ------------------------------------------------------------------- helpers


def _rank_within_date(scores: pd.Series) -> pd.Series:
    """Members produce scores on different scales; rank makes them blendable."""
    return scores.groupby(level="date").rank(pct=True) - 0.5


def _mean_rank_ic(prediction: pd.Series, actual: pd.Series) -> float:
    frame = pd.DataFrame({"p": prediction, "a": actual}).dropna()
    if frame.empty:
        return 0.0
    per_date = frame.groupby(level="date").apply(
        lambda block: block["p"].corr(block["a"], method="spearman") if len(block) > 5 else np.nan
    )
    return float(per_date.mean(skipna=True) or 0.0)


def build(config: ModelConfig) -> Estimator:
    seed = config.seed
    if config.kind == "lgbm":
        return GradientBoosted(config, seed=seed)
    if config.kind == "lgbm_rank":
        return GradientBoosted(config, objective="rank_xendcg", seed=seed)
    if config.kind == "ridge":
        return RidgeLinear(config)
    if config.kind == "ensemble":
        return Ensemble(
            {
                "lgbm": GradientBoosted(config, seed=seed),
                "lgbm_rank": GradientBoosted(config, objective="rank_xendcg", seed=seed),
                "ridge": RidgeLinear(config),
            }
        )
    raise ValueError(f"Unknown model kind '{config.kind}'")
