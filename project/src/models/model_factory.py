"""Factory for constructing ranker models from config."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class Ranker(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: ...
    def predict(self, X: pd.DataFrame) -> pd.Series: ...


def build_ranker(config: dict) -> Ranker:
    """Instantiate a ranker from the 'model' section of the YAML config."""
    model_type = config["type"]

    if model_type in ("ridge", "elasticnet"):
        from src.models.linear_ranker import LinearRanker
        return LinearRanker(
            model_type=model_type,
            alpha=config.get("alpha", 1.0),
            l1_ratio=config.get("l1_ratio", 0.5),
        )

    if model_type == "lightgbm":
        from src.models.gbm_ranker import GBMConfig, GBMRanker
        gbm_cfg = config.get("lightgbm", {})
        return GBMRanker(config=GBMConfig(**gbm_cfg))

    raise ValueError(f"Unknown model type: {model_type}")
