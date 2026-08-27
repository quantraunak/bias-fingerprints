"""Typed configuration loaded from a single YAML file.

One dataclass per pipeline stage. Nothing here has behaviour; if a value needs
to be derived, it is derived at the point of use so the config stays readable
as a description of the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"


@dataclass(frozen=True)
class DataConfig:
    start: str
    end: str
    benchmark: str
    min_history_days: int
    max_missing_pct: float
    min_price: float
    min_dollar_volume: float
    sec_user_agent: str


@dataclass(frozen=True)
class LabelConfig:
    horizon_days: int
    target: str  # "rank" | "zscore" | "raw"


@dataclass(frozen=True)
class ModelConfig:
    kind: str  # "lgbm" | "ridge" | "ensemble"
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    feature_fraction: float
    bagging_fraction: float
    lambda_l2: float
    ridge_alphas: tuple[float, ...]
    train_years: int
    embargo_days: int
    n_folds: int
    sample_every_n_days: int
    signal_smoothing: int
    seed: int = 7


@dataclass(frozen=True)
class PortfolioConfig:
    quantile: float
    gross_leverage: float
    max_weight: float
    beta_tolerance: float
    turnover_penalty: float
    information_coefficient: float
    cov_window: int
    beta_window: int
    solver: str
    vol_target: float
    dd_threshold: float
    dd_min_scale: float
    # "deciles" truncates to the tails before optimising; "full" hands the
    # optimiser the whole tradable cross-section and lets the risk term decide
    # how concentrated to be.
    selection: str = "deciles"
    max_cross_section: int = 400
    industry_tolerance: float = -1.0   # negative disables the constraint
    risk_in_objective: bool = True


@dataclass(frozen=True)
class CostConfig:
    commission_bps: float
    slippage_bps: float
    stress_bps: tuple[float, ...]
    # Annualised stock-borrow bps on SHORT notional. 40bp is general collateral
    # for liquid large caps; hard-to-borrow names run multiples of it, so this
    # is a floor rather than a central estimate.
    financing_bps: float = 40.0


@dataclass(frozen=True)
class Config:
    data: DataConfig
    label: LabelConfig
    model: ModelConfig
    portfolio: PortfolioConfig
    costs: CostConfig
    oos_start: str
    rebalance: str
    factors: list[str] = field(default_factory=list)

    @staticmethod
    def load(path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        return Config(
            data=DataConfig(**raw["data"]),
            label=LabelConfig(**raw["label"]),
            model=ModelConfig(
                **{**raw["model"], "ridge_alphas": tuple(raw["model"]["ridge_alphas"])}
            ),
            portfolio=PortfolioConfig(**raw["portfolio"]),
            costs=CostConfig(**{**raw["costs"], "stress_bps": tuple(raw["costs"]["stress_bps"])}),
            oos_start=raw["oos_start"],
            rebalance=raw["rebalance"],
            factors=raw["factors"],
        )
