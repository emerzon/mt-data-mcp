from __future__ import annotations

from typing import Literal

TuningMetricLiteral = Literal[
    "avg_rmse",
    "avg_mae",
    "avg_directional_accuracy",
    "win_rate",
    "max_drawdown",
    "sharpe_ratio",
    "calmar_ratio",
    "annual_return",
    "avg_return_per_trade",
    "avg_win_loss_ratio",
    "kelly_fraction",
    "half_kelly_fraction",
]
TuningModeLiteral = Literal["auto", "min", "max"]


TUNING_METRIC_DIRECTIONS: dict[str, Literal["min", "max"]] = {
    "avg_rmse": "min",
    "avg_mae": "min",
    "avg_directional_accuracy": "max",
    "win_rate": "max",
    "max_drawdown": "min",
    "sharpe_ratio": "max",
    "calmar_ratio": "max",
    "annual_return": "max",
    "avg_return_per_trade": "max",
    "avg_win_loss_ratio": "max",
    "kelly_fraction": "max",
    "half_kelly_fraction": "max",
}

ANNUALIZED_TUNING_METRICS = frozenset(
    {"sharpe_ratio", "calmar_ratio", "annual_return"}
)
MIN_ANNUALIZED_TUNING_TRADES = 30


def resolve_tuning_mode(metric: str, mode: str = "auto") -> Literal["min", "max"]:
    metric_key = str(metric or "").strip().lower()
    mode_key = str(mode or "auto").strip().lower()
    if mode_key == "max":
        return "max"
    if mode_key == "min":
        return "min"
    return TUNING_METRIC_DIRECTIONS.get(metric_key, "min")
