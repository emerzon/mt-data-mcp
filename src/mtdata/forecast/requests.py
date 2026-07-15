from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ..shared.schema import (
    DenoiseSpec,
    DetailLiteral,
    ForecastLibraryLiteral,
    TimeframeLiteral,
    reject_removed_field,
)
from ..utils.barriers import (
    normalize_trade_direction,
    validate_barrier_unit_family_exclusivity,
)

MAX_FORECAST_HORIZON = 500
MAX_BACKTEST_STEPS = 200
MAX_BACKTEST_SPACING = 10_000


def _normalize_trade_direction_alias(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized, error = normalize_trade_direction(value)
    if error is None and normalized is not None:
        return normalized
    return value


class ForecastGenerateRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    library: ForecastLibraryLiteral = "native"
    method: str = "theta"
    horizon: int = Field(
        12,
        ge=1,
        le=MAX_FORECAST_HORIZON,
        description="Number of future bars to forecast at the requested timeframe.",
    )
    lookback: Optional[int] = Field(None, ge=1)
    as_of: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    ci_alpha: Optional[float] = Field(
        None,
        ge=0.0,
        le=0.5,
        description="Interval tail probability; confidence is 1 - ci_alpha. Use None to omit intervals.",
    )
    quantity: Literal["price", "return", "volatility"] = Field(
        "price",
        description="Forecast target: price levels, returns, or volatility.",
    )
    proxy: Optional[Literal["squared_return", "abs_return", "log_r2"]] = None
    denoise: Optional[DenoiseSpec] = None
    features: Optional[Dict[str, Any]] = None
    dimred_method: Optional[str] = None
    dimred_params: Optional[Dict[str, Any]] = None
    target_spec: Optional[Dict[str, Any]] = None
    async_mode: bool = Field(
        False,
        description="When True, heavy methods submit a background training task and return a task_id instead of blocking.",
    )
    model_id: Optional[str] = Field(
        None,
        description=(
            "Canonical trained model ID (method/data_scope/params_hash) returned by "
            "forecast_train or forecast_models_list. Skips training when found."
        ),
    )
    detail: DetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_target(cls, values: Any) -> Any:
        return reject_removed_field(values, field_name="target", replacement="quantity")

    @model_validator(mode="after")
    def _validate_time_window(self) -> "ForecastGenerateRequest":
        if self.as_of and (self.start or self.end):
            raise ValueError("as_of cannot be combined with start/end")
        return self


def _normalize_methods_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    methods = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    return methods or None


class ForecastBacktestRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON, description="Bars forecast after each backtest anchor.")
    steps: int = Field(5, ge=1, le=MAX_BACKTEST_STEPS, description="Number of rolling-origin backtest anchors to run.")
    spacing: int = Field(20, ge=1, le=MAX_BACKTEST_SPACING, description="Bars between consecutive rolling-origin anchors.")
    start: Optional[str] = None
    end: Optional[str] = None
    methods: Optional[List[str]] = None
    params_per_method: Optional[Dict[str, Any]] = None
    quantity: Literal["price", "return", "volatility"] = "price"
    denoise: Optional[DenoiseSpec] = None
    params: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    dimred_method: Optional[str] = None
    dimred_params: Optional[Dict[str, Any]] = None
    slippage_bps: float = 0.0
    trade_threshold: float = Field(0.0, ge=0.0)
    detail: DetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _normalize_methods_field(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        out = dict(values)
        # Singular `method` is not accepted; use plural `methods` only.
        reject_removed_field(out, field_name="method", replacement="methods")
        if "methods" in out:
            out["methods"] = _normalize_methods_value(out["methods"])
        return out

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_target(cls, values: Any) -> Any:
        return reject_removed_field(values, field_name="target", replacement="quantity")


class StrategyBacktestRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    strategy: Literal["sma_cross", "ema_cross", "rsi_reversion"] = "sma_cross"
    lookback: int = Field(500, ge=5)
    start: Optional[str] = None
    end: Optional[str] = None
    detail: DetailLiteral = "compact"
    position_mode: Literal["long_only", "long_short"] = "long_short"
    fast_period: int = Field(10, ge=1)
    slow_period: int = Field(30, ge=2)
    rsi_length: int = Field(14, ge=1)
    oversold: float = Field(30.0, gt=0.0, lt=100.0)
    overbought: float = Field(70.0, gt=0.0, lt=100.0)
    max_hold_bars: Optional[int] = Field(None, ge=1)
    cost_model: Literal["mt5_observed", "fixed"] = "mt5_observed"
    spread_bps: Optional[float] = Field(None, ge=0.0)
    slippage_bps: float = 1.0

    @model_validator(mode="after")
    def _validate_strategy_thresholds(self) -> "StrategyBacktestRequest":
        if self.strategy in {"sma_cross", "ema_cross"} and self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be less than slow_period")
        if self.oversold >= self.overbought:
            raise ValueError("oversold must be less than overbought")
        return self


class ForecastConformalIntervalsRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    method: str = "theta"
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON)
    steps: int = Field(
        50,
        ge=1,
        le=MAX_BACKTEST_STEPS,
        description="Number of rolling-origin calibration anchors; default 50 for stabler interval quantiles.",
    )
    spacing: int = Field(20, ge=1, le=MAX_BACKTEST_SPACING, description="Bars between consecutive calibration anchors.")
    ci_alpha: float = Field(
        0.1,
        gt=0.0,
        lt=1.0,
        description=(
            "Residual-quantile alpha for rolling-backtest absolute-error bands "
            "(not a true conformal coverage guarantee). 0.10 ≈ 90% empirical "
            "target, 0.05 ≈ 95%. Values outside 0.05-0.20 are warned."
        ),
    )
    denoise: Optional[DenoiseSpec] = None
    params: Optional[Dict[str, Any]] = None
    detail: DetailLiteral = "compact"

    @model_validator(mode="after")
    def _validate_spacing(self) -> "ForecastConformalIntervalsRequest":
        if self.steps > 1 and self.spacing < self.horizon:
            raise ValueError(
                "spacing must be greater than or equal to horizon when steps > 1 "
                f"(got spacing={self.spacing}, horizon={self.horizon})"
            )
        return self


class ForecastTuneGeneticRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    method: Optional[str] = "theta"
    methods: Optional[List[str]] = None
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON, description="Bars forecast after each tuning backtest anchor.")
    steps: int = Field(5, ge=1, le=MAX_BACKTEST_STEPS, description="Number of rolling-origin backtest anchors per trial.")
    spacing: int = Field(20, ge=1, le=MAX_BACKTEST_SPACING, description="Bars between consecutive tuning backtest anchors.")
    search_space: Optional[Dict[str, Any]] = None
    metric: str = "avg_rmse"
    mode: str = "min"
    population: int = Field(12, ge=1)
    generations: int = Field(10, ge=1)
    crossover_rate: float = 0.6
    mutation_rate: float = 0.3
    seed: int = 42
    trade_threshold: float = 0.0
    denoise: Optional[DenoiseSpec] = None
    features: Optional[Dict[str, Any]] = None
    dimred_method: Optional[str] = None
    dimred_params: Optional[Dict[str, Any]] = None
    detail: DetailLiteral = "compact"


class ForecastTuneOptunaRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    method: Optional[str] = "theta"
    methods: Optional[List[str]] = None
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON, description="Bars forecast after each tuning backtest anchor.")
    steps: int = Field(5, ge=1, le=MAX_BACKTEST_STEPS, description="Number of rolling-origin backtest anchors per trial.")
    spacing: int = Field(20, ge=1, le=MAX_BACKTEST_SPACING, description="Bars between consecutive tuning backtest anchors.")
    search_space: Optional[Dict[str, Any]] = None
    metric: str = "avg_rmse"
    mode: str = "min"
    n_trials: int = Field(40, ge=1)
    timeout: Optional[float] = None
    n_jobs: int = 1
    sampler: Literal["tpe", "random", "cmaes"] = "tpe"
    pruner: Literal["median", "none", "hyperband", "percentile"] = "median"
    study_name: Optional[str] = None
    storage: Optional[str] = None
    seed: int = 42
    trade_threshold: float = 0.0
    denoise: Optional[DenoiseSpec] = None
    features: Optional[Dict[str, Any]] = None
    dimred_method: Optional[str] = None
    dimred_params: Optional[Dict[str, Any]] = None
    detail: DetailLiteral = "compact"


class ForecastBarrierProbRequest(BaseModel):
    model_config = {"populate_by_name": True, "extra": "forbid"}

    symbol: str
    timeframe: TimeframeLiteral = "H1"
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON)
    method: str = "mc_gbm_bb"
    direction: str = "long"
    same_bar_policy: Literal["sl_first", "tp_first", "neutral"] = "sl_first"
    tp_abs: Optional[float] = Field(None, description="Take-profit absolute price. Do not combine with percent or tick barriers.")
    sl_abs: Optional[float] = Field(None, description="Stop-loss absolute price. Do not combine with percent or tick barriers.")
    tp_pct: Optional[float] = Field(None, description="Take-profit percent move, e.g. 2.0 for 2%. Do not combine with price or tick barriers.")
    sl_pct: Optional[float] = Field(None, description="Stop-loss percent move, e.g. 1.0 for 1%. Do not combine with price or tick barriers.")
    tp_ticks: Optional[float] = Field(None, description="Take-profit distance in trade ticks. Do not combine with price or percent barriers.")
    sl_ticks: Optional[float] = Field(None, description="Stop-loss distance in trade ticks. Do not combine with price or percent barriers.")
    params: Optional[Dict[str, Any]] = None
    denoise: Optional[DenoiseSpec] = None
    barrier: float = 0.0
    mu: Optional[float] = None
    sigma: Optional[float] = None
    detail: DetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_mc_method(cls, values: Any) -> Any:
        return reject_removed_field(values, field_name="mc_method", replacement="method")

    @model_validator(mode="before")
    @classmethod
    def _validate_barrier_unit_families(cls, values: Any) -> Any:
        return validate_barrier_unit_family_exclusivity(values)

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_trade_direction_alias(value)


class ForecastOptimizeHintsRequest(BaseModel):
    symbol: str
    timeframe: Optional[TimeframeLiteral] = None
    timeframes: Optional[List[TimeframeLiteral]] = None
    methods: Optional[List[str]] = None
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON, description="Bars forecast after each optimization backtest anchor.")
    steps: int = Field(5, ge=1, le=MAX_BACKTEST_STEPS, description="Number of rolling-origin backtest anchors per candidate.")
    spacing: int = Field(20, ge=1, le=MAX_BACKTEST_SPACING, description="Bars between consecutive optimization backtest anchors.")
    population: int = Field(8, ge=1, le=100)
    generations: int = Field(5, ge=1, le=100)
    crossover_rate: float = Field(0.6, ge=0.0, le=1.0)
    mutation_rate: float = Field(0.3, ge=0.0, le=1.0)
    fitness_metric: str = Field(
        "composite",
        description=(
            "Optimization objective. Composite uses trading metrics when available "
            "and falls back to forecast accuracy for flat backtests."
        ),
    )
    fitness_weights: Optional[Dict[str, float]] = None
    seed: int = 42
    max_search_time_seconds: Optional[float] = None
    denoise: Optional[DenoiseSpec] = None
    features: Optional[Dict[str, Any]] = None
    include_feature_genes: bool = False
    top_n: int = Field(5, ge=1, le=20)
    dimred_method: Optional[str] = None
    dimred_params: Optional[Dict[str, Any]] = None
    detail: DetailLiteral = "compact"


class ForecastBarrierOptimizeRequest(BaseModel):
    model_config = {"populate_by_name": True, "extra": "forbid"}

    symbol: str
    timeframe: TimeframeLiteral = "H1"
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON)
    method: str = "auto"
    direction: str = "long"
    same_bar_policy: Literal["sl_first", "tp_first", "neutral"] = "sl_first"
    mode: str = "pct"
    params: Optional[Dict[str, Any]] = None
    denoise: Optional[DenoiseSpec] = None
    objective: str = "ev"
    top_k: Optional[int] = None
    viable_only: bool = True
    tradable_only: bool = False
    min_ev: Optional[float] = None
    min_edge: Optional[float] = None
    min_kelly: Optional[float] = None
    grid_style: str = "fixed"
    preset: Optional[str] = None
    search_profile: str = "medium"
    detail: DetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_output(cls, values: Any) -> Any:
        values = reject_removed_field(values, field_name="output", replacement="extras")
        values = reject_removed_field(values, field_name="output_mode", replacement="extras")
        return reject_removed_field(values, field_name="format", replacement="json")

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_trade_direction_alias(value)


class ForecastVolatilityEstimateRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON)
    method: str = "ewma"
    proxy: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    as_of: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    denoise: Optional[DenoiseSpec] = None
    detail: DetailLiteral = "compact"

    @model_validator(mode="after")
    def _validate_time_window(self) -> "ForecastVolatilityEstimateRequest":
        if self.as_of and (self.start or self.end):
            raise ValueError("as_of cannot be combined with start/end")
        return self
