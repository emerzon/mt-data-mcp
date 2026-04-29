from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from ..shared.schema import (
    CompactFullDetailLiteral,
    CompactStandardFullDetailLiteral,
    DenoiseSpec,
    ForecastLibraryLiteral,
    TimeframeLiteral,
)
from ..utils.barriers import (
    normalize_trade_direction,
    validate_barrier_unit_family_exclusivity,
)


def _reject_removed_field(values: Any, *, field_name: str, replacement: str) -> Any:
    if isinstance(values, dict) and field_name in values:
        raise ValueError(f"{field_name} was removed; use {replacement}")
    return values


def _normalize_direction_alias(value: Optional[str]) -> Optional[str]:
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
    horizon: int = Field(12, ge=1)
    lookback: Optional[int] = Field(None, ge=1)
    as_of: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    ci_alpha: Optional[float] = Field(0.05, ge=0.0, le=0.5)
    quantity: Literal["price", "return", "volatility"] = "price"
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
        description="Explicit trained-model params_hash to use for prediction. Skips training if the model exists in the store.",
    )
    detail: CompactStandardFullDetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_target(cls, values: Any) -> Any:
        return _reject_removed_field(values, field_name="target", replacement="quantity")


class ForecastBacktestRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    horizon: int = Field(12, ge=1)
    steps: int = Field(5, ge=1)
    spacing: int = Field(20, ge=1)
    methods: Optional[List[str]] = None
    params_per_method: Optional[Dict[str, Any]] = None
    quantity: Literal["price", "return", "volatility"] = "price"
    denoise: Optional[DenoiseSpec] = None
    params: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    dimred_method: Optional[str] = None
    dimred_params: Optional[Dict[str, Any]] = None
    slippage_bps: float = 0.0
    trade_threshold: float = 0.0
    detail: CompactFullDetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_target(cls, values: Any) -> Any:
        return _reject_removed_field(values, field_name="target", replacement="quantity")


class StrategyBacktestRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    strategy: Literal["sma_cross", "ema_cross", "rsi_reversion"] = "sma_cross"
    lookback: int = Field(200, ge=5)
    detail: CompactFullDetailLiteral = "compact"
    position_mode: Literal["long_only", "long_short"] = "long_short"
    fast_period: int = Field(10, ge=1)
    slow_period: int = Field(30, ge=2)
    rsi_length: int = Field(14, ge=1)
    oversold: float = Field(30.0, gt=0.0, lt=100.0)
    overbought: float = Field(70.0, gt=0.0, lt=100.0)
    max_hold_bars: Optional[int] = Field(None, ge=1)
    slippage_bps: float = 0.0

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
    horizon: int = Field(12, ge=1)
    steps: int = Field(25, ge=1)
    spacing: int = Field(20, ge=1)
    ci_alpha: float = Field(0.1, gt=0.0, lt=1.0)
    denoise: Optional[DenoiseSpec] = None
    params: Optional[Dict[str, Any]] = None

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
    horizon: int = Field(12, ge=1)
    steps: int = Field(5, ge=1)
    spacing: int = Field(20, ge=1)
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
    detail: CompactFullDetailLiteral = "compact"
    detail: CompactFullDetailLiteral = "compact"


class ForecastTuneOptunaRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    method: Optional[str] = "theta"
    methods: Optional[List[str]] = None
    horizon: int = Field(12, ge=1)
    steps: int = Field(5, ge=1)
    spacing: int = Field(20, ge=1)
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
    detail: CompactFullDetailLiteral = "compact"
    detail: CompactFullDetailLiteral = "compact"


class ForecastBarrierProbRequest(BaseModel):
    model_config = {"populate_by_name": True}

    symbol: str
    timeframe: TimeframeLiteral = "H1"
    horizon: int = Field(12, ge=1)
    method: str = "hmm_mc"
    direction: str = "long"
    tp_abs: Optional[float] = None
    sl_abs: Optional[float] = None
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None
    tp_ticks: Optional[float] = Field(
        None,
        validation_alias=AliasChoices("tp_ticks", "tp_pips"),
    )
    sl_ticks: Optional[float] = Field(
        None,
        validation_alias=AliasChoices("sl_ticks", "sl_pips"),
    )
    params: Optional[Dict[str, Any]] = None
    denoise: Optional[DenoiseSpec] = None
    barrier: float = 0.0
    mu: Optional[float] = None
    sigma: Optional[float] = None
    detail: CompactStandardFullDetailLiteral = "compact"

    @property
    def tp_pips(self) -> Optional[float]:
        """Legacy alias for tick-size barrier distance."""
        return self.tp_ticks

    @property
    def sl_pips(self) -> Optional[float]:
        """Legacy alias for tick-size barrier distance."""
        return self.sl_ticks

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_mc_method(cls, values: Any) -> Any:
        return _reject_removed_field(values, field_name="mc_method", replacement="method")

    @model_validator(mode="before")
    @classmethod
    def _validate_barrier_unit_families(cls, values: Any) -> Any:
        return validate_barrier_unit_family_exclusivity(values)

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_direction_alias(value)


class ForecastOptimizeHintsRequest(BaseModel):
    symbol: str
    timeframe: Optional[TimeframeLiteral] = None
    timeframes: Optional[List[TimeframeLiteral]] = None
    methods: Optional[List[str]] = None
    horizon: int = Field(12, ge=1)
    steps: int = Field(5, ge=1)
    spacing: int = Field(20, ge=1)
    population: int = Field(20, ge=1, le=100)
    generations: int = Field(15, ge=1, le=100)
    crossover_rate: float = Field(0.6, ge=0.0, le=1.0)
    mutation_rate: float = Field(0.3, ge=0.0, le=1.0)
    fitness_metric: str = "composite"
    fitness_weights: Optional[Dict[str, float]] = None
    seed: int = 42
    max_search_time_seconds: Optional[float] = None
    denoise: Optional[DenoiseSpec] = None
    features: Optional[Dict[str, Any]] = None
    include_feature_genes: bool = False
    top_n: int = Field(5, ge=1, le=20)
    dimred_method: Optional[str] = None
    dimred_params: Optional[Dict[str, Any]] = None
    detail: CompactFullDetailLiteral = "compact"
    detail: CompactFullDetailLiteral = "compact"


class ForecastBarrierOptimizeRequest(BaseModel):
    model_config = {"populate_by_name": True, "extra": "forbid"}

    symbol: str
    timeframe: TimeframeLiteral = "H1"
    horizon: int = Field(12, ge=1)
    method: str = "auto"
    direction: str = "long"
    mode: str = "pct"
    params: Optional[Dict[str, Any]] = None
    denoise: Optional[DenoiseSpec] = None
    objective: str = "ev"
    top_k: Optional[int] = None
    viable_only: bool = True
    grid_style: str = "fixed"
    preset: Optional[str] = None
    search_profile: str = "medium"
    detail: CompactStandardFullDetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_output(cls, values: Any) -> Any:
        values = _reject_removed_field(values, field_name="output", replacement="detail")
        values = _reject_removed_field(values, field_name="output_mode", replacement="detail")
        return _reject_removed_field(values, field_name="format", replacement="detail")

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_direction_alias(value)

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode_alias(cls, value: Optional[str]) -> Optional[str]:
        text = str(value or "").strip().lower()
        if text == "ticks":
            return "pips"
        return value


class ForecastVolatilityEstimateRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    horizon: int = Field(12, ge=1)
    method: str = "ewma"
    proxy: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    as_of: Optional[str] = None
    denoise: Optional[DenoiseSpec] = None
    detail: CompactFullDetailLiteral = "compact"
