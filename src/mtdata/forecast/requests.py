from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from ..shared.schema import DenoiseSpec, ForecastLibraryLiteral, TimeframeLiteral


def _reject_removed_field(values: Any, *, field_name: str, replacement: str) -> Any:
    if isinstance(values, dict) and field_name in values:
        raise ValueError(f"{field_name} was removed; use {replacement}")
    return values


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
    detail: Literal["compact", "full"] = "compact"

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_target(cls, values: Any) -> Any:
        return _reject_removed_field(values, field_name="target", replacement="quantity")


class ForecastConformalIntervalsRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    method: str = "theta"
    horizon: int = Field(12, ge=1)
    steps: int = Field(25, ge=1)
    spacing: int = Field(10, ge=1)
    ci_alpha: float = Field(0.1, gt=0.0, lt=1.0)
    denoise: Optional[DenoiseSpec] = None
    params: Optional[Dict[str, Any]] = None


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


class ForecastBarrierProbRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    horizon: int = Field(12, ge=1)
    method: str = "hmm_mc"
    direction: str = "long"
    tp_abs: Optional[float] = None
    sl_abs: Optional[float] = None
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None
    tp_pips: Optional[float] = None
    sl_pips: Optional[float] = None
    params: Optional[Dict[str, Any]] = None
    denoise: Optional[DenoiseSpec] = None
    barrier: float = 0.0
    mu: Optional[float] = None
    sigma: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_mc_method(cls, values: Any) -> Any:
        return _reject_removed_field(values, field_name="mc_method", replacement="method")


class ForecastBarrierOptimizeRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    horizon: int = Field(12, ge=1)
    method: str = "auto"
    direction: str = "long"
    mode: str = "pct"
    tp_min: float = 0.25
    tp_max: float = 1.5
    tp_steps: int = Field(7, ge=1)
    sl_min: float = 0.25
    sl_max: float = 2.5
    sl_steps: int = Field(9, ge=1)
    params: Optional[Dict[str, Any]] = None
    denoise: Optional[DenoiseSpec] = None
    objective: str = "ev"
    return_grid: bool = True
    top_k: Optional[int] = None
    output: str = "summary"
    viable_only: bool = False
    concise: bool = False
    grid_style: str = "fixed"
    preset: Optional[str] = None
    vol_window: int = Field(250, ge=1)
    vol_min_mult: float = 0.5
    vol_max_mult: float = 4.0
    vol_steps: int = Field(7, ge=1)
    vol_sl_extra: float = 1.8
    vol_floor_pct: float = 0.15
    vol_floor_pips: float = 8.0
    ratio_min: float = 0.5
    ratio_max: float = 4.0
    ratio_steps: int = Field(8, ge=1)
    refine: bool = False
    refine_radius: float = 0.3
    refine_steps: int = Field(5, ge=1)
    min_prob_win: Optional[float] = None
    max_prob_no_hit: Optional[float] = None
    max_median_time: Optional[float] = None
    fast_defaults: bool = False
    search_profile: str = "medium"


class ForecastVolatilityEstimateRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    horizon: int = Field(1, ge=1)
    method: str = "ewma"
    proxy: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    as_of: Optional[str] = None
    denoise: Optional[DenoiseSpec] = None
