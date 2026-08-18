"""Pydantic request models for the Web API transport."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..forecast.requests import (
    MAX_BACKTEST_SPACING,
    MAX_BACKTEST_STEPS,
    MAX_FORECAST_HORIZON,
    ForecastBacktestRequest,
    ForecastGenerateRequest,
    ForecastVolatilityEstimateRequest,
)
from ..shared.schema import (
    DetailLiteral,
    DimensionalityReductionSpec,
    ForecastLibraryLiteral,
    TimeframeLiteral,
    reject_removed_field,
)
from .trading.ideas_requests import DEFAULT_RISK_PCT, TradeIdeaComposeRequest


class _ForecastWebBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForecastPriceBody(_ForecastWebBody):
    symbol: str
    timeframe: str = Field("H1")
    library: ForecastLibraryLiteral = Field("native")
    method: str = Field("theta")
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON)
    lookback: Optional[int] = Field(None, ge=1)
    as_of: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    ci_alpha: float = Field(0.0, ge=0.0, le=0.5)
    quantity: Literal["price", "return", "volatility"] = Field("price")
    denoise: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    dimred: Optional[DimensionalityReductionSpec] = None
    target_spec: Optional[Dict[str, Any]] = None
    async_mode: bool = Field(
        False,
        description=(
            "Submit trainable methods to the background task runtime and return "
            "a task_id instead of waiting for training."
        ),
    )
    model_id: Optional[str] = Field(
        None,
        description="Stored model ID to reuse instead of training a new artifact.",
    )
    detail: DetailLiteral = Field("compact")

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_fields(cls, values: Any) -> Any:
        return reject_removed_field(values, field_name="target", replacement="quantity")

    def to_domain_request(self) -> ForecastGenerateRequest:
        return ForecastGenerateRequest(
            symbol=self.symbol,
            timeframe=self.timeframe,
            library=self.library,
            method=self.method,
            horizon=self.horizon,
            lookback=self.lookback,
            as_of=self.as_of,
            start=self.start,
            end=self.end,
            params=self.params,
            ci_alpha=self.ci_alpha,
            quantity=self.quantity,
            denoise=self.denoise,
            features=self.features,
            dimred=self.dimred,
            target_spec=self.target_spec,
            async_mode=self.async_mode,
            model_id=self.model_id,
            detail=self.detail,
        )


class ForecastVolBody(_ForecastWebBody):
    symbol: str
    timeframe: str = Field("H1")
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON)
    method: str = Field("ewma")
    proxy: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    as_of: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    denoise: Optional[Dict[str, Any]] = None
    detail: DetailLiteral = Field("compact")

    def to_domain_request(self) -> ForecastVolatilityEstimateRequest:
        return ForecastVolatilityEstimateRequest(
            symbol=self.symbol,
            timeframe=self.timeframe,
            horizon=self.horizon,
            method=self.method,
            proxy=self.proxy,
            params=self.params,
            as_of=self.as_of,
            start=self.start,
            end=self.end,
            denoise=self.denoise,
            detail=self.detail,
        )


class BacktestBody(_ForecastWebBody):
    symbol: str
    timeframe: str = Field("H1")
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON)
    steps: int = Field(5, ge=1, le=MAX_BACKTEST_STEPS)
    spacing: int = Field(20, ge=1, le=MAX_BACKTEST_SPACING)
    methods: Optional[list[str]] = None
    params_per_method: Optional[Dict[str, Any]] = None
    quantity: Literal["price", "return", "volatility"] = Field("price")
    denoise: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    dimred: Optional[DimensionalityReductionSpec] = None
    slippage_bps: float = 0.0
    trade_threshold: float = Field(0.0, ge=0.0)
    detail: DetailLiteral = Field("compact")

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_fields(cls, values: Any) -> Any:
        return reject_removed_field(values, field_name="target", replacement="quantity")

    def to_domain_request(self) -> ForecastBacktestRequest:
        return ForecastBacktestRequest(
            symbol=self.symbol,
            timeframe=self.timeframe,
            horizon=self.horizon,
            steps=self.steps,
            spacing=self.spacing,
            methods=self.methods,
            params_per_method=self.params_per_method,
            quantity=self.quantity,
            denoise=self.denoise,
            params=self.params,
            features=self.features,
            dimred=self.dimred,
            slippage_bps=self.slippage_bps,
            trade_threshold=self.trade_threshold,
            detail=self.detail,
        )


class TradeIdeaBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    timeframe: TimeframeLiteral = Field("H1")
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON)
    direction: Literal["auto", "long", "short"] = Field("auto")
    template: Literal["quick", "standard"] = Field("quick")
    risk_pct: float = Field(DEFAULT_RISK_PCT, gt=0.0, le=100.0)
    as_of: Optional[str] = None
    detail: DetailLiteral = Field("compact")

    def to_domain_request(self) -> TradeIdeaComposeRequest:
        return TradeIdeaComposeRequest(
            symbol=self.symbol,
            timeframe=self.timeframe,
            horizon=self.horizon,
            direction=self.direction,
            template=self.template,
            risk_pct=self.risk_pct,
            as_of=self.as_of,
            detail=self.detail,
        )


class ToolInvokeBody(BaseModel):
    """Generic MCP tool invocation from the Web UI tool runner."""

    arguments: Dict[str, Any] = Field(default_factory=dict)
    confirm: bool = Field(
        False,
        description="Required true for live trade mutations and destructive model/task tools.",
    )
