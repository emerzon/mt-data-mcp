"""Pydantic request models for the Web API transport."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from ..forecast.requests import ForecastBacktestRequest, ForecastGenerateRequest


def _normalize_legacy_target(values: Any) -> Any:
    if not isinstance(values, dict):
        return values

    data = dict(values)
    legacy_target = data.pop("target", None)
    quantity = data.get("quantity")

    quantity_text = str(quantity).strip().lower() if quantity is not None else None
    legacy_target_text = str(legacy_target).strip().lower() if legacy_target is not None else None

    if quantity_text is None and legacy_target_text:
        data["quantity"] = legacy_target_text
        return data

    if (
        quantity_text is not None
        and legacy_target_text is not None
        and quantity_text != "volatility"
        and quantity_text != legacy_target_text
    ):
        raise ValueError("target is deprecated; use quantity, and if both are provided they must match")

    return data


class ForecastPriceBody(BaseModel):
    symbol: str
    timeframe: str = Field("H1")
    method: str = Field("theta")
    horizon: int = Field(12, ge=1)
    lookback: Optional[int] = Field(None, ge=1)
    as_of: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    ci_alpha: Optional[float] = Field(0.05, ge=0.0, le=0.5)
    quantity: Literal["price", "return", "volatility"] = Field("price")
    denoise: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    dimred_method: Optional[str] = None
    dimred_params: Optional[Dict[str, Any]] = None
    target_spec: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_target_alias(cls, values: Any) -> Any:
        return _normalize_legacy_target(values)

    def to_domain_request(self) -> ForecastGenerateRequest:
        return ForecastGenerateRequest(
            symbol=self.symbol,
            timeframe=self.timeframe,
            library="native",
            model=self.method,
            horizon=self.horizon,
            lookback=self.lookback,
            as_of=self.as_of,
            model_params=self.params,
            ci_alpha=self.ci_alpha,
            quantity=self.quantity,
            denoise=self.denoise,
            features=self.features,
            dimred_method=self.dimred_method,
            dimred_params=self.dimred_params,
            target_spec=self.target_spec,
            method=self.method,
        )


class ForecastVolBody(BaseModel):
    symbol: str
    timeframe: str = Field("H1")
    horizon: int = Field(1, ge=1)
    method: str = Field("ewma")
    proxy: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    as_of: Optional[str] = None
    denoise: Optional[Dict[str, Any]] = None


class BacktestBody(BaseModel):
    symbol: str
    timeframe: str = Field("H1")
    horizon: int = Field(12, ge=1)
    steps: int = Field(5, ge=1)
    spacing: int = Field(20, ge=1)
    methods: Optional[list[str]] = None
    params_per_method: Optional[Dict[str, Any]] = None
    quantity: Literal["price", "return", "volatility"] = Field("price")
    denoise: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    dimred_method: Optional[str] = None
    dimred_params: Optional[Dict[str, Any]] = None
    slippage_bps: float = 0.0
    trade_threshold: float = 0.0
    detail: Literal["compact", "full"] = Field("compact")

    @model_validator(mode="before")
    @classmethod
    def _normalize_target_alias(cls, values: Any) -> Any:
        return _normalize_legacy_target(values)

    def to_domain_request(self) -> ForecastBacktestRequest:
        target: Literal["price", "return"] = "price"
        if self.quantity == "return":
            target = "return"

        return ForecastBacktestRequest(
            symbol=self.symbol,
            timeframe=self.timeframe,
            horizon=self.horizon,
            steps=self.steps,
            spacing=self.spacing,
            methods=self.methods,
            params_per_method=self.params_per_method,
            quantity=self.quantity,
            target=target,
            denoise=self.denoise,
            params=self.params,
            features=self.features,
            dimred_method=self.dimred_method,
            dimred_params=self.dimred_params,
            slippage_bps=self.slippage_bps,
            trade_threshold=self.trade_threshold,
            detail=self.detail,
        )
