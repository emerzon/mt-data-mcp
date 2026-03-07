"""Pydantic request models for the Web API transport."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ..forecast.requests import ForecastBacktestRequest


class ForecastPriceBody(BaseModel):
    symbol: str
    timeframe: str = Field("H1")
    method: str = Field("theta")
    horizon: int = Field(12, ge=1)
    lookback: Optional[int] = Field(None, ge=1)
    as_of: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    ci_alpha: Optional[float] = Field(0.05, ge=0.0, le=0.5)
    quantity: str = Field("price")
    target: str = Field("price")
    denoise: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    dimred_method: Optional[str] = None
    dimred_params: Optional[Dict[str, Any]] = None
    target_spec: Optional[Dict[str, Any]] = None


class ForecastVolBody(BaseModel):
    symbol: str
    timeframe: str = Field("H1")
    horizon: int = Field(1, ge=1)
    method: str = Field("ewma")
    proxy: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    as_of: Optional[str] = None
    denoise: Optional[Dict[str, Any]] = None


class BacktestBody(ForecastBacktestRequest):
    pass
