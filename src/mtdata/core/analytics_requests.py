"""Validated requests for advanced MT5-native analytics tools."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ..shared.schema import (
    DetailLiteral,
    TimeframeLiteral,
    normalize_optional_symbol,
    normalize_required_symbol,
    validate_complete_time_window,
)


class MarketMicrostructureRequest(BaseModel):
    symbol: str
    start: Optional[str] = None
    end: Optional[str] = None
    minutes_back: int = 60
    max_ticks: int = Field(10_000, ge=20, le=50_000)
    bucket_seconds: int = Field(60, ge=1, le=86_400)
    detail: DetailLiteral = "compact"

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, value: str) -> str:
        return normalize_required_symbol(value)

    @model_validator(mode="after")
    def _window(self) -> "MarketMicrostructureRequest":
        validate_complete_time_window(self.start, self.end)
        return self


class TradeExecutionQualityRequest(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None
    minutes_back: int = Field(43_200, gt=0)
    symbol: Optional[str] = None
    side: Optional[Literal["buy", "sell"]] = None
    magic: Optional[int] = None
    limit: int = Field(200, ge=1, le=1_000)
    benchmark: Literal["arrival_quote", "order_price"] = "arrival_quote"
    benchmark_fallback: Literal["skip", "order_price"] = "skip"
    quote_window_seconds: int = Field(5, ge=1, le=60)
    markout_seconds: List[int] = Field(default_factory=lambda: [1, 5, 30])
    min_sample: int = Field(30, ge=1)
    detail: DetailLiteral = "compact"

    @field_validator("symbol")
    @classmethod
    def _optional_symbol(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_symbol(value)

    @field_validator("markout_seconds")
    @classmethod
    def _markouts(cls, value: List[int]) -> List[int]:
        normalized = sorted({int(item) for item in value})
        if not normalized or normalized[0] <= 0 or normalized[-1] > 3600:
            raise ValueError("markout_seconds must contain values from 1 to 3600")
        return normalized

    @model_validator(mode="after")
    def _window(self) -> "TradeExecutionQualityRequest":
        validate_complete_time_window(self.start, self.end)
        return self


class StrategyCandidate(BaseModel):
    id: str
    type: Literal["builtin_strategy", "forecast_threshold"]
    strategy: Optional[Literal["sma_cross", "ema_cross", "rsi_reversion"]] = None
    method: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    horizon: int = Field(1, ge=1, le=100)
    long_above: float = 0.0
    short_below: float = 0.0

    @model_validator(mode="after")
    def _source(self) -> "StrategyCandidate":
        if self.type == "builtin_strategy" and not self.strategy:
            raise ValueError("builtin_strategy candidates require strategy")
        if self.type == "forecast_threshold" and not str(self.method or "").strip():
            raise ValueError("forecast_threshold candidates require method")
        if self.short_below > self.long_above:
            raise ValueError("short_below must be <= long_above")
        return self


class BarrierSpec(BaseModel):
    horizon: int = Field(12, ge=1, le=200)
    tp_pct: float = Field(0.5, gt=0.0)
    sl_pct: float = Field(0.5, gt=0.0)
    same_bar_policy: Literal["sl_first", "tp_first", "neutral"] = "sl_first"


class StrategyValidateRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    lookback: int = Field(3_000, ge=200, le=50_000)
    start: Optional[str] = None
    end: Optional[str] = None
    candidates: List[StrategyCandidate] = Field(min_length=1, max_length=10)
    n_splits: int = Field(5, ge=2, le=10)
    barrier: BarrierSpec = Field(default_factory=BarrierSpec)
    purge_bars: Optional[int] = Field(None, ge=0)
    embargo_bars: Optional[int] = Field(None, ge=0)
    cost_model: Literal["current_spread_proxy", "fixed"] = "current_spread_proxy"
    spread_bps: Optional[float] = Field(None, ge=0.0)
    commission_bps: float = Field(0.0, ge=0.0)
    slippage_bps: float = Field(0.0, ge=0.0)
    bootstrap_samples: int = Field(500, ge=100, le=5_000)
    significance_alpha: float = Field(0.05, gt=0.0, lt=0.5)
    min_positive_fold_share: float = Field(0.8, ge=0.0, le=1.0)
    detail: DetailLiteral = "compact"

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, value: str) -> str:
        return normalize_required_symbol(value)

    @model_validator(mode="after")
    def _window(self) -> "StrategyValidateRequest":
        validate_complete_time_window(self.start, self.end)
        return self


class ProposedTrade(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    volume: float = Field(gt=0.0)


class PortfolioRiskDecomposeRequest(BaseModel):
    timeframe: TimeframeLiteral = "H1"
    lookback: int = Field(1_000, ge=100, le=20_000)
    horizon_bars: List[int] = Field(default_factory=lambda: [1, 5])
    confidence: List[float] = Field(default_factory=lambda: [0.95, 0.99])
    method: Literal["filtered_historical", "historical"] = "filtered_historical"
    ewma_half_life: float = Field(60.0, gt=1.0)
    simulations: int = Field(5_000, ge=500, le=50_000)
    seed: int = Field(42, ge=0, le=4_294_967_295)
    proposed_trade: Optional[ProposedTrade] = None
    allow_partial: bool = False
    detail: DetailLiteral = "compact"

    @field_validator("horizon_bars")
    @classmethod
    def _horizons(cls, value: List[int]) -> List[int]:
        out = sorted({int(item) for item in value})
        if not out or out[0] < 1 or out[-1] > 50:
            raise ValueError("horizon_bars must contain values from 1 to 50")
        return out

    @field_validator("confidence")
    @classmethod
    def _confidence(cls, value: List[float]) -> List[float]:
        out = sorted({float(item) for item in value})
        if not out or any(not math.isfinite(item) or not 0.5 < item < 1.0 for item in out):
            raise ValueError("confidence values must be between 0.5 and 1")
        return out


class MarketRelativeStrengthRequest(BaseModel):
    symbols: Optional[str] = None
    group: Optional[str] = None
    universe: Literal["visible", "all"] = "visible"
    timeframe: TimeframeLiteral = "H1"
    horizons: List[int] = Field(default_factory=lambda: [5, 20, 60])
    weights: List[float] = Field(default_factory=lambda: [0.2, 0.3, 0.5])
    volatility_lookback: int = Field(60, ge=10, le=2_000)
    benchmark: Optional[str] = None
    max_symbols: int = Field(100, ge=2, le=500)
    max_spread_pct: Optional[float] = Field(None, ge=0.0)
    min_tick_volume: Optional[int] = Field(None, ge=0)
    limit: int = Field(20, ge=1, le=100)
    detail: DetailLiteral = "compact"

    @model_validator(mode="after")
    def _ranking(self) -> "MarketRelativeStrengthRequest":
        if len(self.weights) != len(self.horizons):
            raise ValueError("weights must have the same length as horizons")
        pairs = sorted((int(horizon), float(weight)) for horizon, weight in zip(self.horizons, self.weights))
        if len({horizon for horizon, _ in pairs}) != len(pairs):
            raise ValueError("horizons must not contain duplicates")
        self.horizons = [horizon for horizon, _ in pairs]
        self.weights = [weight for _, weight in pairs]
        if not self.horizons or self.horizons[0] < 1 or self.horizons[-1] > 2_000:
            raise ValueError("horizons must contain values from 1 to 2000")
        if any(float(item) < 0 or not math.isfinite(float(item)) for item in self.weights):
            raise ValueError("weights must be finite and non-negative")
        total = float(sum(self.weights))
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = [float(item) / total for item in self.weights]
        if self.universe == "all" and not (self.symbols or self.group):
            raise ValueError("universe='all' requires symbols or group")
        return self
