from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..shared.schema import DenoiseSpec, TimeframeLiteral

PatternsDetailLiteral = Literal["compact", "standard", "summary", "full"]
PatternModeLiteral = Literal["candlestick", "classic", "harmonic", "fractal", "elliott", "all"]


class PatternsDetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    timeframe: Optional[TimeframeLiteral] = None
    mode: PatternModeLiteral = "candlestick"
    detail: PatternsDetailLiteral = "compact"
    lookback: int = Field(
        150,
        ge=1,
        description="Historical bars fetched for pattern analysis.",
    )

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower().replace("-", "_")
        return "elliott" if normalized == "elliott_wave" else normalized

    start: Optional[str] = Field(
        None,
        description="Optional UTC-compatible start date/time for the analysis window.",
    )
    end: Optional[str] = Field(
        None,
        description="Optional UTC-compatible end date/time; end-only anchors recent history.",
    )
    min_strength: float = Field(
        0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Candlestick strength threshold from 0.0 to 1.0; default 0.70. "
            "Lower values show more exploratory/noisy patterns, while 0.70+ "
            "keeps stricter high-conviction detections. Classic/fractal modes "
            "use their own mode-specific confidence rules."
        ),
    )
    min_gap: int = Field(3, ge=0)
    robust_only: bool = False
    whitelist: Optional[str] = None
    top_k: int = Field(3, ge=1)
    last_n_bars: Optional[int] = Field(None, ge=1)
    denoise: Optional[DenoiseSpec] = None
    config: Optional[Dict[str, Any]] = None
    engine: Optional[Literal["native", "stock_pattern"]] = None
    ensemble: bool = False
    ensemble_weights: Optional[Dict[str, Any]] = None
    include_series: bool = False
    series_time: Literal["string", "epoch"] = "string"
    include_completed: bool = False

    @model_validator(mode="after")
    def _validate_request(self) -> "PatternsDetectRequest":
        if self.mode == "all" and self.lookback < 150:
            raise ValueError(
                "mode='all' requires lookback >= 150; use a single pattern mode "
                "for smaller analysis windows"
            )
        return self
