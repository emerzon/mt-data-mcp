from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ..shared.schema import DenoiseSpec, TimeframeLiteral

PatternsDetailLiteral = Literal["compact", "standard", "summary", "full"]
PatternModeLiteral = Literal["candlestick", "classic", "harmonic", "fractal", "elliott", "all"]


class PatternsDetectRequest(BaseModel):
    symbol: str
    timeframe: Optional[TimeframeLiteral] = None
    mode: PatternModeLiteral = "candlestick"
    detail: PatternsDetailLiteral = "compact"
    limit: int = Field(150, ge=1)

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

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
        description=(
            "Candlestick strength threshold from 0.0 to 1.0; default 0.70. "
            "Lower values show more exploratory/noisy patterns, while 0.70+ "
            "keeps stricter high-conviction detections. Classic/fractal modes "
            "use their own mode-specific confidence rules."
        ),
    )
    min_gap: int = 3
    robust_only: bool = False
    whitelist: Optional[str] = None
    top_k: int = Field(3, ge=1)
    last_n_bars: Optional[int] = None
    denoise: Optional[DenoiseSpec] = None
    config: Optional[Dict[str, Any]] = None
    engine: Optional[str] = None
    ensemble: bool = False
    ensemble_weights: Optional[Dict[str, Any]] = None
    include_series: bool = False
    series_time: str = "string"
    include_completed: bool = False
    include_confirmed: Optional[bool] = Field(
        None,
        description=(
            "Elliott v2 alias for include_completed; when supplied it takes precedence."
        ),
    )

    @model_validator(mode="after")
    def _resolve_include_confirmed(self) -> "PatternsDetectRequest":
        if self.include_confirmed is not None:
            self.include_completed = bool(self.include_confirmed)
        if self.mode == "all" and self.limit < 150:
            raise ValueError(
                "mode='all' requires limit >= 150; use a single pattern mode "
                "for smaller analysis windows"
            )
        return self
