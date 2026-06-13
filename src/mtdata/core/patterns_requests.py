from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ..shared.schema import DenoiseSpec, TimeframeLiteral


PatternsDetailLiteral = Literal["compact", "standard", "summary", "full"]
PatternModeLiteral = Literal["candlestick", "classic", "harmonic", "fractal", "elliott", "all"]


class PatternsDetectRequest(BaseModel):
    symbol: str
    timeframe: Optional[TimeframeLiteral] = None
    mode: PatternModeLiteral = "candlestick"
    detail: PatternsDetailLiteral = "compact"
    limit: int = 150

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
    top_k: int = 3
    last_n_bars: Optional[int] = None
    denoise: Optional[DenoiseSpec] = None
    config: Optional[Dict[str, Any]] = None
    engine: Optional[str] = None
    ensemble: bool = False
    ensemble_weights: Optional[Dict[str, Any]] = None
    include_series: bool = False
    series_time: str = "string"
    include_completed: bool = False
