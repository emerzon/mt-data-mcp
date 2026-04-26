from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel

from .schema import DenoiseSpec, TimeframeLiteral


PatternsDetailLiteral = Literal["highlights", "compact", "standard", "full"]


class PatternsDetectRequest(BaseModel):
    symbol: str
    timeframe: Optional[TimeframeLiteral] = None
    mode: str = "all"
    detail: PatternsDetailLiteral = "compact"
    limit: int = 500
    min_strength: float = 0.70
    min_gap: int = 3
    robust_only: bool = False
    whitelist: Optional[str] = None
    top_k: int = 1
    last_n_bars: Optional[int] = None
    denoise: Optional[DenoiseSpec] = None
    config: Optional[Dict[str, Any]] = None
    engine: str = "native"
    ensemble: bool = False
    ensemble_weights: Optional[Dict[str, Any]] = None
    include_series: bool = False
    series_time: str = "string"
    include_completed: bool = False
