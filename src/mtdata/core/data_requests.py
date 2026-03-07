from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from .schema import DenoiseSpec, IndicatorSpec, SimplifySpec, TimeframeLiteral


class DataFetchCandlesRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    limit: int = 200
    start: Optional[str] = None
    end: Optional[str] = None
    ohlcv: Optional[str] = None
    indicators: Optional[List[IndicatorSpec]] = None
    denoise: Optional[DenoiseSpec] = None
    simplify: Optional[SimplifySpec] = None


class DataFetchTicksRequest(BaseModel):
    symbol: str
    limit: int = 200
    start: Optional[str] = None
    end: Optional[str] = None
    simplify: Optional[SimplifySpec] = None
    output: Literal["summary", "stats", "rows"] = "summary"
