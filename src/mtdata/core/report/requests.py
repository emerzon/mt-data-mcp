from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel

from ..schema import DenoiseSpec

ReportOutput = Literal["toon", "markdown"]


class ReportGenerateRequest(BaseModel):
    symbol: str
    horizon: Optional[int] = None
    template: str = "basic"
    timeframe: Optional[str] = None
    methods: Optional[Union[str, List[str]]] = None
    denoise: Optional[DenoiseSpec] = None
    params: Optional[Dict[str, Any]] = None
    output: ReportOutput = "toon"
