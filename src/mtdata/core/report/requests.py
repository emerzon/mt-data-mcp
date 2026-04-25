from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, model_validator

from ..schema import CompactStandardFullDetailLiteral, DenoiseSpec, TimeframeLiteral

ReportFormat = Literal["toon", "markdown"]


def _reject_removed_field(values: Any, *, field_name: str, replacement: str) -> Any:
    if isinstance(values, dict) and field_name in values:
        raise ValueError(f"{field_name} was removed; use {replacement}")
    return values


class ReportGenerateRequest(BaseModel):
    symbol: str
    horizon: Optional[int] = None
    template: str = "basic"
    timeframe: Optional[TimeframeLiteral] = None
    methods: Optional[Union[str, List[str]]] = None
    denoise: Optional[DenoiseSpec] = None
    params: Optional[Dict[str, Any]] = None
    format: ReportFormat = "toon"
    detail: CompactStandardFullDetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_output(cls, values: Any) -> Any:
        return _reject_removed_field(values, field_name="output", replacement="format")
