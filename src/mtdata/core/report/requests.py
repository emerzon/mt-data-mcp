from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, model_validator

from ...shared.schema import (
    CompactStandardFullDetailLiteral,
    DenoiseSpec,
    TimeframeLiteral,
    reject_removed_field,
)


class ReportGenerateRequest(BaseModel):
    symbol: str
    horizon: Optional[int] = None
    template: str = "basic"
    timeframe: Optional[TimeframeLiteral] = None
    methods: Optional[Union[str, List[str]]] = None
    denoise: Optional[DenoiseSpec] = None
    params: Optional[Dict[str, Any]] = None
    detail: CompactStandardFullDetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_output(cls, values: Any) -> Any:
        values = reject_removed_field(values, field_name="output", replacement="json")
        return reject_removed_field(values, field_name="format", replacement="json")
