from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from ...shared.schema import (
    CompactStandardFullDetailLiteral,
    DenoiseSpec,
    TimeframeLiteral,
    reject_removed_field,
)

ReportTemplateLiteral = Literal[
    "minimal",
    "basic",
    "advanced",
    "scalping",
    "intraday",
    "swing",
    "position",
]

_REPORT_TEMPLATE_HELP = (
    "Report template: minimal fast context+forecast, basic balanced default, "
    "advanced adds regimes/HAR/conformal, scalping M5 short-term setup, "
    "intraday H1 setup, swing H4/D1 setup, position D1/W1 setup."
)


class ReportGenerateRequest(BaseModel):
    symbol: str
    horizon: Optional[int] = None
    template: ReportTemplateLiteral = Field("basic", description=_REPORT_TEMPLATE_HELP)
    timeframe: Optional[TimeframeLiteral] = None
    start: Optional[str] = None
    end: Optional[str] = None
    methods: Optional[Union[str, List[str]]] = None
    include_sections: Optional[Union[str, List[str]]] = Field(
        None,
        description="Only include these report sections. Accepts a list or comma/space separated names.",
    )
    max_sections: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum number of report sections to include, after include_sections filtering.",
    )
    summary_only: bool = Field(
        False,
        description="Return only summary and metadata; omit detailed report sections.",
    )
    denoise: Optional[DenoiseSpec] = None
    params: Optional[Dict[str, Any]] = None
    detail: CompactStandardFullDetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_output(cls, values: Any) -> Any:
        values = reject_removed_field(values, field_name="output", replacement="json")
        values = reject_removed_field(values, field_name="format", replacement="json")
        if isinstance(values, dict) and isinstance(values.get("template"), str):
            values = dict(values)
            values["template"] = values["template"].strip().lower()
        return values
