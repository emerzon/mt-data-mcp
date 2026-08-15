from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from ...shared.schema import (
    DenoiseSpec,
    DetailLiteral,
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
    "Report template: minimal fast context+forecast (default), basic research "
    "with confluence and a single volatility estimator, advanced adds "
    "regimes/HAR/conformal, scalping M5 quote and session gates, "
    "intraday H1 plus news/session seasonality, swing H4/D1 plus volume "
    "profile and news, position D1/W1 plus volume profile and news. "
    "Typical warm-runtime tiers: minimal about 3-10 seconds; scalping about "
    "15-60 seconds; basic/intraday/swing/position about 30-120 seconds; "
    "advanced about 60-180 seconds. Broker history and enabled methods can "
    "increase these ranges; use max_runtime or section controls to bound work."
)


class ReportGenerateRequest(BaseModel):
    symbol: str
    horizon: Optional[int] = Field(
        None,
        ge=1,
        description="Forecast/report horizon in bars; must be at least 1 when supplied.",
    )
    template: ReportTemplateLiteral = Field("minimal", description=_REPORT_TEMPLATE_HELP)
    timeframe: Optional[TimeframeLiteral] = None
    start: Optional[str] = None
    end: Optional[str] = None
    methods: Optional[Union[str, List[str]]] = None
    include_sections: Optional[Union[str, List[str]]] = Field(
        None,
        description=(
            "Only execute and return these report sections (plus internal dependencies). "
            "Accepts a list or comma/space separated names."
        ),
    )
    max_sections: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Maximum number of report sections to execute and return, after "
            "include_sections filtering."
        ),
    )
    max_runtime: Optional[float] = Field(
        None,
        ge=1.0,
        le=3_600.0,
        description=(
            "Cooperative wall-clock budget in seconds. The runner plans a section "
            "subset to fit and stops scheduling sub-tools after the deadline; an "
            "already-running native/MT5 call cannot be preempted safely."
        ),
    )
    allow_partial: bool = Field(
        True,
        description=(
            "Treat a report with at least one usable section as successful while "
            "retaining section_run_status='partial' and per-section errors."
        ),
    )
    progress: bool = Field(
        False,
        description="Emit report sub-tool progress lines to stderr while the request runs.",
    )
    denoise: Optional[DenoiseSpec] = None
    params: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Template/sub-tool overrides. Common keys: timeframe, context_limit, context_tail, "
            "methods, backtest_steps, backtest_spacing, backtest_rmse_tolerance, "
            "backtest_min_directional_accuracy, patterns_limit, top_k, barrier_method, "
            "search_profile, grid_style, tp_min/tp_max/tp_steps, sl_min/sl_max/sl_steps, "
            "extra_timeframes, pivot_timeframes, spread_max_ticks, spread_max_pips. "
            "Advanced keys: regime_limit, regime_lookback, "
            "cp_threshold, hmm_states, conformal_steps, conformal_spacing, conformal_alpha."
        ),
    )
    detail: DetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_output(cls, values: Any) -> Any:
        values = reject_removed_field(values, field_name="output", replacement="json")
        values = reject_removed_field(values, field_name="format", replacement="json")
        values = reject_removed_field(
            values,
            field_name="summary_only",
            replacement="detail='summary'",
        )
        if isinstance(values, dict) and isinstance(values.get("template"), str):
            values = dict(values)
            values["template"] = values["template"].strip().lower()
        if isinstance(values, dict) and isinstance(values.get("params"), dict):
            params_horizon = values["params"].get("horizon")
            if params_horizon is not None:
                try:
                    valid_horizon = int(params_horizon) >= 1
                except (TypeError, ValueError):
                    valid_horizon = False
                if not valid_horizon:
                    raise ValueError("params.horizon must be an integer greater than or equal to 1")
        return values
