"""Provider-agnostic economic and earnings calendar tool."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import Field

from ..services.research.capabilities import CALENDAR
from ..services.research.payload import stamp_provider
from ..services.research.protocols import CalendarRequest
from ..services.research.registry import get_research_registry
from ..shared.schema import DetailLiteral
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation

logger = logging.getLogger(__name__)

CalendarKind = Literal["economic", "earnings", "dividends"]
CalendarView = Literal["range", "period"]
ResearchSourcePin = Literal["auto", "finviz", "mt5"]


class FinvizCalendarSource:
    """Finviz-backed structured calendar adapter."""

    name = "finviz"

    def is_available(self) -> bool:
        return True

    def fetch_events(self, request: CalendarRequest) -> Dict[str, Any]:
        from .finviz import finviz_earnings, run_finviz_calendar

        if request.view == "period":
            earnings = finviz_earnings
            return earnings(
                period=request.period or "this-week",
                limit=request.limit,
                page=request.page,
                include_elapsed=request.include_elapsed,
                detail=request.detail,
            )
        return run_finviz_calendar(
            calendar=request.kind,
            impact=request.impact,
            country=request.country,
            currency=request.currency,
            start=request.start,
            end=request.end,
            upcoming=request.upcoming,
            limit=request.limit,
            page=request.page,
            detail=request.detail,
        )


def _ensure_calendar_sources() -> None:
    registry = get_research_registry()
    if "finviz" not in registry.known_names(CALENDAR):
        registry.register(FinvizCalendarSource(), capabilities={CALENDAR})


@mcp.tool()
def calendar(
    kind: Annotated[
        CalendarKind,
        Field(description="Event family: economic, earnings, or dividends."),
    ] = "economic",
    view: Annotated[
        CalendarView,
        Field(
            description=(
                "range is the filterable date table; period is the compact "
                "earnings window."
            )
        ),
    ] = "range",
    period: Annotated[
        Optional[Literal["this-week", "next-week", "previous-week", "this-month"]],
        Field(
            description=(
                "Earnings window when view=period. Defaults to this-week when "
                "view=period and this flag is omitted."
            )
        ),
    ] = None,
    impact: Annotated[
        Optional[Literal["low", "medium", "high"]],
        Field(description="Economic impact filter. Ignored for other kinds."),
    ] = None,
    country: Annotated[
        Optional[str],
        Field(description="Economic country filter, such as US."),
    ] = None,
    currency: Annotated[
        Optional[str],
        Field(description="Economic currency filter, such as USD."),
    ] = None,
    start: Annotated[
        Optional[str],
        Field(description="Inclusive range start (YYYY-MM-DD or relative)."),
    ] = None,
    end: Annotated[
        Optional[str],
        Field(description="Inclusive range end (YYYY-MM-DD or relative)."),
    ] = None,
    upcoming: Annotated[
        Optional[bool],
        Field(description="Keep unreleased economic events only."),
    ] = None,
    include_elapsed: Annotated[
        bool,
        Field(description="Include already-released earnings in the period view."),
    ] = False,
    limit: Annotated[int, Field(ge=1, description="Max rows per page.")] = 20,
    page: Annotated[int, Field(ge=1, description="One-based page number.")] = 1,
    detail: DetailLiteral = "compact",
    source: Annotated[
        ResearchSourcePin,
        Field(description="Adapter pin. auto uses every source that can serve this query."),
    ] = "auto",
) -> Dict[str, Any]:
    """Fetch a structured event calendar from available research sources.

    This is the preferred trader-facing calendar tool. It currently serves
    Finviz economic, earnings, and dividend tables. Pin ``source="finviz"``
    when you need that adapter only. MT5 does not yet expose a structured
    impact/country calendar; ``source="mt5"`` returns a capability error
    instead of a fake table.

    ``view="range"`` is the filterable date-range table (economic filters,
    earnings actuals/surprises, dividends). ``view="period"`` is the compact
    this-week / next-week earnings window.

    Parameters
    ----------
    kind : {"economic", "earnings", "dividends"}
        Event family.
    view : {"range", "period"}
        ``range`` (default) uses start/end filters. ``period`` is earnings-only
        and uses ``period``.
    period : str, optional
        Earnings period view: this-week, next-week, previous-week, this-month.
    impact, country, currency, start, end, upcoming
        Economic range filters. Same contract as the former finviz_calendar
        tool.
    source : {"auto", "finviz", "mt5"}
        Adapter pin. ``auto`` uses every source that can serve this query.
    """

    def _run() -> Dict[str, Any]:
        _ensure_calendar_sources()
        registry = get_research_registry()
        adapters, error = registry.resolve_or_error(
            CALENDAR,
            source,
            operation="calendar",
        )
        if error is not None:
            return error
        request = CalendarRequest(
            kind=str(kind),
            view=str(view),
            period=period,
            impact=impact,
            country=country,
            currency=currency,
            start=start,
            end=end,
            upcoming=upcoming,
            include_elapsed=bool(include_elapsed),
            limit=int(limit),
            page=int(page),
            detail=str(detail or "compact"),
        )
        if request.view == "period" and request.kind != "earnings":
            return {
                "success": False,
                "error": "view='period' is only supported for kind='earnings'.",
                "error_code": "calendar_invalid_view",
                "operation": "calendar",
                "valid_values": {"kind": ["earnings"], "view": ["range", "period"]},
            }
        if request.view == "period":
            invalid = [
                name
                for name, value in (
                    ("start", request.start),
                    ("end", request.end),
                    ("impact", request.impact),
                    ("country", request.country),
                    ("currency", request.currency),
                    ("upcoming", request.upcoming),
                )
                if value is not None
            ]
            if invalid:
                return build_error_payload(
                    "Period view does not use "
                    + ", ".join(invalid)
                    + ".",
                    code="incompatible_parameters",
                    operation="calendar",
                    details={"invalid": invalid, "view": "period"},
                    valid_values={
                        "view": ["period"],
                        "controls": [
                            "period",
                            "limit",
                            "page",
                            "include_elapsed",
                            "detail",
                        ],
                    },
                    remediation=(
                        "Drop start/end and economic filters, or switch to "
                        "view=range."
                    ),
                )
        if request.view == "range":
            invalid = []
            if request.period is not None:
                invalid.append("period")
            if request.include_elapsed:
                invalid.append("include_elapsed")
            if invalid:
                return build_error_payload(
                    "Range view does not use "
                    + ", ".join(invalid)
                    + ".",
                    code="incompatible_parameters",
                    operation="calendar",
                    details={"invalid": invalid, "view": "range"},
                    valid_values={
                        "view": ["range"],
                        "controls": [
                            "start",
                            "end",
                            "impact",
                            "country",
                            "currency",
                            "upcoming",
                            "limit",
                            "page",
                            "detail",
                        ],
                    },
                    remediation=(
                        "Drop period/include_elapsed, or switch to view=period "
                        "with kind=earnings."
                    ),
                )
        # Structured calendars do not share a mergeable row schema yet. Use the
        # first available adapter (Finviz is preferred by the registry).
        adapter = adapters[0]
        payload = adapter.fetch_events(request)
        return stamp_provider(payload, provider=str(adapter.name))

    return run_logged_operation(
        logger,
        operation="calendar",
        kind=kind,
        view=view,
        source=source,
        func=_run,
    )


_ensure_calendar_sources()
