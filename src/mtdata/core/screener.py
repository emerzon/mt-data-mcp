"""Provider-agnostic equity screener."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, Literal, Optional, Union

from pydantic import Field

from ..services.research.capabilities import SCREENER
from ..services.research.payload import stamp_provider
from ..services.research.registry import get_research_registry
from ..shared.schema import DetailLiteral
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation

logger = logging.getLogger(__name__)

ResearchSourcePin = Literal["auto", "finviz", "mt5"]
ScreenerView = Literal[
    "overview",
    "valuation",
    "financial",
    "ownership",
    "performance",
    "technical",
]


class FinvizScreenerSource:
    """Finviz-backed equity screener adapter."""

    name = "finviz"

    def is_available(self) -> bool:
        return True

    def list_filters(
        self,
        *,
        search: Optional[str],
        filter_name: Optional[str],
        limit: int,
        offset: int,
        detail: str,
    ) -> Dict[str, Any]:
        from .finviz import finviz_filters_list

        return finviz_filters_list(
            search=search,
            filter_name=filter_name,
            limit=limit,
            offset=offset,
            detail=detail,  # type: ignore[arg-type]
        )

    def screen(
        self,
        *,
        filters: Optional[Union[str, Dict[str, Any]]],
        order: Optional[str],
        limit: int,
        page: int,
        view: str,
        detail: str,
    ) -> Dict[str, Any]:
        from .finviz import finviz_screen

        return finviz_screen(
            filters=filters,
            order=order,
            limit=limit,
            page=page,
            view=view,  # type: ignore[arg-type]
            detail=detail,  # type: ignore[arg-type]
        )


def _ensure_screener_sources() -> None:
    registry = get_research_registry()
    if "finviz" not in registry.known_names(SCREENER):
        registry.register(FinvizScreenerSource(), capabilities={SCREENER})


@mcp.tool()
def screener(
    filters: Annotated[
        Optional[Union[str, Dict[str, Any]]],
        Field(
            description=(
                "Screener filters as JSON, a dict, or provider shorthand. "
                "Filter names are provider-defined."
            )
        ),
    ] = None,
    order: Annotated[
        Optional[str],
        Field(description="Sort key, for example -marketcap or price."),
    ] = None,
    view: Annotated[
        ScreenerView,
        Field(description="Screener column set."),
    ] = "overview",
    list_filters: Annotated[
        bool,
        Field(description="List valid filter names and values instead of screening."),
    ] = False,
    search: Annotated[
        Optional[str],
        Field(description="Filter-catalog search when list_filters is true."),
    ] = None,
    filter_name: Annotated[
        Optional[str],
        Field(description="Exact filter name to describe when list_filters is true."),
    ] = None,
    limit: Annotated[int, Field(ge=1, description="Max rows per page.")] = 20,
    page: Annotated[int, Field(ge=1, description="One-based results page.")] = 1,
    offset: Annotated[
        int,
        Field(ge=0, description="Zero-based offset for the filter catalog."),
    ] = 0,
    detail: DetailLiteral = "compact",
    source: Annotated[
        ResearchSourcePin,
        Field(
            description="Adapter pin. auto uses every source that can serve this query."
        ),
    ] = "auto",
) -> Dict[str, Any]:
    """Screen equities or list valid screener filters.

    Filter keys stay provider-defined. Finviz is the current adapter;
    ``source="mt5"`` returns a capability error.
    """

    def _run() -> Dict[str, Any]:
        _ensure_screener_sources()
        adapters, error = get_research_registry().resolve_or_error(
            SCREENER,
            source,
            operation="screener",
        )
        if error is not None:
            return error
        adapter = adapters[0]
        if list_filters:
            payload = adapter.list_filters(
                search=search,
                filter_name=filter_name,
                limit=int(limit),
                offset=int(offset),
                detail=str(detail or "compact"),
            )
        else:
            payload = adapter.screen(
                filters=filters,
                order=order,
                limit=int(limit),
                page=int(page),
                view=str(view),
                detail=str(detail or "compact"),
            )
        return stamp_provider(payload, provider=str(adapter.name))

    return run_logged_operation(
        logger,
        operation="screener",
        list_filters=list_filters,
        source=source,
        func=_run,
    )


_ensure_screener_sources()
