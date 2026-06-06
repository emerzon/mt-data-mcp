"""Tool discovery catalog."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..shared.schema import CompactFullDetailLiteral
from ._mcp_instance import mcp
from ._mcp_tools import registered_tool_catalog
from .execution_logging import run_logged_operation

logger = logging.getLogger(__name__)


@mcp.tool()
def tools_list(
    category: Optional[str] = None,
    search: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    include_related: bool = False,
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    """List mtdata tools with filters, pagination, and optional parameter summaries."""

    def _run() -> Dict[str, Any]:
        catalog = registered_tool_catalog(detail=detail)
        tools = catalog.get("tools") if isinstance(catalog, dict) else []
        if not isinstance(tools, list):
            return catalog
        try:
            offset_value = int(offset or 0)
        except (TypeError, ValueError):
            return {"error": "offset must be a non-negative integer."}
        if offset_value < 0:
            return {"error": "offset must be a non-negative integer."}
        limit_value: Optional[int] = None
        if limit is not None:
            try:
                limit_value = int(limit)
            except (TypeError, ValueError):
                return {"error": "limit must be a positive integer."}
            if limit_value < 1:
                return {"error": "limit must be a positive integer."}
        category_filter = str(category or "").strip().lower()
        search_filter = str(search or "").strip().lower()
        known_categories = {
            str(row.get("category") or "").strip().lower()
            for row in tools
            if isinstance(row, dict) and str(row.get("category") or "").strip()
        }
        detail_mode = str(catalog.get("detail") or detail or "compact").strip().lower()
        filtered = []
        for row in tools:
            if not isinstance(row, dict):
                continue
            row_category = str(row.get("category") or "").strip().lower()
            haystack = " ".join(
                str(row.get(key) or "")
                for key in ("name", "category", "description")
            ).lower()
            if category_filter and row_category != category_filter:
                continue
            if search_filter and search_filter not in haystack:
                continue
            filtered.append(row)

        start = min(offset_value, len(filtered))
        if limit_value is None:
            paged = filtered[start:]
        else:
            paged = filtered[start : start + limit_value]
        has_more = start + len(paged) < len(filtered)

        gated_tools: list[Dict[str, Any]] = []
        slimmed: list[Dict[str, Any]] = []
        compact_mode = detail_mode == "compact"
        row_optional_keys = (
            "enabled",
            "enable_env",
            "status",
            "why_disabled",
            "recommended_alternative",
        )
        for row in paged:
            out_row = dict(row)
            if not include_related:
                out_row.pop("related_tools", None)
            if compact_mode:
                gated = {
                    key: out_row.get(key)
                    for key in row_optional_keys
                    if key in out_row
                }
                if gated:
                    gated["name"] = str(out_row.get("name") or "")
                    gated_tools.append(gated)
                    for key in row_optional_keys:
                        out_row.pop(key, None)
            slimmed.append(out_row)

        categories: Dict[str, list[str]] = {}
        for row in filtered:
            row_category = str(row.get("category") or "other")
            categories.setdefault(row_category, []).append(str(row.get("name") or ""))
        catalog = dict(catalog)
        catalog["tools"] = slimmed
        catalog["categories"] = categories
        catalog["count"] = len(slimmed)
        catalog["total_count"] = len(filtered)
        catalog["offset"] = offset_value
        catalog["limit"] = limit_value
        catalog["has_more"] = has_more
        catalog["filters"] = {
            "category": category_filter or None,
            "search": search_filter or None,
        }
        if category_filter and category_filter not in known_categories:
            catalog["warning"] = (
                f"Unknown category '{category}'. Valid categories: "
                + ", ".join(sorted(known_categories))
                + "."
            )
        if compact_mode and gated_tools:
            catalog["gated_tools"] = gated_tools
        return catalog

    return run_logged_operation(
        logger,
        operation="tools_list",
        category=category,
        search=search,
        limit=limit,
        offset=offset,
        include_related=include_related,
        detail=detail,
        func=_run,
    )
