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
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    """List mtdata tools with categories, purpose text, and optional parameter summaries."""

    def _run() -> Dict[str, Any]:
        catalog = registered_tool_catalog(detail=detail)
        tools = catalog.get("tools") if isinstance(catalog, dict) else []
        if not isinstance(tools, list):
            return catalog
        category_filter = str(category or "").strip().lower()
        search_filter = str(search or "").strip().lower()
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
        if len(filtered) != len(tools):
            categories: Dict[str, list[str]] = {}
            for row in filtered:
                row_category = str(row.get("category") or "other")
                categories.setdefault(row_category, []).append(str(row.get("name") or ""))
            catalog = dict(catalog)
            catalog["tools"] = filtered
            catalog["categories"] = categories
            catalog["count"] = len(filtered)
            catalog["filters"] = {
                "category": category_filter or None,
                "search": search_filter or None,
            }
        return catalog

    return run_logged_operation(
        logger,
        operation="tools_list",
        category=category,
        search=search,
        detail=detail,
        func=_run,
    )
