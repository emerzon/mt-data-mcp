"""Denoise discovery MCP tools."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..shared.schema import CompactFullDetailLiteral
from ..utils.denoise import get_denoise_methods_data
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .output_contract import normalize_output_verbosity_detail

logger = logging.getLogger(__name__)


_DENOISE_METHOD_DEFAULT_LIMIT = 30


def _summary_denoise_method(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": row.get("method"),
        "available": bool(row.get("available", False)),
    }


def _denoise_methods(*, available_only: bool = False) -> List[Dict[str, Any]]:
    data = get_denoise_methods_data()
    rows = data.get("methods") if isinstance(data, dict) else []
    methods = [dict(row) for row in rows if isinstance(row, dict)]
    if available_only:
        methods = [row for row in methods if bool(row.get("available", False))]
    return methods


@mcp.tool()
def denoise_list_methods(
    detail: CompactFullDetailLiteral = "compact",
    available_only: bool = False,
    limit: int = _DENOISE_METHOD_DEFAULT_LIMIT,
) -> Dict[str, Any]:
    """List denoise methods, optional dependencies, causality support, and auto params."""

    def _run() -> Dict[str, Any]:
        detail_mode = normalize_output_verbosity_detail(detail)
        methods = _denoise_methods(available_only=available_only)
        if detail_mode == "full":
            return {
                "success": True,
                "detail": detail_mode,
                "available_only": bool(available_only),
                "count": len(methods),
                "methods": methods,
            }
        limit_value = max(1, int(limit or _DENOISE_METHOD_DEFAULT_LIMIT))
        visible = methods[:limit_value]
        hidden = max(0, len(methods) - len(visible))
        out = {
            "success": True,
            "detail": detail_mode,
            "available_only": bool(available_only),
            "count": len(visible),
            "total": len(methods),
            "limit": limit_value,
            "has_more": hidden > 0,
            "methods_hidden": hidden,
            "columns": ["method", "available"],
            "methods": [_summary_denoise_method(row) for row in visible],
            "describe_hint": "Use denoise_describe(method) for params and descriptions.",
        }
        if hidden > 0:
            out["list_all_hint"] = f"Pass limit={len(methods)} to list every method."
        return out

    return run_logged_operation(
        logger,
        operation="denoise_list_methods",
        detail=detail,
        available_only=available_only,
        limit=limit,
        func=_run,
    )


@mcp.tool()
def denoise_describe(method: str) -> Dict[str, Any]:
    """Describe one denoise method and its supported options."""

    def _run() -> Dict[str, Any]:
        wanted = str(method or "").strip().lower()
        methods = _denoise_methods()
        for row in methods:
            if str(row.get("method") or "").strip().lower() == wanted:
                return {"success": True, "method": row}
        return build_error_payload(
            f"Unknown denoise method {method!r}.",
            code="denoise_method_unknown",
            operation="denoise_describe",
            details={"available_methods": [row.get("method") for row in methods]},
        )

    return run_logged_operation(
        logger,
        operation="denoise_describe",
        method=method,
        func=_run,
    )
