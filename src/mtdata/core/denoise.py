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


def _compact_denoise_method(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": row.get("method"),
        "available": bool(row.get("available", False)),
        "requires": row.get("requires") or None,
        "supports_causal": bool(row.get("supports_causal", False)),
        "has_auto_params": bool(row.get("has_auto_params", False)),
        "params": row.get("params") or [],
        "description": row.get("description"),
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
        return {
            "success": True,
            "detail": detail_mode,
            "available_only": bool(available_only),
            "count": len(methods),
            "columns": [
                "method",
                "available",
                "requires",
                "supports_causal",
                "has_auto_params",
                "params",
                "description",
            ],
            "methods": [_compact_denoise_method(row) for row in methods],
        }

    return run_logged_operation(
        logger,
        operation="denoise_list_methods",
        detail=detail,
        available_only=available_only,
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
