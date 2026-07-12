"""Denoise discovery MCP tools."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..shared.schema import DetailLiteral
from ..utils.denoise import get_denoise_methods_data
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .output_contract import build_pagination_meta, normalize_output_detail

logger = logging.getLogger(__name__)


_DENOISE_METHOD_DEFAULT_LIMIT = 30


def _summary_denoise_method(row: Dict[str, Any]) -> Dict[str, Any]:
    supports = row.get("supports") if isinstance(row.get("supports"), dict) else {}
    causality = supports.get("causality") if isinstance(supports, dict) else None
    params = row.get("params") if isinstance(row.get("params"), list) else []
    return {
        "method": row.get("method"),
        "available": bool(row.get("available", False)),
        "causality": list(causality) if isinstance(causality, list) else [],
        "requires": row.get("requires") or None,
        "params": list(params),
    }


def _denoise_methods(*, available_only: bool = False) -> List[Dict[str, Any]]:
    data = get_denoise_methods_data()
    rows = data.get("methods") if isinstance(data, dict) else []
    methods = [dict(row) for row in rows if isinstance(row, dict)]
    if available_only:
        methods = [row for row in methods if bool(row.get("available", False))]
    return methods


def _filter_denoise_methods(
    methods: List[Dict[str, Any]],
    *,
    causality: Optional[str] = None,
    no_extras: bool = False,
) -> List[Dict[str, Any]]:
    filtered = list(methods)
    if no_extras:
        filtered = [row for row in filtered if not str(row.get("requires") or "").strip()]
    causality_value = str(causality or "").strip().lower()
    if causality_value:
        if causality_value not in {"causal", "zero_phase"}:
            raise ValueError("Invalid causality filter. Use 'causal' or 'zero_phase'.")
        filtered = [
            row
            for row in filtered
            if causality_value
            in (
                row.get("supports", {}).get("causality", [])
                if isinstance(row.get("supports"), dict)
                else []
            )
        ]
    return filtered


@mcp.tool()
def denoise_list_methods(
    detail: DetailLiteral = "compact",
    available_only: bool = False,
    causality: Optional[str] = None,
    no_extras: bool = False,
    limit: int = _DENOISE_METHOD_DEFAULT_LIMIT,
) -> Dict[str, Any]:
    """List denoise methods, optional dependencies, causality support, and auto params."""

    def _run() -> Dict[str, Any]:
        detail_mode = normalize_output_detail(detail)
        methods = _filter_denoise_methods(
            _denoise_methods(available_only=available_only),
            causality=causality,
            no_extras=no_extras,
        )
        if detail_mode == "full":
            pagination = build_pagination_meta(
                total=len(methods),
                returned=len(methods),
                offset=0,
                limit=None,
            )
            return {
                "success": True,
                "detail": detail_mode,
                "available_only": bool(available_only),
                "causality": str(causality).strip().lower() if causality else None,
                "no_extras": bool(no_extras),
                "count": len(methods),
                "pagination": pagination,
                "methods": methods,
            }
        limit_value = max(1, int(limit or _DENOISE_METHOD_DEFAULT_LIMIT))
        visible = methods[:limit_value]
        hidden = max(0, len(methods) - len(visible))
        compact_mode = detail_mode != "standard"
        method_rows = (
            [
                {
                    "method": row.get("method"),
                    "available": bool(row.get("available", False)),
                    "causality": list(
                        row.get("supports", {}).get("causality", [])
                        if isinstance(row.get("supports"), dict)
                        else []
                    ),
                }
                for row in visible
            ]
            if compact_mode
            else [_summary_denoise_method(row) for row in visible]
        )
        columns = ["method", "available", "causality"]
        if not compact_mode:
            columns.extend(["requires", "params"])
        out = {
            "success": True,
            "detail": detail_mode,
            "available_only": bool(available_only),
            "count": len(visible),
            "total": len(methods),
            "limit": limit_value,
            "has_more": hidden > 0,
            "methods_hidden": hidden,
            "pagination": build_pagination_meta(
                total=len(methods),
                returned=len(visible),
                offset=0,
                limit=limit_value,
            ),
            "columns": columns,
            "causality": str(causality).strip().lower() if causality else None,
            "no_extras": bool(no_extras),
            "methods": method_rows,
            "describe_hint": "Use denoise_describe(method) for descriptions and defaults.",
        }
        if hidden > 0:
            out["list_all_hint"] = f"Pass limit={len(methods)} to list every method."
        return out

    return run_logged_operation(
        logger,
        operation="denoise_list_methods",
        detail=detail,
        available_only=available_only,
        causality=causality,
        no_extras=no_extras,
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
                described = dict(row)
                supports = described.get("supports")
                causality = supports.get("causality") if isinstance(supports, dict) else None
                if isinstance(causality, list):
                    has_causal = "causal" in causality
                    has_zero_phase = "zero_phase" in causality
                    if has_zero_phase and not has_causal:
                        described["causality_note"] = (
                            "Zero-phase (non-causal) only: uses future bars, so it introduces "
                            "look-ahead bias. Do not use for live signals or forward testing."
                        )
                    elif has_zero_phase and has_causal:
                        described["causality_note"] = (
                            "Supports causal and zero-phase modes. Use causal for live signals; "
                            "zero-phase uses future bars (look-ahead) and is analysis-only."
                        )
                    elif has_causal:
                        described["causality_note"] = (
                            "Causal: uses only past/current bars; safe for live signals."
                        )
                return {"success": True, "method": described}
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
