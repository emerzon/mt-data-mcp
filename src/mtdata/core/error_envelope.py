"""Shared error-envelope and transport logging helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import uuid4


_ERROR_GUIDANCE: Dict[str, Dict[str, Any]] = {
    "mt5_connection_error": {
        "remediation": "Ensure MetaTrader 5 is running, logged in, and reachable.",
        "related_tools": ["symbols_list"],
    },
    "symbol_not_found": {
        "remediation": (
            "Use symbols_list with a search term, then retry with a broker-listed "
            "symbol."
        ),
        "related_tools": ["symbols_list"],
    },
    "unsupported_method": {
        "remediation": "Run forecast_list_methods and choose an available method.",
        "related_tools": ["forecast_list_methods"],
    },
    "invalid_method": {
        "remediation": "Run forecast_list_methods and choose an available method.",
        "related_tools": ["forecast_list_methods"],
    },
    "method_unavailable": {
        "remediation": "Run forecast_list_methods and choose an available method.",
        "related_tools": ["forecast_list_methods"],
    },
    "dependency_missing": {
        "remediation": (
            "Install the optional dependency group required by this method, then retry."
        ),
    },
    "insufficient_data": {
        "remediation": (
            "Increase the lookback, request more bars, or use a longer timeframe."
        ),
    },
    "forecast_task_not_found": {
        "remediation": (
            "Use forecast_task_list to inspect active and recent forecast tasks."
        ),
        "related_tools": ["forecast_task_list"],
    },
    "forecast_task_cancel_failed": {
        "remediation": (
            "Use forecast_task_status to verify the task state, or forecast_task_list "
            "to inspect active tasks."
        ),
        "related_tools": ["forecast_task_status", "forecast_task_list"],
    },
    "forecast_model_not_found": {
        "remediation": "Use forecast_models_list to inspect stored forecast models.",
        "related_tools": ["forecast_models_list"],
    },
}


def new_request_id() -> str:
    return uuid4().hex[:12]


def _default_error_guidance(
    *,
    code: str,
    operation: Optional[str],
) -> Dict[str, Any]:
    code_text = str(code or "").strip().lower()
    operation_text = str(operation or "").strip().lower()
    if code_text in _ERROR_GUIDANCE:
        return dict(_ERROR_GUIDANCE[code_text])
    if code_text.endswith("_connection_error"):
        return dict(_ERROR_GUIDANCE["mt5_connection_error"])
    if operation_text.startswith("forecast_") or code_text.startswith("forecast_"):
        return {
            "remediation": (
                "Check forecast inputs and use forecast_list_methods to inspect "
                "available methods."
            ),
            "related_tools": ["forecast_list_methods"],
        }
    if "insufficient" in code_text:
        return dict(_ERROR_GUIDANCE["insufficient_data"])
    return {}


def _apply_error_guidance(
    payload: Dict[str, Any],
    *,
    code: str,
    operation: Optional[str],
    remediation: Optional[str] = None,
    related_tools: Optional[list[str]] = None,
    valid_values: Optional[Dict[str, Any]] = None,
    example: Optional[str] = None,
    documentation: Optional[str] = None,
) -> None:
    guidance = _default_error_guidance(code=code, operation=operation)
    if remediation:
        guidance["remediation"] = str(remediation)
    if related_tools:
        guidance["related_tools"] = list(related_tools)
    if valid_values:
        guidance["valid_values"] = dict(valid_values)
    if example:
        guidance["example"] = str(example)
    if documentation:
        guidance["documentation"] = str(documentation)

    for key, value in guidance.items():
        if key in payload or value in (None, "", [], {}):
            continue
        payload[key] = value


def normalize_error_payload(
    payload: Dict[str, Any],
    *,
    default_code: Optional[str] = None,
    request_id: Optional[str] = None,
    operation: Optional[str] = None,
) -> Dict[str, Any]:
    error_text = payload.get("error")
    if not isinstance(error_text, str) or not error_text.strip():
        return payload

    out = dict(payload)
    error_code = str(out.get("error_code") or default_code or "tool_error").strip()
    rid = str(out.get("request_id") or "").strip() or (request_id or new_request_id())
    operation_value = str(out.get("operation") or operation or "").strip()

    normalized: Dict[str, Any] = {
        "success": False,
        "error": str(error_text),
        "error_code": error_code,
        "request_id": rid,
    }
    if operation_value:
        normalized["operation"] = operation_value
    for key, value in out.items():
        if key in normalized or key in {
            "success",
            "error",
            "error_code",
            "request_id",
            "operation",
        }:
            continue
        normalized[key] = value
    _apply_error_guidance(
        normalized,
        code=error_code,
        operation=operation_value or None,
    )
    return normalized


def build_error_payload(
    message: Any,
    *,
    code: str,
    request_id: Optional[str] = None,
    operation: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    remediation: Optional[str] = None,
    related_tools: Optional[list[str]] = None,
    valid_values: Optional[Dict[str, Any]] = None,
    example: Optional[str] = None,
    documentation: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "success": False,
        "error": str(message),
        "error_code": str(code),
        "request_id": request_id or new_request_id(),
    }
    if operation:
        payload["operation"] = str(operation)
    _apply_error_guidance(
        payload,
        code=str(code),
        operation=operation,
        remediation=remediation,
        related_tools=related_tools,
        valid_values=valid_values,
        example=example,
        documentation=documentation,
    )
    if details:
        payload["details"] = dict(details)
    return payload


def log_transport_exception(
    logger: logging.Logger,
    *,
    transport: str,
    operation: str,
    request_id: str,
    exc: BaseException,
) -> None:
    logger.exception(
        "transport=%s operation=%s request_id=%s failed: %s",
        transport,
        operation,
        request_id,
        exc,
    )
