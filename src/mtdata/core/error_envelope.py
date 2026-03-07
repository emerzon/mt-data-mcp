"""Shared error-envelope and transport logging helpers."""

from __future__ import annotations

import logging
from uuid import uuid4
from typing import Any, Dict, Optional


def new_request_id() -> str:
    return uuid4().hex[:12]


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
    out.setdefault("success", False)
    if default_code and not str(out.get("error_code") or "").strip():
        out["error_code"] = default_code
    rid = str(out.get("request_id") or "").strip() or (request_id or new_request_id())
    out["request_id"] = rid
    if operation and not str(out.get("operation") or "").strip():
        out["operation"] = operation
    return out


def build_error_payload(
    message: Any,
    *,
    code: str,
    request_id: Optional[str] = None,
    operation: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "success": False,
        "error": str(message),
        "error_code": str(code),
        "request_id": request_id or new_request_id(),
    }
    if operation:
        payload["operation"] = str(operation)
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
