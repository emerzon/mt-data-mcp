"""Trading comment normalization helpers."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional

from . import validation


_MT5_COMMENT_MAX_LENGTH = 31
_MT5_COMMENT_SANITIZE_RE = re.compile(r"[^A-Za-z0-9 _.!-]+")


def _sanitize_trade_comment_text(value: Any) -> str:
    try:
        text = str(value or "").strip()
    except Exception:
        return ""
    if not text:
        return ""
    text = _MT5_COMMENT_SANITIZE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_trade_comment(comment: Optional[str], *, default: str, suffix: str = "") -> str:
    """Return an MT5-safe comment string."""
    base = _sanitize_trade_comment_text(comment)
    if not base:
        base = _sanitize_trade_comment_text(default) or "MCP"

    suffix_text = _sanitize_trade_comment_text(suffix)
    full = f"{base}{suffix_text}" if suffix_text else base
    try:
        if len(full) > _MT5_COMMENT_MAX_LENGTH:
            if suffix_text:
                suffix_kept = suffix_text[:_MT5_COMMENT_MAX_LENGTH]
                allowed_base = _MT5_COMMENT_MAX_LENGTH - len(suffix_kept)
                if allowed_base <= 0 and base:
                    allowed_base = 1
                    suffix_kept = suffix_kept[: _MT5_COMMENT_MAX_LENGTH - allowed_base]
                if allowed_base > 0:
                    full = f"{base[:allowed_base]}{suffix_kept}"
                else:
                    full = suffix_kept
            else:
                full = base[:_MT5_COMMENT_MAX_LENGTH]
    except Exception:
        full = (_sanitize_trade_comment_text(default) or "MCP")[:_MT5_COMMENT_MAX_LENGTH]
    return full.strip()


def _comment_sanitization_info(comment: Optional[str], applied_comment: str) -> Optional[Dict[str, Any]]:
    """Return metadata when a user-supplied comment is sanitized."""
    if comment is None:
        return None
    try:
        requested = str(comment).strip()
    except Exception:
        requested = ""
    if not requested:
        return None
    sanitized = _sanitize_trade_comment_text(requested)
    if not sanitized or sanitized == requested:
        return None
    return {
        "requested": requested,
        "applied": applied_comment,
    }


def _comment_truncation_info(comment: Optional[str], applied_comment: str) -> Optional[Dict[str, Any]]:
    """Return metadata when a user-supplied comment is truncated."""
    if comment is None:
        return None
    try:
        requested = str(comment).strip()
    except Exception:
        requested = ""
    if not requested or requested == applied_comment:
        return None
    return {
        "requested": requested,
        "applied": applied_comment,
        "max_length": _MT5_COMMENT_MAX_LENGTH,
    }


def _comment_row_metadata(comment: Any) -> Dict[str, Any]:
    """Surface broker comment limits in list views."""
    if comment is None:
        return {}
    try:
        if isinstance(comment, (int, float)) and not math.isfinite(float(comment)):
            return {}
    except Exception:
        pass
    try:
        text = str(comment).strip()
    except Exception:
        text = ""
    if not text or text.lower() == "nan":
        return {}
    return {
        "comment_visible_length": len(text),
        "comment_max_length": _MT5_COMMENT_MAX_LENGTH,
        "comment_may_be_truncated": len(text) >= _MT5_COMMENT_MAX_LENGTH,
    }


def _invalid_comment_error_text(result: Any, last_error: Any) -> Optional[str]:
    texts = []
    try:
        result_comment = getattr(result, "comment", None)
    except Exception:
        result_comment = None
    if isinstance(result_comment, str) and result_comment.strip():
        texts.append(result_comment.strip())
    if isinstance(last_error, tuple):
        if len(last_error) >= 2 and isinstance(last_error[1], str) and last_error[1].strip():
            texts.append(last_error[1].strip())
        elif last_error:
            texts.append(str(last_error))
    elif isinstance(last_error, str) and last_error.strip():
        texts.append(last_error.strip())
    elif last_error not in (None, False):
        texts.append(str(last_error))
    combined = " | ".join(text for text in texts if text)
    lowered = combined.lower()
    if "invalid" in lowered and "comment" in lowered:
        return combined or 'Invalid "comment" argument'
    return None


def _send_order_with_comment_fallback(mt5: Any, request: Dict[str, Any]) -> tuple[Any, Optional[Dict[str, Any]], Any]:
    result = mt5.order_send(request)
    last_error = validation._safe_last_error(mt5)
    invalid_comment = _invalid_comment_error_text(result, last_error)
    if invalid_comment is None:
        return result, None, last_error

    fallback_requests = []
    minimal_comment = _normalize_trade_comment("MCP", default="MCP")
    if request.get("comment") != minimal_comment:
        req_short = dict(request)
        req_short["comment"] = minimal_comment
        fallback_requests.append(("minimal", req_short))
    if "comment" in request:
        req_nocomment = dict(request)
        req_nocomment.pop("comment", None)
        fallback_requests.append(("none", req_nocomment))

    strategies = [strategy for strategy, _req in fallback_requests]
    for strategy, alt_request in fallback_requests:
        alt_result = mt5.order_send(alt_request)
        alt_last_error = validation._safe_last_error(mt5)
        if alt_result is not None and getattr(alt_result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
            return (
                alt_result,
                {
                    "used": True,
                    "strategy": strategy,
                    "invalid_comment_error": invalid_comment,
                    "request": alt_request,
                },
                alt_last_error,
            )

    return (
        result,
        {
            "used": False,
            "attempted": bool(fallback_requests),
            "strategies": strategies,
            "invalid_comment_error": invalid_comment,
        },
        last_error,
    )
