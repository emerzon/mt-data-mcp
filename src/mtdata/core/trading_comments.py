"""Trading comment normalization helpers."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional


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
                allowed_base = _MT5_COMMENT_MAX_LENGTH - len(suffix_text)
                if allowed_base > 0:
                    full = f"{base[:allowed_base]}{suffix_text}"
                else:
                    full = base[:_MT5_COMMENT_MAX_LENGTH]
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
        "comment_may_be_truncated": True,
    }
