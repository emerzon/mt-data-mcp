"""Trading comment normalization helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


_MT5_COMMENT_MAX_LENGTH = 31


def _normalize_trade_comment(comment: Optional[str], *, default: str, suffix: str = "") -> str:
    """Return an MT5-safe comment string."""
    try:
        base = str(comment or "").strip()
    except Exception:
        base = ""
    if not base:
        base = str(default or "").strip() or "MCP"

    full = f"{base}{suffix}" if suffix else base
    try:
        if len(full) > _MT5_COMMENT_MAX_LENGTH:
            if suffix:
                allowed_base = _MT5_COMMENT_MAX_LENGTH - len(suffix)
                if allowed_base > 0:
                    full = f"{base[:allowed_base]}{suffix}"
                else:
                    full = base[:_MT5_COMMENT_MAX_LENGTH]
            else:
                full = base[:_MT5_COMMENT_MAX_LENGTH]
    except Exception:
        full = str(default or "MCP")[:_MT5_COMMENT_MAX_LENGTH]
    return full


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
