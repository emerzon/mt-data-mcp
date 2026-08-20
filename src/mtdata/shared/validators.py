from __future__ import annotations

import difflib
from typing import Any, Iterable, Mapping, Optional


def invalid_timeframe_error(
    timeframe: Any,
    timeframe_map: Mapping[str, Any],
) -> str:
    return f"Invalid timeframe: {timeframe}. Valid options: {list(timeframe_map)}"


def unsupported_timeframe_seconds_error(timeframe: Any) -> str:
    return f"Unsupported timeframe seconds for {timeframe}"


def unknown_mapping_keys_error(
    values: Mapping[str, Any],
    allowed_keys: Iterable[str],
    *,
    subject: str,
    error_code: str = "unknown_parameter",
) -> Optional[dict[str, Any]]:
    """Build a structured error for keys a free-form public mapping cannot consume."""
    valid = sorted({str(key) for key in allowed_keys})
    unknown = sorted(str(key) for key in values if str(key) not in valid)
    if not unknown:
        return None
    suggestions = {
        key: matches
        for key in unknown
        if (
            matches := difflib.get_close_matches(
                key,
                valid,
                n=3,
                cutoff=0.6,
            )
        )
    }
    suggestion_text = ""
    if suggestions:
        rendered = "; ".join(
            f"{key}: {', '.join(matches)}"
            for key, matches in suggestions.items()
        )
        suggestion_text = f" Close matches: {rendered}."
    return {
        "success": False,
        "error": (
            f"Unknown {subject} key(s): {', '.join(unknown)}. "
            f"Valid keys: {', '.join(valid) if valid else '(none)'}."
            f"{suggestion_text}"
        ),
        "error_code": error_code,
        "unknown_keys": unknown,
        "valid_keys": valid,
        **({"suggestions": suggestions} if suggestions else {}),
        "remediation": (
            "Correct or remove the unknown key(s); use only keys listed in valid_keys."
        ),
    }
