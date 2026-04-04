"""CLI formatting utilities."""
import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timezone

CLI_FORMAT_JSON = "json"
CLI_FORMAT_TOON = "toon"


def _normalize_cli_formatter(fmt: Optional[str]) -> str:
    """Normalize CLI format string."""
    if not fmt:
        return CLI_FORMAT_TOON
    fmt_lower = str(fmt).lower().strip()
    if fmt_lower in ("json", "j"):
        return CLI_FORMAT_JSON
    if fmt_lower in ("toon", "t", "minimal", "min", "compact", "c"):
        return CLI_FORMAT_TOON
    return CLI_FORMAT_TOON


def _resolve_cli_formatter(args: Any, tool_info: Optional[Dict[str, Any]] = None) -> str:
    """Resolve formatter from args and tool info."""
    fmt = getattr(args, "format", None) or getattr(args, "fmt", None)
    if fmt:
        return _normalize_cli_formatter(fmt)
    if tool_info:
        defaults = tool_info.get("defaults", {})
        if defaults.get("detail") == "compact":
            return CLI_FORMAT_TOON
    return CLI_FORMAT_TOON


def _json_default(obj: Any) -> Any:
    """JSON serializer for non-standard types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _sanitize_json_compat(obj: Any) -> Any:
    """Sanitize object for JSON compatibility."""
    if isinstance(obj, dict):
        return {k: _sanitize_json_compat(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_compat(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _format_time_minimal(ts: Any) -> str:
    """Minimal time formatting."""
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)


def _format_result_minimal(result: Any, cmd_name: str = "") -> str:
    """Format result in minimal/toon style."""
    if result is None:
        return "(no result)"
    if isinstance(result, dict):
        if result.get("success") is False:
            error = result.get("error", "Unknown error")
            return f"Error: {error}"
        # Format successful response
        lines = []
        for key, value in result.items():
            if key in ("success", "schema_version"):
                continue
            if isinstance(value, (list, dict)):
                lines.append(f"{key}: [{len(value) if isinstance(value, list) else len(value)} items]")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines) if lines else "(empty result)"
    if isinstance(result, list):
        if not result:
            return "(empty list)"
        return f"[{len(result)} items]"
    return str(result)


def _format_result_for_cli(result: Any, *, fmt: str, verbose: bool, cmd_name: str) -> str:
    """Format result for CLI output."""
    fmt_s = _normalize_cli_formatter(fmt)
    if fmt_s == CLI_FORMAT_JSON:
        payload = {"text": result} if isinstance(result, str) else result
        payload = _sanitize_json_compat(payload)
        return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
    # TOON/minimal format
    return _format_result_minimal(result, cmd_name)


# Backward compatibility exports
__all__ = [
    "CLI_FORMAT_JSON",
    "CLI_FORMAT_TOON",
    "_normalize_cli_formatter",
    "_resolve_cli_formatter",
    "_json_default",
    "_sanitize_json_compat",
    "_format_time_minimal",
    "_format_result_minimal",
    "_format_result_for_cli",
]
