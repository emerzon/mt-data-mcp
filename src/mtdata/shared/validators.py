from __future__ import annotations

from typing import Any, Mapping


def invalid_timeframe_error(
    timeframe: Any,
    timeframe_map: Mapping[str, Any],
) -> str:
    return f"Invalid timeframe: {timeframe}. Valid options: {list(timeframe_map.keys())}"


def unsupported_timeframe_seconds_error(timeframe: Any) -> str:
    return f"Unsupported timeframe seconds for {timeframe}"
