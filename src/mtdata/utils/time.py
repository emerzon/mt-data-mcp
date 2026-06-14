"""Canonical time formatting helpers."""

from datetime import datetime, timezone
from typing import Any, Optional


def format_epoch_utc(value: Any) -> Optional[str]:
    """Format epoch seconds as second-resolution RFC 3339 UTC."""
    try:
        timestamp = float(value)
        return (
            datetime.fromtimestamp(timestamp, timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (OSError, OverflowError, TypeError, ValueError):
        return None
