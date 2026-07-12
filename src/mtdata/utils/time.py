"""Canonical time formatting and client-timezone helpers."""

from datetime import datetime, timezone
from typing import Any, Optional

from ..shared.constants import TIME_DISPLAY_FORMAT


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


def _format_time_minimal(epoch_seconds: float) -> str:
    """Format epoch seconds as a minute-resolution RFC 3339 UTC string."""
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return dt.strftime(TIME_DISPLAY_FORMAT)


def _format_time_minimal_local(epoch_seconds: float) -> str:
    """Format epoch seconds in client-local time with an explicit offset."""
    try:
        tz = _resolve_client_tz()
        dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).astimezone(tz)
        return _format_datetime_minute_explicit(dt)
    except Exception:
        return _format_time_minimal(epoch_seconds)


def _format_time_explicit(epoch_seconds: float) -> str:
    """Format UTC epoch seconds with an embedded timezone marker."""
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return _format_datetime_minute_explicit(dt)


def _format_time_explicit_local(epoch_seconds: float) -> str:
    """Format epoch seconds in local/client time with an embedded offset."""
    try:
        tz = _resolve_client_tz()
        dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).astimezone(tz)
        return _format_datetime_minute_explicit(dt)
    except Exception:
        return _format_time_explicit(epoch_seconds)


def _format_datetime_minute_explicit(dt: datetime) -> str:
    return _format_datetime_explicit(dt, timespec="minutes")


def _format_datetime_second_explicit(dt: datetime) -> str:
    return _format_datetime_explicit(dt, timespec="seconds")


def _format_datetime_explicit(dt: datetime, *, timespec: str) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    text = dt.isoformat(timespec=timespec)
    return f"{text[:-6]}Z" if text.endswith("+00:00") else text


def _use_client_tz() -> bool:
    """Return True when a client timezone is configured."""
    return _resolve_client_tz() is not None


def _resolve_client_tz():
    """Return the configured client timezone, if any."""
    from ..bootstrap.settings import mt5_config

    try:
        return mt5_config.get_client_tz()
    except Exception:
        return None
