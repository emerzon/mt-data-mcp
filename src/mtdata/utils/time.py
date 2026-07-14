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


def format_relative_time(value: datetime, *, now: Optional[datetime] = None) -> str:
    """Format a datetime as a compact past- or future-relative label."""
    timestamp = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    current = now or datetime.now(timezone.utc)
    current = (
        current.astimezone(timezone.utc)
        if current.tzinfo
        else current.replace(tzinfo=timezone.utc)
    )
    delta_seconds = int(round((current - timestamp).total_seconds()))
    if abs(delta_seconds) < 60:
        return "just now"

    seconds = abs(delta_seconds)
    for unit_seconds, unit_name in (
        (30 * 86400, "month"),
        (7 * 86400, "week"),
        (86400, "day"),
        (3600, "hour"),
        (60, "minute"),
    ):
        if seconds >= unit_seconds:
            amount = max(1, seconds // unit_seconds)
            label = f"{amount} {unit_name}{'' if amount == 1 else 's'}"
            return f"in {label}" if delta_seconds < 0 else f"{label} ago"
    return "just now"


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


def _format_time_second_explicit(epoch_seconds: float) -> str:
    """Format UTC epoch seconds at quote/event precision."""
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return _format_datetime_second_explicit(dt)


def _format_time_second_explicit_local(epoch_seconds: float) -> str:
    """Format local/client epoch seconds at quote/event precision."""
    try:
        tz = _resolve_client_tz()
        dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).astimezone(tz)
        return _format_datetime_second_explicit(dt)
    except Exception:
        return _format_time_second_explicit(epoch_seconds)


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
