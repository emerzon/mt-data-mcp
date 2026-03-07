"""Trading expiration and broker-time normalization helpers."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Union

from .config import mt5_config


ExpirationValue = Union[int, float, str, datetime]
_GTC_EXPIRATION_TOKENS = {"GTC", "GOOD_TILL_CANCEL", "GOOD_TILL_CANCELLED", "NONE", "NO_EXPIRATION"}
_SIMPLE_RELATIVE_PATTERN = re.compile(r"^(?:in\s+)?(\d+(?:\.\d+)?)\s*([a-zA-Z]+)$", re.IGNORECASE)


def _to_server_time_naive(dt: datetime) -> datetime:
    """Convert a datetime into broker/server-local naive time."""
    try:
        server_tz = mt5_config.get_server_tz()
        client_tz = mt5_config.get_client_tz()
    except Exception:
        server_tz = None
        client_tz = None

    aware = dt
    try:
        if dt.tzinfo is None:
            if client_tz is not None:
                aware = client_tz.localize(dt) if hasattr(client_tz, "localize") else dt.replace(tzinfo=client_tz)
            else:
                aware = dt.replace(tzinfo=timezone.utc)
    except Exception:
        aware = dt.replace(tzinfo=timezone.utc)

    if server_tz is not None:
        try:
            server_aware = aware.astimezone(server_tz)
            return server_aware.replace(tzinfo=None)
        except Exception:
            pass

    try:
        offset_sec = int(mt5_config.get_time_offset_seconds())
    except Exception:
        offset_sec = 0
    try:
        utc_dt = aware.astimezone(timezone.utc)
    except Exception:
        utc_dt = aware if aware.tzinfo is not None else aware.replace(tzinfo=timezone.utc)
    server_dt = utc_dt + timedelta(seconds=offset_sec)
    return server_dt.replace(tzinfo=None)


def _server_time_naive_to_mt5_timestamp(dt: datetime) -> int:
    """Convert a server-local naive datetime into an MT5-compatible timestamp."""
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    dt = dt.replace(microsecond=0)
    return int((dt - datetime(1970, 1, 1)).total_seconds())


def _normalize_pending_expiration(expiration: Optional[ExpirationValue]) -> Tuple[Optional[int], bool]:
    """Convert user-supplied expiration data into an MT5-compatible timestamp."""
    if expiration is None:
        return None, False

    if isinstance(expiration, datetime):
        server_dt = _to_server_time_naive(expiration)
        return _server_time_naive_to_mt5_timestamp(server_dt), True

    if isinstance(expiration, (int, float)):
        if not math.isfinite(expiration) or expiration <= 0:
            return None, True
        try:
            server_dt = _to_server_time_naive(datetime.fromtimestamp(expiration, tz=timezone.utc))
            return _server_time_naive_to_mt5_timestamp(server_dt), True
        except (OverflowError, OSError) as exc:
            raise ValueError(f"Expiration timestamp out of range: {expiration}") from exc

    if isinstance(expiration, str):
        cleaned = expiration.strip().strip('"').strip("'")
        if cleaned == "":
            return None, False

        upper_cleaned = cleaned.upper()
        if upper_cleaned in _GTC_EXPIRATION_TOKENS:
            return None, True

        match = _SIMPLE_RELATIVE_PATTERN.match(cleaned)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            delta = None
            if unit in ("s", "sec", "secs", "second", "seconds"):
                delta = timedelta(seconds=value)
            elif unit in ("m", "min", "mins", "minute", "minutes"):
                delta = timedelta(minutes=value)
            elif unit in ("h", "hr", "hrs", "hour", "hours"):
                delta = timedelta(hours=value)
            elif unit in ("d", "day", "days"):
                delta = timedelta(days=value)
            elif unit in ("w", "wk", "weeks"):
                delta = timedelta(weeks=value)
            if delta is not None:
                server_dt = _to_server_time_naive(datetime.now(timezone.utc) + delta)
                return _server_time_naive_to_mt5_timestamp(server_dt), True

        try:
            import dateparser  # type: ignore

            dt = dateparser.parse(
                cleaned,
                settings={
                    "RETURN_AS_TIMEZONE_AWARE": False,
                    "PREFER_DATES_FROM": "future",
                    "RELATIVE_BASE": datetime.now(),
                },
            )
            if dt is not None:
                server_dt = _to_server_time_naive(dt)
                return _server_time_naive_to_mt5_timestamp(server_dt), True
        except Exception:
            pass

        try:
            numeric = float(cleaned)
            if not math.isfinite(numeric) or numeric <= 0:
                return None, True
            try:
                server_dt = _to_server_time_naive(datetime.fromtimestamp(numeric, tz=timezone.utc))
                return _server_time_naive_to_mt5_timestamp(server_dt), True
            except (OverflowError, OSError) as exc:
                raise ValueError(f"Expiration timestamp out of range: {expiration}") from exc
        except ValueError:
            try:
                server_dt = _to_server_time_naive(datetime.fromisoformat(cleaned))
                return _server_time_naive_to_mt5_timestamp(server_dt), True
            except ValueError as exc:
                raise ValueError(f"Unsupported expiration format: {expiration}") from exc

    raise TypeError(f"Unsupported expiration type: {type(expiration).__name__}")
