"""Trading expiration and broker-time normalization helpers."""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Union

from ..config import mt5_config
from ...shared.constants import TIMEFRAME_SECONDS
from ...shared.validators import unsupported_timeframe_seconds_error


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


def _server_time_naive_to_utc(dt: datetime) -> datetime:
    """Convert a server-local naive datetime into UTC."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc)

    try:
        server_tz = mt5_config.get_server_tz()
    except Exception:
        server_tz = None

    if server_tz is not None:
        try:
            aware_server = server_tz.localize(dt, is_dst=None) if hasattr(server_tz, "localize") else dt.replace(tzinfo=server_tz)
        except Exception as exc:
            exc_name = exc.__class__.__name__
            try:
                if hasattr(server_tz, "localize") and exc_name in {"NonExistentTimeError", "AmbiguousTimeError"}:
                    standard_candidate = server_tz.localize(dt, is_dst=False)
                    daylight_candidate = server_tz.localize(dt, is_dst=True)
                    standard_utc = standard_candidate.astimezone(timezone.utc)
                    daylight_utc = daylight_candidate.astimezone(timezone.utc)
                    if exc_name == "NonExistentTimeError":
                        return max(standard_utc, daylight_utc)
                    return min(standard_utc, daylight_utc)
                aware_server = server_tz.localize(dt, is_dst=True) if hasattr(server_tz, "localize") else dt.replace(tzinfo=server_tz)
            except Exception:
                try:
                    aware_server = server_tz.localize(dt, is_dst=False) if hasattr(server_tz, "localize") else dt.replace(tzinfo=server_tz)
                except Exception:
                    aware_server = dt.replace(tzinfo=server_tz)
        return aware_server.astimezone(timezone.utc)

    try:
        offset_sec = int(mt5_config.get_time_offset_seconds())
    except Exception:
        offset_sec = 0
    return (dt - timedelta(seconds=offset_sec)).replace(tzinfo=timezone.utc)


def _next_candle_close_server_time(timeframe: str, *, now_utc: Optional[datetime] = None) -> datetime:
    """Return the next candle close in server-local naive time."""
    tf = str(timeframe or "").upper().strip()
    if tf not in TIMEFRAME_SECONDS:
        valid = ", ".join(sorted(TIMEFRAME_SECONDS.keys()))
        raise ValueError(f"Invalid timeframe '{timeframe}'. Valid options: {valid}")

    current_utc = now_utc or datetime.now(timezone.utc)
    if current_utc.tzinfo is None:
        current_utc = current_utc.replace(tzinfo=timezone.utc)
    else:
        current_utc = current_utc.astimezone(timezone.utc)

    server_now = _to_server_time_naive(current_utc)
    server_now = server_now.replace(tzinfo=None)

    if tf == "MN1":
        if server_now.month == 12:
            return server_now.replace(year=server_now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return server_now.replace(month=server_now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)

    if tf == "W1":
        days_until_next_monday = (7 - server_now.weekday()) % 7
        if days_until_next_monday == 0:
            days_until_next_monday = 7
        return (server_now + timedelta(days=days_until_next_monday)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

    if tf == "D1":
        return (server_now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    interval_seconds = int(TIMEFRAME_SECONDS[tf])
    if interval_seconds <= 0:
        raise ValueError(unsupported_timeframe_seconds_error(tf))

    day_start = server_now.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed_seconds = max(0.0, (server_now - day_start).total_seconds())
    next_slot = int(math.floor(elapsed_seconds / float(interval_seconds))) + 1
    return day_start + timedelta(seconds=float(next_slot * interval_seconds))


def _next_candle_wait_payload(
    timeframe: str,
    *,
    buffer_seconds: float = 1.0,
    now_utc: Optional[datetime] = None,
) -> dict:
    """Build timing metadata for the next candle close without sleeping."""
    current_utc = now_utc or datetime.now(timezone.utc)
    if current_utc.tzinfo is None:
        current_utc = current_utc.replace(tzinfo=timezone.utc)
    else:
        current_utc = current_utc.astimezone(timezone.utc)

    next_close_server = _next_candle_close_server_time(timeframe, now_utc=current_utc)
    next_close_utc = _server_time_naive_to_utc(next_close_server)
    wait_seconds = max(
        0.0,
        float((next_close_utc - current_utc).total_seconds()) + max(0.0, float(buffer_seconds)),
    )

    try:
        server_tz_name = getattr(mt5_config, "server_tz_name", None) or f"UTC{int(mt5_config.get_time_offset_seconds()) / 3600:+g}"
    except Exception:
        server_tz_name = "UTC"

    return {
        "timeframe": str(timeframe).upper().strip(),
        "buffer_seconds": float(buffer_seconds),
        "sleep_seconds": float(wait_seconds),
        "started_at_utc": current_utc.isoformat(),
        "next_candle_close_utc": next_close_utc.isoformat(),
        "next_candle_close_server": next_close_server.isoformat(),
        "server_timezone": str(server_tz_name),
    }


def _sleep_until_next_candle(
    timeframe: str,
    *,
    buffer_seconds: float = 1.0,
    sleep_impl=time.sleep,
    now_utc: Optional[datetime] = None,
) -> dict:
    """Sleep until the next candle closes and return timing metadata."""
    payload = _next_candle_wait_payload(
        timeframe,
        buffer_seconds=buffer_seconds,
        now_utc=now_utc,
    )
    sleep_seconds = float(payload.get("sleep_seconds", 0.0) or 0.0)
    sleep_impl(sleep_seconds)
    payload["status"] = "completed"
    payload["slept"] = True
    payload["slept_seconds"] = sleep_seconds
    payload["remaining_seconds"] = 0.0
    return payload


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
