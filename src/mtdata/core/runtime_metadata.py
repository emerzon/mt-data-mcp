from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo


def _safe_tz_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    name = getattr(value, "zone", None) or getattr(value, "key", None)
    if isinstance(name, str):
        text = name.strip()
        return text or None
    tzname = getattr(value, "tzname", None)
    if callable(tzname):
        try:
            current_name = tzname(datetime.now(timezone.utc).astimezone(value))
        except Exception:
            try:
                current_name = tzname(None)
            except Exception:
                current_name = None
        if isinstance(current_name, str):
            text = current_name.strip()
            if text:
                return text
    std_name = getattr(value, "_std_abbr", None)
    if isinstance(std_name, str):
        text = std_name.strip()
        if text:
            return text
    if hasattr(value, "utcoffset"):
        try:
            text = str(value).strip()
        except Exception:
            return None
        return text or None
    return None


def _safe_now_iso(tzinfo: Any) -> Optional[str]:
    if tzinfo is None:
        return None
    try:
        return datetime.now(tzinfo).isoformat()
    except Exception:
        return None


def _resolve_tzinfo(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "utcoffset"):
        return value
    name = _safe_tz_name(value)
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except Exception:
        return None


def _prune_empty(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, subval in value.items():
            cleaned = _prune_empty(subval)
            if cleaned is None:
                continue
            if isinstance(cleaned, dict) and not cleaned:
                continue
            out[key] = cleaned
        return out
    return value
def _offset_tz_name(offset_seconds: Optional[int]) -> Optional[str]:
    if offset_seconds is None:
        return None
    try:
        return timezone(timedelta(seconds=int(offset_seconds))).tzname(None)
    except Exception:
        return None


def build_runtime_timezone_meta(
    result: Any,
    *,
    mt5_config: Any = None,
    include_local: bool = True,
    include_now: bool = True,
) -> Dict[str, Any]:
    """Build cross-interface timezone/runtime metadata for rendered outputs."""
    cfg = mt5_config
    if cfg is None:
        try:
            from .config import mt5_config as default_mt5_config
        except Exception:
            default_mt5_config = None
        cfg = default_mt5_config

    server_tz_config = None
    server_tz_resolved = None
    client_tz_config = None
    client_tz_resolved = None
    server_offset_seconds = None
    server_tz_obj = None
    client_tz_obj = None
    if cfg is not None:
        server_tz_config = _safe_tz_name(getattr(cfg, "server_tz_name", None))
        client_tz_config = _safe_tz_name(getattr(cfg, "client_tz_name", None))
        try:
            server_tz_obj = cfg.get_server_tz()
            server_tz_resolved = _safe_tz_name(server_tz_obj)
        except Exception:
            server_tz_obj = None
            server_tz_resolved = None
        try:
            client_tz_obj = cfg.get_client_tz()
            client_tz_resolved = _safe_tz_name(client_tz_obj)
        except Exception:
            client_tz_obj = None
            client_tz_resolved = None
        try:
            server_offset_seconds = int(cfg.get_time_offset_seconds())
        except Exception:
            server_offset_seconds = None

    offset_env = os.getenv("MT5_TIME_OFFSET_MINUTES")
    server_offset_minutes = None
    if offset_env is not None:
        try:
            server_offset_minutes = int(offset_env)
        except Exception:
            server_offset_minutes = None
    if server_offset_seconds is None and isinstance(server_offset_minutes, int):
        server_offset_seconds = int(server_offset_minutes) * 60

    server_source = "none"
    if server_tz_config:
        server_source = "MT5_SERVER_TZ"
    elif server_offset_minutes is not None:
        server_source = "MT5_TIME_OFFSET_MINUTES"

    server_tzinfo = _resolve_tzinfo(server_tz_obj) or _resolve_tzinfo(server_tz_resolved or server_tz_config)
    client_tzinfo = _resolve_tzinfo(client_tz_obj) or _resolve_tzinfo(client_tz_resolved or client_tz_config)

    utc_now = _safe_now_iso(timezone.utc) if include_now else None
    server_now = None
    if include_now:
        if server_tzinfo is not None:
            server_now = _safe_now_iso(server_tzinfo)
        elif server_source == "MT5_TIME_OFFSET_MINUTES" and server_offset_seconds is not None:
            server_now = _safe_now_iso(timezone(timedelta(seconds=server_offset_seconds)))

    client_now = _safe_now_iso(client_tzinfo) if include_now and client_tzinfo is not None else None

    server_tz_value = server_tz_resolved or server_tz_config
    if server_tz_value is None and server_source == "MT5_TIME_OFFSET_MINUTES":
        server_tz_value = _offset_tz_name(server_offset_seconds)

    client_tz_value = client_tz_resolved or client_tz_config
    used_tz = client_tz_value or "UTC"

    runtime_meta = {
        "utc": {
            "tz": "UTC",
            "now": utc_now,
        },
        "server": {
            "source": server_source,
            "tz": server_tz_value,
            "offset_seconds": (
                server_offset_seconds
                if server_offset_seconds is not None and (server_source != "none" or server_offset_seconds != 0)
                else None
            ),
            "now": server_now if include_now else None,
        },
        "client": {
            "tz": client_tz_value,
            "now": client_now if include_now else None,
        },
        "used": {
            "tz": used_tz,
        },
    }
    return _prune_empty(runtime_meta)
