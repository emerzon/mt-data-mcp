import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..shared.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..shared.symbols import is_probably_forex_symbol
from ..shared.schema import DetailLiteral, TimeframeLiteral
from ..shared.validators import (
    invalid_timeframe_error,
    unsupported_timeframe_seconds_error,
)
from ..utils.mt5 import (
    MT5ConnectionError,
    _mt5_copy_rates_from,
    _mt5_copy_rates_range,
    _symbol_ready_guard,
    ensure_mt5_connection_or_raise,
    get_symbol_info_cached,
    mt5,
)
from ..utils.coercion import safe_float as _safe_float
from ..utils.time import (
    _format_time_minimal,
    _format_time_minimal_local,
    _resolve_client_tz,
)
from ..utils.utils import (
    _parse_end_datetime,
    _parse_start_datetime,
)
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway
from .output_contract import normalize_output_detail

logger = logging.getLogger(__name__)


_TEMPORAL_RELIABLE_GROUP_BARS = 30
_TEMPORAL_SAMPLE_WARNING_LIMIT = 10
_PERCENTAGE_POINTS_UNIT = "percentage_points (1.0 = 1%)"
_TEMPORAL_DEFAULT_LOOKBACK_DAYS = {
    "dow": 210,
    "hour": 60,
    "session": 60,
    "month": 730,
    "all": 365,
}
_TEMPORAL_DEFAULT_LOOKBACK_FLOOR = 200
_TEMPORAL_DEFAULT_LOOKBACK_CAP = 20_000


_DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
_PAGINATION_KEYS = (
    "total_count",
    "offset",
    "limit",
    "has_more",
    "more_available",
    "truncated",
)
_SESSION_ORDER = {
    "asia": 0,
    "london": 1,
    "london_ny_overlap": 2,
    "ny": 3,
    "off_session": 4,
}
_SESSION_DEFINITION = {
    "basis": "dst_aware_market_sessions",
    "asia": "Tokyo open to London open",
    "london": "London open to New York open",
    "london_ny_overlap": "New York open to London close",
    "ny": "London close to New York close",
    "off_session": "New York close to next Tokyo open",
    "market_timezones": {
        "tokyo": "Asia/Tokyo",
        "london": "Europe/London",
        "new_york": "America/New_York",
    },
    "market_local_hours": {
        "tokyo_open": "09:00",
        "london_open": "08:00",
        "london_close": "16:00",
        "new_york_open": "09:30",
        "new_york_close": "16:00",
    },
}
_FX_SESSION_DEFINITION = {
    **_SESSION_DEFINITION,
    "calendar": "fx",
    "market_local_hours": {
        "tokyo_open": "09:00",
        "london_open": "08:00",
        "london_close": "17:00",
        "new_york_open": "08:00",
        "new_york_close": "17:00",
    },
}
_SESSION_DEFINITION["calendar"] = "equity"
_TOKYO_TZ = ZoneInfo("Asia/Tokyo")
_LONDON_TZ = ZoneInfo("Europe/London")
_NEW_YORK_TZ = ZoneInfo("America/New_York")


def _error_response(
    message: str,
    stage: str,
    *,
    context: Optional[Dict[str, Any]] = None,
    details: Optional[Any] = None,
    bars: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"error": message, "stage": stage}
    if context:
        payload["context"] = context
    if details is not None:
        payload["details"] = details
    if bars is not None:
        payload["bars"] = int(bars)
    if filters is not None:
        payload["filters"] = filters
    return payload


def _normalize_group_by(value: Optional[str]) -> str:
    if value is None:
        return "dow"
    v = str(value).strip().lower()
    if v in ("day_of_week", "weekday", "week_day", "day", "daily", "dow", "wday"):
        return "dow"
    if v in ("hour", "hours", "hr", "time", "time_of_day"):
        return "hour"
    if v in ("month", "months", "mo"):
        return "month"
    if v in ("session", "sessions", "market_session", "trading_session"):
        return "session"
    if v in ("all", "none", "overall"):
        return "all"
    return v


def _default_temporal_lookback(timeframe: str, group_by: str) -> int:
    seconds_per_bar = TIMEFRAME_SECONDS.get(str(timeframe or "").strip().upper())
    if not seconds_per_bar:
        return _TEMPORAL_DEFAULT_LOOKBACK_FLOOR
    days = _TEMPORAL_DEFAULT_LOOKBACK_DAYS.get(
        str(group_by or "").strip().lower(),
        _TEMPORAL_DEFAULT_LOOKBACK_DAYS["dow"],
    )
    bars = int((int(days) * 86_400 + int(seconds_per_bar) - 1) // int(seconds_per_bar))
    return max(
        _TEMPORAL_DEFAULT_LOOKBACK_FLOOR,
        min(_TEMPORAL_DEFAULT_LOOKBACK_CAP, bars),
    )


def _timezone_label(value: Any, *, default: str = "UTC") -> str:
    try:
        label = getattr(value, "key", None) or getattr(value, "zone", None) or str(value)
    except Exception:
        return default
    text = str(label or "").strip()
    return text or default


def _session_boundary(
    day: date,
    *,
    market_tz: ZoneInfo,
    hour: int,
    minute: int,
    analysis_tz: Any,
) -> datetime:
    return datetime(
        day.year,
        day.month,
        day.day,
        hour,
        minute,
        tzinfo=market_tz,
    ).astimezone(analysis_tz)


def _session_boundaries_for_day(
    day: date,
    analysis_tz: Any,
    session_calendar: str = "equity",
) -> Dict[str, datetime]:
    fx = session_calendar == "fx"
    return {
        "asia_open": _session_boundary(
            day,
            market_tz=_TOKYO_TZ,
            hour=9,
            minute=0,
            analysis_tz=analysis_tz,
        ),
        "london_open": _session_boundary(
            day,
            market_tz=_LONDON_TZ,
            hour=8,
            minute=0,
            analysis_tz=analysis_tz,
        ),
        "ny_open": _session_boundary(
            day,
            market_tz=_NEW_YORK_TZ,
            hour=8 if fx else 9,
            minute=0 if fx else 30,
            analysis_tz=analysis_tz,
        ),
        "london_close": _session_boundary(
            day,
            market_tz=_LONDON_TZ,
            hour=17 if fx else 16,
            minute=0,
            analysis_tz=analysis_tz,
        ),
        "ny_close": _session_boundary(
            day,
            market_tz=_NEW_YORK_TZ,
            hour=17 if fx else 16,
            minute=0,
            analysis_tz=analysis_tz,
        ),
    }


def _market_session_label(
    value: Any,
    *,
    analysis_tz: Any = timezone.utc,
    boundary_cache: Optional[Dict[date, Dict[str, datetime]]] = None,
    session_calendar: str = "equity",
) -> str:
    if not isinstance(value, datetime):
        return "unknown"
    dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    try:
        dt_analysis = dt.astimezone(analysis_tz or timezone.utc)
    except Exception:
        return "unknown"
    cache = boundary_cache if boundary_cache is not None else {}
    anchor_date = dt_analysis.date()
    for day in (
        anchor_date - timedelta(days=1),
        anchor_date,
        anchor_date + timedelta(days=1),
    ):
        boundaries = cache.get(day)
        if boundaries is None:
            boundaries = _session_boundaries_for_day(
                day,
                analysis_tz or timezone.utc,
                session_calendar,
            )
            cache[day] = boundaries
        if boundaries["asia_open"] <= dt_analysis < boundaries["london_open"]:
            return "asia"
        if boundaries["london_open"] <= dt_analysis < boundaries["ny_open"]:
            return "london"
        if boundaries["ny_open"] <= dt_analysis < boundaries["london_close"]:
            return "london_ny_overlap"
        if boundaries["london_close"] <= dt_analysis < boundaries["ny_close"]:
            return "ny"
    return "off_session"


def _session_definition_for_clock(clock_name: str, session_calendar: str = "equity") -> Dict[str, Any]:
    source = _FX_SESSION_DEFINITION if session_calendar == "fx" else _SESSION_DEFINITION
    out = dict(source)
    out["clock"] = clock_name or "UTC"
    return out


def _parse_mapped_value(
    value: Optional[str],
    *,
    minimum: int,
    maximum: int,
    mapping: Dict[str, int],
    numeric_aliases: Optional[Dict[int, int]] = None,
) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text.isdigit():
        num = int(text)
        if numeric_aliases and num in numeric_aliases:
            return numeric_aliases[num]
        if minimum <= num <= maximum:
            return num
        return None
    return mapping.get(text)


def _parse_weekday(value: Optional[str]) -> Optional[int]:
    mapping = {
        "mon": 0, "monday": 0,
        "tue": 1, "tues": 1, "tuesday": 1,
        "wed": 2, "wednesday": 2,
        "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
        "fri": 4, "friday": 4,
        "sat": 5, "saturday": 5,
        "sun": 6, "sunday": 6,
    }
    return _parse_mapped_value(
        value,
        minimum=0,
        maximum=6,
        mapping=mapping,
        numeric_aliases={7: 6},
    )


def _parse_month(value: Optional[str]) -> Optional[int]:
    mapping = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    return _parse_mapped_value(
        value,
        minimum=1,
        maximum=12,
        mapping=mapping,
    )


def _parse_time_token(token: str) -> Optional[int]:
    text = token.strip()
    if not text:
        return None
    parts = text.split(":")
    if not parts:
        return None
    try:
        hour = int(parts[0])
    except Exception:
        return None
    minute = 0
    if len(parts) > 1:
        try:
            minute = int(parts[1])
        except Exception:
            return None
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour * 60 + minute


def _parse_time_range(value: Optional[str]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    if value is None:
        return None, None, None
    text = str(value).strip().lower()
    if not text:
        return None, None, None
    if "to" in text:
        parts = [p.strip() for p in text.split("to") if p.strip()]
    elif "-" in text:
        parts = [p.strip() for p in text.split("-") if p.strip()]
    else:
        return None, None, "Invalid time_range. Use 'HH:MM-HH:MM'."
    if len(parts) != 2:
        return None, None, "Invalid time_range. Use 'HH:MM-HH:MM'."
    start_min = _parse_time_token(parts[0])
    end_min = _parse_time_token(parts[1])
    if start_min is None or end_min is None:
        return None, None, "Invalid time_range. Use 'HH:MM-HH:MM'."
    if start_min == end_min:
        return None, None, "time_range start and end must differ."
    return start_min, end_min, None


def _time_label(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _rounded_temporal_float(value: Any, *, digits: int = 6) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    return round(float(number), digits)


def _stats_for_group(df: pd.DataFrame, volume_col: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "bars": int(len(df)),
    }
    ret = pd.to_numeric(df.get("__return"), errors="coerce")
    ret = ret[pd.notna(ret)]
    n = int(ret.shape[0])
    out["return_observations"] = n
    if n > 0:
        out["avg_return_pct"] = _rounded_temporal_float(ret.mean())
        out["median_return_pct"] = _rounded_temporal_float(ret.median())
        out["volatility_pct"] = _rounded_temporal_float(ret.std(ddof=0))
        out["avg_abs_return_pct"] = _rounded_temporal_float(ret.abs().mean())
        win_rate = _safe_float(round((ret > 0).sum() / float(n), 4))
        out["win_rate"] = win_rate
        out["win_rate_pct"] = (
            _safe_float(round(float(win_rate) * 100.0, 2))
            if win_rate is not None
            else None
        )
    else:
        out["avg_return_pct"] = None
        out["median_return_pct"] = None
        out["volatility_pct"] = None
        out["avg_abs_return_pct"] = None
        out["win_rate"] = None
        out["win_rate_pct"] = None

    if "__range" in df.columns:
        rng = pd.to_numeric(df["__range"], errors="coerce")
        rng = rng[pd.notna(rng)]
        out["avg_range"] = (
            _rounded_temporal_float(rng.mean(), digits=8) if len(rng) else None
        )
    if "__range_pct" in df.columns:
        rngp = pd.to_numeric(df["__range_pct"], errors="coerce")
        rngp = rngp[pd.notna(rngp)]
        out["avg_range_pct"] = _rounded_temporal_float(rngp.mean()) if len(rngp) else None
    if volume_col and volume_col in df.columns:
        vol = pd.to_numeric(df[volume_col], errors="coerce")
        vol = vol[pd.notna(vol)]
        out["avg_volume"] = (
            _rounded_temporal_float(vol.mean(), digits=2) if len(vol) else None
        )
    return out


def _compact_temporal_stats(
    row: Dict[str, Any],
    *,
    include_group: bool = True,
) -> Dict[str, Any]:
    keys = (
        "group_label",
        "bars",
        "avg_return_pct",
        "median_return_pct",
        "win_rate_pct",
        "volatility_pct",
    )
    if include_group and row.get("group") is not None:
        keys = ("group", *keys)
    return {key: row.get(key) for key in keys if row.get(key) is not None}


def _standard_temporal_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    out = _compact_temporal_stats(row, include_group=True)
    for key in ("return_observations", "avg_abs_return_pct", "avg_range_pct", "avg_volume"):
        value = row.get(key)
        if value is not None:
            out[key] = value
    return out


def _paginate_temporal_rows(
    rows: List[Dict[str, Any]],
    *,
    limit: Optional[int],
    offset: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    total_count = len(rows)
    paged = rows[offset:]
    if limit is not None:
        paged = paged[:limit]

    if limit is None and offset == 0:
        return paged, {}

    more_available = max(0, total_count - offset - len(paged))
    meta: Dict[str, Any] = {
        "total_count": total_count,
        "offset": offset,
        "has_more": more_available > 0,
        "more_available": more_available,
    }
    if limit is not None:
        meta["limit"] = limit
    if more_available > 0:
        meta["truncated"] = True
    return paged, meta


def _copy_pagination_meta(source: Dict[str, Any], target: Dict[str, Any]) -> None:
    for key in _PAGINATION_KEYS:
        if key in source:
            target[key] = source[key]


def _flatten_temporal_dimension_groups(
    groups: List[Any],
    *,
    formatter: Any,
    best_keys: Tuple[str, ...],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    flat_groups: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []
    pagination: Dict[str, Dict[str, Any]] = {}
    for item in groups:
        if not isinstance(item, dict):
            continue
        dimension = str(item.get("dimension") or "")
        breakdown = item.get("breakdown")
        if not dimension or not isinstance(breakdown, list):
            continue
        formatted_rows = [
            formatter(row)
            for row in breakdown
            if isinstance(row, dict)
        ]
        for row in formatted_rows:
            flat_row = {"dimension": dimension}
            flat_row.update(row)
            flat_groups.append(flat_row)
        page_meta: Dict[str, Any] = {}
        _copy_pagination_meta(item, page_meta)
        if page_meta:
            pagination[dimension] = page_meta
        best = max(
            (
                row
                for row in formatted_rows
                if row.get("avg_return_pct") is not None
            ),
            key=lambda row: float(row.get("avg_return_pct") or 0.0),
            default=None,
        )
        if best:
            best_row = {"dimension": dimension}
            best_row.update(
                {
                    key: best[key]
                    for key in best_keys
                    if key in best
                }
            )
            best_rows.append(best_row)
    return flat_groups, best_rows, pagination


def _temporal_sample_warnings(groups: Any) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    def _add_row(row: Dict[str, Any], *, dimension: Optional[str] = None) -> None:
        bars = int(row.get("bars", 0) or 0)
        if bars >= _TEMPORAL_RELIABLE_GROUP_BARS:
            return
        warning = {
            "group_label": row.get("group_label"),
            "bars": bars,
            "recommended_min_bars": _TEMPORAL_RELIABLE_GROUP_BARS,
        }
        if dimension:
            warning["dimension"] = dimension
        if row.get("group") is not None:
            warning["group"] = row.get("group")
        rows.append(warning)

    if isinstance(groups, list):
        for item in groups:
            if not isinstance(item, dict):
                continue
            breakdown = item.get("breakdown")
            if isinstance(breakdown, list):
                dimension = str(item.get("dimension") or "")
                for row in breakdown:
                    if isinstance(row, dict):
                        _add_row(row, dimension=dimension or None)
            else:
                _add_row(item)

    if not rows:
        return {}
    shown = rows[:_TEMPORAL_SAMPLE_WARNING_LIMIT]
    return {
        "sample_warnings": shown,
        "sample_warning_count": len(rows),
        "sample_notice": (
            "Some temporal groups have small samples; increase lookback or set "
            "min_bars for stricter filtering."
        ),
    }


def _temporal_group_count(groups: Any) -> int:
    if not isinstance(groups, list):
        return 0
    total = 0
    for item in groups:
        if not isinstance(item, dict):
            continue
        breakdown = item.get("breakdown")
        if isinstance(breakdown, list):
            total += len([row for row in breakdown if isinstance(row, dict)])
        else:
            total += 1
    return total


def _base_temporal_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "success",
            "symbol",
            "timeframe",
            "group_by",
            "return_mode",
            "units",
            "timezone",
            "session_definition",
            "lookback",
            "lookback_source",
            "lookback_note",
            "bars",
            "start",
            "end",
            "groups_analyzed",
            "groups_excluded",
            "min_bars_applied",
            "overall_basis",
            *_PAGINATION_KEYS,
        )
        if key in payload
    }


def _drop_compact_temporal_duplicate_units(out: Dict[str, Any]) -> None:
    units = out.get("units")
    if not isinstance(units, dict):
        return
    compact_units = {
        key: value
        for key, value in units.items()
        if key != "win_rate"
    }
    out["units"] = compact_units


def _compact_temporal_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = _base_temporal_payload(payload)
    _drop_compact_temporal_duplicate_units(out)
    groups = payload.get("groups")
    if isinstance(groups, list) and groups:
        if all(isinstance(row, dict) and "dimension" in row for row in groups):
            compact_groups, best_rows, pagination = _flatten_temporal_dimension_groups(
                groups,
                formatter=_compact_temporal_stats,
                best_keys=("group", "group_label", "avg_return_pct", "win_rate", "win_rate_pct"),
            )
            out["groups"] = compact_groups
            if best_rows:
                out["best"] = best_rows
            if pagination:
                out["dimension_pagination"] = pagination
            return out
        compact_groups = [
            _compact_temporal_stats(row)
            for row in groups
            if isinstance(row, dict)
        ]
        out["groups"] = compact_groups
        best = max(
            (
                row
                for row in compact_groups
                if row.get("avg_return_pct") is not None
            ),
            key=lambda row: float(row.get("avg_return_pct") or 0.0),
            default=None,
        )
        if best:
            out["best"] = {
                key: best[key]
                for key in ("group", "group_label", "avg_return_pct", "win_rate", "win_rate_pct")
                if key in best
            }
    elif isinstance(payload.get("overall"), dict):
        out["overall"] = _compact_temporal_stats(payload["overall"])
    for key in (
        "warnings",
        "excluded_groups",
        "sample_warnings",
        "sample_warning_count",
        "sample_notice",
    ):
        value = payload.get(key)
        if value:
            out[key] = value
    return out


def _summary_temporal_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    compact = _compact_temporal_payload(payload)
    out = {
        key: compact[key]
        for key in (
            "success",
            "symbol",
            "timeframe",
            "group_by",
            "return_mode",
            "units",
            "timezone",
            "lookback",
            "lookback_source",
            "lookback_note",
            "bars",
            "start",
            "end",
            "groups_analyzed",
            "groups_excluded",
            "min_bars_applied",
        )
        if key in compact
    }
    groups = compact.get("groups")
    if isinstance(groups, list):
        if all(isinstance(row, dict) and "dimension" in row for row in groups):
            pagination = compact.get("dimension_pagination")
            if isinstance(pagination, dict):
                out["group_counts"] = {
                    str(dimension): int(meta.get("total_count") or 0)
                    for dimension, meta in pagination.items()
                    if isinstance(meta, dict)
                }
            else:
                counts: Dict[str, int] = {}
                for row in groups:
                    dimension = str(row.get("dimension") or "")
                    counts[dimension] = counts.get(dimension, 0) + 1
                out["group_counts"] = counts
        else:
            out["group_count"] = len(groups)
    if compact.get("best") not in (None, "", [], {}):
        out["best"] = compact["best"]
    overall = payload.get("overall")
    if isinstance(overall, dict):
        out["overall"] = {
            key: overall.get(key)
            for key in ("bars", "avg_return_pct", "win_rate_pct", "volatility_pct")
            if overall.get(key) is not None
        }
    for key in (
        "warnings",
        "excluded_groups",
        "sample_warnings",
        "sample_warning_count",
        "sample_notice",
    ):
        value = payload.get(key)
        if value:
            out[key] = value
    return out


def _standard_temporal_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = _base_temporal_payload(payload)
    groups = payload.get("groups")
    if isinstance(groups, list) and groups:
        best = None
        if all(isinstance(row, dict) and "dimension" in row for row in groups):
            standard_groups, best_rows, pagination = _flatten_temporal_dimension_groups(
                groups,
                formatter=_standard_temporal_stats,
                best_keys=("group_label", "avg_return_pct", "win_rate", "win_rate_pct"),
            )
            out["groups"] = standard_groups
            if best_rows:
                out["best"] = best_rows
            if pagination:
                out["dimension_pagination"] = pagination
        else:
            standard_groups = [
                _standard_temporal_stats(row)
                for row in groups
                if isinstance(row, dict)
            ]
            out["groups"] = standard_groups
            best = max(
                (
                    row
                    for row in standard_groups
                    if row.get("avg_return_pct") is not None
                ),
                key=lambda row: float(row.get("avg_return_pct") or 0.0),
                default=None,
            )
        if best:
            out["best"] = {
                key: best[key]
                for key in ("group_label", "avg_return_pct", "win_rate", "win_rate_pct")
                if key in best
            }
    overall = payload.get("overall")
    if isinstance(overall, dict):
        out["overall"] = _standard_temporal_stats(overall)
    for key in (
        "volume_source",
        "filters",
        "warnings",
        "excluded_groups",
        "sample_warnings",
        "sample_warning_count",
        "sample_notice",
    ):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            out[key] = value
    return out


def _fetch_rates(
    symbol: str,
    timeframe: str,
    limit: int,
    start: Optional[str],
    end: Optional[str],
    gateway: Any = None,
) -> Tuple[Optional[Any], Optional[str]]:
    mt5_gateway = gateway or create_mt5_gateway(
        adapter=mt5,
        ensure_connection_impl=ensure_mt5_connection_or_raise,
    )
    if timeframe not in TIMEFRAME_MAP:
        return None, invalid_timeframe_error(timeframe, TIMEFRAME_MAP)
    mt5_tf = TIMEFRAME_MAP[timeframe]

    if start and end:
        start_dt = _parse_start_datetime(start)
        end_dt = _parse_end_datetime(end)
        if not start_dt or not end_dt:
            return None, "Invalid start/end date format."
        if start_dt > end_dt:
            return None, "start must be before end."
        rates = _mt5_copy_rates_range(symbol, mt5_tf, start_dt, end_dt)
        return rates, None

    if start:
        start_dt = _parse_start_datetime(start)
        if not start_dt:
            return None, "Invalid start date format."
        seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe)
        if not seconds_per_bar:
            return None, unsupported_timeframe_seconds_error(timeframe)
        end_dt = start_dt + timedelta(seconds=seconds_per_bar * max(int(limit), 1))
        rates = _mt5_copy_rates_range(symbol, mt5_tf, start_dt, end_dt)
        return rates, None

    if end:
        end_dt = _parse_end_datetime(end)
        if not end_dt:
            return None, "Invalid end date format."
        rates = _mt5_copy_rates_from(symbol, mt5_tf, end_dt, int(limit))
        return rates, None

    tick = mt5_gateway.symbol_info_tick(symbol)
    if tick is not None and getattr(tick, "time", None):
        t_utc = float(tick.time)
        to_dt = datetime.fromtimestamp(t_utc, tz=timezone.utc).replace(tzinfo=None)
    else:
        to_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    rates = _mt5_copy_rates_from(symbol, mt5_tf, to_dt, int(limit))
    return rates, None


@mcp.tool()
def temporal_analyze(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    lookback: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    group_by: Literal["dow", "hour", "month", "session", "all"] = "dow",
    session_calendar: Literal["auto", "fx", "equity"] = "auto",
    day_of_week: Optional[str] = None,
    month: Optional[str] = None,
    time_range: Optional[str] = None,
    return_mode: Literal["pct", "log"] = "pct",  # type: ignore
    min_bars: Optional[int] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Temporal analysis by day-of-week, hour, market session, or month.

    `lookback` controls the number of historical bars used when start/end are
    omitted. When omitted, the tool chooses a timeframe-aware seasonal window.

    Filters:
    - day_of_week: 0-6 or names like Mon, Tuesday
    - month: 1-12 or names like Jan, September
    - time_range: 'HH:MM-HH:MM' (start inclusive, end exclusive; analysis
      timezone if configured via client timezone, wraps midnight like 22:00-02:00)
    - min_bars: exclude grouped rows below this sample size. When omitted for
      day-of-week analysis, sparse weekend groups are auto-filtered.
    - limit/offset: page grouped output rows; does not change the analysis
      window or overall statistics.
    - volume: uses real_volume when available and non-zero, else tick_volume

    Returns grouped averages for returns and volatility plus simple extras.
    Group keys are numeric for dow/hour/month; session uses DST-aware market
    session labels (asia, london, london_ny_overlap, ny, off_session)
    evaluated in the same analysis timezone used for hour and time_range.
    Example: temporal_analyze(symbol="EURUSD", group_by="dow")
    Use group_by='all' to return day-of-week, hour, month, and session
    breakdowns in one call. The overall sample statistics are included when
    detail is 'standard' or 'full'.
    """
    def _run() -> Dict[str, Any]:  # noqa: C901
        context: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "group_by": group_by,
            "session_calendar": session_calendar,
            "return_mode": return_mode,
            "lookback": lookback,
            "start": start,
            "end": end,
            "detail": detail,
            "offset": offset,
        }
        if min_bars is not None:
            context["min_bars"] = min_bars
        if limit is not None:
            context["limit"] = limit
        try:
            mt5_gateway = create_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            group_norm = _normalize_group_by(group_by)
            if group_norm not in ("dow", "hour", "month", "session", "all"):
                return _error_response(
                    "Invalid group_by. Use: dow, hour, month, session, all.",
                    stage="validate",
                    context=context,
                )
            context["group_by"] = group_norm
            session_calendar_value = str(session_calendar or "auto").strip().lower()
            if session_calendar_value not in {"auto", "fx", "equity"}:
                return _error_response(
                    "Invalid session_calendar. Use: auto, fx, equity.",
                    stage="validate",
                    context=context,
                )
            resolved_session_calendar = (
                "fx"
                if session_calendar_value == "auto" and is_probably_forex_symbol(symbol)
                else "equity" if session_calendar_value == "auto" else session_calendar_value
            )
            lookback_defaulted = lookback is None
            try:
                effective_lookback = (
                    _default_temporal_lookback(timeframe, group_norm)
                    if lookback_defaulted
                    else int(lookback)
                )
            except Exception:
                return _error_response(
                    "lookback must be an integer.",
                    stage="validate",
                    context=context,
                )
            context["lookback"] = effective_lookback
            if lookback_defaulted:
                context["lookback_source"] = "auto"
            if effective_lookback <= 1 and not (start and end):
                return _error_response(
                    "lookback must be >= 2 for return calculations.",
                    stage="validate",
                    context=context,
                )
            requested_detail = str(detail or "compact").strip().lower()
            detail_mode = normalize_output_detail(detail, default="compact")
            if requested_detail not in {"compact", "standard", "summary", "full"}:
                return _error_response(
                    "detail must be one of: compact, standard, summary, full.",
                    stage="validate",
                    context=context,
                )

            limit_value: Optional[int] = None
            if limit is not None:
                try:
                    limit_value = int(float(limit))
                except Exception:
                    return _error_response(
                        "limit must be a positive integer.",
                        stage="validate",
                        context=context,
                    )
                if limit_value <= 0:
                    return _error_response(
                        "limit must be a positive integer.",
                        stage="validate",
                        context=context,
                    )
            try:
                offset_value = int(float(offset or 0))
            except Exception:
                return _error_response(
                    "offset must be a non-negative integer.",
                    stage="validate",
                    context=context,
                )
            if offset_value < 0:
                return _error_response(
                    "offset must be a non-negative integer.",
                    stage="validate",
                    context=context,
                )
            context["offset"] = offset_value
            if limit_value is not None:
                context["limit"] = limit_value

            dow_val = _parse_weekday(day_of_week)
            if day_of_week is not None and dow_val is None:
                return _error_response(
                    "Invalid day_of_week. Use 0-6 or day name (e.g., Mon).",
                    stage="validate",
                    context=context,
                    details={"day_of_week": day_of_week},
                )

            month_val = _parse_month(month)
            if month is not None and month_val is None:
                return _error_response(
                    "Invalid month. Use 1-12 or month name (e.g., Jan).",
                    stage="validate",
                    context=context,
                    details={"month": month},
                )

            tr_start, tr_end, tr_err = _parse_time_range(time_range)
            if tr_err:
                return _error_response(
                    tr_err,
                    stage="validate",
                    context=context,
                    details={"time_range": time_range},
                )

            min_bars_value: Optional[int] = None
            if min_bars is not None:
                try:
                    min_bars_value = int(min_bars)
                except Exception:
                    return _error_response(
                        "min_bars must be a non-negative integer.",
                        stage="validate",
                        context=context,
                    )
                if min_bars_value < 0:
                    return _error_response(
                        "min_bars must be a non-negative integer.",
                        stage="validate",
                        context=context,
                    )

            filters: Dict[str, Any] = {}
            if dow_val is not None:
                filters["day_of_week"] = {"value": dow_val, "label": _DOW_LABELS[dow_val]}
            if month_val is not None:
                filters["month"] = {"value": month_val, "label": _MONTH_LABELS[month_val - 1]}
            if tr_start is not None and tr_end is not None:
                filters["time_range"] = {
                    "start": _time_label(tr_start),
                    "end": _time_label(tr_end),
                    "end_exclusive": True,
                    "wraps_midnight": bool(tr_start > tr_end),
                }

            info_before = get_symbol_info_cached(symbol)
            with _symbol_ready_guard(symbol, info_before=info_before) as (err, _info):
                if err:
                    return _error_response(err, stage="symbol", context=context)

                rates, fetch_err = _fetch_rates(
                    symbol,
                    timeframe,
                    effective_lookback,
                    start,
                    end,
                    gateway=mt5_gateway,
                )
                if fetch_err:
                    return _error_response(fetch_err, stage="fetch", context=context)

            if rates is None or len(rates) < 2:
                return _error_response(
                    f"Failed to get rates for {symbol}.",
                    stage="fetch",
                    context=context,
                    details={"mt5_error": mt5.last_error()},
                )

            df = pd.DataFrame(rates)
            if df.empty:
                return _error_response("No data available.", stage="fetch", context=context, bars=0)

            try:
                df["__epoch"] = df["time"].astype(float)
            except Exception:
                return _error_response("Failed to normalize bar times.", stage="process", context=context)

            if not end:
                tf_secs = TIMEFRAME_SECONDS.get(timeframe)
                if tf_secs:
                    now_ts = datetime.now(timezone.utc).timestamp()
                    last_epoch = float(df["__epoch"].iloc[-1])
                    if 0 <= (now_ts - last_epoch) < float(tf_secs) and len(df) > 1:
                        df = df.iloc[:-1]

            if len(df) < 2:
                return _error_response(
                    "Insufficient data after trimming live bars.",
                    stage="trim",
                    context=context,
                    bars=len(df),
                )

            client_tz = _resolve_client_tz()
            use_client_tz = client_tz is not None
            analysis_tz = client_tz if use_client_tz else timezone.utc
            dt_utc = pd.to_datetime(df["__epoch"], unit="s", utc=True)
            dt = dt_utc.dt.tz_convert(client_tz) if use_client_tz else dt_utc
            df["__dt"] = dt
            session_boundary_cache: Dict[date, Dict[str, datetime]] = {}
            df["__session"] = dt.map(
                lambda value: _market_session_label(
                    value,
                    analysis_tz=analysis_tz,
                    boundary_cache=session_boundary_cache,
                    session_calendar=resolved_session_calendar,
                )
            )

            if "close" not in df.columns:
                return _error_response(
                    "Rates data missing close prices.",
                    stage="process",
                    context=context,
                )

            close = pd.to_numeric(df["close"], errors="coerce").astype(float)
            if return_mode == "log":
                close_safe = close.where(close > 0)
                ret = np.log(close_safe / close_safe.shift(1)) * 100.0
            else:
                ret = close.pct_change() * 100.0
            df["__return"] = ret

            if "high" in df.columns and "low" in df.columns:
                high = pd.to_numeric(df["high"], errors="coerce").astype(float)
                low = pd.to_numeric(df["low"], errors="coerce").astype(float)
                df["__range"] = high - low
                close_safe = close.where(close > 0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    df["__range_pct"] = (df["__range"] / close_safe) * 100.0

            volume_col = None
            if "real_volume" in df.columns:
                rv = pd.to_numeric(df["real_volume"], errors="coerce")
                if rv.notna().any() and float(rv.fillna(0.0).sum()) > 0.0:
                    volume_col = "real_volume"
            if volume_col is None and "tick_volume" in df.columns:
                volume_col = "tick_volume"

            if dow_val is not None:
                df = df[df["__dt"].dt.weekday == dow_val]
            if month_val is not None:
                df = df[df["__dt"].dt.month == month_val]
            if tr_start is not None and tr_end is not None:
                mins = df["__dt"].dt.hour * 60 + df["__dt"].dt.minute
                if tr_start < tr_end:
                    mask = (mins >= tr_start) & (mins < tr_end)
                else:
                    mask = (mins >= tr_start) | (mins < tr_end)
                df = df.loc[mask]

            if len(df) < 2:
                return _error_response(
                    "Insufficient data after applying filters.",
                    stage="filter",
                    context=context,
                    bars=len(df),
                    filters=filters,
                )

            def _groups_for_dimension(dimension: str) -> List[Dict[str, Any]]:
                grouped = df.copy()
                if dimension == "dow":
                    grouped["__group"] = grouped["__dt"].dt.weekday
                elif dimension == "month":
                    grouped["__group"] = grouped["__dt"].dt.month
                elif dimension == "hour":
                    grouped["__group"] = grouped["__dt"].dt.hour
                elif dimension == "session":
                    grouped["__group"] = grouped["__session"]
                else:
                    return []

                out_rows: List[Dict[str, Any]] = []
                sort_groups = dimension != "session"
                for key, grp in grouped.groupby("__group", sort=sort_groups):
                    row = _stats_for_group(grp, volume_col)
                    if dimension == "dow":
                        key_int = int(key)
                        row["group"] = key_int + 1
                        row["group_label"] = (
                            _DOW_LABELS[key_int] if 0 <= key_int <= 6 else str(key)
                        )
                    elif dimension == "month":
                        key_int = int(key)
                        row["group"] = key_int
                        row["group_label"] = (
                            _MONTH_LABELS[key_int - 1]
                            if 1 <= key_int <= 12
                            else str(key)
                        )
                    elif dimension == "hour":
                        key_int = int(key)
                        row["group"] = key_int
                        row["group_label"] = f"{key_int:02d}:00"
                    else:
                        key_text = str(key)
                        row["group"] = key_text
                        row["group_label"] = key_text.replace("_", " ")
                        key_int = _SESSION_ORDER.get(key_text, 99)
                    row["_group_key"] = key_int
                    out_rows.append(row)
                if dimension == "session":
                    out_rows.sort(key=lambda row: int(row.get("_group_key", 99)))
                return out_rows

            groups_out: List[Dict[str, Any]] = []
            grouped_dimensions: List[Dict[str, Any]] = []

            if group_norm in {"dow", "month", "hour", "session"}:
                groups_out = _groups_for_dimension(group_norm)
                if groups_out:
                    if group_norm == "dow":
                        df["__group"] = df["__dt"].dt.weekday
                    elif group_norm == "month":
                        df["__group"] = df["__dt"].dt.month
                    elif group_norm == "hour":
                        df["__group"] = df["__dt"].dt.hour
                    else:
                        df["__group"] = df["__session"].map(
                            lambda value: _SESSION_ORDER.get(str(value), 99)
                        )
            elif group_norm == "all":
                for dimension in ("dow", "hour", "month", "session"):
                    breakdown = _groups_for_dimension(dimension)
                    if breakdown:
                        grouped_dimensions.append(
                            {
                                "dimension": dimension,
                                "breakdown": breakdown,
                            }
                        )

            excluded_groups: List[Dict[str, Any]] = []
            auto_min_bars = False
            if (
                min_bars_value is None
                and group_norm in {"dow", "all"}
                and dow_val is None
            ):
                dow_groups = groups_out
                if group_norm == "all":
                    dow_groups = next(
                        (
                            item.get("breakdown", [])
                            for item in grouped_dimensions
                            if item.get("dimension") == "dow"
                        ),
                        [],
                    )
                bar_counts = [int(row.get("bars", 0) or 0) for row in dow_groups]
                median_bars = float(np.median(bar_counts)) if bar_counts else 0.0
                min_bars_value = max(2, int(median_bars * 0.25))
                auto_min_bars = True

            analysis_df = df
            if min_bars_value is not None and min_bars_value > 0 and groups_out:
                excluded = [
                    row
                    for row in groups_out
                    if int(row.get("bars", 0) or 0) < min_bars_value
                ]
                included = [
                    row
                    for row in groups_out
                    if int(row.get("bars", 0) or 0) >= min_bars_value
                ]
                if excluded and included:
                    excluded_keys = {row.get("_group_key") for row in excluded}
                    excluded_groups = [
                        {
                            "group": row.get("group"),
                            "group_label": row.get("group_label"),
                            "bars": int(row.get("bars", 0) or 0),
                            "min_bars": int(min_bars_value),
                            "auto": bool(auto_min_bars),
                        }
                        for row in excluded
                    ]
                    groups_out = included
                    analysis_df = df[~df["__group"].isin(excluded_keys)]

            if min_bars_value is not None and min_bars_value > 0 and grouped_dimensions:
                for item in grouped_dimensions:
                    dimension = str(item.get("dimension") or "")
                    if auto_min_bars and dimension != "dow":
                        continue
                    breakdown = item.get("breakdown")
                    if not isinstance(breakdown, list):
                        continue
                    excluded = [
                        row
                        for row in breakdown
                        if int(row.get("bars", 0) or 0) < min_bars_value
                    ]
                    included = [
                        row
                        for row in breakdown
                        if int(row.get("bars", 0) or 0) >= min_bars_value
                    ]
                    if excluded:
                        item["breakdown"] = included
                        excluded_groups.extend(
                            {
                                "dimension": dimension,
                                "group": row.get("group"),
                                "group_label": row.get("group_label"),
                                "bars": int(row.get("bars", 0) or 0),
                                "min_bars": int(min_bars_value),
                                "auto": bool(auto_min_bars),
                            }
                            for row in excluded
                        )

            groups_out = [
                {key: value for key, value in row.items() if key != "_group_key"}
                for row in groups_out
            ]
            for item in grouped_dimensions:
                breakdown = item.get("breakdown")
                if isinstance(breakdown, list):
                    item["breakdown"] = [
                        {
                            key: value
                            for key, value in row.items()
                            if key != "_group_key"
                        }
                        for row in breakdown
                    ]
            pagination_meta: Dict[str, Any] = {}
            if grouped_dimensions and (limit_value is not None or offset_value):
                paged_dimensions = []
                for item in grouped_dimensions:
                    breakdown = item.get("breakdown")
                    if not isinstance(breakdown, list):
                        paged_dimensions.append(item)
                        continue
                    paged_breakdown, item_meta = _paginate_temporal_rows(
                        breakdown,
                        limit=limit_value,
                        offset=offset_value,
                    )
                    paged_item = {
                        "dimension": item.get("dimension"),
                        "breakdown": paged_breakdown,
                    }
                    paged_item.update(item_meta)
                    paged_dimensions.append(paged_item)
                grouped_dimensions = paged_dimensions
            elif groups_out:
                groups_out, pagination_meta = _paginate_temporal_rows(
                    groups_out,
                    limit=limit_value,
                    offset=offset_value,
                )
            overall = _stats_for_group(analysis_df, volume_col)

            start_epoch = float(analysis_df["__epoch"].iloc[0])
            end_epoch = float(analysis_df["__epoch"].iloc[-1])
            start_str = _format_time_minimal_local(start_epoch) if use_client_tz else _format_time_minimal(start_epoch)
            end_str = _format_time_minimal_local(end_epoch) if use_client_tz else _format_time_minimal(end_epoch)

            tz_name = _timezone_label(analysis_tz)

            payload: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "group_by": group_norm,
                "return_mode": return_mode,
                "units": {
                    "avg_return_pct": _PERCENTAGE_POINTS_UNIT,
                    "median_return_pct": _PERCENTAGE_POINTS_UNIT,
                    "avg_abs_return_pct": _PERCENTAGE_POINTS_UNIT,
                    "win_rate": "fraction",
                    "win_rate_pct": _PERCENTAGE_POINTS_UNIT,
                    "avg_range_pct": _PERCENTAGE_POINTS_UNIT,
                    "volatility_pct": "percentage_point_return_stddev_per_bar",
                },
                "timezone": tz_name,
                "lookback": effective_lookback,
                "lookback_source": "auto" if lookback_defaulted else "request",
                "bars": int(len(analysis_df)),
                "start": start_str,
                "end": end_str,
                "filters": filters,
                "overall": overall,
                "overall_basis": (
                    "full_filtered_sample_before_per_dimension_group_exclusions"
                    if group_norm == "all"
                    else "sample_after_group_exclusions"
                ),
                "volume_source": volume_col,
            }
            if lookback_defaulted:
                payload["lookback_note"] = (
                    f"Auto lookback selected {int(effective_lookback)} bars for "
                    f"{group_norm} analysis on {timeframe}; pass --lookback for a "
                    "smaller, faster sample."
                )
            group_payload_source = grouped_dimensions if grouped_dimensions else groups_out
            payload["groups_analyzed"] = _temporal_group_count(group_payload_source)
            payload["groups_excluded"] = len(excluded_groups)
            if min_bars_value is not None:
                payload["min_bars_applied"] = int(min_bars_value or 0)
            if group_norm in {"session", "all"}:
                payload["session_calendar"] = resolved_session_calendar
                payload["session_calendar_source"] = (
                    "symbol_inference" if session_calendar_value == "auto" else "request"
                )
                payload["session_definition"] = _session_definition_for_clock(
                    tz_name,
                    resolved_session_calendar,
                )
            if min_bars_value is not None:
                filters["min_bars"] = {
                    "value": int(min_bars_value or 0),
                    "auto": bool(auto_min_bars),
                    "source": "auto" if auto_min_bars else "request",
                    "purpose": "exclude grouped rows below this sample size",
                }
            payload.update(pagination_meta)
            sample_context = _temporal_sample_warnings(
                grouped_dimensions if grouped_dimensions else groups_out
            )
            if sample_context:
                payload.update(sample_context)
            if excluded_groups:
                payload["excluded_groups"] = excluded_groups
                if group_norm == "all":
                    payload["warnings"] = [
                        "Sparse temporal groups below min_bars were excluded from "
                        "their grouped breakdowns only; overall statistics use the "
                        "full filtered sample."
                    ]
                else:
                    payload["warnings"] = [
                        "Sparse temporal groups below min_bars were excluded from "
                        "grouped results and overall statistics."
                    ]
            if grouped_dimensions:
                payload["groups"] = grouped_dimensions
            elif groups_out:
                payload["groups"] = groups_out
            if detail_mode == "summary":
                return _summary_temporal_payload(payload)
            if detail_mode == "compact":
                return _compact_temporal_payload(payload)
            if detail_mode == "standard":
                return _standard_temporal_payload(payload)
            return payload
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return _error_response(
                f"Error computing temporal analysis: {str(exc)}",
                stage="internal",
                context=context,
            )

    return run_logged_operation(
        logger,
        operation="temporal_analyze",
        symbol=symbol,
        timeframe=timeframe,
        group_by=group_by,
        lookback=lookback,
        limit=limit,
        offset=offset,
        func=_run,
    )
