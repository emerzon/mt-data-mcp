from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List, Literal, Tuple
import math

import numpy as np
import pandas as pd

import MetaTrader5 as mt5

from .schema import TimeframeLiteral
from .constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from .server import mcp, _auto_connect_wrapper
from ..utils.mt5 import (
    _mt5_copy_rates_from,
    _mt5_copy_rates_range,
    _mt5_epoch_to_utc,
    _ensure_symbol_ready,
    get_symbol_info_cached,
)
from ..utils.utils import (
    _parse_start_datetime,
    _format_time_minimal,
    _format_time_minimal_local,
    _resolve_client_tz,
)


_DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _normalize_group_by(value: Optional[str]) -> str:
    if value is None:
        return "dow"
    v = str(value).strip().lower()
    if v in ("weekday", "week_day", "day", "dow", "wday"):
        return "dow"
    if v in ("hour", "hr", "time", "time_of_day"):
        return "hour"
    if v in ("month", "mo"):
        return "month"
    if v in ("all", "none", "overall"):
        return "all"
    return v


def _parse_weekday(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text.isdigit():
        num = int(text)
        if 0 <= num <= 6:
            return num
        if 1 <= num <= 7:
            return num - 1
    mapping = {
        "mon": 0, "monday": 0,
        "tue": 1, "tues": 1, "tuesday": 1,
        "wed": 2, "wednesday": 2,
        "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
        "fri": 4, "friday": 4,
        "sat": 5, "saturday": 5,
        "sun": 6, "sunday": 6,
    }
    if text in mapping:
        return mapping[text]
    return None


def _parse_month(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text.isdigit():
        num = int(text)
        if 1 <= num <= 12:
            return num
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
    if text in mapping:
        return mapping[text]
    return None


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


def _safe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _stats_for_group(df: pd.DataFrame, volume_col: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "bars": int(len(df)),
    }
    ret = pd.to_numeric(df.get("__return"), errors="coerce")
    ret = ret[pd.notna(ret)]
    n = int(ret.shape[0])
    out["returns"] = n
    if n > 0:
        out["avg_return"] = _safe_float(ret.mean())
        out["median_return"] = _safe_float(ret.median())
        out["volatility"] = _safe_float(ret.std(ddof=0))
        out["avg_abs_return"] = _safe_float(ret.abs().mean())
        out["win_rate"] = _safe_float((ret > 0).sum() / float(n))
    else:
        out["avg_return"] = None
        out["median_return"] = None
        out["volatility"] = None
        out["avg_abs_return"] = None
        out["win_rate"] = None

    if "__range" in df.columns:
        rng = pd.to_numeric(df["__range"], errors="coerce")
        rng = rng[pd.notna(rng)]
        out["avg_range"] = _safe_float(rng.mean()) if len(rng) else None
    if "__range_pct" in df.columns:
        rngp = pd.to_numeric(df["__range_pct"], errors="coerce")
        rngp = rngp[pd.notna(rngp)]
        out["avg_range_pct"] = _safe_float(rngp.mean()) if len(rngp) else None
    if volume_col and volume_col in df.columns:
        vol = pd.to_numeric(df[volume_col], errors="coerce")
        vol = vol[pd.notna(vol)]
        out["avg_volume"] = _safe_float(vol.mean()) if len(vol) else None
    return out


def _fetch_rates(
    symbol: str,
    timeframe: str,
    limit: int,
    start: Optional[str],
    end: Optional[str],
) -> Tuple[Optional[Any], Optional[str]]:
    if timeframe not in TIMEFRAME_MAP:
        return None, f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"
    mt5_tf = TIMEFRAME_MAP[timeframe]

    if start and end:
        start_dt = _parse_start_datetime(start)
        end_dt = _parse_start_datetime(end)
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
            return None, f"Unsupported timeframe seconds for {timeframe}"
        end_dt = start_dt + timedelta(seconds=seconds_per_bar * max(int(limit), 1))
        rates = _mt5_copy_rates_range(symbol, mt5_tf, start_dt, end_dt)
        return rates, None

    if end:
        end_dt = _parse_start_datetime(end)
        if not end_dt:
            return None, "Invalid end date format."
        rates = _mt5_copy_rates_from(symbol, mt5_tf, end_dt, int(limit))
        return rates, None

    tick = mt5.symbol_info_tick(symbol)
    if tick is not None and getattr(tick, "time", None):
        t_utc = _mt5_epoch_to_utc(float(tick.time))
        to_dt = datetime.utcfromtimestamp(t_utc)
    else:
        to_dt = datetime.utcnow()
    rates = _mt5_copy_rates_from(symbol, mt5_tf, to_dt, int(limit))
    return rates, None


@mcp.tool()
@_auto_connect_wrapper
def temporal_analyze(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    limit: int = 1000,
    start: Optional[str] = None,
    end: Optional[str] = None,
    group_by: Literal["dow", "hour", "month", "all"] = "dow",  # type: ignore
    day_of_week: Optional[str] = None,
    month: Optional[str] = None,
    time_range: Optional[str] = None,
    return_mode: Literal["pct", "log"] = "pct",  # type: ignore
) -> Dict[str, Any]:
    """Temporal analysis by day-of-week, hour, or month.

    Filters:
    - day_of_week: 0-6 or names like Mon, Tuesday
    - month: 1-12 or names like Jan, September
    - time_range: 'HH:MM-HH:MM' (local/client timezone if configured)

    Returns grouped averages for returns and volatility plus simple extras.
    Use group_by='all' for a single overall summary.
    """
    try:
        if limit is None:
            limit = 0
        limit = int(limit)
        if limit <= 1 and not (start and end):
            return {"error": "limit must be >= 2 for return calculations."}

        group_norm = _normalize_group_by(group_by)
        if group_norm not in ("dow", "hour", "month", "all"):
            return {"error": "Invalid group_by. Use: dow, hour, month, all."}

        dow_val = _parse_weekday(day_of_week)
        if day_of_week is not None and dow_val is None:
            return {"error": "Invalid day_of_week. Use 0-6 or day name (e.g., Mon)."}

        month_val = _parse_month(month)
        if month is not None and month_val is None:
            return {"error": "Invalid month. Use 1-12 or month name (e.g., Jan)."}

        tr_start, tr_end, tr_err = _parse_time_range(time_range)
        if tr_err:
            return {"error": tr_err}

        info_before = get_symbol_info_cached(symbol)
        was_visible = bool(info_before.visible) if info_before is not None else None
        err = _ensure_symbol_ready(symbol)
        if err:
            return {"error": err}

        try:
            rates, fetch_err = _fetch_rates(symbol, timeframe, limit, start, end)
            if fetch_err:
                return {"error": fetch_err}
        finally:
            if was_visible is False:
                try:
                    mt5.symbol_select(symbol, False)
                except Exception:
                    pass

        if rates is None or len(rates) < 2:
            return {"error": f"Failed to get rates for {symbol}: {mt5.last_error()}"}

        df = pd.DataFrame(rates)
        if df.empty:
            return {"error": "No data available"}

        try:
            df["__epoch"] = df["time"].astype(float).apply(_mt5_epoch_to_utc)
        except Exception:
            return {"error": "Failed to normalize bar times."}

        if not end:
            tf_secs = TIMEFRAME_SECONDS.get(timeframe)
            if tf_secs:
                now_ts = datetime.utcnow().timestamp()
                last_epoch = float(df["__epoch"].iloc[-1])
                if 0 <= (now_ts - last_epoch) < float(tf_secs) and len(df) > 1:
                    df = df.iloc[:-1]

        if len(df) < 2:
            return {"error": "Insufficient data after trimming live bars."}

        client_tz = _resolve_client_tz()
        use_client_tz = client_tz is not None
        dt_utc = pd.to_datetime(df["__epoch"], unit="s", utc=True)
        if use_client_tz:
            dt = dt_utc.dt.tz_convert(client_tz)
        else:
            dt = dt_utc
        df["__dt"] = dt

        if "close" not in df.columns:
            return {"error": "Rates data missing close prices."}

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
            with np.errstate(divide="ignore", invalid="ignore"):
                df["__range_pct"] = (df["__range"] / close) * 100.0

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
            return {"error": "Insufficient data after applying filters."}

        overall = _stats_for_group(df, volume_col)
        groups_out: List[Dict[str, Any]] = []

        if group_norm == "dow":
            df["__group"] = df["__dt"].dt.weekday
            for key, grp in df.groupby("__group", sort=True):
                row = _stats_for_group(grp, volume_col)
                row["group"] = _DOW_LABELS[int(key)] if 0 <= int(key) <= 6 else str(key)
                row["group_key"] = int(key)
                groups_out.append(row)
        elif group_norm == "month":
            df["__group"] = df["__dt"].dt.month
            for key, grp in df.groupby("__group", sort=True):
                label = _MONTH_LABELS[int(key) - 1] if 1 <= int(key) <= 12 else str(key)
                row = _stats_for_group(grp, volume_col)
                row["group"] = label
                row["group_key"] = int(key)
                groups_out.append(row)
        elif group_norm == "hour":
            df["__group"] = df["__dt"].dt.hour
            for key, grp in df.groupby("__group", sort=True):
                row = _stats_for_group(grp, volume_col)
                row["group"] = f"{int(key):02d}:00"
                row["group_key"] = int(key)
                groups_out.append(row)

        start_epoch = float(df["__epoch"].iloc[0])
        end_epoch = float(df["__epoch"].iloc[-1])
        start_str = _format_time_minimal_local(start_epoch) if use_client_tz else _format_time_minimal(start_epoch)
        end_str = _format_time_minimal_local(end_epoch) if use_client_tz else _format_time_minimal(end_epoch)

        filters: Dict[str, Any] = {}
        if dow_val is not None:
            filters["day_of_week"] = {"value": dow_val, "label": _DOW_LABELS[dow_val]}
        if month_val is not None:
            filters["month"] = {"value": month_val, "label": _MONTH_LABELS[month_val - 1]}
        if tr_start is not None and tr_end is not None:
            filters["time_range"] = {
                "start": _time_label(tr_start),
                "end": _time_label(tr_end),
                "wraps_midnight": bool(tr_start > tr_end),
            }

        tz_name = "UTC"
        if use_client_tz:
            try:
                tz_name = getattr(client_tz, "zone", None) or str(client_tz)
            except Exception:
                tz_name = "local"

        payload: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "group_by": group_norm,
            "return_mode": return_mode,
            "timezone": tz_name,
            "bars": int(len(df)),
            "start": start_str,
            "end": end_str,
            "filters": filters,
            "overall": overall,
        }
        if groups_out:
            payload["groups"] = groups_out
        return payload
    except Exception as e:
        return {"error": f"Error computing temporal analysis: {str(e)}"}
