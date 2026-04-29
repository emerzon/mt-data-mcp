import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ..shared.validators import (
    invalid_timeframe_error,
    unsupported_timeframe_seconds_error,
)
from ..utils.mt5 import (
    MT5ConnectionError,
    _mt5_copy_rates_from,
    _mt5_copy_rates_range,
    _mt5_epoch_to_utc as _mt5_epoch_to_utc_compat,
    _symbol_ready_guard,
    ensure_mt5_connection_or_raise,
    get_symbol_info_cached,
    mt5,
)
from ..utils.utils import (
    _format_time_minimal,
    _format_time_minimal_local,
    _parse_start_datetime,
    _resolve_client_tz,
    _safe_float,
)
from ._mcp_instance import mcp
from .constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from .execution_logging import run_logged_operation
from .mt5_gateway import get_mt5_gateway
from .output_contract import normalize_output_verbosity_detail
from .schema import CompactFullDetailLiteral, TimeframeLiteral

logger = logging.getLogger(__name__)


def _mt5_epoch_to_utc(value: float) -> float:
    """Backward-compatible patch target; MT5 reads are normalized upstream."""
    return _mt5_epoch_to_utc_compat(value)



_DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


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
    if v in ("all", "none", "overall"):
        return "all"
    return v


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


def _compact_temporal_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("group", "bars", "avg_return", "median_return", "win_rate", "volatility")
    return {key: row.get(key) for key in keys if row.get(key) is not None}


def _compact_temporal_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        key: payload[key]
        for key in (
            "success",
            "symbol",
            "timeframe",
            "group_by",
            "return_mode",
            "timezone",
            "bars",
            "start",
            "end",
        )
        if key in payload
    }
    groups = payload.get("groups")
    if isinstance(groups, list) and groups:
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
                if row.get("avg_return") is not None
            ),
            key=lambda row: float(row.get("avg_return") or 0.0),
            default=None,
        )
        if best:
            out["best"] = {
                key: best[key]
                for key in ("group", "avg_return", "win_rate")
                if key in best
            }
    elif isinstance(payload.get("overall"), dict):
        out["overall"] = _compact_temporal_stats(payload["overall"])
    for key in ("warnings", "excluded_groups"):
        value = payload.get(key)
        if value:
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
    mt5_gateway = gateway or get_mt5_gateway(
        adapter=mt5,
        ensure_connection_impl=ensure_mt5_connection_or_raise,
    )
    if timeframe not in TIMEFRAME_MAP:
        return None, invalid_timeframe_error(timeframe, TIMEFRAME_MAP)
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
            return None, unsupported_timeframe_seconds_error(timeframe)
        end_dt = start_dt + timedelta(seconds=seconds_per_bar * max(int(limit), 1))
        rates = _mt5_copy_rates_range(symbol, mt5_tf, start_dt, end_dt)
        return rates, None

    if end:
        end_dt = _parse_start_datetime(end)
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
    lookback: int = 1000,
    start: Optional[str] = None,
    end: Optional[str] = None,
    group_by: Literal[
        "dow",
        "day_of_week",
        "weekday",
        "day",
        "daily",
        "hour",
        "hours",
        "month",
        "months",
        "all",
    ] = "dow",  # type: ignore
    day_of_week: Optional[str] = None,
    month: Optional[str] = None,
    time_range: Optional[str] = None,
    return_mode: Literal["pct", "log"] = "pct",  # type: ignore
    min_bars: Optional[int] = None,
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    """Temporal analysis by day-of-week, hour, or month.

    `lookback` controls the number of historical bars used when start/end are
    omitted.

    Filters:
    - day_of_week: 0-6 or names like Mon, Tuesday
    - month: 1-12 or names like Jan, September
    - time_range: 'HH:MM-HH:MM' (start inclusive, end exclusive; local/client
      timezone if configured, wraps midnight like 22:00-02:00)
    - min_bars: exclude grouped rows below this sample size. When omitted for
      day-of-week analysis, sparse weekend groups are auto-filtered.
    - volume: uses real_volume when available and non-zero, else tick_volume

    Returns grouped averages for returns and volatility plus simple extras.
    Example: temporal_analyze(symbol="EURUSD", group_by="dow")
    Use group_by='all' for a single overall summary.
    """
    def _run() -> Dict[str, Any]:  # noqa: C901
        context: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "group_by": group_by,
            "return_mode": return_mode,
            "lookback": lookback,
            "start": start,
            "end": end,
            "detail": detail,
        }
        if min_bars is not None:
            context["min_bars"] = min_bars
        try:
            mt5_gateway = get_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            effective_lookback = 0 if lookback is None else int(lookback)
            context["lookback"] = effective_lookback
            if effective_lookback <= 1 and not (start and end):
                return _error_response(
                    "lookback must be >= 2 for return calculations.",
                    stage="validate",
                    context=context,
                )

            group_norm = _normalize_group_by(group_by)
            if group_norm not in ("dow", "hour", "month", "all"):
                return _error_response(
                    "Invalid group_by. Use: dow, hour, month, all.",
                    stage="validate",
                    context=context,
                )
            context["group_by"] = group_norm
            requested_detail = str(detail or "compact").strip().lower()
            detail_mode = normalize_output_verbosity_detail(detail, default="compact")
            if requested_detail not in {"compact", "standard", "summary", "full"}:
                return _error_response(
                    "detail must be one of: compact, standard, summary, full.",
                    stage="validate",
                    context=context,
                )

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
            dt_utc = pd.to_datetime(df["__epoch"], unit="s", utc=True)
            dt = dt_utc.dt.tz_convert(client_tz) if use_client_tz else dt_utc
            df["__dt"] = dt

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

            groups_out: List[Dict[str, Any]] = []

            if group_norm == "dow":
                df["__group"] = df["__dt"].dt.weekday
                for key, grp in df.groupby("__group", sort=True):
                    row = _stats_for_group(grp, volume_col)
                    row["group"] = _DOW_LABELS[int(key)] if 0 <= int(key) <= 6 else str(key)
                    row["_group_key"] = int(key)
                    groups_out.append(row)
            elif group_norm == "month":
                df["__group"] = df["__dt"].dt.month
                for key, grp in df.groupby("__group", sort=True):
                    label = _MONTH_LABELS[int(key) - 1] if 1 <= int(key) <= 12 else str(key)
                    row = _stats_for_group(grp, volume_col)
                    row["group"] = label
                    row["_group_key"] = int(key)
                    groups_out.append(row)
            elif group_norm == "hour":
                df["__group"] = df["__dt"].dt.hour
                for key, grp in df.groupby("__group", sort=True):
                    row = _stats_for_group(grp, volume_col)
                    row["group"] = f"{int(key):02d}:00"
                    row["_group_key"] = int(key)
                    groups_out.append(row)

            excluded_groups: List[Dict[str, Any]] = []
            auto_min_bars = False
            if (
                min_bars_value is None
                and group_norm == "dow"
                and dow_val is None
                and groups_out
            ):
                bar_counts = [int(row.get("bars", 0) or 0) for row in groups_out]
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
                            "bars": int(row.get("bars", 0) or 0),
                            "min_bars": int(min_bars_value),
                            "auto": bool(auto_min_bars),
                        }
                        for row in excluded
                    ]
                    groups_out = included
                    analysis_df = df[~df["__group"].isin(excluded_keys)]

            groups_out = [
                {key: value for key, value in row.items() if key != "_group_key"}
                for row in groups_out
            ]
            overall = _stats_for_group(analysis_df, volume_col)

            start_epoch = float(analysis_df["__epoch"].iloc[0])
            end_epoch = float(analysis_df["__epoch"].iloc[-1])
            start_str = _format_time_minimal_local(start_epoch) if use_client_tz else _format_time_minimal(start_epoch)
            end_str = _format_time_minimal_local(end_epoch) if use_client_tz else _format_time_minimal(end_epoch)

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
                "bars": int(len(analysis_df)),
                "start": start_str,
                "end": end_str,
                "filters": filters,
                "overall": overall,
                "volume_source": volume_col,
            }
            if excluded_groups:
                filters["min_bars"] = {
                    "value": int(min_bars_value or 0),
                    "auto": bool(auto_min_bars),
                }
                payload["excluded_groups"] = excluded_groups
                payload["warnings"] = [
                    "Sparse temporal groups below min_bars were excluded from "
                    "grouped results and overall summary."
                ]
            if groups_out:
                payload["groups"] = groups_out
            if detail_mode == "compact":
                return _compact_temporal_payload(payload)
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
        func=_run,
    )
