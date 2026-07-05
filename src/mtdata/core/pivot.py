
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from ..forecast.common import fetch_history as _fetch_history
from ..shared.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..shared.schema import (
    _PIVOT_METHODS,
    AutoTimeframeLiteral,
    CompactStandardFullDetailLiteral,
    PivotMethodLiteral,
    TimeframeLiteral,
)
from ..shared.validators import (
    invalid_timeframe_error,
    unsupported_timeframe_seconds_error,
)
from ..utils.level_confluence import build_level_confluence_payload
from ..utils.mt5 import (
    MT5ConnectionError,
    _mt5_copy_rates_from,
    _symbol_ready_guard,
    ensure_mt5_connection_or_raise,
    mt5,
)
from ..utils.pivot_points import compute_pivot_method_levels, compute_pivot_methods
from ..utils.support_resistance import (
    compact_support_resistance_payload,
    compute_support_resistance_levels,
    full_support_resistance_payload,
    get_auto_support_resistance_timeframes,
    merge_support_resistance_results,
    standard_support_resistance_payload,
)
from ..utils.utils import (
    _format_time_minimal,
    _format_time_minimal_local,
    _positive_float_attr,
    _resolve_client_tz,
    _use_client_tz,
)
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway
from .output_contract import normalize_output_extras
from .runtime_metadata import display_timezone_label
from .volume_profile import compute_volume_profile_payload

logger = logging.getLogger(__name__)


_LEVEL_PRICE_FIELD_NAMES = frozenset(
    {
        "value",
        "price",
        "reference_price",
        "current_price",
        "nearest_support",
        "nearest_resistance",
        "low",
        "high",
        "width",
        "distance",
        "range",
    }
)


def _symbol_price_digits(info: Any) -> Optional[int]:
    try:
        digits = int(info.digits)
    except Exception:
        return None
    if digits < 0 or digits > 15:
        return None
    return digits


def _round_level_price(value: Any, *, digits: int) -> Any:
    if isinstance(value, bool):
        return value
    try:
        number = float(value)
    except Exception:
        return value
    if not math.isfinite(number):
        return value
    return round(number, max(0, int(digits)))


def _round_level_payload_prices(value: Any, *, digits: Optional[int], key: Optional[str] = None) -> Any:
    if digits is None:
        return value
    if isinstance(value, dict):
        return {
            item_key: _round_level_payload_prices(
                item_value,
                digits=digits,
                key=str(item_key),
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [
            _round_level_payload_prices(item, digits=digits, key=key)
            for item in value
        ]
    if key in _LEVEL_PRICE_FIELD_NAMES or (isinstance(key, str) and key.endswith("_price")):
        return _round_level_price(value, digits=digits)
    return value


def _confluence_volume_profile_window(
    sr_timeframe: str,
    lookback: int,
) -> tuple[str, int]:
    timeframe = str(sr_timeframe or "H1").strip().upper()
    if timeframe == "AUTO":
        timeframe = max(
            get_auto_support_resistance_timeframes(),
            key=lambda value: int(TIMEFRAME_SECONDS.get(value, 0) or 0),
        )
    seconds = int(TIMEFRAME_SECONDS.get(timeframe, 0) or 0)
    minutes_per_bar = max(1, int(math.ceil(seconds / 60.0)))
    return timeframe, max(1, int(lookback) * minutes_per_bar)


_PIVOT_METHOD_INFO: Dict[str, Dict[str, str]] = {
    "classic": {
        "method_description": "PP=(H+L+C)/3; R/S levels extend arithmetically from the prior bar range.",
        "intended_use": (
            "Timeframe-matched classic pivot context from the last completed source bar; "
            "use D1 for conventional daily floor-trader pivots."
        ),
    },
    "fibonacci": {
        "method_description": "PP=(H+L+C)/3; R/S levels use 0.382, 0.618, and 1.000 range multiples.",
        "intended_use": "Traders who align pivot levels with Fibonacci retracement/extension zones.",
    },
    "camarilla": {
        "method_description": "Levels are close-centered using 1.1 * prior range fractions; includes R1-R4/S1-S4.",
        "intended_use": "Intraday mean-reversion/breakout context; R3/S3 and R4/S4 are commonly watched.",
    },
    "woodie": {
        "method_description": "PP=(H+L+2*C)/4, weighting the close more heavily than classic pivots.",
        "intended_use": "Close-sensitive intraday pivot context.",
    },
    "demark": {
        "method_description": "Uses open/close relationship to choose X, then computes PP, R1, and S1.",
        "intended_use": "Directional single-level pivot context from the prior bar.",
    },
}


def _tick_reference_price(tick: Any) -> Optional[float]:
    if tick is None:
        return None
    bid = _positive_float_attr(tick, "bid")
    ask = _positive_float_attr(tick, "ask")
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    return _positive_float_attr(tick, "last", "bid", "ask")


def _resolve_support_resistance_timeframes(timeframe: Optional[str]) -> tuple[str, List[str]]:
    raw = str(timeframe or "auto").strip()
    if not raw or raw.lower() == "auto":
        return "auto", list(get_auto_support_resistance_timeframes())
    normalized = raw.upper()
    if normalized not in TIMEFRAME_MAP:
        raise RuntimeError(invalid_timeframe_error(normalized, TIMEFRAME_MAP))
    return normalized, [normalized]


def compute_support_resistance_payload(
    *,
    fetch_history_impl,
    symbol: str,
    timeframe: Optional[str],
    limit: int,
    tolerance_pct: float,
    min_touches: int,
    max_levels: int,
    reaction_bars: int,
    adx_period: int,
    decay_half_life_bars: Optional[int],
    max_distance_pct: Optional[float],
    volume_weighting: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    requested_timeframe, timeframes = _resolve_support_resistance_timeframes(timeframe)
    multi_timeframe = len(timeframes) > 1
    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    partial_warnings: List[Dict[str, Any]] = []
    per_timeframe_min_touches = 1 if multi_timeframe else int(min_touches)
    per_timeframe_max_levels = max(int(max_levels), 1) if not multi_timeframe else max(int(max_levels) * 2, 6)

    for tf in timeframes:
        try:
            history_kwargs: Dict[str, Any] = {
                "symbol": symbol,
                "timeframe": tf,
                "need": int(limit),
            }
            if start or end:
                history_kwargs.update({"start": start, "end": end})
            frame = fetch_history_impl(**history_kwargs)
            if frame is None:
                raise RuntimeError("No history available")
            if len(frame) > int(limit):
                frame = frame.iloc[-int(limit):].copy()
            result = compute_support_resistance_levels(
                frame,
                symbol=symbol,
                timeframe=tf,
                limit=int(limit),
                tolerance_pct=float(tolerance_pct),
                min_touches=int(per_timeframe_min_touches),
                max_levels=int(per_timeframe_max_levels),
                reaction_bars=int(reaction_bars),
                adx_period=int(adx_period),
                decay_half_life_bars=None if decay_half_life_bars is None else int(decay_half_life_bars),
                max_distance_pct=None if max_distance_pct is None else float(max_distance_pct),
                volume_weighting=str(volume_weighting),
            )
            if (result.get("levels") or []) or not multi_timeframe:
                results.append(result)
        except Exception as exc:
            error_text = f"{tf}: {exc}"
            errors.append(error_text)
            partial_warnings.append(
                {
                    "code": "timeframe_failed",
                    "timeframe": tf,
                    "message": error_text,
                }
            )
            if not multi_timeframe:
                raise

    if not results:
        if errors:
            raise RuntimeError("; ".join(errors))
        raise RuntimeError("No history available")

    if not multi_timeframe:
        return results[0]

    merged = merge_support_resistance_results(
        results,
        symbol=symbol,
        timeframe=requested_timeframe,
        limit=int(limit),
        tolerance_pct=float(tolerance_pct),
        min_touches=int(min_touches),
        max_levels=int(max_levels),
        reaction_bars=int(reaction_bars),
        adx_period=int(adx_period),
        decay_half_life_bars=None if decay_half_life_bars is None else int(decay_half_life_bars),
        max_distance_pct=None if max_distance_pct is None else float(max_distance_pct),
        volume_weighting=str(volume_weighting),
    )
    if partial_warnings:
        merged["warnings"] = list(merged.get("warnings") or []) + partial_warnings
    return merged


@mcp.tool()
def pivot_compute_points(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "D1",
    method: Optional[PivotMethodLiteral] = None,
    detail: CompactStandardFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    """Compute pivot point levels from the last completed bar on `timeframe`.
    Parameters: symbol, timeframe, method, detail

    Defaults to D1 because daily pivots are the common floor-trader convention.
    Compact detail returns classic pivots while standard/full include every
    supported method. Set `method` to return only one pivot method.
    Use `support_resistance_levels` for complementary data-driven levels from
    historical retests and reactions.
    """
    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            mt5 = create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise)
            mt5.ensure_connection()
            method_filter = str(method).strip().lower() if method is not None else None
            if method_filter and method_filter not in _PIVOT_METHODS:
                return {
                    "error": (
                        f"Invalid pivot method: {method_filter}. "
                        f"Valid methods: {', '.join(_PIVOT_METHODS)}"
                    )
                }
            if timeframe not in TIMEFRAME_MAP:
                return {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP)}
            mt5_tf = TIMEFRAME_MAP[timeframe]
            tf_secs = TIMEFRAME_SECONDS.get(timeframe)
            if not tf_secs:
                return {"error": unsupported_timeframe_seconds_error(timeframe)}

            with _symbol_ready_guard(symbol) as (err, _info_before):
                if err:
                    return {"error": err}
                system_now_dt = datetime.now(timezone.utc)
                system_now_ts = system_now_dt.timestamp()
                server_now_dt = system_now_dt
                server_now_ts = system_now_ts
                _tick = mt5.symbol_info_tick(symbol)
                if _tick is not None and getattr(_tick, "time", None):
                    t_utc = float(_tick.time)
                    freshness_limit = float(max(tf_secs, 300))
                    if abs(system_now_ts - t_utc) <= freshness_limit:
                        server_now_ts = t_utc
                        server_now_dt = datetime.fromtimestamp(server_now_ts, tz=timezone.utc)
                rates = _mt5_copy_rates_from(symbol, mt5_tf, server_now_dt, 5)

            if rates is None or len(rates) == 0:
                return {"error": f"Failed to get rates for {symbol}: {mt5.last_error()}"}

            now_ts = server_now_ts
            latest = rates[-1]
            if (float(latest["time"]) + tf_secs) <= now_ts:
                src = latest
            elif len(rates) >= 2:
                src = rates[-2]
            else:
                return {"error": "No completed bars available to compute pivot points"}

            def _has_field(row, name: str) -> bool:
                try:
                    if isinstance(row, dict):
                        return name in row
                    dt = getattr(row, 'dtype', None)
                    names = getattr(dt, 'names', None) if dt is not None else None
                    return bool(names and name in names)
                except Exception:
                    return False

            H = float(src["high"]) if _has_field(src, "high") else float("nan")
            L = float(src["low"]) if _has_field(src, "low") else float("nan")
            C = float(src["close"]) if _has_field(src, "close") else float("nan")
            O = float(src["open"]) if _has_field(src, "open") else C
            if any(math.isnan(v) for v in (H, L, C)):
                return {"error": "Pivot calculation requires high, low, and close prices"}

            period_start = float(src["time"]) if _has_field(src, "time") else float("nan")
            period_end = period_start + float(tf_secs)

            digits = int(getattr(_info_before, "digits", 0) or 0) if _info_before is not None else 0

            def _round(v: float) -> float:
                try:
                    return round(float(v), digits) if digits >= 0 else float(v)
                except Exception:
                    return float(v)

            def _round_context(v: float) -> float:
                try:
                    return round(float(v), max(int(digits) + 2, 8))
                except Exception:
                    return float(v)

            rng = H - L
            price_increment = _positive_float_attr(
                _info_before,
                "trade_tick_size",
                "point",
            )
            if price_increment is None and digits >= 0:
                price_increment = 10.0 ** (-int(digits))

            def _degenerate_levels_info(levels: Dict[str, float]) -> Dict[str, Any]:
                values = [
                    float(value)
                    for value in levels.values()
                    if isinstance(value, (int, float)) and math.isfinite(float(value))
                ]
                if not values:
                    return {}
                unique_count = len(set(values))
                reasons: List[str] = []
                if len(values) >= 3 and unique_count < 3:
                    reasons.append(
                        f"Only {unique_count} unique rounded level price(s) remain."
                    )
                if (
                    price_increment is not None
                    and math.isfinite(rng)
                    and rng < 2.0 * price_increment
                ):
                    reasons.append(
                        "Source bar range "
                        f"({_round_context(rng)}) is smaller than 2x price increment "
                        f"({_round_context(price_increment)})."
                    )
                if not reasons:
                    return {}
                return {
                    "levels_degenerate": True,
                    "reason": " ".join(reasons)
                    + " Pivot levels may appear identical after rounding.",
                    "source_range": _round_context(rng),
                    "price_increment": _round_context(price_increment)
                    if price_increment is not None
                    else None,
                    "digits": digits,
                    "unique_level_count": unique_count,
                }

            def _compute_method(method_name: str):
                method_info = compute_pivot_method_levels(
                    method_name,
                    open_price=O,
                    high_price=H,
                    low_price=L,
                    close_price=C,
                    digits=digits,
                )
                if not method_info:
                    return None
                return {
                    **method_info,
                    **_PIVOT_METHOD_INFO.get(str(method_info.get("method")), {}),
                }

            methods_out = []
            levels_by_method: Dict[str, Dict[str, float]] = {}
            pivot_values: Dict[str, float] = {}
            for method_name in _PIVOT_METHODS:
                if method_filter and method_name != method_filter:
                    continue
                method_info = _compute_method(method_name)
                if not method_info:
                    continue
                methods_out.append(method_info)
                levels_by_method[method_info["method"]] = method_info["levels"]
                pivot_val = method_info.get('pivot')
                if isinstance(pivot_val, (int, float)):
                    pivot_values[method_info["method"]] = float(pivot_val)

            method_names = [info["method"] for info in methods_out]
            present_levels = set()
            for info in methods_out:
                for lvl in info["levels"].keys():
                    present_levels.add(str(lvl))
            import re as _re
            rs_nums = set()
            for name in list(present_levels):
                m = _re.match(r"^([RS])(\d+)$", str(name))
                if m:
                    try:
                        rs_nums.add(int(m.group(2)))
                    except Exception:
                        pass
            max_n = max(rs_nums) if rs_nums else 0
            include_pivot_row = bool(pivot_values)
            level_sequence: List[str] = []
            for n in range(max_n, 0, -1):
                rn = f"R{n}"
                if rn in present_levels:
                    level_sequence.append(rn)
            if not include_pivot_row and 'PP' in present_levels:
                level_sequence.append('PP')
            for n in range(1, max_n + 1):
                sn = f"S{n}"
                if sn in present_levels:
                    level_sequence.append(sn)
            consumed = set(level_sequence) | ({'PP'} if include_pivot_row else set())
            leftovers = sorted([lv for lv in present_levels if lv not in consumed])
            level_sequence.extend(leftovers)
            levels_table: List[Dict[str, Any]] = []
            for lvl in level_sequence:
                if not str(lvl).startswith('R'):
                    continue
                row: Dict[str, Any] = {"level": lvl}
                for name in method_names:
                    level_map = levels_by_method.get(name, {})
                    val = level_map.get(lvl)
                    if val is not None:
                        row[name] = val
                levels_table.append(row)
            if include_pivot_row:
                pivot_row: Dict[str, Any] = {"level": "PP"}
                for name in method_names:
                    if name in pivot_values:
                        pivot_row[name] = pivot_values.get(name)
                levels_table.append(pivot_row)
            elif 'PP' in level_sequence:
                row: Dict[str, Any] = {"level": 'PP'}
                for name in method_names:
                    level_map = levels_by_method.get(name, {})
                    val = level_map.get('PP')
                    if val is not None:
                        row[name] = val
                levels_table.append(row)
            for lvl in level_sequence:
                if not str(lvl).startswith('S'):
                    continue
                row: Dict[str, Any] = {"level": lvl}
                for name in method_names:
                    level_map = levels_by_method.get(name, {})
                    val = level_map.get(lvl)
                    if val is not None:
                        row[name] = val
                levels_table.append(row)
            for lvl in leftovers:
                row: Dict[str, Any] = {"level": lvl}
                for name in method_names:
                    level_map = levels_by_method.get(name, {})
                    val = level_map.get(lvl)
                    if val is not None:
                        row[name] = val
                levels_table.append(row)

            _use_ctz = _use_client_tz()
            timezone_label = display_timezone_label(
                use_client_tz=_use_ctz,
                fallback="UTC",
                resolve_client_tz=_resolve_client_tz,
            )
            start_str = _format_time_minimal_local(period_start) if _use_ctz else _format_time_minimal(period_start)
            end_str = _format_time_minimal_local(period_end) if _use_ctz else _format_time_minimal(period_end)
            period_note = None
            if str(timeframe).upper() in {"D1", "W1", "MN1"}:
                period_note = (
                    "MT5 daily/weekly/monthly bar periods follow broker/server "
                    "session boundaries; UTC timestamps may not align to UTC "
                    "calendar midnight."
                )

            detail_value = str(detail).strip().lower()
            if detail_value in {"summary", "summary_only"}:
                detail_value = "compact"
            elif detail_value not in {"compact", "standard", "full"}:
                detail_value = "compact"

            payload: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "period": {
                    "start": start_str,
                    "end": end_str,
                },
                "calculation_basis": {
                    "source_bar": f"last completed {timeframe} bar",
                    "session_boundary": "MT5 broker/session calendar",
                    "display_timezone": timezone_label,
                },
                "levels_note": (
                    "null cells mean that pivot method does not define that level. "
                    "Camarilla levels are centered on the close price, so S1 may be above PP and R1 may be below PP."
                ),
                "method_descriptions": {
                    name: dict(_PIVOT_METHOD_INFO.get(name, {}))
                    for name in method_names
                },
                "levels": levels_table,
            }
            if period_note:
                payload["period_note"] = period_note
            payload["timezone"] = timezone_label
            if detail_value == "compact":
                compact_method_name = method_filter or "classic"
                selected_method = next(
                    (
                        info
                        for info in methods_out
                        if str(info.get("method")).strip().lower() == compact_method_name
                    ),
                    methods_out[0] if methods_out else None,
                )
                compact_levels = (
                    dict(selected_method.get("levels", {}))
                    if isinstance(selected_method, dict)
                    else {}
                )
                compact_payload: Dict[str, Any] = {
                    "success": True,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "period": payload["period"],
                    "method": (
                        selected_method.get("method")
                        if isinstance(selected_method, dict)
                        else "classic"
                    ),
                    "pivot": (
                        selected_method.get("pivot")
                        if isinstance(selected_method, dict)
                        else None
                    ),
                    "levels": compact_levels,
                }
                if period_note:
                    compact_payload["period_note"] = period_note
                compact_payload["timezone"] = timezone_label
                degenerate_info = _degenerate_levels_info(compact_payload["levels"])
                if degenerate_info:
                    compact_payload.update(degenerate_info)
                return compact_payload
            payload["detail"] = "full" if detail_value == "full" else "standard"
            if detail_value == "full":
                payload["methods"] = methods_out
            return payload
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error computing pivot points: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="pivot_compute_points",
        symbol=symbol,
        timeframe=timeframe,
        method=method,
        detail=detail,
        func=_run,
    )


@mcp.tool()
def confluence_levels(  # noqa: C901
    symbol: str,
    pivot_timeframe: TimeframeLiteral = "D1",
    sr_timeframe: AutoTimeframeLiteral = "auto",
    lookback: int = 200,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tolerance_pct: float = 0.0015,
    tolerance_points: Optional[float] = None,
    min_touches: int = 2,
    max_levels: int = 5,
    max_distance_pct: Optional[float] = 5.0,
    min_source_families: int = 1,
    pivot_method: Optional[PivotMethodLiteral] = None,
    volume_weighting: Literal["off", "auto"] = "off",
    reaction_bars: int = 6,
    adx_period: int = 14,
    decay_half_life_bars: Optional[int] = None,
    volume_profile_source: Literal["auto", "ticks", "m1_bars"] = "auto",
    volume_profile_max_tick_window_days: int = 7,
    volume_profile_max_ticks: int = 50_000,
    detail: CompactStandardFullDetailLiteral = "compact",
    extras: Optional[str] = None,
) -> Dict[str, Any]:
    """Find nearby high-probability price zones where multiple level methods agree.

    Combines formula pivot levels, touch-derived support/resistance, and
    Fibonacci swing levels. Defaults use daily pivots and auto-timeframe S/R.
    Single-family clusters are returned but score lower than multi-family
    confluence. Use `min_source_families=2` to require independent agreement.
    """

    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            gateway = create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise)
            gateway.ensure_connection()

            pivot_tf = str(pivot_timeframe or "D1").strip().upper()
            sr_tf = str(sr_timeframe or "auto").strip()
            method_filter = str(pivot_method).strip().lower() if pivot_method is not None else None
            if method_filter and method_filter not in _PIVOT_METHODS:
                return {
                    "error": (
                        f"Invalid pivot method: {method_filter}. "
                        f"Valid methods: {', '.join(_PIVOT_METHODS)}"
                    )
                }
            if pivot_tf not in TIMEFRAME_MAP:
                return {"error": invalid_timeframe_error(pivot_tf, TIMEFRAME_MAP)}
            mt5_tf = TIMEFRAME_MAP[pivot_tf]
            tf_secs = TIMEFRAME_SECONDS.get(pivot_tf)
            if not tf_secs:
                return {"error": unsupported_timeframe_seconds_error(pivot_tf)}
            if tolerance_points is not None and float(tolerance_points) < 0.0:
                return {"error": "tolerance_points must be non-negative"}
            if float(tolerance_pct) < 0.0:
                return {"error": "tolerance_pct must be non-negative"}

            def _has_field(row, name: str) -> bool:
                try:
                    if isinstance(row, dict):
                        return name in row
                    dt = getattr(row, "dtype", None)
                    names = getattr(dt, "names", None) if dt is not None else None
                    return bool(names and name in names)
                except Exception:
                    return False

            with _symbol_ready_guard(symbol) as (err, info_before):
                if err:
                    return {"error": err}
                system_now_dt = datetime.now(timezone.utc)
                system_now_ts = system_now_dt.timestamp()
                server_now_dt = system_now_dt
                server_now_ts = system_now_ts
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None and getattr(tick, "time", None):
                    tick_time = float(tick.time)
                    freshness_limit = float(max(tf_secs, 300))
                    if abs(system_now_ts - tick_time) <= freshness_limit:
                        server_now_ts = tick_time
                        server_now_dt = datetime.fromtimestamp(server_now_ts, tz=timezone.utc)
                rates = _mt5_copy_rates_from(symbol, mt5_tf, server_now_dt, 5)

            if rates is None or len(rates) == 0:
                return {"error": f"Failed to get rates for {symbol}: {mt5.last_error()}"}

            latest = rates[-1]
            if (float(latest["time"]) + tf_secs) <= server_now_ts:
                source_bar = latest
            elif len(rates) >= 2:
                source_bar = rates[-2]
            else:
                return {"error": "No completed bars available to compute pivot points"}

            high = float(source_bar["high"]) if _has_field(source_bar, "high") else float("nan")
            low = float(source_bar["low"]) if _has_field(source_bar, "low") else float("nan")
            close = float(source_bar["close"]) if _has_field(source_bar, "close") else float("nan")
            open_ = float(source_bar["open"]) if _has_field(source_bar, "open") else close
            if any(math.isnan(value) for value in (high, low, close)):
                return {"error": "Pivot calculation requires high, low, and close prices"}

            digits = int(getattr(info_before, "digits", 0) or 0) if info_before is not None else 0
            price_increment = _positive_float_attr(info_before, "trade_tick_size", "point")
            if price_increment is None and digits >= 0:
                price_increment = 10.0 ** (-int(digits))

            requested_methods = [method_filter] if method_filter else list(_PIVOT_METHODS)
            pivot_methods = compute_pivot_methods(
                open_price=open_,
                high_price=high,
                low_price=low,
                close_price=close,
                digits=digits,
                methods=requested_methods,
            )
            for method_info in pivot_methods:
                method_name = str(method_info.get("method") or "")
                method_info.update(_PIVOT_METHOD_INFO.get(method_name, {}))

            sr_payload = compute_support_resistance_payload(
                fetch_history_impl=_fetch_history,
                symbol=symbol,
                timeframe=sr_tf,
                limit=int(lookback),
                start=start,
                end=end,
                tolerance_pct=float(tolerance_pct),
                min_touches=int(min_touches),
                max_levels=max(1, int(max_levels)),
                max_distance_pct=None if max_distance_pct is None else float(max_distance_pct),
                volume_weighting=str(volume_weighting),
                reaction_bars=int(reaction_bars),
                adx_period=int(adx_period),
                decay_half_life_bars=None if decay_half_life_bars is None else int(decay_half_life_bars),
            )

            reference_price = _tick_reference_price(tick)
            reference_price_source = "live_tick"
            if reference_price is None:
                reference_price_source = "last_completed_bar_close"
                try:
                    sr_current = sr_payload.get("current_price")
                    reference_price = float(sr_current) if sr_current is not None else None
                except Exception:
                    reference_price = None
            if reference_price is None or not math.isfinite(float(reference_price)):
                reference_price = close
                reference_price_source = "last_completed_bar_close"

            detail_value = str(detail).strip().lower()
            if normalize_output_extras(extras):
                detail_value = "full"
            if detail_value in {"summary", "summary_only"}:
                detail_value = "compact"
            volume_profile_payload: Optional[Dict[str, Any]]
            vp_timeframe, max_m1_bars = _confluence_volume_profile_window(
                sr_tf,
                int(lookback),
            )
            volume_profile_payload = compute_volume_profile_payload(
                symbol=symbol,
                start=start,
                end=end,
                timeframe=vp_timeframe,
                limit=int(lookback),
                source=volume_profile_source,
                price_source="mid",
                volume_source="auto",
                bucket_points=None,
                bucket_count=80,
                max_buckets=120,
                value_area_pct=0.70,
                reference_price=float(reference_price),
                max_tick_window_days=int(volume_profile_max_tick_window_days),
                max_ticks=int(volume_profile_max_ticks),
                max_m1_bars=max_m1_bars,
                detail="compact",
            )

            payload = build_level_confluence_payload(
                symbol=symbol,
                pivot_timeframe=pivot_tf,
                sr_timeframe=str(sr_payload.get("timeframe") or sr_tf),
                pivot_methods=pivot_methods,
                support_resistance_payload=sr_payload,
                reference_price=float(reference_price),
                tolerance_pct=float(tolerance_pct),
                tolerance_points=tolerance_points,
                price_increment=price_increment,
                max_levels=int(max_levels),
                max_distance_pct=None if max_distance_pct is None else float(max_distance_pct),
                min_source_families=max(1, int(min_source_families)),
                detail=detail_value,
                volume_profile_payload=volume_profile_payload,
            )
            payload["reference_price_source"] = reference_price_source
            if reference_price_source == "live_tick":
                payload["reference_price_as_of"] = (
                    datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
                )
            else:
                payload.setdefault("warnings", []).append(
                    "reference_price is the latest completed bar close (no live tick available); "
                    "the proximity of price to support/resistance reflects the analysis window, "
                    "not a live quote."
                )
            period_start = float(source_bar["time"]) if _has_field(source_bar, "time") else float("nan")
            if math.isfinite(period_start):
                _use_ctz = _use_client_tz()
                payload["pivot_period"] = {
                    "start": _format_time_minimal_local(period_start) if _use_ctz else _format_time_minimal(period_start),
                    "end": _format_time_minimal_local(period_start + float(tf_secs))
                    if _use_ctz
                    else _format_time_minimal(period_start + float(tf_secs)),
                }
                payload["timezone"] = display_timezone_label(
                    use_client_tz=_use_ctz,
                    fallback="UTC",
                    resolve_client_tz=_resolve_client_tz,
                )
            else:
                payload["timezone"] = "UTC"
            if detail_value != "compact":
                payload["calculation_basis"] = {
                    "pivot_source_bar": f"last completed {pivot_tf} bar",
                    "support_resistance_timeframe": str(sr_payload.get("timeframe") or sr_tf),
                    "reference_price": "latest tick midpoint/last when available, else S/R current price or pivot close",
                    "volume_profile": (
                        f"{volume_profile_payload.get('source')} source"
                        if isinstance(volume_profile_payload, dict) and volume_profile_payload.get("success")
                        else "unavailable"
                    ),
                }
            warnings = sr_payload.get("warnings")
            if isinstance(warnings, list) and warnings:
                payload["warnings"] = list(warnings)
            digits_value = _symbol_price_digits(info_before)
            if digits_value is not None:
                payload["price_precision"] = digits_value
                payload = _round_level_payload_prices(payload, digits=digits_value)
            return payload
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error computing confluence levels: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="confluence_levels",
        symbol=symbol,
        pivot_timeframe=pivot_timeframe,
        sr_timeframe=sr_timeframe,
        lookback=lookback,
        start=start,
        end=end,
        tolerance_pct=tolerance_pct,
        tolerance_points=tolerance_points,
        min_touches=min_touches,
        max_levels=max_levels,
        max_distance_pct=max_distance_pct,
        min_source_families=min_source_families,
        pivot_method=pivot_method,
        volume_weighting=volume_weighting,
        reaction_bars=reaction_bars,
        adx_period=adx_period,
        decay_half_life_bars=decay_half_life_bars,
        volume_profile_source=volume_profile_source,
        volume_profile_max_tick_window_days=volume_profile_max_tick_window_days,
        volume_profile_max_ticks=volume_profile_max_ticks,
        detail=detail,
        extras=extras,
        func=_run,
    )


@mcp.tool()
def support_resistance_levels(
    symbol: str,
    timeframe: AutoTimeframeLiteral = "H1",
    lookback: int = 200,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tolerance_pct: float = 0.0015,
    min_touches: int = 2,
    max_levels: int = 4,
    max_distance_pct: Optional[float] = 5.0,
    volume_weighting: Literal["off", "auto"] = "off",
    reaction_bars: int = 6,
    adx_period: int = 14,
    decay_half_life_bars: Optional[int] = None,
    detail: CompactStandardFullDetailLiteral = "compact",
    extras: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect support/resistance levels around the current price from historical structure.

    Set `timeframe="auto"` to merge levels from M15, H1, H4, and D1.
    `lookback` caps the historical bars used to detect levels after applying
    any optional `start`/`end` time window.
    Use `detail="compact"` for the nearest-level summary, `detail="standard"`
    for compact actionable supports/resistances/levels plus Fibonacci swing
    levels, and `detail="full"` for the raw diagnostic payload. The default
    `max_distance_pct=5.0` keeps returned levels near current price; pass
    `None` for all levels.
    Set `extras="metadata"` to return the full diagnostic payload without
    changing every command call site to `detail="full"`.
    Level `type` reflects current price geometry; `dominant_source` reflects
    whether historical tests mostly behaved as support or resistance.
    Use `pivot_compute_points` for complementary formula-based PP/R/S levels
    from the last completed OHLC bar.

    Score combines:
    - repeated tests of a level
    - bounce strength after each test (normalized by ATR)
    - pre-test ADX trend strength
    - exponential time decay so recent tests matter more
    - ATR-filtered Fibonacci retracement/extension levels from the most relevant completed swing
    """

    def _run() -> Dict[str, Any]:
        try:
            gateway = create_mt5_gateway(
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            gateway.ensure_connection()
            symbol_info = gateway.symbol_info(symbol)
            digits_value = _symbol_price_digits(symbol_info)
            result = compute_support_resistance_payload(
                fetch_history_impl=_fetch_history,
                symbol=symbol,
                timeframe=timeframe,
                limit=int(lookback),
                start=start,
                end=end,
                tolerance_pct=float(tolerance_pct),
                min_touches=int(min_touches),
                max_levels=int(max_levels),
                max_distance_pct=None if max_distance_pct is None else float(max_distance_pct),
                volume_weighting=str(volume_weighting),
                reaction_bars=int(reaction_bars),
                adx_period=int(adx_period),
                decay_half_life_bars=None if decay_half_life_bars is None else int(decay_half_life_bars),
            )
            detail_value = str(detail).strip().lower()
            if normalize_output_extras(extras):
                detail_value = "full"
            if detail_value in {"summary", "summary_only"}:
                detail_value = "compact"
            if detail_value == "compact":
                payload = compact_support_resistance_payload(result)
            elif detail_value == "standard":
                payload = standard_support_resistance_payload(result)
            else:
                detail_value = "full"
                payload = full_support_resistance_payload(result)
            payload["detail"] = detail_value
            payload.setdefault("timezone", "UTC")
            if digits_value is not None:
                payload["price_precision"] = digits_value
                payload = _round_level_payload_prices(payload, digits=digits_value)
            return payload
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error computing support/resistance levels: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="support_resistance_levels",
        symbol=symbol,
        timeframe=timeframe,
        lookback=lookback,
        start=start,
        end=end,
        tolerance_pct=tolerance_pct,
        min_touches=min_touches,
        max_levels=max_levels,
        max_distance_pct=max_distance_pct,
        volume_weighting=volume_weighting,
        reaction_bars=reaction_bars,
        adx_period=adx_period,
        decay_half_life_bars=decay_half_life_bars,
        detail=detail,
        extras=extras,
        func=_run,
    )
