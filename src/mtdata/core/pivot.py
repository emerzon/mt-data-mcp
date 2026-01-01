
from datetime import datetime
from typing import Any, Dict, List
import math

from .schema import TimeframeLiteral, _PIVOT_METHODS
from .constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..utils.mt5 import _mt5_copy_rates_from, _mt5_epoch_to_utc, _symbol_ready_guard
from ..utils.utils import _format_time_minimal, _format_time_minimal_local, _use_client_tz
from .server import mcp, _auto_connect_wrapper
import MetaTrader5 as mt5


@mcp.tool()
@_auto_connect_wrapper
def pivot_compute_points(
    symbol: str,
    timeframe: TimeframeLiteral = "D1",
) -> Dict[str, Any]:
    """Compute pivot point levels from the last completed bar on `timeframe`.
    Parameters: symbol, timeframe

    Returns JSON with shared source data plus levels for every supported pivot method.
    """
    try:
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_tf = TIMEFRAME_MAP[timeframe]
        tf_secs = TIMEFRAME_SECONDS.get(timeframe)
        if not tf_secs:
            return {"error": f"Unsupported timeframe seconds for {timeframe}"}

        with _symbol_ready_guard(symbol) as (err, _info_before):
            if err:
                return {"error": err}
            _tick = mt5.symbol_info_tick(symbol)
            if _tick is not None and getattr(_tick, "time", None):
                t_utc = _mt5_epoch_to_utc(float(_tick.time))
                server_now_dt = datetime.utcfromtimestamp(t_utc)
                server_now_ts = t_utc
            else:
                server_now_dt = datetime.utcnow()
                server_now_ts = server_now_dt.timestamp()
            rates = _mt5_copy_rates_from(symbol, mt5_tf, server_now_dt, 5)

        if rates is None or len(rates) == 0:
            return {"error": f"Failed to get rates for {symbol}: {mt5.last_error()}"} 

        now_ts = server_now_ts
        if len(rates) >= 2:
            src = rates[-2]
        else:
            only = rates[-1]
            if (float(only["time"]) + tf_secs) <= now_ts:
                src = only
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
        period_start = _mt5_epoch_to_utc(period_start)
        period_end = period_start + float(tf_secs)

        digits = int(getattr(_info_before, "digits", 0) or 0) if _info_before is not None else 0

        def _round(v: float) -> float:
            try:
                return round(float(v), digits) if digits >= 0 else float(v)
            except Exception:
                return float(v)

        rng = H - L

        def _compute_method(method_name: str):
            name = method_name.lower().strip()
            if name == "classic":
                PP = (H + L + C) / 3.0
                levels_raw = {
                    "PP": PP,
                    "R1": 2 * PP - L,
                    "S1": 2 * PP - H,
                    "R2": PP + rng,
                    "S2": PP - rng,
                    "R3": H + 2 * (PP - L),
                    "S3": L - 2 * (H - PP),
                }
                pivot_val = PP
            elif name == "fibonacci":
                PP = (H + L + C) / 3.0
                levels_raw = {
                    "PP": PP,
                    "R1": PP + 0.382 * rng,
                    "S1": PP - 0.382 * rng,
                    "R2": PP + 0.618 * rng,
                    "S2": PP - 0.618 * rng,
                    "R3": PP + rng,
                    "S3": PP - rng,
                }
                pivot_val = PP
            elif name == "camarilla":
                k = 1.1
                levels_raw = {
                    "PP": (H + L + C) / 3.0,
                    "R1": C + (k * rng) / 12.0,
                    "S1": C - (k * rng) / 12.0,
                    "R2": C + (k * rng) / 6.0,
                    "S2": C - (k * rng) / 6.0,
                    "R3": C + (k * rng) / 4.0,
                    "S3": C - (k * rng) / 4.0,
                    "R4": C + (k * rng) / 2.0,
                    "S4": C - (k * rng) / 2.0,
                }
                pivot_val = levels_raw["PP"]
            elif name == "woodie":
                PP = (H + L + 2 * C) / 4.0
                levels_raw = {
                    "PP": PP,
                    "R1": 2 * PP - L,
                    "S1": 2 * PP - H,
                    "R2": PP + rng,
                    "S2": PP - rng,
                }
                pivot_val = PP
            elif name == "demark":
                if C < O:
                    X = H + 2 * L + C
                elif C > O:
                    X = 2 * H + L + C
                else:
                    X = H + L + 2 * C
                PP = X / 4.0
                levels_raw = {
                    "PP": PP,
                    "R1": X / 2.0 - L,
                    "S1": X / 2.0 - H,
                }
                pivot_val = PP
            else:
                return None

            levels = {k: _round(v) for k, v in levels_raw.items()}
            return {
                "method": name,
                "pivot": _round(pivot_val) if pivot_val is not None else None,
                "levels": levels,
            }

        methods_out = []
        levels_by_method: Dict[str, Dict[str, float]] = {}
        pivot_values: Dict[str, float] = {}
        for method_name in _PIVOT_METHODS:
            method_info = _compute_method(method_name)
            if not method_info:
                continue
            methods_out.append(method_info)
            levels_by_method[method_info["method"]] = method_info["levels"]
            pivot_val = method_info.get('pivot')
            if isinstance(pivot_val, (int, float)):
                pivot_values[method_info["method"]] = float(pivot_val)

        method_names = [info["method"] for info in methods_out]
        # Determine logical level ordering across all methods
        present_levels = set()
        for info in methods_out:
            for lvl in info["levels"].keys():
                present_levels.add(str(lvl))
        # Collect numeric R/S tiers
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
        # Build preferred level sequence as seen on charts:
        # R(max) .. R1, (pivot|PP), S1 .. S(max)
        include_pivot_row = bool(pivot_values)
        level_sequence: List[str] = []
        # Resistances top→bottom
        for n in range(max_n, 0, -1):
            rn = f"R{n}"
            if rn in present_levels:
                level_sequence.append(rn)
        # Center pivot (omit PP if explicit pivot row is included)
        if not include_pivot_row and 'PP' in present_levels:
            level_sequence.append('PP')
        # Supports top→bottom after pivot
        for n in range(1, max_n + 1):
            sn = f"S{n}"
            if sn in present_levels:
                level_sequence.append(sn)
        # Append any remaining non-standard levels (e.g., method-specific extras) in sorted order
        consumed = set(level_sequence) | ({'PP'} if include_pivot_row else set())
        leftovers = sorted([lv for lv in present_levels if lv not in consumed])
        level_sequence.extend(leftovers)
        # Assemble rows in chart order: R... -> (PP) -> S... -> leftovers
        levels_table: List[Dict[str, Any]] = []
        # Resistances top→bottom
        for lvl in level_sequence:
            if not str(lvl).startswith('R'):
                continue
            row: Dict[str, Any] = {"level": lvl}
            for name in method_names:
                level_map = levels_by_method.get(name, {})
                row[name] = level_map.get(lvl)
            levels_table.append(row)
        # Central pivot between R and S
        if include_pivot_row:
            pivot_row: Dict[str, Any] = {"level": "PP"}
            for name in method_names:
                pivot_row[name] = pivot_values.get(name)
            levels_table.append(pivot_row)
        elif 'PP' in level_sequence:
            row: Dict[str, Any] = {"level": 'PP'}
            for name in method_names:
                level_map = levels_by_method.get(name, {})
                row[name] = level_map.get('PP')
            levels_table.append(row)
        # Supports top→bottom after pivot
        for lvl in level_sequence:
            if not str(lvl).startswith('S'):
                continue
            row: Dict[str, Any] = {"level": lvl}
            for name in method_names:
                level_map = levels_by_method.get(name, {})
                row[name] = level_map.get(lvl)
            levels_table.append(row)
        # Any leftover non-standard levels
        for lvl in leftovers:
            row: Dict[str, Any] = {"level": lvl}
            for name in method_names:
                level_map = levels_by_method.get(name, {})
                row[name] = level_map.get(lvl)
            levels_table.append(row)

        _use_ctz = _use_client_tz()
        start_str = _format_time_minimal_local(period_start) if _use_ctz else _format_time_minimal(period_start)
        end_str = _format_time_minimal_local(period_end) if _use_ctz else _format_time_minimal(period_end)

        payload: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "period": {
                "start": start_str,
                "end": end_str,
            },
            "levels": levels_table,
        }
        if not _use_ctz:
            payload["timezone"] = "UTC"
        return payload
    except Exception as e:
        return {"error": f"Error computing pivot points: {str(e)}"}

