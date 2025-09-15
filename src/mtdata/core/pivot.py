
from datetime import datetime
from typing import Any, Dict, Optional

from .schema import TimeframeLiteral, PivotMethodLiteral, _PIVOT_METHODS
from .constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..utils.mt5 import _mt5_copy_rates_from, _mt5_epoch_to_utc
from ..utils.utils import _format_time_minimal_util, _format_time_minimal_local_util, _use_client_tz_util
from .server import mcp, _auto_connect_wrapper, _ensure_symbol_ready
import MetaTrader5 as mt5

@mcp.tool()
@_auto_connect_wrapper
def compute_pivot_points(
    symbol: str,
    timeframe: TimeframeLiteral = "D1",
    method: PivotMethodLiteral = "classic",
    timezone: str = "auto",
) -> Dict[str, Any]:
    """Compute pivot point levels from the last completed bar on `timeframe`.
    Parameters: symbol, timeframe, method, timezone

    - `timeframe`: Timeframe to source H/L/C from (e.g., D1, W1, MN1).
    - `method`: One of classic, fibonacci, camarilla, woodie, demark.

    Returns JSON with period info, source H/L/C, and computed levels.
    """
    try:
        # Validate timeframe
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_tf = TIMEFRAME_MAP[timeframe]
        tf_secs = TIMEFRAME_SECONDS.get(timeframe)
        if not tf_secs:
            return {"error": f"Unsupported timeframe seconds for {timeframe}"}

        method_l = str(method).lower().strip()
        if method_l not in _PIVOT_METHODS:
            return {"error": f"Invalid method: {method}. Valid options: {list(_PIVOT_METHODS)}"}

        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = mt5.symbol_info(symbol)
        _was_visible = bool(_info_before.visible) if _info_before is not None else None
        err = _ensure_symbol_ready(symbol)
        if err:
            return {"error": err}

        try:
            # Use server tick time to avoid local/server time drift; normalize to UTC
            _tick = mt5.symbol_info_tick(symbol)
            if _tick is not None and getattr(_tick, "time", None):
                t_utc = _mt5_epoch_to_utc(float(_tick.time))
                server_now_dt = datetime.utcfromtimestamp(t_utc)
                server_now_ts = t_utc
            else:
                server_now_dt = datetime.utcnow()
                server_now_ts = server_now_dt.timestamp()
            # Fetch last few bars up to server time and select last closed
            rates = _mt5_copy_rates_from(symbol, mt5_tf, server_now_dt, 5)
        finally:
            # Restore original visibility if we changed it
            if _was_visible is False:
                try:
                    mt5.symbol_select(symbol, False)
                except Exception:
                    pass

        if rates is None or len(rates) == 0:
            return {"error": f"Failed to get rates for {symbol}: {mt5.last_error()}"}

        # Identify last closed bar robustly:
        # - If we have at least 2 bars, use the second-to-last (last closed),
        #   since the last element is typically the forming bar.
        # - If only 1 bar, verify it's closed via time-based check.
        now_ts = server_now_ts
        if len(rates) >= 2:
            src = rates[-2]
        else:
            only = rates[-1]
            if (float(only["time"]) + tf_secs) <= now_ts:
                src = only
            else:
                return {"error": "No completed bars available to compute pivot points"}

        # Access fields robustly for both dicts and NumPy structured rows
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
        period_start = float(src["time"]) if _has_field(src, "time") else float("nan")
        period_start = _mt5_epoch_to_utc(period_start)
        period_end = period_start + float(tf_secs)

        # Round levels to symbol precision if available
        digits = int(getattr(_info_before, "digits", 0) or 0)
        def _round(v: float) -> float:
            try:
                return round(float(v), digits) if digits >= 0 else float(v)
            except Exception:
                return float(v)

        levels: Dict[str, float] = {}
        pp_val: Optional[float] = None

        if method_l == "classic":
            PP = (H + L + C) / 3.0
            R1 = 2 * PP - L
            S1 = 2 * PP - H
            R2 = PP + (H - L)
            S2 = PP - (H - L)
            # Use the common R3/S3 variant
            R3 = H + 2 * (PP - L)
            S3 = L - 2 * (H - PP)
            pp_val = PP
            levels = {
                "PP": _round(PP),
                "R1": _round(R1), "S1": _round(S1),
                "R2": _round(R2), "S2": _round(S2),
                "R3": _round(R3), "S3": _round(S3),
            }
        elif method_l == "fibonacci":
            PP = (H + L + C) / 3.0
            rng = (H - L)
            R1 = PP + 0.382 * rng
            R2 = PP + 0.618 * rng
            R3 = PP + 1.000 * rng
            S1 = PP - 0.382 * rng
            S2 = PP - 0.618 * rng
            S3 = PP - 1.000 * rng
            pp_val = PP
            levels = {
                "PP": _round(PP),
                "R1": _round(R1), "S1": _round(S1),
                "R2": _round(R2), "S2": _round(S2),
                "R3": _round(R3), "S3": _round(S3),
            }
        elif method_l == "camarilla":
            rng = (H - L)
            k = 1.1
            R1 = C + (k * rng) / 12.0
            R2 = C + (k * rng) / 6.0
            R3 = C + (k * rng) / 4.0
            R4 = C + (k * rng) / 2.0
            S1 = C - (k * rng) / 12.0
            S2 = C - (k * rng) / 6.0
            S3 = C - (k * rng) / 4.0
            S4 = C - (k * rng) / 2.0
            pp_val = (H + L + C) / 3.0
            levels = {
                "PP": _round(pp_val),
                "R1": _round(R1), "S1": _round(S1),
                "R2": _round(R2), "S2": _round(S2),
                "R3": _round(R3), "S3": _round(S3),
                "R4": _round(R4), "S4": _round(S4),
            }
        elif method_l == "woodie":
            PP = (H + L + 2 * C) / 4.0
            R1 = 2 * PP - L
            S1 = 2 * PP - H
            R2 = PP + (H - L)
            S2 = PP - (H - L)
            pp_val = PP
            levels = {
                "PP": _round(PP),
                "R1": _round(R1), "S1": _round(S1),
                "R2": _round(R2), "S2": _round(S2),
            }
        elif method_l == "demark":
            # DeMark uses open/close relationship to form X
            # If we can't fetch open, approximate using the bar's 'open' if present
            O = float(src["open"]) if _has_field(src, "open") else C
            if C < O:
                X = H + 2 * L + C
            elif C > O:
                X = 2 * H + L + C
            else:
                X = H + L + 2 * C
            PP = X / 4.0
            R1 = X / 2.0 - L
            S1 = X / 2.0 - H
            pp_val = PP
            levels = {
                "PP": _round(PP),
                "R1": _round(R1), "S1": _round(S1),
            }

        # Format times per display preference
        _use_ctz = _use_client_tz_util(timezone)
        start_str = _format_time_minimal_local_util(period_start) if _use_ctz else _format_time_minimal_util(period_start)
        end_str = _format_time_minimal_local_util(period_end) if _use_ctz else _format_time_minimal_util(period_end)

        payload: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "method": method_l,
            "timeframe": timeframe,
            "period": {
                "start": start_str,
                "end": end_str,
            },
            "source": {
                "high": _round(H),
                "low": _round(L),
                "close": _round(C),
                "range": _round(H - L),
                "pivot_basis": _round(pp_val) if pp_val is not None else None,
            },
            "levels": levels,
        }
        if not _use_ctz:
            payload["timezone"] = "UTC"
        return payload
    except Exception as e:
        return {"error": f"Error computing pivot points: {str(e)}"}
