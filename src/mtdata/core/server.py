#!/usr/bin/env python3
import logging
import atexit
import functools
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List, Tuple, Literal
from typing_extensions import TypedDict
import io
import csv
import time
import re
import inspect
import pydoc
import pandas as pd
import numpy as np
import math
try:
    import pywt as _pywt  # type: ignore
except Exception:
    _pywt = None  # optional
try:
    from PyEMD import EMD as _EMD, EEMD as _EEMD, CEEMDAN as _CEEMDAN  # type: ignore
except Exception:
    _EMD = _EEMD = _CEEMDAN = None  # optional
import warnings
import pandas_ta as pta
import dateparser
import MetaTrader5 as mt5
from mcp.server.fastmcp import FastMCP
from .config import mt5_config
from .constants import (
    SERVICE_NAME,
    GROUP_SEARCH_THRESHOLD,
    TICKS_LOOKBACK_DAYS,
    DATA_READY_TIMEOUT,
    DATA_POLL_INTERVAL,
    FETCH_RETRY_ATTEMPTS,
    FETCH_RETRY_DELAY,
    SANITY_BARS_TOLERANCE,
    TI_NAN_RETRY_ATTEMPTS,
    TI_NAN_WARMUP_FACTOR,
    TI_NAN_WARMUP_MIN_ADD,
    PRECISION_REL_TOL,
    PRECISION_ABS_TOL,
    PRECISION_MAX_DECIMALS,
    SIMPLIFY_DEFAULT_RATIO,
    SIMPLIFY_DEFAULT_MIN_POINTS,
    SIMPLIFY_DEFAULT_MAX_POINTS,
    TIMEFRAME_MAP,
    TIMEFRAME_SECONDS,
)
mcp = FastMCP(SERVICE_NAME)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import split-out helpers to keep server.py thin
from ..utils.utils import (
    _csv_from_rows as _csv_from_rows_util,
    _format_time_minimal as _format_time_minimal_util,
    _format_time_minimal_local as _format_time_minimal_local_util,
    _use_client_tz as _use_client_tz_util,
    _resolve_client_tz as _resolve_client_tz_util,
    _time_format_from_epochs as _time_format_from_epochs_util,
    _maybe_strip_year as _maybe_strip_year_util,
    _style_time_format as _style_time_format_util,
    _optimal_decimals as _optimal_decimals_util,
    _format_float as _format_float_util,
    _format_numeric_rows_from_df as _format_numeric_rows_from_df_util,
    _parse_start_datetime as _parse_start_datetime_util,
)
from ..utils.simplify import _simplify_dataframe_rows as _simplify_dataframe_rows_util
from ..utils.indicators import (
    _list_ta_indicators as _list_ta_indicators_util,
    _parse_ti_specs as _parse_ti_specs_util,
    _apply_ta_indicators as _apply_ta_indicators_util,
    _estimate_warmup_bars as _estimate_warmup_bars_util,
)
from ..utils.denoise import _apply_denoise as _apply_denoise_util
from ..utils.symbol import _extract_group_path as _extract_group_path_util

_TIMEFRAME_CHOICES = tuple(sorted(TIMEFRAME_MAP.keys()))
TimeframeLiteral = Literal[_TIMEFRAME_CHOICES]  # type: ignore

# Build a Literal for single OHLCV letters; the parameter will be a list of these
OhlcvCharLiteral = Literal['O', 'H', 'L', 'C', 'V']  # type: ignore

from ..utils.mt5 import (
    _mt5_epoch_to_utc,
    _to_server_naive_dt,
    _normalize_times_in_struct,
    _mt5_copy_rates_from,
    _mt5_copy_rates_range,
    _mt5_copy_ticks_from,
    _mt5_copy_ticks_range,
    MT5Connection,
    mt5_connection,
    _auto_connect_wrapper,
)

atexit.register(mt5_connection.disconnect)


# Flexible datetime parsing helper using dateparser
def _parse_start_datetime(value: str) -> Optional[datetime]:
    """Thin wrapper delegating to utils._parse_start_datetime"""
    return _parse_start_datetime_util(value)


# Helpers
def _ensure_symbol_ready(symbol: str) -> Optional[str]:
    """Ensure a symbol is selected and tick info is available. Returns error string or None."""
    info_before = mt5.symbol_info(symbol)
    was_visible = bool(info_before.visible) if info_before is not None else None
    if not mt5.symbol_select(symbol, True):
        return f"Failed to select symbol {symbol}: {mt5.last_error()}"
    # If we just made it visible, wait for fresh tick data to arrive (poll up to timeout)
    if was_visible is False:
        deadline = time.time() + DATA_READY_TIMEOUT
        while time.time() < deadline:
            tick = mt5.symbol_info_tick(symbol)
            if tick and (getattr(tick, 'time', 0) or getattr(tick, 'bid', 0) or getattr(tick, 'ask', 0)):
                break
            time.sleep(DATA_POLL_INTERVAL)
    # Final check
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return f"Failed to refresh {symbol} data: {mt5.last_error()}"
    return None
def _csv_from_rows(headers: List[str], rows: List[List[Any]]) -> Dict[str, str]:
    return _csv_from_rows_util(headers, rows)

def _format_time_minimal(epoch_seconds: float) -> str:
    return _format_time_minimal_util(epoch_seconds)

def _format_time_minimal_local(epoch_seconds: float) -> str:
    return _format_time_minimal_local_util(epoch_seconds)

def _use_client_tz(client_tz_param: object) -> bool:
    return _use_client_tz_util(client_tz_param)

def _resolve_client_tz(client_tz_param: object):
    return _resolve_client_tz_util(client_tz_param)

def _time_format_from_epochs(epochs: List[float]) -> str:
    return _time_format_from_epochs_util(epochs)

def _maybe_strip_year(fmt: str, epochs: List[float]) -> str:
    return _maybe_strip_year_util(fmt, epochs)

def _style_time_format(fmt: str) -> str:
    return _style_time_format_util(fmt)

def _time_format_from_epochs_local(epochs: List[float]) -> str:
    """Choose a consistent local-time format for a series of UTC epoch timestamps.

    Uses CLIENT_TZ if configured; otherwise system local timezone. Mirrors _time_format_from_epochs
    but bases the decision on local-time components.
    """
    try:
        tz = mt5_config.get_client_tz()
    except Exception:
        tz = None
    any_sec = False
    any_min = False
    any_hour = False
    for e in epochs:
        try:
            dt = datetime.fromtimestamp(e, tz=timezone.utc).astimezone(tz) if tz else datetime.fromtimestamp(e).astimezone()
        except Exception:
            # fallback to UTC
            dt = datetime.utcfromtimestamp(e)
        if dt.second != 0:
            any_sec = True
            break
        if dt.minute != 0:
            any_min = True
        if dt.hour != 0:
            any_hour = True
    if any_sec:
        return "%Y-%m-%d %H:%M:%S"
    if any_min:
        return "%Y-%m-%d %H:%M"
    if any_hour:
        return "%Y-%m-%d %H"
    return "%Y-%m-%d"

def _maybe_strip_year_local(fmt: str, epochs: List[float]) -> str:
    """If all timestamps are in the same year in client/local tz, remove the year from the format."""
    try:
        tz = mt5_config.get_client_tz()
    except Exception:
        tz = None
    try:
        years = set()
        for e in epochs:
            try:
                dt = datetime.fromtimestamp(e, tz=timezone.utc).astimezone(tz) if tz else datetime.fromtimestamp(e).astimezone()
            except Exception:
                dt = datetime.utcfromtimestamp(e)
            years.add(dt.year)
        if len(years) == 1 and fmt.startswith("%Y-"):
            return fmt[3:]
    except Exception:
        pass
    return fmt

def _extract_group_path(sym) -> str:
    return _extract_group_path_util(sym)


# ---- Numeric formatting helpers ----
def _optimal_decimals(values: List[float], rel_tol: float = PRECISION_REL_TOL, abs_tol: float = PRECISION_ABS_TOL,
                      max_decimals: int = PRECISION_MAX_DECIMALS) -> int:
    return _optimal_decimals_util(values, rel_tol, abs_tol, max_decimals)


def _format_float(v: float, d: int) -> str:
    return _format_float_util(v, d)


def _format_numeric_rows_from_df(df: pd.DataFrame, headers: List[str]) -> List[List[str]]:
    return _format_numeric_rows_from_df_util(df, headers)


# ---- Timeseries simplification helpers ----
def _lttb_select_indices(x: List[float], y: List[float], n_out: int) -> List[int]:
    """Largest-Triangle-Three-Buckets (LTTB) downsampling.

    Returns the indices of selected points (monotonic, includes first/last).
    """
    try:
        m = len(x)
        if n_out >= m or n_out <= 2 or m <= 2:
            return list(range(m))
        # Ensure monotonic x
        # Assumes x are sorted; if not, sort and map back
        # For our usage, x is time so already sorted
        idxs: List[int] = [0]
        bucket_size = (m - 2) / float(n_out - 2)
        a = 0  # previously selected index
        for i in range(0, n_out - 2):
            # Current bucket range
            start = int(math.floor(1 + i * bucket_size))
            end = int(math.floor(1 + (i + 1) * bucket_size))
            end = min(end, m - 1)
            if start >= end:
                start = max(min(start, m - 2), 1)
                end = start + 1

            # Next bucket average (for triangle base point)
            next_start = int(math.floor(1 + (i + 1) * bucket_size))
            next_end = int(math.floor(1 + (i + 2) * bucket_size))
            next_end = min(next_end, m - 1)
            if next_start >= next_end:
                next_start = max(min(next_start, m - 1), 1)
                next_end = next_start
            avg_x = 0.0
            avg_y = 0.0
            cnt = max(1, next_end - next_start)
            for j in range(next_start, next_end):
                avg_x += x[j]
                avg_y += y[j]
            avg_x /= float(cnt)
            avg_y /= float(cnt)

            ax = x[a]
            ay = y[a]
            # Pick point in [start, end) maximizing triangle area with A and avg
            max_area = -1.0
            max_idx = start
            for j in range(start, end):
                area = abs((ax - avg_x) * (y[j] - ay) - (ax - x[j]) * (avg_y - ay))
                if area > max_area:
                    max_area = area
                    max_idx = j
            idxs.append(max_idx)
            a = max_idx

        idxs.append(m - 1)
        # Ensure strictly increasing and unique
        out = []
        last = -1
        for ix in sorted(set(idxs)):
            if ix > last:
                out.append(ix)
                last = ix
        return out
    except Exception:
        # Fallback: no simplification
        return list(range(len(x)))


def _default_target_points(total: int) -> int:
    """Default target points when simplify requested without explicit points/ratio.

    Uses SIMPLIFY_DEFAULT_RATIO bounded by [SIMPLIFY_DEFAULT_MIN_POINTS, SIMPLIFY_DEFAULT_MAX_POINTS].
    """
    try:
        t = int(round(total * SIMPLIFY_DEFAULT_RATIO))
        t = max(SIMPLIFY_DEFAULT_MIN_POINTS, min(SIMPLIFY_DEFAULT_MAX_POINTS, t))
        return max(3, min(t, total))
    except Exception:
        return max(3, min(SIMPLIFY_DEFAULT_MIN_POINTS, total))


def _choose_simplify_points(total: int, spec: Dict[str, Any]) -> int:
    """Determine target number of points from a simplify spec.

    Supports keys: 'points', 'max_points', 'target_points', or 'ratio' (0..1).
    Enforces bounds [3, total]. Returns total if no effective reduction requested.
    """
    try:
        if not spec:
            return total
        n = None
        for k in ("points", "max_points", "target_points"):
            if k in spec and spec[k] is not None:
                try:
                    n = int(spec[k])
                    break
                except Exception:
                    pass
        if n is None and "ratio" in spec and spec["ratio"] is not None:
            try:
                r = float(spec["ratio"])
                if r > 0 and r < 1:
                    n = int(max(3, round(total * r)))
            except Exception:
                pass
        if n is None:
            # If method specified or spec present, use default target
            if spec and ("method" in spec or len(spec) > 0):
                return _default_target_points(total)
            return total
        n = max(3, min(int(n), total))
        return n
    except Exception:
        return total


def _point_line_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Vertical distance from P to line y(x) through (x1,y1)-(x2,y2).

    For time series, x is monotonic (time), so using vertical deviation is sensible
    and keeps epsilon/max_error in the same units as y.
    """
    try:
        dx = x2 - x1
        if dx == 0.0:
            # Degenerate in time; fallback to distance from y to midpoint
            return abs(py - (y1 + y2) / 2.0)
        m = (y2 - y1) / dx
        y_on_line = y1 + m * (px - x1)
        return abs(py - y_on_line)
    except Exception:
        return 0.0


def _rdp_select_indices(x: List[float], y: List[float], epsilon: float) -> List[int]:
    """Douglas–Peucker simplification returning kept indices (always includes first/last)."""
    try:
        n = len(x)
        if n <= 2 or epsilon <= 0:
            return list(range(n))
        stack: List[Tuple[int, int]] = [(0, n - 1)]
        keep = [False] * n
        keep[0] = True
        keep[n - 1] = True
        while stack:
            i0, i1 = stack.pop()
            x0, y0 = x[i0], y[i0]
            x1, y1 = x[i1], y[i1]
            idx = -1
            dmax = 0.0
            for i in range(i0 + 1, i1):
                d = _point_line_distance(x[i], y[i], x0, y0, x1, y1)
                if d > dmax:
                    idx = i
                    dmax = d
            if idx != -1 and dmax > epsilon:
                keep[idx] = True
                stack.append((i0, idx))
                stack.append((idx, i1))
        return [i for i, k in enumerate(keep) if k]
    except Exception:
        return list(range(len(x)))


def _max_line_error(x: List[float], y: List[float], i0: int, i1: int) -> float:
    """Maximum perpendicular distance of points i0..i1 to the line i0-i1."""
    if i1 <= i0 + 1:
        return 0.0
    x0, y0 = x[i0], y[i0]
    x1, y1 = x[i1], y[i1]
    m = 0.0
    for i in range(i0 + 1, i1):
        d = _point_line_distance(x[i], y[i], x0, y0, x1, y1)
        if d > m:
            m = d
    return m


def _pla_select_indices(x: List[float], y: List[float], max_error: Optional[float] = None, segments: Optional[int] = None, points: Optional[int] = None) -> List[int]:
    """Piecewise Linear Approximation; greedy by max perpendicular error or uniform segments.

    Returns indices of breakpoints (includes first/last).
    """
    n = len(x)
    if n <= 2:
        return list(range(n))
    # Uniform segmentation if requested and no error bound
    if (max_error is None or max_error <= 0) and (segments or points):
        if points and (segments is None):
            segments = max(1, int(points) - 1)
        segments = max(1, min(int(segments or 1), n - 1))
        idxs = [0]
        for k in range(1, segments):
            idxs.append(int(round(k * (n - 1) / segments)))
        idxs.append(n - 1)
        return sorted(set(idxs))
    # Greedy by error
    if max_error is None or max_error <= 0:
        # Fallback to keep all
        return list(range(n))
    idxs = [0]
    start = 0
    while start < n - 1:
        end = start + 1
        last_good = end
        while end < n:
            err = _max_line_error(x, y, start, end)
            if err <= max_error:
                last_good = end
                end += 1
            else:
                break
        if last_good == start:
            last_good = start + 1
        idxs.append(last_good)
        start = last_good
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    # Deduplicate increasing
    out = []
    for i in idxs:
        if not out or i > out[-1]:
            out.append(i)
    return out


def _apca_select_indices(y: List[float], max_error: Optional[float] = None, segments: Optional[int] = None, points: Optional[int] = None) -> List[int]:
    """Adaptive Piecewise Constant Approximation; greedy by max absolute deviation or uniform.

    Returns indices of breakpoints (includes first/last).
    """
    n = len(y)
    if n <= 2:
        return list(range(n))
    if (max_error is None or max_error <= 0) and (segments or points):
        if points and (segments is None):
            segments = max(1, int(points) - 1)
        segments = max(1, min(int(segments or 1), n - 1))
        idxs = [0]
        for k in range(1, segments):
            idxs.append(int(round(k * (n - 1) / segments)))
        idxs.append(n - 1)
        return sorted(set(idxs))
    if max_error is None or max_error <= 0:
        return list(range(n))
    idxs = [0]
    start = 0
    while start < n - 1:
        end = start + 1
        last_good = end
        while end < n:
            seg = y[start:end + 1]
            mean = sum(seg) / float(len(seg))
            err = max(abs(v - mean) for v in seg)
            if err <= max_error:
                last_good = end
                end += 1
            else:
                break
        if last_good == start:
            last_good = start + 1
        idxs.append(last_good)
        start = last_good
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    out = []
    for i in idxs:
        if not out or i > out[-1]:
            out.append(i)
    return out


def _select_indices_for_timeseries(x: List[float], y: List[float], spec: Optional[Dict[str, Any]]) -> Tuple[List[int], str, Dict[str, Any]]:
    """Select representative indices according to simplify spec.

    Returns (indices, method_used, params_meta).
    """
    meta: Dict[str, Any] = {}
    if not spec:
        return list(range(len(x))), "none", meta
    method = str(spec.get("method", "lttb")).lower().strip()
    if method in ("lttb", "default"):
        n_out = _choose_simplify_points(len(x), spec)
        idxs = _lttb_select_indices(x, y, n_out)
        meta.update({"points": n_out})
        return idxs, "lttb", meta
    if method == "rdp":
        eps = spec.get("epsilon", None)
        if eps is None:
            eps = spec.get("tolerance", None)
        if eps is None:
            eps = spec.get("eps", None)
        try:
            eps_val = float(eps) if eps is not None else None
        except Exception:
            eps_val = None
        # If epsilon provided, use it directly
        if eps_val is not None and eps_val > 0:
            idxs = _rdp_select_indices(x, y, eps_val)
            meta.update({"epsilon": eps_val})
            return idxs, "rdp", meta
        # Else, auto-tune epsilon to hit target points
        target = spec.get("points") or spec.get("target_points") or spec.get("max_points") or None
        if target is None and spec.get("ratio") is not None:
            try:
                r = float(spec.get("ratio"))
                if r > 0 and r < 1:
                    target = max(3, int(round(len(x) * r)))
            except Exception:
                pass
        # Provide default target when none supplied
        if target is None:
            target = _default_target_points(len(x))
        try:
            target_n = int(target) if target is not None else None
        except Exception:
            target_n = None
        if target_n is not None and target_n < len(x):
            idxs, eps_used = _rdp_autotune_epsilon(x, y, target_n)
            meta.update({"epsilon": eps_used, "points": len(idxs), "auto_tuned": True})
            return idxs, "rdp", meta
        # Fallback to LTTB if neither epsilon nor target provided
        n_out = _choose_simplify_points(len(x), spec)
        idxs = _lttb_select_indices(x, y, n_out)
        meta.update({"points": n_out, "fallback": "rdp->lttb"})
        return idxs, "lttb", meta
    if method == "pla":
        # Prefer error bound; else segments/points uniform
        max_error = spec.get("max_error", None)
        try:
            me = float(max_error) if max_error is not None else None
        except Exception:
            me = None
        segments = spec.get("segments", None)
        points = spec.get("points", None) or spec.get("target_points", None) or spec.get("max_points", None)
        # If max_error provided, use it
        if me is not None and me > 0:
            idxs = _pla_select_indices(x, y, me, None, None)
            meta.update({"max_error": me})
            return idxs, "pla", meta
        # If segments specified, do uniform segmentation
        if segments is not None:
            idxs = _pla_select_indices(x, y, None, segments, None)
            meta.update({"segments": segments})
            return idxs, "pla", meta
        # If points specified, auto-tune max_error to hit points
        try:
            p = int(points) if points is not None else None
        except Exception:
            p = None
        if p is None:
            p = _default_target_points(len(x))
        if p is not None and p < len(x):
            idxs, me_used = _pla_autotune_max_error(x, y, p)
            meta.update({"max_error": me_used, "points": len(idxs), "auto_tuned": True})
            return idxs, "pla", meta
        # Fallback to LTTB
        n_out = _choose_simplify_points(len(x), spec)
        idxs = _lttb_select_indices(x, y, n_out)
        meta.update({"points": n_out, "fallback": "pla->lttb"})
        return idxs, "lttb", meta
    if method == "apca":
        max_error = spec.get("max_error", None)
        try:
            me = float(max_error) if max_error is not None else None
        except Exception:
            me = None
        segments = spec.get("segments", None)
        points = spec.get("points", None) or spec.get("target_points", None) or spec.get("max_points", None)
        if me is not None and me > 0:
            idxs = _apca_select_indices(y, me, None, None)
            meta.update({"max_error": me})
            return idxs, "apca", meta
        if segments is not None:
            idxs = _apca_select_indices(y, None, segments, None)
            meta.update({"segments": segments})
            return idxs, "apca", meta
        try:
            p = int(points) if points is not None else None
        except Exception:
            p = None
        if p is None:
            p = _default_target_points(len(x))
        if p is not None and p < len(x):
            idxs, me_used = _apca_autotune_max_error(y, p)
            meta.update({"max_error": me_used, "points": len(idxs), "auto_tuned": True})
            return idxs, "apca", meta
        # Fallback to LTTB
        n_out = _choose_simplify_points(len(x), spec)
        idxs = _lttb_select_indices(x, y, n_out)
        meta.update({"points": n_out, "fallback": "apca->lttb"})
        return idxs, "lttb", meta
    # Unknown method -> default LTTB
    n_out = _choose_simplify_points(len(x), spec)
    idxs = _lttb_select_indices(x, y, n_out)
    meta.update({"points": n_out, "fallback": f"{method}->lttb"})
    return idxs, "lttb", meta


def _rdp_autotune_epsilon(x: List[float], y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    """Binary search epsilon so RDP keeps ~target_points. Returns (indices, epsilon)."""
    n = len(x)
    target = max(3, min(int(target_points), n))
    if target >= n:
        return list(range(n)), 0.0
    # Upper bound: max vertical residual to line between endpoints
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    dx = x1 - x0
    if dx == 0:
        base = sum(y) / float(n)
        hi = max(abs(v - base) for v in y) if n > 0 else 1.0
    else:
        m = (y1 - y0) / dx
        hi = 0.0
        for i in range(n):
            yline = y0 + m * (x[i] - x0)
            hi = max(hi, abs(y[i] - yline))
    if hi <= 0:
        rng = (max(y) - min(y)) if n > 1 else 1.0
        hi = max(1e-12, rng)
    lo = 0.0
    best_idxs = list(range(n))
    best_eps = lo
    best_diff = abs(len(best_idxs) - target)
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        idxs = _rdp_select_indices(x, y, mid)
        cnt = len(idxs)
        diff = abs(cnt - target)
        if diff < best_diff or (diff == best_diff and mid < best_eps):
            best_idxs = idxs
            best_eps = mid
            best_diff = diff
        if cnt > target:
            lo = mid if mid > lo else lo + (hi - lo) * 0.5
        elif cnt < target:
            hi = mid
        else:
            break
        if hi - lo <= 1e-12:
            break
    return best_idxs, float(best_eps)


def _pla_autotune_max_error(x: List[float], y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    """Binary search max_error so PLA keeps ~target_points. Returns (indices, max_error)."""
    n = len(x)
    target = max(3, min(int(target_points), n))
    if target >= n:
        return list(range(n)), 0.0
    # Upper bound: max vertical residual to endpoints line (single segment acceptance)
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    dx = x1 - x0
    if dx == 0:
        base = sum(y) / float(n)
        hi = max(abs(v - base) for v in y) if n > 0 else 1.0
    else:
        m = (y1 - y0) / dx
        hi = 0.0
        for i in range(n):
            yline = y0 + m * (x[i] - x0)
            hi = max(hi, abs(y[i] - yline))
    if hi <= 0:
        rng = (max(y) - min(y)) if n > 1 else 1.0
        hi = max(1e-12, rng)
    lo = 0.0
    best_idxs = list(range(n))
    best_me = lo
    best_diff = abs(len(best_idxs) - target)
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        idxs = _pla_select_indices(x, y, mid, None, None)
        cnt = len(idxs)
        diff = abs(cnt - target)
        if diff < best_diff or (diff == best_diff and mid < best_me):
            best_idxs = idxs
            best_me = mid
            best_diff = diff
        if cnt > target:
            lo = mid if mid > lo else lo + (hi - lo) * 0.5
        elif cnt < target:
            hi = mid
        else:
            break
        if hi - lo <= 1e-12:
            break
    return best_idxs, float(best_me)


def _apca_autotune_max_error(y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    """Binary search max_error so APCA keeps ~target_points. Returns (indices, max_error)."""
    n = len(y)
    target = max(3, min(int(target_points), n))
    if target >= n:
        return list(range(n)), 0.0
    base = sum(y) / float(n)
    hi = max(abs(v - base) for v in y) if n > 0 else 1.0
    if hi <= 0:
        rng = (max(y) - min(y)) if n > 1 else 1.0
        hi = max(1e-12, rng)
    lo = 0.0
    best_idxs = list(range(n))
    best_me = lo
    best_diff = abs(len(best_idxs) - target)
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        idxs = _apca_select_indices(y, mid, None, None)
        cnt = len(idxs)
        diff = abs(cnt - target)
        if diff < best_diff or (diff == best_diff and mid < best_me):
            best_idxs = idxs
            best_me = mid
            best_diff = diff
        if cnt > target:
            lo = mid if mid > lo else lo + (hi - lo) * 0.5
        elif cnt < target:
            hi = mid
        else:
            break
        if hi - lo <= 1e-12:
            break
    return best_idxs, float(best_me)


def _simplify_dataframe_rows(df: pd.DataFrame, headers: List[str], simplify: Optional[Dict[str, Any]]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Apply row-wise downsampling to reduce output size while keeping readability.

    Always represents all numeric columns: union indices selected per column, then refine
    to the requested target size using a composite metric across columns.

    Returns (possibly reduced) DataFrame and a meta dict describing the reduction.
    """
    if not simplify:
        return df, None
    try:
        total = len(df)
        if total <= 3:
            return df, None
        method = str(simplify.get("method", "lttb")).lower().strip()
        # Determine number of points
        n_out = _choose_simplify_points(total, simplify)
        if n_out >= total:
            return df, None
        # Build candidate numeric columns (from headers first)
        def _numeric_columns_from_headers() -> List[str]:
            cols: List[str] = []
            for h in headers:
                if h in ('time',) or h.startswith('_'):
                    continue
                try:
                    if h in df.columns and pd.api.types.is_numeric_dtype(df[h]):
                        cols.append(h)
                except Exception:
                    continue
            if not cols:
                for c in df.columns:
                    if c in ('time',) or str(c).startswith('_'):
                        continue
                    try:
                        if pd.api.types.is_numeric_dtype(df[c]):
                            cols.append(c)
                    except Exception:
                        continue
            return cols

        x = [float(v) for v in df['__epoch'].tolist()]
        # Always multi-column selection
        cols = _numeric_columns_from_headers()
        if not cols:
            return df, None
        # Allocate points roughly evenly across columns (keeping first/last overall)
        k = max(1, len(cols))
        base_points = max(3, int(round(n_out / k)))
        idx_set: set = set([0, total - 1])
        method_used_overall = None
        params_meta_overall: Dict[str, Any] = {}
        for c in cols:
            y = [float(v) if v is not None and not pd.isna(v) else 0.0 for v in df[c].tolist()]
            sub_spec = dict(simplify)
            sub_spec['points'] = base_points
            idxs, method_used, params_meta = _select_indices_for_timeseries(x, y, sub_spec)
            for i in idxs:
                if 0 <= int(i) < total:
                    idx_set.add(int(i))
            method_used_overall = method_used
            try:
                if params_meta:
                    for k2, v2 in params_meta.items():
                        params_meta_overall.setdefault(k2, v2)
            except Exception:
                pass
        idxs_union = sorted(idx_set)
        # Build composite normalized series for refinement/top-up
        mins: Dict[str, float] = {}
        ranges: Dict[str, float] = {}
        for c in cols:
            try:
                series = [float(v) if v is not None and not pd.isna(v) else 0.0 for v in df[c].tolist()]
                mn = min(series)
                mx = max(series)
                ranges[c] = max(1e-12, mx - mn)
                mins[c] = mn
            except Exception:
                ranges[c] = 1.0
                mins[c] = 0.0
        comp: List[float] = []
        for i in range(total):
            s = 0.0
            for c in cols:
                try:
                    vv = (float(df.iloc[i][c]) - mins[c]) / ranges[c]
                except Exception:
                    vv = 0.0
                s += abs(vv)
            comp.append(s)
        if len(idxs_union) > n_out:
            refined = _lttb_select_indices(x, comp, n_out)
            idxs_final = sorted(set(refined))
        elif len(idxs_union) < n_out:
            refined = _lttb_select_indices(x, comp, n_out)
            merged = sorted(set(idxs_union).union(refined))
            if len(merged) > n_out:
                keep = set([0, total - 1])
                candidates = [(comp[i], i) for i in merged if i not in keep]
                candidates.sort(reverse=True)
                for _, i in candidates:
                    keep.add(i)
                    if len(keep) >= n_out:
                        break
                idxs_final = sorted(keep)
            else:
                idxs_final = merged
        else:
            idxs_final = idxs_union

        idxs_final = [i for i in idxs_final if 0 <= i < total]
        if len(idxs_final) >= total:
            return df, None
        reduced = df.iloc[idxs_final].copy()
        meta = {
            "method": (method_used_overall or str(simplify.get("method", "lttb")).lower()),
            "columns": cols,
            "multi_column": True,
            "original_rows": total,
            "returned_rows": len(reduced),
        }
        try:
            if params_meta_overall:
                meta.update(params_meta_overall)
        except Exception:
            pass
        return reduced, meta
    except Exception:
        return df, None


def _to_float_or_nan(x: Any) -> float:
    """Best-effort conversion to float; returns NaN on None or conversion failure.

    Handles scalars, sequences, numpy arrays, and pandas Series gracefully.
    """
    try:
        if x is None:
            return float('nan')
        # If already a float/int
        if isinstance(x, (float, int)):
            return float(x)
        # If numpy scalar
        try:
            import numpy as _np  # local import to avoid circulars
            if isinstance(x, _np.generic):
                return float(x)
        except Exception:
            pass
        # If sequence/array/Series, take first element
        try:
            if hasattr(x, '__len__') and len(x) > 0:
                # pandas Series/DataFrame .iloc if present
                if hasattr(x, 'iloc'):
                    try:
                        return float(x.iloc[0])
                    except Exception:
                        pass
                return float(list(x)[0])
        except Exception:
            pass
        # Last resort
        return float(x)
    except Exception:
        return float('nan')


# ---- Technical Indicators (dynamic discovery and application) ----
def _list_ta_indicators() -> List[Dict[str, Any]]:
    """Dynamically list TA indicators available via pandas_ta.

    Returns a list of dicts with: name, params (name,type,default), description.
    """
    # Create a minimal DataFrame to get the .ta accessor
    tmp = pd.DataFrame({
        'open': [1], 'high': [1], 'low': [1], 'close': [1], 'volume': [1]
    })
    ind_list: List[Dict[str, Any]] = []
    seen = set()
    for attr in dir(tmp.ta):
        if attr.startswith('_'):
            continue
        func = getattr(tmp.ta, attr, None)
        if not callable(func):
            continue
        name = attr.lower()
        if name in seen:
            continue
        seen.add(name)
        # Prefer original pandas_ta function (for better docs) if available
        lib_func = getattr(pta, name, None)
        target_for_sig = lib_func if callable(lib_func) else func
        try:
            sig = inspect.signature(target_for_sig)
        except (TypeError, ValueError):
            continue
        # Collect parameters excluding self and implicit OHLCV columns
        params = []
        for p in sig.parameters.values():
            if p.name in {"self", "open", "high", "low", "close", "volume"}:
                continue
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            entry = {"name": p.name}
            if p.default is not inspect._empty and p.default is not None:
                entry["default"] = p.default
            params.append(entry)
        # Full help text: prefer library function docs if present
        try:
            if callable(lib_func):
                raw = pydoc.render_doc(lib_func)
                desc = _clean_help_text(raw, func_name=name, func=lib_func)
            else:
                raw = pydoc.render_doc(func)
                desc = _clean_help_text(raw, func_name=name, func=func)
        except Exception:
            # Fallback to raw docstring
            desc = inspect.getdoc(lib_func or func) or ''

        # Try to infer missing defaults from the doc text
        try:
            doc_text = inspect.getdoc(lib_func or func) or raw if 'raw' in locals() else ''
            _infer_defaults_from_doc(name, doc_text, params)
        except Exception:
            pass
        # Derive category from module path (e.g., pandas_ta.momentum.rsi -> momentum)
        category = ''
        try:
            mod = (lib_func or func).__module__
            parts = mod.split('.')
            if len(parts) >= 2 and parts[0] == 'pandas_ta':
                # usually pandas_ta.<category>.<func>
                category = parts[1]
        except Exception:
            category = ''

        ind_list.append({
            "name": name,
            "params": params,
            "description": desc,
            "category": category,
        })
    # Sort by name
    ind_list.sort(key=lambda x: x["name"])
    return ind_list


def _infer_defaults_from_doc(func_name: str, doc_text: str, params: List[Dict[str, Any]]):
    """Infer parameter defaults from doc_text when signature uses None but docs specify defaults.

    Attempts two strategies:
    1) Parse a signature-like line: func_name(param=123, other=4.5)
    2) Parse prose patterns: 'param ... Default: 20' or 'param=20' in descriptions
    """
    if not doc_text:
        return
    text = doc_text
    # Remove overstrikes just in case
    text = re.sub(r'.\x08', '', text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Strategy 1: look for a signature line containing func_name(
    sig_line = None
    for ln in lines:
        if ln.startswith(func_name + '(') or re.match(rf"^\s*{re.escape(func_name)}\s*\(.*\)", ln):
            sig_line = ln
            break
    if sig_line:
        inside = sig_line[sig_line.find('(') + 1 : sig_line.rfind(')')] if '(' in sig_line and ')' in sig_line else ''
        for part in re.split(r'[\s,]+', inside):
            if '=' in part:
                k, v = part.split('=', 1)
                k = k.strip()
                v = v.strip().strip(',)')
                num = _try_number(v)
                if num is not None:
                    for p in params:
                        if p.get('name') == k and 'default' not in p:
                            p['default'] = num
    # Strategy 2: prose patterns like 'length ... Default: 20'
    for p in params:
        if 'default' in p:
            continue
        k = p.get('name')
        if not k:
            continue
        m = re.search(rf"{re.escape(k)}[^\n]*?(?:Default|default)\s*:?[\s]*([0-9]+(?:\.[0-9]+)?)", text)
        if m:
            p['default'] = _try_number(m.group(1))


def _try_number(s: str):
    try:
        if '.' in s:
            return float(s)
        return int(s)
    except Exception:
        return None


def _clean_help_text(text: str, func_name: Optional[str] = None, func: Optional[Any] = None) -> str:
    """Return full pydoc help text (cleaned), starting at the function signature.

    - Always uses the rendered pydoc text provided as `text`
    - Removes overstrike sequences
    - Drops everything before the first signature line
    - Cleans trailing "method of ... instance" blurb on the signature line and direct next line
    """
    if not isinstance(text, str):
        return ''
    cleaned = re.sub(r'.\x08', '', text)
    lines = [ln.rstrip() for ln in cleaned.splitlines()]
    # Find signature line
    sig_re = re.compile(rf"^\s*{re.escape(func_name)}\s*\(.*\)") if func_name else re.compile(r"^\s*\w+\s*\(.*\)")
    start = 0
    for i, ln in enumerate(lines):
        if sig_re.match(ln):
            start = i
            break
    kept = lines[start:]
    if kept:
        kept[0] = re.sub(r"\s+method of.*", "", kept[0], flags=re.IGNORECASE)
        if len(kept) > 1 and re.search(r"method of", kept[1], re.IGNORECASE):
            kept.pop(1)
    return "\n".join(kept).strip()


def _parse_ti_specs(spec: str) -> List[Tuple[str, List[Any], Dict[str, Any]]]:
    """Parse TI spec string into a list of (name, args, kwargs).

    Supported formats:
        "sma(14), ema(length=50), macd(12,26,9)"
    Returns empty list on parse error.
    """
    results: List[Tuple[str, List[Any], Dict[str, Any]]] = []
    
    # Split by commas but respect parentheses
    parts = []
    current_part = ""
    paren_depth = 0
    
    for char in spec:
        if char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1
        elif char == ',' and paren_depth == 0:
            if current_part.strip():
                parts.append(current_part.strip())
            current_part = ""
            continue
        current_part += char
    
    if current_part.strip():
        parts.append(current_part.strip())
    
    for part in parts:
        # Tolerate surrounding quotes and whitespace
        part = part.strip().strip("\"'")
        if not part:
            continue
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)(?:\((.*)\))?$", part)
        if not m:
            continue
        raw_name = m.group(1)
        inner = (m.group(2) or '').strip().strip("\"'")
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        if inner:
            for token in re.split(r"[\s,;]+", inner):
                if not token:
                    continue
                if '=' in token:
                    k, v = token.split('=', 1)
                    kwargs[k.strip()] = _coerce_scalar(v.strip())
                else:
                    args.append(_coerce_scalar(token))
        # Flex: detect trailing numeric length in the indicator name (e.g., RSI_48 or EMA21)
        base_name = raw_name
        mlen = re.match(r"^([A-Za-z]+)[_-]?(\d+)$", raw_name)
        if mlen and not inner:
            base_name = mlen.group(1)
            try:
                length_val = int(mlen.group(2))
                args = [length_val] + args
            except Exception:
                pass
        name = base_name.lower()
        results.append((name, args, kwargs))
    return results


def _coerce_scalar(val: str) -> Any:
    """Best-effort type coercion for numeric strings."""
    # Try int
    try:
        return int(val)
    except Exception:
        pass
    # Try float
    try:
        return float(val)
    except Exception:
        pass
    # Strip quotes if present
    if ((val.startswith('"') and val.endswith('"')) or
        (val.startswith("'") and val.endswith("'"))):
        return val[1:-1]
    return val


def _apply_ta_indicators(df: pd.DataFrame, ti_spec: str) -> List[str]:
    return _apply_ta_indicators_util(df, ti_spec)


def _estimate_warmup_bars(ti_spec: Optional[str]) -> int:
    return _estimate_warmup_bars_util(ti_spec)


# Build category Literal before tool registration so MCP captures it in the schema
try:
    _CATEGORY_CHOICES = sorted({it.get('category') for it in _list_ta_indicators() if it.get('category')})
except Exception:
    _CATEGORY_CHOICES = []

if _CATEGORY_CHOICES:
    # Create a Literal type alias dynamically
    CategoryLiteral = Literal[tuple(_CATEGORY_CHOICES)]  # type: ignore
else:
    CategoryLiteral = str  # fallback

# Build indicator name Literal so details endpoint has enum name choices
try:
    _INDICATOR_NAME_CHOICES = sorted({it.get('name') for it in _list_ta_indicators() if it.get('name')})
except Exception:
    _INDICATOR_NAME_CHOICES = []

if _INDICATOR_NAME_CHOICES:
    IndicatorNameLiteral = Literal[tuple(_INDICATOR_NAME_CHOICES)]  # type: ignore
else:
    IndicatorNameLiteral = str  # fallback

class IndicatorSpec(TypedDict, total=False):
    """Structured TI spec: name with optional numeric params."""
    name: IndicatorNameLiteral  # type: ignore
    params: List[float]

# ---- Denoising (spec + application) ----
# Allowed denoising methods for first phase (no extra dependencies)
_DENOISE_METHODS = (
    "none",        # no-op
    "ema",         # exponential moving average
    "sma",         # simple moving average
    "median",      # rolling median
    "lowpass_fft", # zero-phase FFT low-pass
    "wavelet",     # wavelet shrinkage (PyWavelets optional)
    "emd",         # empirical mode decomposition (PyEMD optional)
    "eemd",        # ensemble EMD (PyEMD optional)
    "ceemdan",     # complementary EEMD with adaptive noise (PyEMD optional)
)

try:
    DenoiseMethodLiteral = Literal[_DENOISE_METHODS]  # type: ignore
except Exception:
    DenoiseMethodLiteral = str  # fallback for typing

class DenoiseSpec(TypedDict, total=False):
    method: DenoiseMethodLiteral  # type: ignore
    params: Dict[str, Any]
    columns: List[str]
    when: Literal['pre_ti', 'post_ti']  # type: ignore
    causality: Literal['causal', 'zero_phase']  # type: ignore
    keep_original: bool
    suffix: str

# ---- Simple parsers / normalizers ----
def _normalize_ohlcv_arg(ohlcv: Optional[object]) -> Optional[set]:
    """Parse friendly OHLCV spec into a set of letters {O,H,L,C,V}.

    Accepts:
    - None: return None (use default columns)
    - String presets: 'close', 'price' -> C; 'ohlc' -> O,H,L,C; 'ohlcv'/'all' -> O,H,L,C,V
    - Comma/space mixed strings: 'open,high low close', 'O H L C V'
    - Compact letters: 'cl', 'ohlcv'
    - List/Tuple of strings: ['open','close'] or ['O','C']
    """
    if ohlcv is None:
        return None
    mapping = {
        'o': 'O', 'open': 'O',
        'h': 'H', 'high': 'H',
        'l': 'L', 'low': 'L',
        'c': 'C', 'close': 'C', 'price': 'C',
        'v': 'V', 'vol': 'V', 'volume': 'V', 'tickvolume': 'V', 'tick_volume': 'V',
    }
    presets = {
        'ohlc': {'O','H','L','C'},
        'ohlcv': {'O','H','L','C','V'},
        'all': {'O','H','L','C','V'},
        'close': {'C'},
        'price': {'C'},
    }
    letters: set = set()
    try:
        if isinstance(ohlcv, str):
            s = ohlcv.strip()
            key = s.lower().replace('-', '').replace('_','')
            if key in presets:
                return set(presets[key])
            # split by non-alphas
            try:
                import re as _re
                tokens = [t for t in _re.split(r"[^a-zA-Z]+", key) if t]
            except Exception:
                tokens = [key]
            if not tokens:
                tokens = [key]
            for t in tokens:
                if t in mapping:
                    letters.add(mapping[t])
                else:
                    for ch in t:
                        if ch in mapping:
                            letters.add(mapping[ch])
            return letters or None
        if isinstance(ohlcv, (list, tuple, set)):
            for it in ohlcv:
                if it is None:
                    continue
                s = str(it).strip().lower().replace('-', '').replace('_','')
                if s in presets:
                    letters |= presets[s]
                    continue
                if s in mapping:
                    letters.add(mapping[s])
                else:
                    for ch in s:
                        if ch in mapping:
                            letters.add(mapping[ch])
            return letters or None
    except Exception:
        return None
    return None

# ---- Pivot Point methods (enums) ----
_PIVOT_METHODS = (
    "classic",
    "fibonacci",
    "camarilla",
    "woodie",
    "demark",
)

try:
    PivotMethodLiteral = Literal[_PIVOT_METHODS]  # type: ignore
except Exception:
    PivotMethodLiteral = str  # fallback for typing

def _denoise_series(
    s: pd.Series,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    causality: Optional[str] = None,
) -> pd.Series:
    """Apply denoising to a single numeric Series and return the result.

    Supported methods (no external deps): ema, sma, median, lowpass_fft.
    - ema: params {span:int|None, alpha:float|None}
    - sma: params {window:int}
    - median: params {window:int}
    - lowpass_fft: params {cutoff_ratio:float in (0, 0.5], taper:bool}
    - wavelet: params {wavelet:str, level:int|None, threshold:float|"auto", mode:"soft"|"hard"}
    - emd/eemd/ceemdan: params {drop_imfs:list[int], keep_imfs:list[int], max_imfs:int, noise_strength:float, trials:int, random_state:int}
    """
    if params is None:
        params = {}
    method = (method or 'none').lower()
    if method == 'none':
        return s

    # Ensure float for numeric stability
    x = s.astype('float64')

    if method == 'ema':
        span = params.get('span')
        alpha = params.get('alpha')
        if alpha is None and span is None:
            span = 10
        # Base forward EMA
        y_fwd = x.ewm(span=span, alpha=alpha, adjust=False).mean()
        if (causality or '').lower() == 'zero_phase':
            # Two-pass EMA (forward + backward) to approximate zero-phase
            y_bwd = y_fwd.iloc[::-1].ewm(span=span, alpha=alpha, adjust=False).mean().iloc[::-1]
            return y_bwd
        return y_fwd

    if method == 'sma':
        window = int(params.get('window', 10))
        if window <= 1:
            return x
        if (causality or 'causal').lower() == 'causal':
            return x.rolling(window=window, min_periods=1).mean()
        # zero-phase via symmetric convolution with reflection padding
        k = np.ones(window, dtype=float) / float(window)
        pad = window // 2
        xpad = np.pad(x.to_numpy(), (pad, pad), mode='reflect')
        y = np.convolve(xpad, k, mode='same')[pad:-pad]
        return pd.Series(y, index=x.index)

    if method == 'median':
        window = int(params.get('window', 7))
        if window <= 1:
            return x
        center = (causality or '').lower() == 'zero_phase'
        return x.rolling(window=window, min_periods=1, center=center).median()

    if method == 'lowpass_fft':
        # Zero-phase by construction; ignore 'causal' and document behavior
        cutoff_ratio = float(params.get('cutoff_ratio', 0.1))
        cutoff_ratio = max(1e-6, min(0.5, cutoff_ratio))
        # Fill NaNs before FFT to avoid propagation
        xnp = x.fillna(method='ffill').fillna(method='bfill').fillna(0.0).to_numpy()
        n = len(xnp)
        if n <= 2:
            return x
        X = np.fft.rfft(xnp)
        freqs = np.fft.rfftfreq(n, d=1.0)  # normalized per-sample
        cutoff = cutoff_ratio * 0.5 * 2.0  # interpret ratio vs. Nyquist; clamp in [0,0.5]
        mask = freqs <= cutoff
        X_filtered = X * mask
        y = np.fft.irfft(X_filtered, n=n)
        return pd.Series(y, index=x.index)

    if method == 'wavelet':
        if _pywt is None:
            return s
        wname = str(params.get('wavelet', 'db4'))
        mode = str(params.get('mode', 'soft')).lower()
        thr = params.get('threshold', 'auto')
        # choose decomposition level if not provided
        try:
            w = _pywt.Wavelet(wname)
            max_level = _pywt.dwt_max_level(len(x), w.dec_len)
        except Exception:
            w = _pywt.Wavelet('db4') if _pywt else None
            max_level = 4
        level = int(params.get('level', max(1, min(5, max_level))))
        # Fill NaNs
        xnp = x.fillna(method='ffill').fillna(method='bfill').fillna(0.0).to_numpy()
        coeffs = _pywt.wavedec(xnp, w, mode='symmetric', level=level)
        cA, cDs = coeffs[0], coeffs[1:]
        if thr == 'auto':
            # universal threshold using first detail level
            d1 = cDs[-1] if len(cDs) > 0 else cA
            sigma = np.median(np.abs(d1 - np.median(d1))) / 0.6745 if len(d1) else 0.0
            lam = sigma * np.sqrt(2.0 * np.log(len(xnp) + 1e-9))
        else:
            try:
                lam = float(thr)
            except Exception:
                lam = 0.0
        new_coeffs = [cA]
        for d in cDs:
            if mode == 'hard':
                d_new = d * (np.abs(d) >= lam)
            else:
                d_new = np.sign(d) * np.maximum(np.abs(d) - lam, 0.0)
            new_coeffs.append(d_new)
        y = _pywt.waverec(new_coeffs, w, mode='symmetric')
        if len(y) != len(xnp):
            y = y[:len(xnp)]
        return pd.Series(y, index=x.index)

    if method in ('emd', 'eemd', 'ceemdan'):
        if _EMD is None and _EEMD is None and _CEEMDAN is None:
            return s
        xnp = x.fillna(method='ffill').fillna(method='bfill').fillna(0.0).to_numpy()
        max_imfs = params.get('max_imfs')
        if isinstance(max_imfs, str) and str(max_imfs).lower() == 'auto':
            max_imfs = None
        drop_imfs = params.get('drop_imfs')
        keep_imfs = params.get('keep_imfs')
        ns = params.get('noise_strength', 0.2)
        trials = int(params.get('trials', 100))
        rng = params.get('random_state')
        # Sensible default for number of IMFs: ~log2(n), capped [2,10]
        n = len(xnp)
        if max_imfs is None:
            try:
                est = int(np.ceil(np.log2(max(8, n))))
            except Exception:
                est = 6
            max_imfs = max(2, min(10, est))

        try:
            if method == 'eemd' and _EEMD is not None:
                decomp = _EEMD()
                if rng is not None:
                    decomp.trials = trials
                    decomp.noise_seed(rng)
                else:
                    decomp.trials = trials
                decomp.noise_strength = ns
                imfs = decomp.eemd(xnp, max_imf=max_imfs)
            elif method == 'ceemdan' and _CEEMDAN is not None:
                decomp = _CEEMDAN()
                if rng is not None:
                    decomp.random_seed = rng
                decomp.noise_strength = ns
                imfs = decomp.ceemdan(xnp, max_imf=int(max_imfs) if max_imfs is not None else None)
            else:
                # fallback to plain EMD
                decomp = _EMD() if _EMD is not None else (_EEMD() if _EEMD is not None else _CEEMDAN())
                imfs = decomp.emd(xnp, max_imf=int(max_imfs) if max_imfs is not None else None) if hasattr(decomp, 'emd') else decomp.eemd(xnp, max_imf=int(max_imfs) if max_imfs is not None else None)
        except Exception:
            return s

        if imfs is None or len(imfs) == 0:
            return s
        imfs = np.atleast_2d(imfs)
        # Residual (trend) component not returned explicitly; reconstruct it
        resid = xnp - imfs.sum(axis=0)
        k_all = list(range(imfs.shape[0]))
        if isinstance(keep_imfs, (list, tuple)) and len(keep_imfs) > 0:
            k_sel = [k for k in keep_imfs if 0 <= int(k) < imfs.shape[0]]
        elif isinstance(drop_imfs, (list, tuple)) and len(drop_imfs) > 0:
            drop = {int(k) for k in drop_imfs}
            k_sel = [k for k in k_all if k not in drop]
        else:
            # Default: drop the first IMF (highest frequency)
            k_sel = [k for k in k_all if k != 0]
        y = resid + imfs[k_sel].sum(axis=0) if len(k_sel) > 0 else resid
        return pd.Series(y, index=x.index)

    # Unknown method: return original
    return s


def _apply_denoise(
    df: pd.DataFrame,
    spec: Optional[DenoiseSpec],
    default_when: str = 'post_ti',
) -> List[str]:
    """Apply denoising per spec to selected columns in-place.

    Returns list of columns added (if any). May also overwrite columns when keep_original=False.
    """
    added_cols: List[str] = []
    if not spec or not isinstance(spec, dict):
        return added_cols
    method = str(spec.get('method', 'none')).lower()
    if method == 'none':
        return added_cols
    params = spec.get('params') or {}
    cols = spec.get('columns') or ['close']
    when = str(spec.get('when') or default_when)
    causality = str(spec.get('causality') or ('causal' if when == 'pre_ti' else 'zero_phase'))
    keep_original = bool(spec.get('keep_original')) if 'keep_original' in spec else (when != 'pre_ti')
    suffix = str(spec.get('suffix') or '_dn')

    for col in cols:
        if col not in df.columns:
            continue
        try:
            y = _denoise_series(df[col], method=method, params=params, causality=causality)
        except Exception:
            continue
        if keep_original:
            new_col = f"{col}{suffix}"
            df[new_col] = y
            added_cols.append(new_col)
        else:
            df[col] = y
    return added_cols


@mcp.tool()
def list_denoise_methods() -> Dict[str, Any]:
    """Return JSON with supported denoise methods, availability, and parameter docs."""
    try:
        from denoise_utils import get_denoise_methods_data
        return get_denoise_methods_data()
    except Exception as e:
        return {"error": f"Error listing denoise methods: {e}"}

@mcp.tool()
def list_forecast_methods() -> Dict[str, Any]:
    """Return JSON describing supported forecast methods, params, and defaults."""
    try:
        from ..utils.forecast import get_forecast_methods_data
        return get_forecast_methods_data(_SM_ETS_AVAILABLE, _SM_SARIMAX_AVAILABLE)
    except Exception as e:
        return {"error": f"Error listing forecast methods: {e}"}

@mcp.tool()
def list_indicators(search_term: Optional[str] = None, category: Optional[CategoryLiteral] = None) -> Dict[str, Any]:  # type: ignore
    """List indicators as CSV with columns: name,category. Optional filters: search_term, category."""
    try:
        items = _list_ta_indicators()
        if search_term:
            q = search_term.strip().lower()
            filtered = []
            for it in items:
                name = it.get('name', '').lower()
                desc = (it.get('description') or '').lower()
                cat = (it.get('category') or '').lower()
                if q in name or q in desc or q in cat:
                    filtered.append(it)
            items = filtered
        if category:
            cat_q = category.strip().lower()
            items = [it for it in items if (it.get('category') or '').lower() == cat_q]
        items.sort(key=lambda x: (x.get('category') or '', x.get('name') or ''))
        rows = [[it.get('name',''), it.get('category','')] for it in items]
        return _csv_from_rows(["name", "category"], rows)
    except Exception as e:
        return {"error": f"Error listing indicators: {e}"}


# Note: category annotation is set at definition time above to be captured in the MCP schema

@mcp.tool()
def describe_indicator(name: IndicatorNameLiteral) -> Dict[str, Any]:  # type: ignore
    """Return detailed indicator information (name, category, params, description)."""
    try:
        items = _list_ta_indicators()
        target = next((it for it in items if it.get('name','').lower() == str(name).lower()), None)
        if not target:
            return {"error": f"Indicator '{name}' not found"}
        return {"success": True, "indicator": target}
    except Exception as e:
        return {"error": f"Error getting indicator details: {e}"}

# Removed grouping helper; get_symbols is simplified to CSV list only

@mcp.tool()
@_auto_connect_wrapper
def list_symbols(
    search_term: Optional[str] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """List symbols as CSV with columns: name,group,description.

    - If `search_term` is provided, matches group name, then symbol name, then description.
    - If omitted, returns only visible symbols. When searching, includes non‑visible matches.
    - `limit` caps the number of returned rows.
    """
    try:
        search_strategy = "none"
        matched_symbols = []
        
        if search_term:
            search_upper = search_term.upper()
            
            # Strategy 1: Search for matching group names first
            all_symbols = mt5.symbols_get()
            if all_symbols is None:
                return {"error": f"Failed to get symbols: {mt5.last_error()}"}
            
            # Get all unique groups
            groups = {}
            for symbol in all_symbols:
                group_path = _extract_group_path(symbol)
                if group_path not in groups:
                    groups[group_path] = []
                groups[group_path].append(symbol)
            
            # Strategy 1: Try group search first, but only if it looks like a group name
            # (avoid matching individual symbol groups for currency searches)
            matching_groups = []
            group_search_threshold = GROUP_SEARCH_THRESHOLD  # centralized threshold
            
            for group_name in groups.keys():
                if search_upper in group_name.upper():
                    matching_groups.append(group_name)
            
            # If we find many groups with the search term, it's probably a symbol search (like EUR, USD)
            # If we find few groups, it's probably a real group search (like Majors, Forex)
            if matching_groups and len(matching_groups) <= group_search_threshold:
                search_strategy = "group_match"
                # Use symbols from matching groups
                for group_name in matching_groups:
                    matched_symbols.extend(groups[group_name])
            else:
                # Strategy 2: Partial match in symbol names (primary strategy for currencies)
                symbol_name_matches = []
                for symbol in all_symbols:
                    if search_upper in symbol.name.upper():
                        symbol_name_matches.append(symbol)
                
                if symbol_name_matches:
                    search_strategy = "symbol_name_match"
                    matched_symbols = symbol_name_matches
                elif matching_groups:  # Fall back to group matches if we had many
                    search_strategy = "group_match"
                    for group_name in matching_groups:
                        matched_symbols.extend(groups[group_name])
                else:
                    # Strategy 3: Partial match in descriptions
                    description_matches = []
                    for symbol in all_symbols:
                        # Check symbol description
                        if hasattr(symbol, 'description') and symbol.description:
                            if search_upper in symbol.description.upper():
                                description_matches.append(symbol)
                                continue
                        
                        # Check group path as description
                        group_path = getattr(symbol, 'path', '')
                        if search_upper in group_path.upper():
                            description_matches.append(symbol)
                    
                    if description_matches:
                        search_strategy = "description_match"
                        matched_symbols = description_matches
                    else:
                        search_strategy = "no_match"
                        matched_symbols = []
        else:
            # No search term - return all symbols
            search_strategy = "all"
            matched_symbols = list(mt5.symbols_get() or [])
        
        # Build symbol list with visibility rule
        only_visible = False if search_term else True
        symbol_list = []
        for symbol in matched_symbols:
            if only_visible and not symbol.visible:
                continue
            symbol_list.append({
                "name": symbol.name,
                "group": _extract_group_path(symbol),
                "description": symbol.description,
            })
        
        # Apply limit
        if limit and limit > 0:
            symbol_list = symbol_list[:limit]
        # Convert to CSV format using proper escaping
        rows = [[s["name"], s["group"], s["description"]] for s in symbol_list]
        return _csv_from_rows(["name", "group", "description"], rows)
    except Exception as e:
        return {"error": f"Error getting symbols: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def list_symbol_groups(search_term: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """List group paths as CSV with a single column: group.

    - Filters by `search_term` (substring, case‑insensitive) when provided.
    - Sorted by group size (desc); `limit` caps the number of groups.
    """
    try:
        # Get all symbols first
        all_symbols = mt5.symbols_get()
        if all_symbols is None:
            return {"error": f"Failed to get symbols: {mt5.last_error()}"}
        
        # Collect unique groups and counts
        groups = {}
        for symbol in all_symbols:
            group_path = _extract_group_path(symbol)
            if group_path not in groups:
                groups[group_path] = {"count": 0}
            groups[group_path]["count"] += 1
        
        # Filter by search term if provided
        filtered_items = list(groups.items())
        if search_term:
            q = search_term.strip().lower()
            filtered_items = [(k, v) for (k, v) in filtered_items if q in (k or '').lower()]

        # Sort groups by count (most symbols first)
        filtered_items.sort(key=lambda x: x[1]["count"], reverse=True)

        # Apply limit
        if limit and limit > 0:
            filtered_items = filtered_items[:limit]

        # Build CSV with only group names
        group_names = [name for name, _ in filtered_items]
        rows = [[g] for g in group_names]
        return _csv_from_rows(["group"], rows)
    except Exception as e:
        return {"error": f"Error getting symbol groups: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def describe_symbol(symbol: str) -> Dict[str, Any]:
    """Return symbol information as JSON for `symbol`.
       Includes information such as Symbol Description, Swap Values, Tick Size/Value, etc.
    """
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {"error": f"Symbol {symbol} not found"}
        
        # Build symbol info dynamically: include all available attributes
        # except excluded ones; skip empty/default values when possible.
        symbol_data = {}
        excluded = {"spread", "ask", "bid", "visible", "custom"}
        for attr in dir(symbol_info):
            if attr.startswith('_'):
                continue
            if attr in excluded:
                continue
            try:
                value = getattr(symbol_info, attr)
            except Exception:
                continue
            # Skip callables and descriptors
            if callable(value):
                continue
            # Skip empty/defaults for readability
            if value is None:
                continue
            if isinstance(value, str) and value == "":
                continue
            if isinstance(value, (int, float)) and value == 0:
                continue
            symbol_data[attr] = value
        
        return {
            "success": True,
            "symbol": symbol_data
        }
    except Exception as e:
        return {"error": f"Error getting symbol info: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def fetch_candles(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    limit: int = 10,
    start: Optional[str] = None,
    end: Optional[str] = None,
    ohlcv: Optional[str] = None,
    indicators: Optional[List[IndicatorSpec]] = None,
    denoise: Optional[DenoiseSpec] = None,
    simplify: Optional[Dict[str, Any]] = None,
    timezone: str = "auto",
) -> Dict[str, Any]:
    """Return historical candles as CSV.
       Can include OHLCV data, optionally along with technical indicators.
       Returns the last candles by default, unless a date range is specified.
         Parameters:
         - symbol: The symbol to retrieve data for (e.g., "EURUSD").
         - timeframe: The timeframe to use (e.g., "H1", "M30").
         - limit: The maximum number of bars to return when not using a date range (default 10).
         - start: Optional start date (e.g., "2025-08-29" or "yesterday 14:00").
         - end: Optional end date.
         - ohlcv: Optional fields to include.
             Accepts friendly forms like: 'close', 'price', 'ohlc', 'ohlcv', 'all',
             compact 'cl' or letters 'OHLCV', or names 'open,high,low,close'.
         - indicators: Optional technical indicators to include (e.g., "rsi(20),macd(12,26,9),ema(26)")
         - denoise: Optional denoising spec to smooth selected columns either pre‑ or post‑TI
         - simplify: Optional dict to reduce rows while preserving shape across all numeric columns.
             keys:
               - method: 'lttb' (default), 'rdp', 'pla', 'apca'
               - For 'lttb': points/max_points/target_points or ratio (0..1)
               - For 'rdp': epsilon (tolerance in y-units); if missing, auto-tunes when points/ratio provided, else falls back to LTTB
               - For 'pla': max_error (in y-units), or points/segments for uniform segmentation
               - For 'apca': max_error (in y-units), or points/segments for uniform segmentation
       The full list of supported technical indicators can be retrieved from `get_indicators`.
    """
    try:
        # Validate timeframe using the shared map
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_timeframe = TIMEFRAME_MAP[timeframe]
        
        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = mt5.symbol_info(symbol)
        _was_visible = bool(_info_before.visible) if _info_before is not None else None
        err = _ensure_symbol_ready(symbol)
        if err:
            return {"error": err}
        
        try:
            # Normalize TI spec from structured list to string for internal processing
            ti_spec = None
            if ti is not None:
                if isinstance(ti, (list, tuple)):
                    parts = []
                    for item in ti:
                        if isinstance(item, dict) and 'name' in item:
                            nm = str(item['name'])
                            params = item.get('params') or []
                            if isinstance(params, (list, tuple)) and len(params) > 0:
                                args_str = ",".join(str(_coerce_scalar(str(p))) for p in params)
                                parts.append(f"{nm}({args_str})")
                            else:
                                parts.append(nm)
                        else:
                            parts.append(str(item))
                    ti_spec = ",".join(parts)
                else:
                    ti_spec = str(ti)
            # Determine warmup bars if technical indicators requested
            warmup_bars = _estimate_warmup_bars(ti_spec)

            if start_datetime and end_datetime:
                from_date = _parse_start_datetime(start_datetime)
                to_date = _parse_start_datetime(end_datetime)
                if not from_date or not to_date:
                    return {"error": "Invalid date format. Try '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00'."}
                if from_date > to_date:
                    return {"error": "start_datetime must be before end_datetime"}
                # Expand range backward by warmup bars for TI calculation
                seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe, 60)
                from_date_internal = from_date - timedelta(seconds=seconds_per_bar * warmup_bars)
                rates = None
                for _ in range(FETCH_RETRY_ATTEMPTS):
                    rates = _mt5_copy_rates_range(symbol, mt5_timeframe, from_date_internal, to_date)
                    if rates is not None and len(rates) > 0:
                        # Sanity: last bar should be close to end
                        last_t = rates[-1]["time"]
                        seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe, 60)
                        if last_t >= (to_date.timestamp() - seconds_per_bar * SANITY_BARS_TOLERANCE):
                            break
                    time.sleep(FETCH_RETRY_DELAY)
            elif start_datetime:
                from_date = _parse_start_datetime(start_datetime)
                if not from_date:
                    return {"error": "Invalid date format. Try '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00'."}
                # Fetch forward from the provided start by using a to_date window
                seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe)
                if not seconds_per_bar:
                    return {"error": f"Unable to determine timeframe seconds for {timeframe}"}
                to_date = from_date + timedelta(seconds=seconds_per_bar * (candles + 2))
                # Expand backward for warmup
                from_date_internal = from_date - timedelta(seconds=seconds_per_bar * warmup_bars)
                rates = None
                for _ in range(FETCH_RETRY_ATTEMPTS):
                    rates = _mt5_copy_rates_range(symbol, mt5_timeframe, from_date_internal, to_date)
                    if rates is not None and len(rates) > 0:
                        # Sanity: last bar should be close to computed to_date
                        last_t = rates[-1]["time"]
                        if last_t >= (to_date.timestamp() - seconds_per_bar * SANITY_BARS_TOLERANCE):
                            break
                    time.sleep(FETCH_RETRY_DELAY)
            elif end_datetime:
                to_date = _parse_start_datetime(end_datetime)
                if not to_date:
                    return {"error": "Invalid date format. Try '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00'."}
                # Get the last 'count' bars ending at end_datetime
                rates = None
                seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe, 60)
                for _ in range(FETCH_RETRY_ATTEMPTS):
                    rates = _mt5_copy_rates_from(symbol, mt5_timeframe, to_date, candles + warmup_bars)
                    if rates is not None and len(rates) > 0:
                        # Sanity: last bar near end
                        last_t = rates[-1]["time"]
                        if last_t >= (to_date.timestamp() - seconds_per_bar * SANITY_BARS_TOLERANCE):
                            break
                    time.sleep(FETCH_RETRY_DELAY)
            else:
                # Use copy_rates_from with current time (now) to force fresh data retrieval
                utc_now = datetime.utcnow()
                rates = None
                seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe, 60)
                for _ in range(FETCH_RETRY_ATTEMPTS):
                    rates = _mt5_copy_rates_from(symbol, mt5_timeframe, utc_now, candles + warmup_bars)
                    if rates is not None and len(rates) > 0:
                        last_t = rates[-1]["time"]
                        if last_t >= (utc_now.timestamp() - seconds_per_bar * SANITY_BARS_TOLERANCE):
                            break
                    time.sleep(FETCH_RETRY_DELAY)
        finally:
            # Restore original visibility if we changed it
            if _was_visible is False:
                try:
                    mt5.symbol_select(symbol, False)
                except Exception:
                    pass
        
        if rates is None:
            return {"error": f"Failed to get rates for {symbol}: {mt5.last_error()}"}

        # Generate CSV-like format with dynamic column filtering
        if len(rates) == 0:
            return {"error": "No data available"}
        
        # Check which optional columns have meaningful data (at least one non-zero/different value)
        tick_volumes = [int(rate["tick_volume"]) for rate in rates]
        spreads = [int(rate["spread"]) for rate in rates]
        real_volumes = [int(rate["real_volume"]) for rate in rates]
        
        has_tick_volume = len(set(tick_volumes)) > 1 or any(v != 0 for v in tick_volumes)
        has_spread = len(set(spreads)) > 1 or any(v != 0 for v in spreads)
        has_real_volume = len(set(real_volumes)) > 1 or any(v != 0 for v in real_volumes)
        
        # Determine requested columns (O,H,L,C,V) if provided
        requested: Optional[set] = _normalize_ohlcv_arg(ohlcv)
        
        # Build header dynamically
        headers = ["time"]
        if requested is not None:
            # Include only requested subset
            if "O" in requested:
                headers.append("open")
            if "H" in requested:
                headers.append("high")
            if "L" in requested:
                headers.append("low")
            if "C" in requested:
                headers.append("close")
            if "V" in requested:
                headers.append("tick_volume")
        else:
            # Default: OHLC always; include extras if meaningful
            headers.extend(["open", "high", "low", "close"])
            if has_tick_volume:
                headers.append("tick_volume")
            if has_spread:
                headers.append("spread")
            if has_real_volume:
                headers.append("real_volume")
        
        csv_header = ",".join(headers)
        csv_rows = []
        
        # Construct DataFrame to support indicators and consistent CSV building
        df = pd.DataFrame(rates)
        # Normalize MT5 epochs to UTC if configured
        try:
            if 'time' in df.columns:
                df['time'] = df['time'].astype(float).apply(_mt5_epoch_to_utc)
        except Exception:
            pass
        # Keep epoch for filtering and convert readable time; ensure 'volume' exists for TA
        df['__epoch'] = df['time']
        _use_ctz = _use_client_tz(client_tz)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df["time"] = df["time"].apply(_format_time_minimal_local if _use_ctz else _format_time_minimal)
        if 'volume' not in df.columns and 'tick_volume' in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df['volume'] = df['tick_volume']

        # Optional pre-TI denoising (in-place by default)
        if denoise and str(denoise.get('when', 'post_ti')).lower() == 'pre_ti':
            _apply_denoise(df, denoise, default_when='pre_ti')

        # Apply technical indicators if requested (dynamic)
        ti_cols: List[str] = []
        if ti_spec:
            ti_cols = _apply_ta_indicators(df, ti_spec)
            headers.extend([c for c in ti_cols if c not in headers])

        # Build final header list when not using OHLCV subset
        if requested is None:
            # headers already includes OHLC and optional extras
            pass

        # Filter out warmup region to return the intended target window only
        if start_datetime and end_datetime:
            # Keep within original [from_date, to_date]
            target_from = _parse_start_datetime(start_datetime).timestamp()
            target_to = _parse_start_datetime(end_datetime).timestamp()
            df = df.loc[(df['__epoch'] >= target_from) & (df['__epoch'] <= target_to)].copy()
        elif start_datetime:
            target_from = _parse_start_datetime(start_datetime).timestamp()
            df = df.loc[df['__epoch'] >= target_from].copy()
            if len(df) > candles:
                df = df.iloc[:candles].copy()
        elif end_datetime:
            if len(df) > candles:
                df = df.iloc[-candles:].copy()
        else:
            if len(df) > candles:
                df = df.iloc[-candles:].copy()

        # If TI requested, check for NaNs and retry once with increased warmup
        if ti_spec and ti_cols:
            try:
                if df[ti_cols].isna().any().any():
                    # Increase warmup and refetch once
                    warmup_bars_retry = max(int(warmup_bars * TI_NAN_WARMUP_FACTOR), warmup_bars + TI_NAN_WARMUP_MIN_ADD)
                    seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe, 60)
                    # Refetch rates with larger warmup
                    if start_datetime and end_datetime:
                        target_from_dt = _parse_start_datetime(start_datetime)
                        target_to_dt = _parse_start_datetime(end_datetime)
                        from_date_internal = target_from_dt - timedelta(seconds=seconds_per_bar * warmup_bars_retry)
                        rates = _mt5_copy_rates_range(symbol, mt5_timeframe, from_date_internal, target_to_dt)
                    elif start_datetime:
                        target_from_dt = _parse_start_datetime(start_datetime)
                        to_date_dt = target_from_dt + timedelta(seconds=seconds_per_bar * (candles + 2))
                        from_date_internal = target_from_dt - timedelta(seconds=seconds_per_bar * warmup_bars_retry)
                        rates = _mt5_copy_rates_range(symbol, mt5_timeframe, from_date_internal, to_date_dt)
                    elif end_datetime:
                        target_to_dt = _parse_start_datetime(end_datetime)
                        rates = _mt5_copy_rates_from(symbol, mt5_timeframe, target_to_dt, candles + warmup_bars_retry)
                    else:
                        utc_now = datetime.utcnow()
                        rates = _mt5_copy_rates_from(symbol, mt5_timeframe, utc_now, candles + warmup_bars_retry)
                    # Rebuild df and indicators with the larger window
                    if rates is not None and len(rates) > 0:
                        df = pd.DataFrame(rates)
                        df['__epoch'] = df['time']
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            df['time'] = df['time'].apply(_format_time_minimal_local if _use_ctz else _format_time_minimal)
                        if 'volume' not in df.columns and 'tick_volume' in df.columns:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                df['volume'] = df['tick_volume']
                        # Optional pre-TI denoising on retried window
                        if denoise and str(denoise.get('when', 'post_ti')).lower() == 'pre_ti':
                            _apply_denoise(df, denoise, default_when='pre_ti')
                        # Re-apply indicators and re-extend headers
                        ti_cols = _apply_ta_indicators(df, ti_spec)
                        headers.extend([c for c in ti_cols if c not in headers])
                        # Re-trim to target window
                        if start_datetime and end_datetime:
                            target_from = _parse_start_datetime(start_datetime).timestamp()
                            target_to = _parse_start_datetime(end_datetime).timestamp()
                            df = df[(df['__epoch'] >= target_from) & (df['__epoch'] <= target_to)]
                        elif start_datetime:
                            target_from = _parse_start_datetime(start_datetime).timestamp()
                            df = df[df['__epoch'] >= target_from]
                            if len(df) > candles:
                                df = df.iloc[:candles]
                        elif end_datetime:
                            if len(df) > candles:
                                df = df.iloc[-candles:]
                        else:
                            if len(df) > candles:
                                df = df.iloc[-candles:]
            except Exception:
                pass

        # Optional post-TI denoising (adds new columns by default)
        if denoise and str(denoise.get('when', 'post_ti')).lower() == 'post_ti':
            added_dn = _apply_denoise(df, denoise, default_when='post_ti')
            for c in added_dn:
                if c not in headers:
                    headers.append(c)

        # Ensure headers are unique and exist in df
        headers = [h for h in headers if h in df.columns or h == 'time']

        # Reformat time consistently across rows for display
        if 'time' in headers and len(df) > 0:
            epochs_list = df['__epoch'].tolist()
            if _use_ctz:
                fmt = _time_format_from_epochs_local(epochs_list)
                fmt = _maybe_strip_year_local(fmt, epochs_list)
                fmt = _style_time_format(fmt)
                tz = _resolve_client_tz(client_tz)
                # Track used tz name and invalid explicit values
                tz_used_name = None
                tz_warning = None
                if isinstance(client_tz, str):
                    vlow = client_tz.strip().lower()
                    if vlow not in ('auto','client','utc',''):
                        try:
                            import pytz  # type: ignore
                            tz_explicit = pytz.timezone(client_tz.strip())
                            tz = tz_explicit
                        except Exception:
                            tz_warning = f"Unknown timezone '{client_tz}', falling back to CLIENT_TZ or system"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if tz is not None:
                        tz_used_name = getattr(tz, 'zone', None) or str(tz)
                        df['time'] = df['__epoch'].apply(lambda t: datetime.fromtimestamp(t, tz=timezone.utc).astimezone(tz).strftime(fmt))
                    else:
                        tz_used_name = 'system'
                        df['time'] = df['__epoch'].apply(lambda t: datetime.fromtimestamp(t, tz=timezone.utc).astimezone().strftime(fmt))
                df.__dict__['_tz_used_name'] = tz_used_name
                df.__dict__['_tz_warning'] = tz_warning
            else:
                fmt = _time_format_from_epochs(epochs_list)
                fmt = _maybe_strip_year(fmt, epochs_list)
                fmt = _style_time_format(fmt)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df['time'] = df['__epoch'].apply(lambda t: datetime.utcfromtimestamp(t).strftime(fmt))

        # Optionally reduce number of rows for readability/output size
        original_rows = len(df)
        df, simplify_meta = _simplify_dataframe_rows(df, headers, simplify)

        # Assemble rows from (possibly reduced) DataFrame for selected headers
        rows = _format_numeric_rows_from_df(df, headers)

        # Build CSV via writer for escaping
        payload = _csv_from_rows(headers, rows)
        payload.update({
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": len(df),
        })
        payload["display_timezone"] = "client" if _use_ctz else "UTC"
        if simplify_meta is not None:
            payload["simplified"] = True
            payload["simplify"] = simplify_meta
            payload["simplify"]["timeframe"] = timeframe
            payload["simplify"]["original_candles"] = original_rows
        return payload
    except Exception as e:
        return {"error": f"Error getting rates: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def fetch_ticks(symbol: str, count: int = 100, start_datetime: Optional[str] = None, simplify: Optional[Dict[str, Any]] = None, client_tz: str = "auto") -> Dict[str, Any]:
    """Return latest ticks as CSV with columns: time,bid,ask and optional last,volume,flags.
    - `count` limits the number of rows; `start_datetime` starts from a flexible date/time.
    - `simplify`: reduce rows while preserving shape across all numeric columns.
        - method: 'lttb' (default), 'rdp', 'pla', 'apca'
        - For 'lttb': points/max_points/target_points or ratio (0..1)
        - For 'rdp': epsilon (y-units); if missing, auto-tunes when points/ratio given, else falls back to LTTB
        - For 'pla': max_error (y-units), or points/segments
        - For 'apca': max_error (y-units), or points/segments
    """
    try:
        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = mt5.symbol_info(symbol)
        _was_visible = bool(_info_before.visible) if _info_before is not None else None
        err = _ensure_symbol_ready(symbol)
        if err:
            return {"error": err}
        
        try:
            if start_datetime:
                from_date = _parse_start_datetime(start_datetime)
                if not from_date:
                    return {"error": "Invalid date format. Try examples like '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00', '2 days ago'."}
                ticks = None
                for _ in range(FETCH_RETRY_ATTEMPTS):
                    ticks = _mt5_copy_ticks_from(symbol, from_date, count, mt5.COPY_TICKS_ALL)
                    if ticks is not None and len(ticks) > 0:
                        break
                    time.sleep(FETCH_RETRY_DELAY)
            else:
                # Get recent ticks from current time (now)
                to_date = datetime.utcnow()
                from_date = to_date - timedelta(days=TICKS_LOOKBACK_DAYS)  # look back a configurable window
                ticks = None
                for _ in range(FETCH_RETRY_ATTEMPTS):
                    ticks = _mt5_copy_ticks_range(symbol, from_date, to_date, mt5.COPY_TICKS_ALL)
                    if ticks is not None and len(ticks) > 0:
                        break
                    time.sleep(FETCH_RETRY_DELAY)
                if ticks is not None and len(ticks) > count:
                    ticks = ticks[-count:]  # Get the last 'count' ticks
        finally:
            # Restore original visibility if we changed it
            if _was_visible is False:
                try:
                    mt5.symbol_select(symbol, False)
                except Exception:
                    pass
        
        if ticks is None:
            return {"error": f"Failed to get ticks for {symbol}: {mt5.last_error()}"}
        
        # Generate CSV-like format with dynamic column filtering
        if len(ticks) == 0:
            return {"error": "No tick data available"}
        
        # Check which optional columns have meaningful data
        lasts = [float(tick["last"]) for tick in ticks]
        volumes = [float(tick["volume"]) for tick in ticks]
        flags = [int(tick["flags"]) for tick in ticks]
        
        has_last = len(set(lasts)) > 1 or any(v != 0 for v in lasts)
        has_volume = len(set(volumes)) > 1 or any(v != 0 for v in volumes)
        has_flags = len(set(flags)) > 1 or any(v != 0 for v in flags)
        
        # Build header dynamically (time, bid, ask are always included)
        headers = ["time", "bid", "ask"]
        if has_last:
            headers.append("last")
        if has_volume:
            headers.append("volume")
        if has_flags:
            headers.append("flags")
        
        # Build data rows with matching columns and escape properly
        # Choose a consistent time format for all rows (strip year if constant)
        # Normalize tick times to UTC
        _epochs = [_mt5_epoch_to_utc(float(t["time"])) for t in ticks]
        _use_ctz = _use_client_tz(client_tz)
        if not _use_ctz:
            fmt = _time_format_from_epochs(_epochs)
            fmt = _maybe_strip_year(fmt, _epochs)
            fmt = _style_time_format(fmt)
        # Optional simplification based on a chosen y-series
        original_count = len(ticks)
        select_indices = list(range(original_count))
        _simp_method_used: Optional[str] = None
        _simp_params_meta: Optional[Dict[str, Any]] = None
        if simplify and original_count > 3:
            try:
                # Always represent all available numeric columns (bid/ask/(last)/(volume))
                cols: List[str] = ['bid', 'ask']
                if has_last:
                    cols.append('last')
                if has_volume:
                    cols.append('volume')
                n_out = _choose_simplify_points(original_count, simplify)
                per = max(3, int(round(n_out / max(1, len(cols)))))
                idx_set: set = set([0, original_count - 1])
                params_accum: Dict[str, Any] = {}
                method_used_overall = None
                for c in cols:
                    series: List[float] = []
                    for t in ticks:
                        v = t.get(c)
                        try:
                            series.append(float(v))
                        except Exception:
                            series.append(float('nan'))
                    sub_spec = dict(simplify)
                    sub_spec['points'] = per
                    idxs, method_used, params_meta = _select_indices_for_timeseries(_epochs, series, sub_spec)
                    method_used_overall = method_used
                    for i in idxs:
                        if 0 <= int(i) < original_count:
                            idx_set.add(int(i))
                    try:
                        if params_meta:
                            for k2, v2 in params_meta.items():
                                params_accum.setdefault(k2, v2)
                    except Exception:
                        pass
                union_idxs = sorted(idx_set)
                # Build composite metric for refinement/top-up
                mins: Dict[str, float] = {}
                ranges: Dict[str, float] = {}
                for c in cols:
                    vals = []
                    for t in ticks:
                        try:
                            vals.append(float(t.get(c)))
                        except Exception:
                            vals.append(0.0)
                    if vals:
                        mn, mx = min(vals), max(vals)
                        ranges[c] = max(1e-12, mx - mn)
                        mins[c] = mn
                    else:
                        ranges[c] = 1.0
                        mins[c] = 0.0
                comp: List[float] = []
                for i in range(original_count):
                    s = 0.0
                    for c in cols:
                        try:
                            vv = (float(ticks[i].get(c)) - mins[c]) / ranges[c]
                        except Exception:
                            vv = 0.0
                        s += abs(vv)
                    comp.append(s)
                if len(union_idxs) > n_out:
                    refined = _lttb_select_indices(_epochs, comp, n_out)
                    select_indices = sorted(set(int(i) for i in refined if 0 <= i < original_count))
                elif len(union_idxs) < n_out:
                    refined = _lttb_select_indices(_epochs, comp, n_out)
                    merged = sorted(set(union_idxs).union(refined))
                    if len(merged) > n_out:
                        keep = set([0, original_count - 1])
                        candidates = [(comp[i], i) for i in merged if i not in keep]
                        candidates.sort(reverse=True)
                        for _, i in candidates:
                            keep.add(i)
                            if len(keep) >= n_out:
                                break
                        select_indices = sorted(keep)
                    else:
                        select_indices = merged
                else:
                    select_indices = union_idxs
                _simp_method_used = method_used_overall or str(simplify.get('method', 'lttb')).lower()
                _simp_params_meta = params_accum
            except Exception:
                select_indices = list(range(original_count))

        rows = []
        for i in select_indices:
            tick = ticks[i]
            if _use_ctz:
                time_str = _format_time_minimal_local(_epochs[i])
            else:
                time_str = datetime.utcfromtimestamp(_epochs[i]).strftime(fmt)
            values = [time_str, str(tick['bid']), str(tick['ask'])]
            if has_last:
                values.append(str(tick['last']))
            if has_volume:
                values.append(str(tick['volume']))
            if has_flags:
                values.append(str(tick['flags']))
            rows.append(values)

        payload = _csv_from_rows(headers, rows)
        payload.update({
            "success": True,
            "symbol": symbol,
            "count": len(rows),
        })
        payload["display_timezone"] = "client" if _use_ctz else "UTC"
        if simplify and original_count > len(rows):
            payload["simplified"] = True
            meta = {
                "method": (_simp_method_used or str((simplify or {}).get('method', 'lttb')).lower()),
                "original_rows": original_count,
                "returned_rows": len(rows),
                "multi_column": True,
                "columns": [c for c in ["bid","ask"] + (["last"] if has_last else []) + (["volume"] if has_volume else [])],
            }
            try:
                if _simp_params_meta:
                    meta.update(_simp_params_meta)
                else:
                    # Return key params if present
                    for key in ("epsilon", "max_error", "segments", "points", "ratio"):
                        if key in (simplify or {}):
                            meta[key] = (simplify or {})[key]
            except Exception:
                pass
            payload["simplify"] = meta
        return payload
    except Exception as e:
        return {"error": f"Error getting ticks: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def detect_candlestick_patterns(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    candles: int = 10,
    client_tz: str = "auto",
) -> Dict[str, Any]:
    """Detect candlestick patterns and return CSV rows of detections.

    Inputs:
    - `symbol`: Trading symbol (e.g., "EURUSD").
    - `timeframe`: One of the supported MT5 timeframes (e.g., "M15", "H1").
    - `candles`: Number of most recent candles to analyze.

    Output CSV columns:
    - `time`: UTC timestamp of the bar (compact format)
    - `pattern`: Human-friendly pattern label (includes direction, e.g., "Bearish ENGULFING BEAR")
    """
    try:
        # Validate timeframe
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_timeframe = TIMEFRAME_MAP[timeframe]

        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = mt5.symbol_info(symbol)
        _was_visible = bool(_info_before.visible) if _info_before is not None else None
        err = _ensure_symbol_ready(symbol)
        if err:
            return {"error": err}

        try:
            # Fetch last `candles` bars from now (UTC anchor)
            utc_now = datetime.utcnow()
            rates = _mt5_copy_rates_from(symbol, mt5_timeframe, utc_now, candles)
        finally:
            # Restore original visibility if we changed it
            if _was_visible is False:
                try:
                    mt5.symbol_select(symbol, False)
                except Exception:
                    pass

        if rates is None:
            return {"error": f"Failed to get rates for {symbol}: {mt5.last_error()}"}
        if len(rates) == 0:
            return {"error": "No candle data available"}

        # Build DataFrame and format time
        df = pd.DataFrame(rates)
        # Normalize epochs to UTC if server offset configured
        try:
            if 'time' in df.columns:
                df['time'] = df['time'].astype(float).apply(_mt5_epoch_to_utc)
        except Exception:
            pass
        epochs = [float(t) for t in df['time'].tolist()] if 'time' in df.columns else []
        _use_ctz = _use_client_tz(client_tz)
        if _use_ctz:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df['time'] = df['time'].apply(_format_time_minimal_local)
        else:
            time_fmt = _time_format_from_epochs(epochs) if epochs else "%Y-%m-%d %H:%M:%S"
            time_fmt = _maybe_strip_year(time_fmt, epochs)
            time_fmt = _style_time_format(time_fmt)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df['time'] = df['time'].apply(lambda t: datetime.utcfromtimestamp(float(t)).strftime(time_fmt))

        # Ensure required OHLC columns exist
        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                return {"error": f"Missing '{col}' data from rates"}

        # Prepare temp DataFrame with DatetimeIndex for pandas_ta compatibility
        try:
            temp = df.copy()
            temp['__epoch'] = [float(e) for e in epochs]
            temp.index = pd.to_datetime(temp['__epoch'], unit='s')
        except Exception:
            temp = df.copy()

        # Discover callable cdl_* methods
        pattern_methods: List[str] = []
        try:
            for attr in dir(temp.ta):
                if not attr.startswith('cdl_'):
                    continue
                func = getattr(temp.ta, attr, None)
                if callable(func):
                    pattern_methods.append(attr)
        except Exception:
            pass

        if not pattern_methods:
            return {"error": "No candlestick pattern detectors (cdl_*) found in pandas_ta. Ensure pandas_ta (and TA-Lib if required) are installed."}

        before_cols = set(temp.columns)
        for name in sorted(pattern_methods):
            try:
                method = getattr(temp.ta, name)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    method(append=True)
            except Exception:
                continue

        # Identify newly added pattern columns
        pattern_cols = [c for c in temp.columns if c not in before_cols and c.lower().startswith('cdl_')]
        if not pattern_cols:
            return {"error": "No candle patterns produced any outputs."}

        # Compile detection rows
        rows: List[List[Any]] = []
        for i in range(len(temp)):
            t = df.iloc[i]['time']
            for col in pattern_cols:
                try:
                    val = temp.iloc[i][col]
                except Exception:
                    continue
                try:
                    v = float(val)
                except Exception:
                    continue
                if pd.isna(v) or v == 0:
                    continue
                direction = 'bullish' if v > 0 else 'bearish'
                # Remove leading 'CDL_' or 'cdl_' prefix from pattern name
                pat = col
                if pat.lower().startswith('cdl_'):
                    pat = pat[len('cdl_'):]
                # Human-friendly label: "Bearish ENGULFING BEAR"
                # Replace underscores with spaces and uppercase the pattern words
                pat_human = pat.replace('_', ' ').strip()
                if pat_human:
                    pat_human = pat_human.upper()
                dir_title = 'Bullish' if v > 0 else 'Bearish'
                pattern_label = f"{dir_title} {pat_human}" if pat_human else dir_title
                rows.append([t, pattern_label])

        # Sort for stable output
        try:
            rows.sort(key=lambda r: (r[0], r[1]))
        except Exception:
            pass

        headers = ["time", "pattern"]
        payload = _csv_from_rows(headers, rows)
        payload.update({
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": int(candles),
        })
        payload["display_timezone"] = "client" if _use_ctz else "UTC"
        return payload
    except Exception as e:
        return {"error": f"Error detecting candlestick patterns: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def compute_pivot_points(
    symbol: str,
    basis_timeframe: TimeframeLiteral = "D1",
    method: PivotMethodLiteral = "classic",
    client_tz: str = "auto",
) -> Dict[str, Any]:
    """Compute pivot point levels from the last completed bar on `basis_timeframe`.

    - `basis_timeframe`: Timeframe to source H/L/C from (e.g., D1, W1, MN1).
    - `method`: One of classic, fibonacci, camarilla, woodie, demark.

    Returns JSON with period info, source H/L/C, and computed levels.
    """
    try:
        # Validate timeframe
        if basis_timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {basis_timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_tf = TIMEFRAME_MAP[basis_timeframe]
        tf_secs = TIMEFRAME_SECONDS.get(basis_timeframe)
        if not tf_secs:
            return {"error": f"Unsupported timeframe seconds for {basis_timeframe}"}

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
        _use_ctz = _use_client_tz(client_tz)
        start_str = _format_time_minimal_local(period_start) if _use_ctz else _format_time_minimal(period_start)
        end_str = _format_time_minimal_local(period_end) if _use_ctz else _format_time_minimal(period_end)

        payload: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "method": method_l,
            "basis_timeframe": basis_timeframe,
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
        payload["display_timezone"] = "client" if _use_ctz else "UTC"
        return payload
    except Exception as e:
        return {"error": f"Error computing pivot points: {str(e)}"}

def _bars_per_year(timeframe: str) -> float:
    try:
        sec = TIMEFRAME_SECONDS.get(timeframe)
        if not sec or sec <= 0:
            return 0.0
        seconds_per_year = 365.0 * 24.0 * 3600.0
        return seconds_per_year / float(sec)
    except Exception:
        return 0.0

def _log_returns_from_closes(closes: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.diff(np.log(closes.astype(float)))
    r = r[~np.isnan(r)]
    r = r[np.isfinite(r)]
    return r

def _parkinson_sigma_sq(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    # (1/(4 ln 2)) * (ln(H/L))^2 per bar
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.log(np.maximum(high, 1e-12) / np.maximum(low, 1e-12))
        v = (x * x) / (4.0 * math.log(2.0))
    v[~np.isfinite(v)] = np.nan
    return v

def _garman_klass_sigma_sq(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    # 0.5*(ln(H/L))^2 - (2 ln 2 - 1)*(ln(C/O))^2
    with np.errstate(divide='ignore', invalid='ignore'):
        hl = np.log(np.maximum(high, 1e-12) / np.maximum(low, 1e-12))
        co = np.log(np.maximum(close, 1e-12) / np.maximum(open_, 1e-12))
        v = 0.5 * (hl * hl) - (2.0 * math.log(2.0) - 1.0) * (co * co)
    v[~np.isfinite(v)] = np.nan
    return v

def _rogers_satchell_sigma_sq(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    # ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
    with np.errstate(divide='ignore', invalid='ignore'):
        hc = np.log(np.maximum(high, 1e-12) / np.maximum(close, 1e-12))
        ho = np.log(np.maximum(high, 1e-12) / np.maximum(open_, 1e-12))
        lc = np.log(np.maximum(low, 1e-12) / np.maximum(close, 1e-12))
        lo = np.log(np.maximum(low, 1e-12) / np.maximum(open_, 1e-12))
        v = (hc * ho) + (lc * lo)
    v[~np.isfinite(v)] = np.nan
    return v

try:
    # Optional GARCH support
    from arch import arch_model as _arch_model  # type: ignore
    _ARCH_AVAILABLE = True
except Exception:
    _ARCH_AVAILABLE = False

# ---- Fast Forecast methods (enums) ----
_FORECAST_METHODS = (
    "naive",
    "seasonal_naive",
    "drift",
    "theta",
    "fourier_ols",
    "ses",
    "holt",
    "holt_winters_add",
    "holt_winters_mul",
    "arima",
    "sarima",
)

try:
    ForecastMethodLiteral = Literal[_FORECAST_METHODS]  # type: ignore
except Exception:
    ForecastMethodLiteral = str  # fallback for typing

# Optional statsmodels for ETS/SARIMA
try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing as _SES, ExponentialSmoothing as _ETS  # type: ignore
    _SM_ETS_AVAILABLE = True
except Exception:
    _SM_ETS_AVAILABLE = False
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX  # type: ignore
    _SM_SARIMAX_AVAILABLE = True
except Exception:
    _SM_SARIMAX_AVAILABLE = False

def _default_seasonality_period(timeframe: str) -> int:
    try:
        sec = TIMEFRAME_SECONDS.get(timeframe)
        if not sec or sec <= 0:
            return 0
        # Intraday: use daily cycle
        if sec < 86400:
            return int(round(86400.0 / float(sec)))
        # Daily: trading week by default (Mon-Fri)
        if timeframe == 'D1':
            return 5
        # Weekly: ~52 weeks
        if timeframe == 'W1':
            return 52
        # Monthly: 12 months
        if timeframe == 'MN1':
            return 12
        return 0
    except Exception:
        return 0

def _next_times_from_last(last_epoch: float, tf_secs: int, horizon: int) -> List[float]:
    base = float(last_epoch)
    step = float(tf_secs)
    return [base + step * (i + 1) for i in range(int(horizon))]

@mcp.tool()
@_auto_connect_wrapper
def forecast_volatility(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon_bars: int = 1,
    method: Literal['ewma','parkinson','gk','rs','garch'] = 'ewma',  # type: ignore
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Forecast volatility over `horizon_bars` using EWMA/range methods/GARCH.

    - `method`: 'ewma' (RiskMetrics), 'parkinson', 'gk' (Garman–Klass), 'rs' (Rogers–Satchell), 'garch' (1,1 if `arch` installed)
    - `params` (optional):
        ewma: {"halflife": int|null, "lambda": float|null, "lookback": int}
        parkinson/gk/rs: {"window": int}
        garch: {"fit_bars": int, "mean": "Zero"|"Constant", "dist": "normal"}
    Returns JSON with current per-bar sigma, annualized sigma, and horizon forecast.
    """
    try:
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_tf = TIMEFRAME_MAP[timeframe]
        tf_secs = TIMEFRAME_SECONDS.get(timeframe)
        if not tf_secs:
            return {"error": f"Unsupported timeframe seconds for {timeframe}"}
        method_l = str(method).lower().strip()
        if method_l not in {'ewma','parkinson','gk','rs','garch'}:
            return {"error": f"Invalid method: {method}"}
        if method_l == 'garch' and not _ARCH_AVAILABLE:
            return {"error": "GARCH requires 'arch' package. Please install 'arch' to enable this method."}

        p = dict(params or {})

        # Determine bars required
        if method_l == 'ewma':
            lookback = int(p.get('lookback', 1500))
            need = max(lookback + 2, 100)
        elif method_l in ('parkinson','gk','rs'):
            window = int(p.get('window', 20))
            need = max(window + 2, 50)
        else:  # garch
            fit_bars = int(p.get('fit_bars', 2000))
            need = max(fit_bars + 2, 500)

        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = mt5.symbol_info(symbol)
        _was_visible = bool(_info_before.visible) if _info_before is not None else None
        err = _ensure_symbol_ready(symbol)
        if err:
            return {"error": err}

        try:
            # Use server time for alignment
            _tick = mt5.symbol_info_tick(symbol)
            if _tick is not None and getattr(_tick, 'time', None):
                t_utc = _mt5_epoch_to_utc(float(_tick.time))
                server_now_dt = datetime.utcfromtimestamp(t_utc)
            else:
                server_now_dt = datetime.utcnow()
            rates = _mt5_copy_rates_from(symbol, mt5_tf, server_now_dt, need)
        finally:
            if _was_visible is False:
                try:
                    mt5.symbol_select(symbol, False)
                except Exception:
                    pass

        if rates is None or len(rates) < 3:
            return {"error": f"Failed to get sufficient rates for {symbol}: {mt5.last_error()}"}

        df = pd.DataFrame(rates)
        # Drop forming last bar; keep closed bars only
        if len(df) >= 2:
            df = df.iloc[:-1]
        if len(df) < 3:
            return {"error": "Not enough closed bars to compute volatility"}

        closes = df['close'].to_numpy(dtype=float)
        highs = df['high'].to_numpy(dtype=float) if 'high' in df.columns else None
        lows = df['low'].to_numpy(dtype=float) if 'low' in df.columns else None
        opens = df['open'].to_numpy(dtype=float) if 'open' in df.columns else None
        last_close = float(closes[-1])

        bars_per_year = _bars_per_year(timeframe)
        ann_factor = math.sqrt(bars_per_year) if bars_per_year > 0 else float('nan')

        sigma_bar = float('nan')
        sigma_ann = float('nan')
        sigma_h_bar = float('nan')  # horizon sigma of sum of returns over k bars
        params_used: Dict[str, Any] = {}

        if method_l == 'ewma':
            r = _log_returns_from_closes(closes)
            if r.size < 5:
                return {"error": "Not enough return observations for EWMA"}
            lam = p.get('lambda')
            hl = p.get('halflife')
            if hl is not None:
                try:
                    hl = float(hl)
                except Exception:
                    hl = None
            if lam is not None:
                try:
                    lam = float(lam)
                except Exception:
                    lam = None
            if lam is not None and (hl is None):
                alpha = 1.0 - float(lam)
                var_series = pd.Series(r).ewm(alpha=alpha, adjust=False).var(bias=False)
                params_used['lambda'] = float(lam)
            else:
                # default halflife if not provided
                if hl is None:
                    # heuristic per timeframe
                    default_hl = 60 if timeframe.startswith('H') else (11 if timeframe == 'D1' else 180)
                    hl = float(p.get('halflife', default_hl))
                var_series = pd.Series(r).ewm(halflife=float(hl), adjust=False).var(bias=False)
                params_used['halflife'] = float(hl)
            v = float(var_series.iloc[-1])
            v = v if math.isfinite(v) and v >= 0 else float('nan')
            sigma_bar = math.sqrt(v) if math.isfinite(v) and v >= 0 else float('nan')
            sigma_h_bar = math.sqrt(max(1, int(horizon_bars)) * v) if math.isfinite(v) and v >= 0 else float('nan')
        elif method_l in ('parkinson','gk','rs'):
            if highs is None or lows is None:
                return {"error": "High/Low data required for range-based estimators"}
            window = int(p.get('window', 20))
            params_used['window'] = window
            if method_l == 'parkinson':
                var_bars = _parkinson_sigma_sq(highs, lows)
            elif method_l == 'gk':
                if opens is None:
                    return {"error": "Open data required for Garman–Klass"}
                var_bars = _garman_klass_sigma_sq(opens, highs, lows, closes)
            else:  # rs
                if opens is None:
                    return {"error": "Open data required for Rogers–Satchell"}
                var_bars = _rogers_satchell_sigma_sq(opens, highs, lows, closes)
            s = pd.Series(var_bars)
            v = float(s.tail(window).mean(skipna=True))
            v = v if math.isfinite(v) and v >= 0 else float('nan')
            sigma_bar = math.sqrt(v) if math.isfinite(v) and v >= 0 else float('nan')
            sigma_h_bar = math.sqrt(max(1, int(horizon_bars)) * v) if math.isfinite(v) and v >= 0 else float('nan')
        else:  # garch
            r = _log_returns_from_closes(closes)
            if r.size < 100:
                return {"error": "Not enough return observations for GARCH (need >=100)"}
            fit_bars = int(p.get('fit_bars', min(2000, r.size)))
            mean_model = str(p.get('mean', 'Zero'))
            dist = str(p.get('dist', 'normal'))
            params_used.update({'fit_bars': fit_bars, 'mean': mean_model, 'dist': dist})
            r_fit = pd.Series(r[-fit_bars:]) * 100.0  # scale to percent
            try:
                am = _arch_model(r_fit, mean=mean_model.lower(), vol='GARCH', p=1, q=1, dist=dist)
                res = am.fit(disp='off')
                # Current conditional variance (percent^2)
                cond_vol = float(res.conditional_volatility.iloc[-1])  # percent
                sigma_bar = cond_vol / 100.0
                # k-step ahead variance forecasts (percent^2)
                fc = res.forecast(horizon=max(1, int(horizon_bars)), reindex=False)
                var_path = np.array(fc.variance.iloc[-1].values, dtype=float)  # shape (horizon,)
                var_sum = float(np.nansum(var_path))  # percent^2
                sigma_h_bar = math.sqrt(var_sum) / 100.0
            except Exception as ex:
                return {"error": f"GARCH fitting error: {ex}"}

        sigma_ann = sigma_bar * ann_factor if math.isfinite(sigma_bar) and math.isfinite(ann_factor) else float('nan')
        sigma_h_ann = sigma_h_bar * ann_factor if math.isfinite(sigma_h_bar) and math.isfinite(ann_factor) else float('nan')

        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method_l,
            "params_used": params_used,
            "bars_used": int(len(df)),
            "horizon_bars": int(horizon_bars),
            "last_close": last_close,
            "sigma_bar_return": sigma_bar,
            "sigma_annual_return": sigma_ann,
            "horizon_sigma_return": sigma_h_bar,
            "horizon_sigma_annual": sigma_h_ann,
        }
    except Exception as e:
        return {"error": f"Error computing volatility forecast: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def forecast(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    method: ForecastMethodLiteral = "theta",
    horizon: int = 12,
    lookback: Optional[int] = None,
    as_of: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    ci_alpha: Optional[float] = 0.05,
    target: Literal['price','return'] = 'price',  # type: ignore
    denoise: Optional[DenoiseSpec] = None,
    client_tz: str = "auto",
) -> Dict[str, Any]:
    """Fast forecasts for the next `horizon_bars` using lightweight methods.

    Methods: naive, seasonal_naive, drift, theta, fourier_ols, ses, holt, holt_winters_add, holt_winters_mul, arima, sarima.
    - `params`: method-specific settings; use `seasonality` inside params when needed (auto if omitted).
    - `target`: 'price' or 'return' (log-return). Price forecasts operate on close prices.
    - `ci_alpha`: confidence level (e.g., 0.05). Set to null to disable intervals.
    """
    try:
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_tf = TIMEFRAME_MAP[timeframe]
        tf_secs = TIMEFRAME_SECONDS.get(timeframe)
        if not tf_secs:
            return {"error": f"Unsupported timeframe seconds for {timeframe}"}

        method_l = str(method).lower().strip()
        if method_l not in _FORECAST_METHODS:
            return {"error": f"Invalid method: {method}. Valid options: {list(_FORECAST_METHODS)}"}

        p = dict(params or {})
        # Prefer explicit seasonality inside params; otherwise auto by timeframe
        m = int(p.get('seasonality')) if p.get('seasonality') is not None else _default_seasonality_period(timeframe)
        if method_l == 'seasonal_naive' and (not m or m <= 0):
            return {"error": "seasonal_naive requires a positive 'seasonality' in params or auto period"}

        # Determine lookback
        if lookback and lookback > 0:
            need = int(lookback) + 2
        else:
            if method_l == 'seasonal_naive':
                need = max(3 * m, int(horizon) + m + 2)
            elif method_l in ('theta', 'fourier_ols'):
                need = max(300, int(horizon) + (2 * m if m else 50))
            else:  # naive, drift
                need = max(100, int(horizon) + 10)

        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = mt5.symbol_info(symbol)
        _was_visible = bool(_info_before.visible) if _info_before is not None else None
        err = _ensure_symbol_ready(symbol)
        if err:
            return {"error": err}

        try:
            # Use explicit as-of time if provided, else server time for alignment
            if as_of:
                to_dt = _parse_start_datetime(as_of)
                if not to_dt:
                    return {"error": "Invalid as_of_datetime. Try '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00'."}
                rates = _mt5_copy_rates_from(symbol, mt5_tf, to_dt, need)
            else:
                _tick = mt5.symbol_info_tick(symbol)
                if _tick is not None and getattr(_tick, 'time', None):
                    t_utc = _mt5_epoch_to_utc(float(_tick.time))
                    server_now_dt = datetime.utcfromtimestamp(t_utc)
                else:
                    server_now_dt = datetime.utcnow()
                rates = _mt5_copy_rates_from(symbol, mt5_tf, server_now_dt, need)
        finally:
            if _was_visible is False:
                try:
                    mt5.symbol_select(symbol, False)
                except Exception:
                    pass

        if rates is None or len(rates) < 3:
            return {"error": f"Failed to get sufficient rates for {symbol}: {mt5.last_error()}"}

        df = pd.DataFrame(rates)
        # Normalize MT5 epochs to UTC if server offset configured
        try:
            if 'time' in df.columns:
                df['time'] = df['time'].astype(float).apply(_mt5_epoch_to_utc)
        except Exception:
            pass
        # Drop forming last bar only when forecasting from 'now'; for historical as_of, keep all
        if as_of is None and len(df) >= 2:
            df = df.iloc[:-1]
        if len(df) < 3:
            return {"error": "Not enough closed bars to compute forecast"}

        # Optionally denoise close
        base_col = 'close'
        if denoise:
            added = _apply_denoise(df, denoise, default_when='pre_ti')
            if len(added) > 0 and f"{base_col}_dn" in added:
                base_col = f"{base_col}_dn"

        y = df[base_col].astype(float).to_numpy()
        t = np.arange(1, len(y) + 1, dtype=float)
        last_time = float(df['time'].iloc[-1])
        future_times = _next_times_from_last(last_time, int(tf_secs), int(horizon))

        # Transform for return modeling if requested
        use_returns = (str(target).lower() == 'return')
        if use_returns:
            with np.errstate(divide='ignore', invalid='ignore'):
                x = np.diff(np.log(np.maximum(y, 1e-12)))
            x = x[np.isfinite(x)]
            if x.size < 5:
                return {"error": "Not enough data to compute return-based forecast"}
            series = x
            origin_price = float(y[-1])
        else:
            series = y
            origin_price = float(y[-1])

        # Ensure finite numeric series for modeling
        series = np.asarray(series, dtype=float)
        series = series[np.isfinite(series)]
        n = len(series)
        if n < 3:
            return {"error": "Series too short for forecasting"}

        # Fit/forecast by method
        fh = int(horizon)
        f_vals = np.zeros(fh, dtype=float)
        pre_ci: Optional[Tuple[np.ndarray, np.ndarray]] = None
        model_fitted: Optional[np.ndarray] = None
        params_used: Dict[str, Any] = {}

        if method_l == 'naive':
            last_val = float(series[-1])
            f_vals[:] = last_val
            params_used = {}

        elif method_l == 'drift':
            # Classic drift: y_{T+h} = y_T + h*(y_T - y_1)/(T-1)
            slope = (float(series[-1]) - float(series[0])) / float(max(1, n - 1))
            f_vals = float(series[-1]) + slope * np.arange(1, fh + 1, dtype=float)
            params_used = {"slope": slope}

        elif method_l == 'seasonal_naive':
            m_eff = int(p.get('seasonality', m) or m)
            if m_eff <= 0 or n < m_eff:
                return {"error": f"Insufficient data for seasonal_naive (m={m_eff})"}
            last_season = series[-m_eff:]
            reps = int(np.ceil(fh / float(m_eff)))
            f_vals = np.tile(last_season, reps)[:fh]
            params_used = {"m": m_eff}

        elif method_l == 'theta':
            # Combine linear trend extrapolation with simple exponential smoothing (fast, fixed alpha)
            alpha = float(p.get('alpha', 0.2))
            # Linear trend via least squares on original series index
            tt = np.arange(1, n + 1, dtype=float)
            A = np.vstack([np.ones(n), tt]).T
            coef, _, _, _ = np.linalg.lstsq(A, series, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            trend_future = a + b * (tt[-1] + np.arange(1, fh + 1, dtype=float))
            # SES on series
            level = float(series[0])
            for v in series[1:]:
                level = alpha * float(v) + (1.0 - alpha) * level
            ses_future = np.full(fh, level, dtype=float)
            f_vals = 0.5 * (trend_future + ses_future)
            params_used = {"alpha": alpha, "trend_slope": b}

        elif method_l == 'fourier_ols':
            m_eff = int(p.get('seasonality', m) or m)
            K = int(p.get('K', min(3, max(1, (m_eff // 2) if m_eff else 2))))
            use_trend = bool(p.get('trend', True))
            tt = np.arange(1, n + 1, dtype=float)
            X_list = [np.ones(n)]
            if use_trend:
                X_list.append(tt)
            for k in range(1, K + 1):
                w = 2.0 * math.pi * k / float(m_eff if m_eff else max(2, n))
                X_list.append(np.sin(w * tt))
                X_list.append(np.cos(w * tt))
            X = np.vstack(X_list).T
            coef, _, _, _ = np.linalg.lstsq(X, series, rcond=None)
            # Future design
            tt_f = tt[-1] + np.arange(1, fh + 1, dtype=float)
            Xf_list = [np.ones(fh)]
            if use_trend:
                Xf_list.append(tt_f)
            for k in range(1, K + 1):
                w = 2.0 * math.pi * k / float(m_eff if m_eff else max(2, n))
                Xf_list.append(np.sin(w * tt_f))
                Xf_list.append(np.cos(w * tt_f))
            Xf = np.vstack(Xf_list).T
            f_vals = Xf @ coef
            params_used = {"m": m_eff, "K": K, "trend": use_trend}

        elif method_l == 'ses':
            if not _SM_ETS_AVAILABLE:
                return {"error": "SES requires statsmodels. Please install 'statsmodels'."}
            alpha = p.get('alpha')
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if alpha is None:
                        res = _SES(series, initialization_method='heuristic').fit(optimized=True)
                    else:
                        res = _SES(series, initialization_method='heuristic').fit(smoothing_level=float(alpha), optimized=False)
                f_vals = res.forecast(fh)
                f_vals = np.asarray(f_vals, dtype=float)
                try:
                    model_fitted = np.asarray(res.fittedvalues, dtype=float)
                except Exception:
                    model_fitted = None
                alpha_used = None
                try:
                    par = getattr(res, 'params', None)
                    if par is not None:
                        # pandas Series or dict-like
                        if hasattr(par, 'get'):
                            val = par.get('smoothing_level', None)
                            if val is None:
                                # fall back to first element if available
                                try:
                                    val = float(par.iloc[0]) if hasattr(par, 'iloc') else float(par[0])
                                except Exception:
                                    val = None
                            alpha_used = val
                        else:
                            # array-like
                            try:
                                alpha_used = float(par[0]) if len(par) > 0 else None
                            except Exception:
                                alpha_used = None
                    if alpha_used is None:
                        mv = getattr(res.model, 'smoothing_level', None)
                        alpha_used = mv if mv is not None else alpha
                except Exception:
                    alpha_used = alpha
                params_used = {"alpha": _to_float_or_nan(alpha_used)}
            except Exception as ex:
                return {"error": f"SES fitting error: {ex}"}

        elif method_l == 'holt':
            if not _SM_ETS_AVAILABLE:
                return {"error": "Holt requires statsmodels. Please install 'statsmodels'."}
            damped = bool(p.get('damped', True))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = _ETS(series, trend='add', damped_trend=damped, initialization_method='heuristic')
                    res = model.fit(optimized=True)
                f_vals = res.forecast(fh)
                f_vals = np.asarray(f_vals, dtype=float)
                try:
                    model_fitted = np.asarray(res.fittedvalues, dtype=float)
                except Exception:
                    model_fitted = None
                params_used = {"damped": damped}
            except Exception as ex:
                return {"error": f"Holt fitting error: {ex}"}

        elif method_l in ('holt_winters_add', 'holt_winters_mul'):
            if not _SM_ETS_AVAILABLE:
                return {"error": "Holt-Winters requires statsmodels. Please install 'statsmodels'."}
            m_eff = int(p.get('seasonality', m) or m)
            if m_eff <= 0:
                return {"error": "Holt-Winters requires a positive seasonality_period"}
            seasonal = 'add' if method_l == 'holt_winters_add' else 'mul'
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = _ETS(series, trend='add', seasonal=seasonal, seasonal_periods=m_eff, initialization_method='heuristic')
                    res = model.fit(optimized=True)
                f_vals = res.forecast(fh)
                f_vals = np.asarray(f_vals, dtype=float)
                try:
                    model_fitted = np.asarray(res.fittedvalues, dtype=float)
                except Exception:
                    model_fitted = None
                params_used = {"seasonal": seasonal, "m": m_eff}
            except Exception as ex:
                return {"error": f"Holt-Winters fitting error: {ex}"}

        elif method_l in ('arima', 'sarima'):
            if not _SM_SARIMAX_AVAILABLE:
                return {"error": "ARIMA/SARIMA require statsmodels. Please install 'statsmodels'."}
            # Defaults: price: d=1, returns: d=0
            d_default = 0 if use_returns else 1
            p_ord = int(p.get('p', 1)); d_ord = int(p.get('d', d_default)); q_ord = int(p.get('q', 1))
            if method_l == 'sarima':
                m_eff = int(p.get('seasonality', m) or m)
                P = int(p.get('P', 0)); D = int(p.get('D', 1 if not use_returns else 0)); Q = int(p.get('Q', 0))
                # SARIMAX requires seasonal period >= 2; fall back to non-seasonal if < 2
                if m_eff is None or m_eff < 2:
                    seas = (0, 0, 0, 0)
                else:
                    seas = (P, D, Q, int(m_eff))
            else:
                seas = (0, 0, 0, 0)
            trend = str(p.get('trend', 'c'))  # 'n' or 'c'
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    endog = pd.Series(series.astype(float))
                    model = _SARIMAX(
                        endog,
                        order=(p_ord, d_ord, q_ord),
                        seasonal_order=seas,
                        trend=trend,
                        enforce_stationarity=True,
                        enforce_invertibility=True,
                    )
                    res = model.fit(method='lbfgs', disp=False, maxiter=100)
                    pred = res.get_forecast(steps=fh)
                f_vals = pred.predicted_mean.to_numpy()
                ci = None
                try:
                    # Use configured CI alpha if provided; default to 0.05
                    _alpha = float(ci_alpha) if ci_alpha is not None else 0.05
                    ci_df = pred.conf_int(alpha=_alpha)
                    ci = (ci_df.iloc[:, 0].to_numpy(), ci_df.iloc[:, 1].to_numpy())
                except Exception:
                    ci = None
                params_used = {"order": (p_ord, d_ord, q_ord), "seasonal_order": seas if method_l=='sarima' else (0,0,0,0), "trend": trend}
                if ci is not None:
                    pre_ci = ci
            except Exception as ex:
                return {"error": f"SARIMAX fitting error: {ex}"}

        # Compute residual scale for intervals (on modeling scale)
        lower = upper = None
        do_ci = (ci_alpha is not None)
        _alpha = float(ci_alpha) if ci_alpha is not None else 0.05
        if do_ci:
            try:
                # Prefer model-provided intervals if available (e.g., SARIMAX)
                if pre_ci is not None:
                    lo, hi = pre_ci
                    lower = np.asarray(lo, dtype=float)
                    upper = np.asarray(hi, dtype=float)
                # Else compute from in-sample residuals
                elif method_l == 'naive':
                    fitted = np.roll(series, 1)[1:]
                    resid = series[1:] - fitted
                elif method_l == 'drift':
                    slope = (float(series[-1]) - float(series[0])) / float(max(1, n - 1))
                    fitted = series[:-1] + slope  # 1-step ahead approx
                    resid = series[1:] - fitted
                elif method_l == 'seasonal_naive':
                    m_eff = int(params_used.get('m', m) or m)
                    if n > m_eff:
                        resid = series[m_eff:] - series[:-m_eff]
                    else:
                        resid = series - np.mean(series)
                elif method_l == 'theta':
                    alpha = float(params_used.get('alpha', 0.2))
                    tt = np.arange(1, n + 1, dtype=float)
                    A = np.vstack([np.ones(n), tt]).T
                    coef, _, _, _ = np.linalg.lstsq(A, series, rcond=None)
                    a, b = float(coef[0]), float(coef[1])
                    trend = a + b * tt
                    level = float(series[0])
                    fitted_ses = [level]
                    for v in series[1:]:
                        level = alpha * float(v) + (1.0 - alpha) * level
                        fitted_ses.append(level)
                    fitted_theta = 0.5 * (trend + np.array(fitted_ses))
                    resid = series - fitted_theta
                elif method_l in ('ses','holt','holt_winters_add','holt_winters_mul') and model_fitted is not None:
                    fitted = model_fitted
                    if fitted.shape[0] > n:
                        fitted = fitted[-n:]
                    elif fitted.shape[0] < n:
                        # pad with last fitted
                        last = fitted[-1] if fitted.size > 0 else float('nan')
                        fitted = np.pad(fitted, (n - fitted.shape[0], 0), mode='edge') if fitted.size > 0 else np.full(n, last)
                    resid = series - fitted
                else:  # fourier_ols fallback
                    tt = np.arange(1, n + 1, dtype=float)
                    m_eff = int(params_used.get('m', m) or m)
                    K = int(params_used.get('K', min(3, (m_eff // 2) if m_eff else 2)))
                    use_trend = bool(params_used.get('trend', True))
                    X_list = [np.ones(n)]
                    if use_trend:
                        X_list.append(tt)
                    for k in range(1, K + 1):
                        w = 2.0 * math.pi * k / float(m_eff if m_eff else max(2, n))
                        X_list.append(np.sin(w * tt))
                        X_list.append(np.cos(w * tt))
                    X = np.vstack(X_list).T
                    coef, _, _, _ = np.linalg.lstsq(X, series, rcond=None)
                    fitted = X @ coef
                    resid = series - fitted
                if pre_ci is None:
                    resid = resid[np.isfinite(resid)]
                    sigma = float(np.std(resid, ddof=1)) if resid.size >= 3 else float('nan')
                    from scipy.stats import norm  # optional if available
                    try:
                        z = float(norm.ppf(1.0 - _alpha / 2.0))
                    except Exception:
                        z = 1.96
                    lower = f_vals - z * sigma
                    upper = f_vals + z * sigma
            except Exception:
                do_ci = False

        # Map back to price if target was returns
        if use_returns:
            # Compose price path multiplicatively from origin_price
            price_path = np.empty(fh, dtype=float)
            price_path[0] = origin_price * math.exp(float(f_vals[0]))
            for i in range(1, fh):
                price_path[i] = price_path[i-1] * math.exp(float(f_vals[i]))
            out_forecast_price = price_path
            if do_ci and lower is not None and upper is not None:
                # Convert return bands to price bands via lognormal mapping per-step
                lower_price = np.empty(fh, dtype=float)
                upper_price = np.empty(fh, dtype=float)
                lower_price[0] = origin_price * math.exp(float(lower[0]))
                upper_price[0] = origin_price * math.exp(float(upper[0]))
                for i in range(1, fh):
                    lower_price[i] = lower_price[i-1] * math.exp(float(lower[i]))
                    upper_price[i] = upper_price[i-1] * math.exp(float(upper[i]))
            else:
                lower_price = upper_price = None
        else:
            out_forecast_price = f_vals
            lower_price = lower
            upper_price = upper

        # Rounding based on symbol digits
        digits = int(getattr(_info_before, "digits", 0) or 0)
        def _round(v: float) -> float:
            try:
                return round(float(v), digits) if digits >= 0 else float(v)
            except Exception:
                return float(v)

        _use_ctz = _use_client_tz(client_tz)
        if _use_ctz:
            times_fmt = [_format_time_minimal_local(ts) for ts in future_times]
        else:
            times_fmt = [_format_time_minimal(ts) for ts in future_times]

        # Training window first/last candle timestamps (used for the forecast)
        try:
            train_first_epoch = float(df['time'].iloc[0])
            train_last_epoch = float(df['time'].iloc[-1])
        except Exception:
            train_first_epoch = float('nan')
            train_last_epoch = float('nan')
        if _use_ctz:
            train_first_time = _format_time_minimal_local(train_first_epoch) if math.isfinite(train_first_epoch) else None
            train_last_time = _format_time_minimal_local(train_last_epoch) if math.isfinite(train_last_epoch) else None
        else:
            train_first_time = _format_time_minimal(train_first_epoch) if math.isfinite(train_first_epoch) else None
            train_last_time = _format_time_minimal(train_last_epoch) if math.isfinite(train_last_epoch) else None

        # Overall forecast trend based on net change over horizon
        try:
            if out_forecast_price.size >= 2:
                delta = float(out_forecast_price[-1] - out_forecast_price[0])
                # Use half a rounding unit as flat threshold
                prec = max(0, int(getattr(_info_before, "digits", 0) or 0))
                unit = 10.0 ** (-prec) if prec <= 12 else 0.0
                thresh = 0.5 * unit if unit > 0 else 0.0
                if delta > thresh:
                    forecast_trend = "up"
                elif delta < -thresh:
                    forecast_trend = "down"
                else:
                    forecast_trend = "flat"
            else:
                forecast_trend = "flat"
        except Exception:
            forecast_trend = "flat"
        payload: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method_l,
            "target": str(target),
            "params_used": params_used,
            "lookback_used": int(len(df)),
            "horizon": int(horizon),
            "seasonality_period": int(m or 0),
            "as_of": as_of or None,
            "train_start": train_first_time,
            "train_end": train_last_time,
            "times": times_fmt,
            "forecast_price": [_round(v) for v in out_forecast_price.tolist()],
        }
        payload["display_timezone"] = "client" if _use_ctz else "UTC"
        payload["forecast_trend"] = forecast_trend
        if use_returns:
            payload["forecast_return"] = [float(v) for v in f_vals.tolist()]
        if do_ci and lower_price is not None and upper_price is not None:
            payload["lower_price"] = [_round(v) for v in lower_price.tolist()]
            payload["upper_price"] = [_round(v) for v in upper_price.tolist()]
            payload["ci_alpha"] = float(_alpha)

        return payload
    except Exception as e:
        return {"error": f"Error computing forecast: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def fetch_market_depth(symbol: str, client_tz: str = "auto") -> Dict[str, Any]:
    """Return DOM if available; otherwise current bid/ask snapshot for `symbol`."""
    try:
        # Ensure symbol is selected
        if not mt5.symbol_select(symbol, True):
            return {"error": f"Failed to select symbol {symbol}: {mt5.last_error()}"}
        
        # Try to get market depth first
        depth = mt5.market_book_get(symbol)
        
        if depth is not None and len(depth) > 0:
            # Process DOM levels
            buy_orders = []
            sell_orders = []
            
            for level in depth:
                order_data = {
                    "price": float(level["price"]),
                    "volume": float(level["volume"]),
                    "volume_real": float(level["volume_real"])
                }
                
                if int(level["type"]) == 0:  # Buy order
                    buy_orders.append(order_data)
                else:  # Sell order
                    sell_orders.append(order_data)
            
            return {
                "success": True,
                "symbol": symbol,
                "type": "full_depth",
                "data": {
                    "buy_orders": buy_orders,
                    "sell_orders": sell_orders
                }
            }
        else:
            # DOM not available, fall back to symbol tick info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"error": f"Symbol {symbol} not found"}
            
            # Get current tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"error": f"Failed to get tick data for {symbol}"}
            
            out = {
                "success": True,
                "symbol": symbol,
                "type": "tick_data",
                "data": {
                    "bid": float(tick.bid) if tick.bid else None,
                    "ask": float(tick.ask) if tick.ask else None,
                    "last": float(tick.last) if tick.last else None,
                    "volume": int(tick.volume) if tick.volume else None,
                    "time": int(_mt5_epoch_to_utc(float(tick.time))) if tick.time else None,
                    "spread": symbol_info.spread,
                    "note": "Full market depth not available, showing current bid/ask"
                }
            }
            try:
                _use_ctz = _use_client_tz(client_tz)
                if tick.time and _use_ctz:
                    out["data"]["time_display"] = _format_time_minimal_local(_mt5_epoch_to_utc(float(tick.time)))
                elif tick.time:
                    out["data"]["time_display"] = _format_time_minimal(_mt5_epoch_to_utc(float(tick.time)))
            except Exception:
                pass
            out["display_timezone"] = "client" if _use_ctz else "UTC"
            return out
    except Exception as e:
        return {"error": f"Error getting market depth: {str(e)}"}

def main():
    """Main entry point for the MCP server"""
    mcp.run()

if __name__ == "__main__":
    main()

# Rebind selected helpers to extracted modules at import time to ensure all call sites
# use the split implementations (lookups occur at call time).
_csv_from_rows = _csv_from_rows_util
_format_time_minimal = _format_time_minimal_util
_format_time_minimal_local = _format_time_minimal_local_util
_use_client_tz = _use_client_tz_util
_resolve_client_tz = _resolve_client_tz_util
_time_format_from_epochs = _time_format_from_epochs_util
_maybe_strip_year = _maybe_strip_year_util
_style_time_format = _style_time_format_util
_optimal_decimals = _optimal_decimals_util
_format_float = _format_float_util
_format_numeric_rows_from_df = _format_numeric_rows_from_df_util
_parse_start_datetime = _parse_start_datetime_util
_simplify_dataframe_rows = _simplify_dataframe_rows_util
_list_ta_indicators = _list_ta_indicators_util
_parse_ti_specs = _parse_ti_specs_util
_apply_ta_indicators = _apply_ta_indicators_util
_estimate_warmup_bars = _estimate_warmup_bars_util
_apply_denoise = _apply_denoise_util
_extract_group_path = _extract_group_path_util
