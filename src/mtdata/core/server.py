#!/usr/bin/env python3
import logging
import atexit
import functools
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any, Dict, Optional, List, Tuple, Literal
from typing_extensions import TypedDict
import io
import os
import json
import csv
import time
import re
import inspect
import pandas as pd
import numpy as np
import math
import warnings
try:
    import pywt as _pywt  # type: ignore
except Exception:
    _pywt = None  # optional
try:
    from PyEMD import EMD as _EMD, EEMD as _EEMD, CEEMDAN as _CEEMDAN  # type: ignore
except Exception:
    _EMD = _EEMD = _CEEMDAN = None  # optional
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
    SIMPLIFY_DEFAULT_METHOD,
    SIMPLIFY_DEFAULT_MODE,
    SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT,
    SIMPLIFY_DEFAULT_RATIO,
    SIMPLIFY_DEFAULT_MIN_POINTS,
    SIMPLIFY_DEFAULT_MAX_POINTS,
    TIMEFRAME_MAP,
    TIMEFRAME_SECONDS,
    TIME_DISPLAY_FORMAT,
)
from .schema import (
    enrich_schema_with_shared_defs as _enrich_schema_with_shared_defs,
    build_minimal_schema as _build_minimal_schema,
    get_function_info as _get_function_info,
    complex_defs as _complex_defs,
)
mcp = FastMCP(SERVICE_NAME)

# Extracted indicator doc helpers
from .indicators_docs import list_ta_indicators as _list_ta_indicators_docs
from .indicators_docs import clean_help_text as _clean_help_text_docs
from .indicators_docs import infer_defaults_from_doc as _infer_defaults_from_doc_docs
 

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
from ..utils.symbol import _extract_group_path as _extract_group_path_util
from ..utils.indicators import (
    _list_ta_indicators as _list_ta_indicators_util,
    _parse_ti_specs as _parse_ti_specs_util,
    _apply_ta_indicators as _apply_ta_indicators_util,
    _estimate_warmup_bars as _estimate_warmup_bars_util,
)
from ..utils.denoise import _apply_denoise as _apply_denoise_util
from ..utils.patterns import PatternIndex as _PatternIndex, build_index as _build_pattern_index, _SeriesStore
from scipy.spatial.ckdtree import cKDTree
from ..patterns.classic import detect_classic_patterns as _detect_classic_patterns, ClassicDetectorConfig as _ClassicCfg
from ..utils.dimred import list_dimred_methods as _list_dimred_methods_util, create_reducer as _create_dimred_reducer
from ..utils.simplify import (
    _choose_simplify_points as _choose_simplify_points_util,
    _lttb_select_indices as _lttb_select_indices_util,
    _rdp_select_indices as _rdp_select_indices_util,
    _pla_select_indices as _pla_select_indices_util,
    _apca_select_indices as _apca_select_indices_util,
    _rdp_autotune_epsilon as _rdp_autotune_epsilon_util,
    _pla_autotune_max_error as _pla_autotune_max_error_util,
    _apca_autotune_max_error as _apca_autotune_max_error_util,
    _select_indices_for_timeseries as _select_indices_for_timeseries_util,
)

def _simplify_dataframe_rows_ext(df: pd.DataFrame, headers: List[str], simplify: Optional[Dict[str, Any]]):
    """Wrapper delegating row simplification to utils.simplify to reduce server size."""
    try:
        from ..utils.simplify import _simplify_dataframe_rows as _impl  # type: ignore
        return _impl(df, headers, simplify)
    except Exception:
        # Fallback to local implementation if util import fails
        return _simplify_dataframe_rows(df, headers, simplify)  # type: ignore

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

# In-memory cache for pattern indices keyed by (timeframe, window_size, future_size, denoise_key)
_PATTERN_INDEX_CACHE: Dict[Tuple[str, int, int, Optional[str]], _PatternIndex] = {}

def _index_cache_path(cache_dir: Optional[str], cache_id: Optional[str]) -> Optional[str]:
    if not cache_id:
        return None
    base = cache_dir or ".mtdata_cache"
    try:
        os.makedirs(os.path.join(base, "pattern_index"), exist_ok=True)
    except Exception:
        pass
    return os.path.join(base, "pattern_index", f"{cache_id}.pkl")

def _save_pattern_index_to_disk(idx: _PatternIndex, cache_dir: Optional[str], cache_id: Optional[str]) -> None:
    path = _index_cache_path(cache_dir, cache_id)
    if not path:
        return
    try:
        import pickle
        payload = {
            'timeframe': idx.timeframe,
            'window_size': int(idx.window_size),
            'future_size': int(idx.future_size),
            'symbols': list(idx.symbols),
            'X': idx.X.astype(np.float32, copy=False),
            'start_end_idx': np.asarray(idx.start_end_idx, dtype=np.int32),
            'labels': np.asarray(idx.labels, dtype=np.int32),
            'series': [
                {
                    'symbol': s.symbol,
                    'time_epoch': np.asarray(s.time_epoch, dtype=np.float64),
                    'close': np.asarray(s.close, dtype=np.float64),
                } for s in getattr(idx, '_series', [])
            ],
            'scale': idx.scale,
            'metric': idx.metric,
            'pca_components': idx.pca_components,
            'dimred_method': getattr(idx, 'dimred_method', 'none'),
            'dimred_params': getattr(idx, 'dimred_params', {}),
            'dimred_reducer': getattr(idx, '_reducer', None),
            'engine': idx.engine,
            'max_bars_per_symbol': int(getattr(idx, 'max_bars_per_symbol', 0)),
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass

def _load_pattern_index_from_disk(cache_dir: Optional[str], cache_id: Optional[str]) -> Optional[_PatternIndex]:
    path = _index_cache_path(cache_dir, cache_id)
    if not path or not os.path.isfile(path):
        return None
    try:
        import pickle
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        # Rebuild tree (use cKDTree)
        X = np.asarray(payload['X'], dtype=np.float32)
        tree = cKDTree(X)
        series = []
        for sd in payload.get('series', []):
            series.append(_SeriesStore(symbol=sd['symbol'], time_epoch=np.asarray(sd['time_epoch'], dtype=np.float64), close=np.asarray(sd['close'], dtype=np.float64)))
        idx = _PatternIndex(
            timeframe=str(payload['timeframe']),
            window_size=int(payload['window_size']),
            future_size=int(payload['future_size']),
            symbols=list(payload['symbols']),
            tree=tree,
            X=X,
            start_end_idx=np.asarray(payload['start_end_idx'], dtype=np.int32),
            labels=np.asarray(payload['labels'], dtype=np.int32),
            series=series,
            scale=str(payload.get('scale','minmax')),
            metric=str(payload.get('metric','euclidean')),
            pca_components=payload.get('pca_components'),
            pca_model=None,
            dimred_method=str(payload.get('dimred_method','none')),
            dimred_params=dict(payload.get('dimred_params', {})),
            reducer=payload.get('dimred_reducer'),
            engine='ckdtree',
            max_bars_per_symbol=int(payload.get('max_bars_per_symbol', 0)),
        )
        return idx
    except Exception:
        return None

def _denoise_cache_key(obj: Optional[Dict[str, Any]]) -> Optional[str]:
    if not obj or not isinstance(obj, dict):
        return None
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(obj)


# Flexible datetime parsing helper using dateparser
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








def _time_format_from_epochs_util_local(epochs: List[float]) -> str:
    """Normalized local/client datetime format (constant for consistency)."""
    return TIME_DISPLAY_FORMAT

def _maybe_strip_year_util_local(fmt: str, epochs: List[float]) -> str:
    """No-op to keep full year for normalized outputs."""
    return fmt



# ---- Numeric formatting helpers ----

def _coerce_scalar(val: str):
    """Coerce a string scalar to int/float when possible; else return original.

    Used to serialize indicator params from structured input into a compact spec string.
    """
    try:
        s = str(val).strip()
    except Exception:
        return val
    if not s:
        return s
    # Try integer
    try:
        # Detect pure integer tokens (including + / -)
        if re.match(r"^[+-]?\d+$", s):
            return int(s)
    except Exception:
        pass
    # Try float
    try:
        return float(s)
    except Exception:
        pass
    # Common boolean-like tokens to numeric flags
    sl = s.lower()
    if sl in ("true", "false"):
        return 1 if sl == "true" else 0
    if sl in ("none", "null"):
        return 0
    return s

# ---- Timeseries simplification helpers ----
def _lttb_select_indices(x: List[float], y: List[float], n_out: int) -> List[int]:
    return _lttb_select_indices_util(x, y, n_out)


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
    from ..utils.simplify import _point_line_distance as _impl
    return _impl(px, py, x1, y1, x2, y2)


def _rdp_select_indices(x: List[float], y: List[float], epsilon: float) -> List[int]:
    return _rdp_select_indices_util(x, y, epsilon)


def _max_line_error(x: List[float], y: List[float], i0: int, i1: int) -> float:
    from ..utils.simplify import _max_line_error as _impl
    return _impl(x, y, i0, i1)


def _pla_select_indices(x: List[float], y: List[float], max_error: Optional[float] = None, segments: Optional[int] = None, points: Optional[int] = None) -> List[int]:
    return _pla_select_indices_util(x, y, max_error, segments, points)


def _apca_select_indices(y: List[float], max_error: Optional[float] = None, segments: Optional[int] = None, points: Optional[int] = None) -> List[int]:
    return _apca_select_indices_util(y, max_error, segments, points)


def _select_indices_for_timeseries(x: List[float], y: List[float], spec: Optional[Dict[str, Any]]) -> Tuple[List[int], str, Dict[str, Any]]:
    return _select_indices_for_timeseries_util(x, y, spec)


def _rdp_autotune_epsilon(x: List[float], y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    return _rdp_autotune_epsilon_util(x, y, target_points, max_iter)


def _pla_autotune_max_error(x: List[float], y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    return _pla_autotune_max_error_util(x, y, target_points, max_iter)


def _apca_autotune_max_error(y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    return _apca_autotune_max_error_util(y, target_points, max_iter)


def _simplify_dataframe_rows(df: pd.DataFrame, headers: List[str], simplify: Optional[Dict[str, Any]]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Reduce or transform rows across numeric columns.

    Modes (simplify['mode']):
    - 'select' (default): pick representative existing rows using the chosen method.
    - 'approximate': partition by selected breakpoints and aggregate numeric columns per segment.
    - 'resample': time-based bucketing by '__epoch' with 'bucket_seconds'; aggregates numeric columns.
    - 'encode': transform per-row representation to a compact schema (e.g., envelope or delta) and
                optionally pre-select rows before encoding.

    Aggregation: mean for numeric columns; first value for non-numeric columns like 'time'.
    """
    if not simplify:
        return df, None
    try:
        total = len(df)
        if total <= 3:
            return df, None
        method = str(simplify.get("method", SIMPLIFY_DEFAULT_METHOD)).lower().strip()
        mode = str(simplify.get("mode", SIMPLIFY_DEFAULT_MODE)).lower().strip() or SIMPLIFY_DEFAULT_MODE
        # If users passed a high-level mode via --simplify (CLI maps to 'method'), map it to mode
        if method in ("encode", "symbolic", "segment"):
            explicit_mode = str(simplify.get("mode", "")).lower().strip()
            if explicit_mode in ("", SIMPLIFY_DEFAULT_MODE, "select"):
                mode = method

        # Helper: numeric columns in requested headers order, then any others
        def _numeric_columns_from_headers() -> List[str]:
            cols: List[str] = []
            for h in headers:
                if h in ('time',) or str(h).startswith('_'):
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

        # Aggregation helper for approximate/resample modes
        def _aggregate_segment(i0: int, i1: int) -> Dict[str, Any]:
            seg = df.iloc[i0:i1]
            row: Dict[str, Any] = {}
            if "time" in df.columns and "time" in headers:
                row["time"] = seg.iloc[0]["time"]
            for col in headers:
                if col == "time":
                    continue
                if col in seg.columns and pd.api.types.is_numeric_dtype(seg[col]):
                    row[col] = float(seg[col].mean())
                elif col in seg.columns:
                    try:
                        row[col] = next((v for v in seg[col].tolist() if pd.notna(v)), seg.iloc[0][col])
                    except Exception:
                        row[col] = seg.iloc[0][col]
            return row

        # Determine target number of points early (used for select and to infer resample/encode)
        n_out = _choose_simplify_points(total, simplify)
        if mode == "resample" and "__epoch" in df.columns:
            bs = simplify.get("bucket_seconds")
            if bs is None or (isinstance(bs, (int, float)) and bs <= 0):
                try:
                    # Infer bucket by total time span / target buckets
                    t0 = float(df["__epoch"].iloc[0])
                    t1 = float(df["__epoch"].iloc[-1])
                    span = max(1.0, t1 - t0)
                    bs = max(1, int(round(span / max(1, n_out))))
                except Exception:
                    # Fallback to rough bucket from count
                    bs = max(1, int(round(total / float(max(1, n_out)))))
            try:
                bs = max(1, int(bs))
            except Exception:
                bs = max(1, int(round(total / float(max(1, n_out)))))
            grp = ((df["__epoch"].astype(float) - float(df["__epoch"].iloc[0])) // bs).astype(int)
            out_rows: List[Dict[str, Any]] = []
            for _, seg in df.groupby(grp):
                i0 = seg.index[0]
                i1 = seg.index[-1] + 1
                out_rows.append(_aggregate_segment(i0, i1))
            out_df = pd.DataFrame(out_rows)
            meta = {
                "mode": "resample",
                "method": method or SIMPLIFY_DEFAULT_METHOD,
                "bucket_seconds": int(bs),
                "original_rows": total,
                "returned_rows": len(out_df),
                "points": len(out_df),
            }
            return out_df.reset_index(drop=True), meta

        # Encode mode: optionally select rows first, then encode candle columns
        if mode == "encode":
            # Determine encoding schema and parameters
            schema = str(simplify.get("schema", "envelope")).lower().strip()

            # Helper: choose selection indices using multi-column union (close prioritized if present)
            def _preselect_indices() -> List[int]:
                if n_out >= total:
                    return list(range(total))
                x = [float(v) for v in df['__epoch'].tolist()]
                # Prefer main candle columns if present, else any numeric
                preferred = [c for c in ["close", "open", "high", "low"] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                cols = preferred or [c for c in df.columns if c not in ("time", "__epoch") and pd.api.types.is_numeric_dtype(df[c])]
                if not cols:
                    return list(range(total))
                base_points = max(3, int(round(n_out / max(1, len(cols)))))
                idx_set: set = set([0, total - 1])
                for c in cols:
                    y = [float(v) if v is not None and not pd.isna(v) else 0.0 for v in df[c].tolist()]
                    sub_spec = dict(simplify)
                    sub_spec['points'] = base_points
                    idxs, _, _ = _select_indices_for_timeseries(x, y, sub_spec)
                    for i in idxs:
                        if 0 <= int(i) < total:
                            idx_set.add(int(i))
                idxs_union = sorted(idx_set)
                if len(idxs_union) == n_out:
                    return idxs_union
                # Refine/top-up using composite
                mins: Dict[str, float] = {}
                ranges: Dict[str, float] = {}
                for c in cols:
                    arr = [float(v) if v is not None and not pd.isna(v) else 0.0 for v in df[c].tolist()]
                    if arr:
                        mn, mx = min(arr), max(arr)
                        ranges[c] = max(1e-12, mx - mn)
                        mins[c] = mn
                    else:
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
                    return sorted(set(refined))
                elif len(idxs_union) < n_out:
                    refined = _lttb_select_indices(x, comp, n_out)
                    merged = sorted(set(idxs_union).union(refined))
                    if len(merged) > n_out:
                        keep = set([0, total - 1])
                        cand = [(comp[i], i) for i in merged if i not in keep]
                        cand.sort(reverse=True)
                        for _, i in cand:
                            keep.add(i)
                            if len(keep) >= n_out:
                                break
                        return sorted(keep)
                    return merged
                return idxs_union

            idxs_final = _preselect_indices()
            out = df.iloc[idxs_final].copy() if len(idxs_final) < total else df.copy()

            # Envelope schema: keep low/high, encode open/close as compact positions
            if schema in ("envelope", "env"):
                try:
                    bits = int(simplify.get("bits", 8))
                except Exception:
                    bits = 8
                # Optional character alphabet to minimize UTF-8 characters
                alphabet = simplify.get("alphabet") or "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                alphabet = str(alphabet)
                levels_bits = max(2, int(2 ** bits))
                levels_alpha = max(2, len(alphabet))
                levels = min(levels_bits, levels_alpha)
                as_chars = bool(simplify.get("as_chars") or simplify.get("chars"))
                # Ensure required columns exist
                needed = ["open", "high", "low", "close"]
                if not all(col in out.columns for col in needed):
                    return out, None
                lo = out["low"].astype(float)
                hi = out["high"].astype(float)
                rng = (hi - lo).replace(0, np.nan)
                o_pos = (((out["open"].astype(float) - lo) / rng).clip(0.0, 1.0).fillna(0.5) * (levels - 1)).round().astype(int)
                c_pos = (((out["close"].astype(float) - lo) / rng).clip(0.0, 1.0).fillna(0.5) * (levels - 1)).round().astype(int)
                # Build compact dataframe keeping all requested non-replaced columns
                if as_chars:
                    # Map positions to single characters from the alphabet (index modulo available)
                    def _map_char(v: int) -> str:
                        try:
                            idx = int(v) % len(alphabet)
                            return alphabet[idx]
                        except Exception:
                            return alphabet[0]
                    o_ser = o_pos.apply(_map_char)
                    c_ser = c_pos.apply(_map_char)
                    o_name, c_name = "o", "c"
                else:
                    o_ser = o_pos
                    c_ser = c_pos
                    o_name, c_name = "o_pos", "c_pos"
                base_cols = ["time", "low", "high", o_name, c_name]
                replaced = {"open", "close"}  # replaced by positions
                # Start with base
                out_df = pd.DataFrame({
                    "time": out["time"] if "time" in out.columns else out.index.astype(str),
                    "low": lo,
                    "high": hi,
                    o_name: o_ser,
                    c_name: c_ser,
                })
                # Append all other requested headers (TI, volumes, spread, etc.) in requested order
                extra_cols: List[str] = []
                for h in headers:
                    if h in ("time", "low", "high"):
                        continue  # already included
                    if h in replaced:
                        continue
                    if h in out.columns and h not in extra_cols:
                        extra_cols.append(h)
                for c in extra_cols:
                    out_df[c] = out[c]
                meta = {
                    "mode": "encode",
                    "schema": "envelope",
                    "bits": bits,
                    "levels": levels,
                    "as_chars": as_chars,
                    "alphabet_len": len(alphabet) if as_chars else None,
                    "original_rows": total,
                    "returned_rows": len(out_df),
                    "points": len(out_df),
                    "headers": base_cols + extra_cols,
                }
                return out_df.reset_index(drop=True), meta

            # Delta schema: emit integer deltas vs previous close with scaling
            if schema in ("delta", "deltas"):
                # Determine scale (tick size); default tries to infer from price magnitude
                try:
                    scale = float(simplify.get("scale", 1e-5))
                except Exception:
                    scale = 1e-5
                if scale <= 0:
                    scale = 1e-5
                # Optional char mode for minimal UTF-8 characters
                alphabet = simplify.get("alphabet") or "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                alphabet = str(alphabet)
                zero_char = str(simplify.get("zero_char", "."))
                as_chars = bool(simplify.get("as_chars") or simplify.get("chars"))
                needed = ["open", "high", "low", "close"]
                if not all(col in out.columns for col in needed):
                    return out, None
                closes = out["close"].astype(float).tolist()
                opens = out["open"].astype(float).tolist()
                highs = out["high"].astype(float).tolist()
                lows = out["low"].astype(float).tolist()
                base_close = float(closes[0]) if closes else float('nan')
                d_open: List[int] = []
                d_high: List[int] = []
                d_low: List[int] = []
                d_close: List[int] = []
                prev_close = base_close
                for i in range(len(out)):
                    d_open.append(int(round((float(opens[i]) - prev_close) / scale)))
                    d_high.append(int(round((float(highs[i]) - prev_close) / scale)))
                    d_low.append(int(round((float(lows[i]) - prev_close) / scale)))
                    d_close.append(int(round((float(closes[i]) - prev_close) / scale)))
                    prev_close = float(closes[i])
                # Optionally map to compact base-N strings with sign, minimizing characters
                def _to_base_str(v: int) -> str:
                    try:
                        if v == 0:
                            return zero_char
                        sign = '-' if v < 0 else '+'
                        n = abs(int(v))
                        digits = []
                        base = max(2, len(alphabet))
                        while n > 0:
                            digits.append(alphabet[n % base])
                            n //= base
                        return sign + ''.join(reversed(digits))
                    except Exception:
                        return '0'

                if as_chars:
                    do_ser = [ _to_base_str(v) for v in d_open ]
                    dh_ser = [ _to_base_str(v) for v in d_high ]
                    dl_ser = [ _to_base_str(v) for v in d_low ]
                    dc_ser = [ _to_base_str(v) for v in d_close ]
                else:
                    do_ser, dh_ser, dl_ser, dc_ser = d_open, d_high, d_low, d_close

                base_cols = ["time", "d_open", "d_high", "d_low", "d_close"]
                out_df = pd.DataFrame({
                    "time": out["time"] if "time" in out.columns else out.index.astype(str),
                    "d_open": do_ser,
                    "d_high": dh_ser,
                    "d_low": dl_ser,
                    "d_close": dc_ser,
                })
                # Keep all other requested headers (TI, volume, spread) except original OHLC
                replaced = {"open", "high", "low", "close"}
                extra_cols: List[str] = []
                for h in headers:
                    if h == "time" or h in replaced:
                        continue
                    if h in out.columns and h not in extra_cols:
                        extra_cols.append(h)
                for c in extra_cols:
                    out_df[c] = out[c]
                meta = {
                    "mode": "encode",
                    "schema": "delta",
                    "scale": scale,
                    "base_close": base_close,
                    "as_chars": as_chars,
                    "alphabet_len": len(alphabet) if as_chars else None,
                    "zero_char": zero_char if as_chars else None,
                    "original_rows": total,
                    "returned_rows": len(out_df),
                    "points": len(out_df),
                    "headers": base_cols + extra_cols,
                }
                return out_df.reset_index(drop=True), meta

            # Unknown schema: no-op
            return out, None

        # Segment mode: ZigZag turning points
        if mode == "segment":
            algo = str(simplify.get("algo", "zigzag")).lower().strip()
            if algo in ("zigzag", "zz"):
                try:
                    thr = simplify.get("threshold_pct", simplify.get("threshold", 0.5))
                    try:
                        threshold_pct = float(thr)
                    except Exception:
                        threshold_pct = 0.5
                    # Choose value column
                    val_col = simplify.get("value_col") or ("close" if "close" in df.columns and pd.api.types.is_numeric_dtype(df["close"]) else None)
                    if not val_col:
                        for c in df.columns:
                            if c in ("time", "__epoch"):
                                continue
                            try:
                                if pd.api.types.is_numeric_dtype(df[c]):
                                    val_col = c
                                    break
                            except Exception:
                                pass
                    if not val_col:
                        return df, None
                    y = df[val_col].astype(float).tolist()
                    times = df["time"].tolist() if "time" in df.columns else list(range(len(df)))
                    n = len(y)
                    if n < 2:
                        return df, None
                    piv_idx: List[int] = []
                    piv_dir: List[str] = []
                    # Initialize pivot and extremes
                    pivot_i = 0
                    pivot_price = float(y[0])
                    trend = None  # 'up' or 'down'
                    last_ext_i = 0
                    last_ext_p = float(y[0])
                    for i in range(1, n):
                        p = float(y[i])
                        if trend is None:
                            change = (p - pivot_price) / pivot_price * 100.0 if pivot_price != 0 else 0.0
                            if abs(change) >= threshold_pct:
                                trend = 'up' if change > 0 else 'down'
                                last_ext_i = i
                                last_ext_p = p
                                # first pivot at start
                                piv_idx.append(pivot_i)
                                piv_dir.append(trend)
                            else:
                                if p > last_ext_p:
                                    last_ext_p = p
                                    last_ext_i = i
                                if p < last_ext_p and trend is None:
                                    last_ext_p = p
                                    last_ext_i = i
                                continue
                        if trend == 'up':
                            if p > last_ext_p:
                                last_ext_p = p
                                last_ext_i = i
                            retr = (last_ext_p - p) / last_ext_p * 100.0 if last_ext_p != 0 else 0.0
                            if retr >= threshold_pct:
                                piv_idx.append(last_ext_i)
                                piv_dir.append('up')
                                trend = 'down'
                                pivot_i = i
                                pivot_price = p
                                last_ext_i = i
                                last_ext_p = p
                        else:  # down
                            if p < last_ext_p:
                                last_ext_p = p
                                last_ext_i = i
                            retr = (p - last_ext_p) / abs(last_ext_p) * 100.0 if last_ext_p != 0 else 0.0
                            if retr >= threshold_pct:
                                piv_idx.append(last_ext_i)
                                piv_dir.append('down')
                                trend = 'up'
                                pivot_i = i
                                pivot_price = p
                                last_ext_i = i
                                last_ext_p = p
                    # Append final extreme
                    if len(piv_idx) == 0 or piv_idx[-1] != last_ext_i:
                        piv_idx.append(last_ext_i)
                        piv_dir.append('up' if trend == 'up' else 'down' if trend == 'down' else 'flat')
                    # Preserve all original columns by subsetting DF at pivot indices, then add ZigZag extras
                    out_df = df.iloc[piv_idx].copy()
                    # Add ZigZag metadata columns
                    zz_values: List[float] = []
                    prev_v = None
                    changes: List[float] = []
                    for j, i in enumerate(piv_idx):
                        v = float(y[i])
                        zz_values.append(v)
                        changes.append(0.0 if prev_v is None else ((v - prev_v) / prev_v * 100.0 if prev_v != 0 else 0.0))
                        prev_v = v
                    out_df['value'] = zz_values
                    out_df['direction'] = piv_dir
                    out_df['change_pct'] = changes
                    # Headers: keep all requested headers, plus ZigZag extras
                    hdrs = list(headers)
                    for extra in ('value','direction','change_pct'):
                        if extra not in hdrs:
                            hdrs.append(extra)
                    meta = {
                        'mode': 'segment',
                        'algo': 'zigzag',
                        'threshold_pct': threshold_pct,
                        'value_col': val_col,
                        'original_rows': total,
                        'returned_rows': len(out_df),
                        'points': len(out_df),
                        'headers': hdrs,
                    }
                    return out_df.reset_index(drop=True), meta
                except Exception:
                    return df, None

        # Symbolic mode: SAX string output per segment
        if mode == "symbolic":
            schema = str(simplify.get("schema", "sax")).lower().strip()
            if schema == 'sax':
                try:
                    try:
                        paa = int(simplify.get('paa', max(8, min(32, n_out))))
                    except Exception:
                        paa = max(8, min(32, n_out))
                    if paa <= 0:
                        paa = max(8, min(32, n_out))
                    # Choose value column
                    val_col = simplify.get("value_col") or ("close" if "close" in df.columns and pd.api.types.is_numeric_dtype(df["close"]) else None)
                    if not val_col:
                        for c in df.columns:
                            if c in ("time", "__epoch"):
                                continue
                            if pd.api.types.is_numeric_dtype(df[c]):
                                val_col = c
                                break
                    if not val_col:
                        return df, None
                    s = df[val_col].astype(float).to_numpy(copy=False)
                    n = len(s)
                    if n == 0:
                        return df, None
                    znorm = bool(simplify.get('znorm', True))
                    x = s.copy()
                    if znorm:
                        mu = float(np.mean(x))
                        sigma = float(np.std(x))
                        if sigma > 0:
                            x = (x - mu) / sigma
                        else:
                            x = x * 0.0
                    # Build contiguous PAA segments
                    seg_sizes = [n // paa] * paa
                    for k in range(n % paa):
                        seg_sizes[k] += 1
                    idx = 0
                    seg_means: List[float] = []
                    seg_bounds: List[Tuple[int,int]] = []
                    for sz in seg_sizes:
                        if sz <= 0:
                            continue
                        j = min(n, idx + sz)
                        if idx >= n:
                            break
                        seg_means.append(float(np.mean(x[idx:j])))
                        seg_bounds.append((idx, j-1))
                        idx = j
                    alphabet = str(simplify.get('alphabet') or 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    a = max(2, len(alphabet))
                    # Breakpoints for standard normal quantiles (approximate inverse CDF)
                    from math import sqrt
                    def _norm_ppf(q: float) -> float:
                        # Acklam rational approximation constants
                        # For robustness and readability
                        a1=-39.69683028665376; a2=220.9460984245205; a3=-275.9285104469687
                        a4=138.3577518672690; a5=-30.66479806614716; a6=2.506628277459239
                        b1=-54.47609879822406; b2=161.5858368580409; b3=-155.6989798598866
                        b4=66.80131188771972; b5=-13.28068155288572
                        c1=-0.007784894002430293; c2=-0.3223964580411365; c3=-2.400758277161838
                        c4=-2.549732539343734; c5=4.374664141464968; c6=2.938163982698783
                        d1=0.007784695709041462; d2=0.3224671290700398; d3=2.445134137142996; d4=3.754408661907416
                        plow=0.02425; phigh=1-plow
                        if q < plow:
                            ql = math.sqrt(-2.0*math.log(q))
                            return (((((c1*ql+c2)*ql+c3)*ql+c4)*ql+c5)*ql+c6)/((((d1*ql+d2)*ql+d3)*ql+d4)*ql+1)
                        if q > phigh:
                            ql = math.sqrt(-2.0*math.log(1.0-q))
                            return -(((((c1*ql+c2)*ql+c3)*ql+c4)*ql+c5)*ql+c6)/((((d1*ql+d2)*ql+d3)*ql+d4)*ql+1)
                        ql = q-0.5
                        r = ql*ql
                        return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*ql/(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)
                    bps = [ _norm_ppf((k+1)/a) for k in range(a-1) ]
                    # Map means to symbols
                    def _symbol_for(v: float) -> str:
                        k = 0
                        for bp in bps:
                            if v > bp:
                                k += 1
                            else:
                                break
                        return alphabet[k]
                    symbols = [ _symbol_for(m) for m in seg_means ]
                    # Build output rows per segment, preserving all requested columns via aggregation
                    times = df['time'].tolist() if 'time' in df.columns else [str(i) for i in range(n)]
                    rows: List[Dict[str, Any]] = []
                    for seg_idx, ((i0, i1), sym, mean_val) in enumerate(zip(seg_bounds, symbols, seg_means), start=1):
                        row = {
                            'segment': seg_idx,
                            'start_time': times[i0],
                            'end_time': times[i1],
                            'symbol': sym,
                            'paa_mean': mean_val,
                        }
                        # Aggregate all requested headers (except 'time') for this segment
                        try:
                            agg_row = _aggregate_segment(i0, i1 + 1)
                            for k, v in agg_row.items():
                                if k == 'time':
                                    continue
                                row[k] = v
                        except Exception:
                            pass
                        rows.append(row)
                    out_df = pd.DataFrame(rows)
                    meta = {
                        'mode': 'symbolic',
                        'schema': 'sax',
                        'paa': len(seg_bounds),
                        'alphabet_len': a,
                        'znorm': znorm,
                        'original_rows': total,
                        'returned_rows': len(out_df),
                        'points': len(out_df),
                        'headers': ['segment','start_time','end_time','symbol','paa_mean'] + [h for h in headers if h != 'time'],
                    }
                    return out_df.reset_index(drop=True), meta
                except Exception:
                    return df, None


        # Default/select: multi-column selection with union + refinement
        x = [float(v) for v in df['__epoch'].tolist()]
        cols = _numeric_columns_from_headers()
        if not cols:
            return df, None
        if n_out >= total:
            return df, None
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
        # Composite normalized series for refinement/top-up
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

        if mode == "approximate":
            # Aggregate between consecutive selected indices
            segments: List[Tuple[int, int]] = []
            for a, b in zip(idxs_final[:-1], idxs_final[1:]):
                if b > a:
                    segments.append((a, b))
            if not segments:
                out_df = df.iloc[idxs_final].reset_index(drop=True)
                meta = {
                    "mode": "approximate",
                    "method": method_used_overall or method or "lttb",
                    "original_rows": total,
                    "returned_rows": len(out_df),
                }
                try:
                    if params_meta_overall:
                        meta.update(params_meta_overall)
                except Exception:
                    pass
                return out_df, meta
            rows: List[Dict[str, Any]] = []
            for a, b in segments:
                rows.append(_aggregate_segment(a, b))
            out_df = pd.DataFrame(rows)
            meta = {
                "mode": "approximate",
                "method": method_used_overall or method or SIMPLIFY_DEFAULT_METHOD,
                "original_rows": total,
                "returned_rows": len(out_df),
                "points": len(out_df),
            }
            try:
                if params_meta_overall:
                    meta.update(params_meta_overall)
            except Exception:
                pass
            return out_df.reset_index(drop=True), meta

        # Default 'select' mode
        reduced = df.iloc[idxs_final].copy()
        meta = {
            "mode": "select",
            "method": (method_used_overall or method or SIMPLIFY_DEFAULT_METHOD),
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
        # Normalize points to actual returned rows for clarity
        meta["points"] = len(reduced)
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
    """Delegate to indicators_docs.list_ta_indicators() for discovery."""
    return _list_ta_indicators_docs()


def _infer_defaults_from_doc(func_name: str, doc_text: str, params: List[Dict[str, Any]]):
    return _infer_defaults_from_doc_docs(func_name, doc_text, params)


def _try_number(s: str):
    from .indicators_docs import _try_number as _tn
    return _tn(s)


def _clean_help_text(text: str, func_name: Optional[str] = None, func: Optional[Any] = None) -> str:
    return _clean_help_text_docs(text, func_name=func_name)










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

# ---- Simplify (schema for MCP) ----
_SIMPLIFY_MODES = (
    'select',        # pick representative existing rows
    'approximate',   # aggregate between selected rows
    'resample',      # time-bucket aggregation
    'encode',        # compact encodings (envelope, delta)
    'segment',       # swing points (e.g., ZigZag)
    'symbolic',      # SAX symbolic representation
)
_SIMPLIFY_METHODS = (
    'lttb', 'rdp', 'pla', 'apca'
)
try:
    SimplifyModeLiteral = Literal[_SIMPLIFY_MODES]  # type: ignore
except Exception:
    SimplifyModeLiteral = str
try:
    SimplifyMethodLiteral = Literal[_SIMPLIFY_METHODS]  # type: ignore
except Exception:
    SimplifyMethodLiteral = str
try:
    EncodeSchemaLiteral = Literal['envelope','delta']  # type: ignore
    SymbolicSchemaLiteral = Literal['sax']  # type: ignore
except Exception:
    EncodeSchemaLiteral = str
    SymbolicSchemaLiteral = str

class SimplifySpec(TypedDict, total=False):
    # Common
    mode: SimplifyModeLiteral  # type: ignore
    method: SimplifyMethodLiteral  # type: ignore
    points: int
    ratio: float
    # RDP/PLA/APCA specifics
    epsilon: float
    max_error: float
    segments: int
    # Resample
    bucket_seconds: int
    # Encode specifics
    schema: EncodeSchemaLiteral  # 'envelope' | 'delta' (or 'sax' when mode='symbolic')
    bits: int
    as_chars: bool
    alphabet: str
    scale: float
    zero_char: str
    # Segment specifics
    algo: Literal['zigzag','zz']  # type: ignore
    threshold_pct: float
    value_col: str
    # Symbolic specifics
    paa: int
    znorm: bool

# Volatility params (concise)
class VolatilityParams(TypedDict, total=False):
    # EWMA
    halflife: Optional[float]
    lambda_: Optional[float]  # use 'lambda_' to avoid reserved word in schema
    lookback: int
    # Parkinson/GK/RS
    window: int
    # GARCH
    fit_bars: int
    mean: Literal['Zero','Constant']  # type: ignore
    dist: Literal['normal']  # type: ignore

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


def _get_denoise_methods_data_safe() -> Dict[str, Any]:
    try:
        from ..utils.denoise import get_denoise_methods_data
        return get_denoise_methods_data()
    except Exception as e:
        return {"error": f"Error listing denoise methods: {e}"}

def _get_forecast_methods_data_safe() -> Dict[str, Any]:
    try:
        from ..utils.forecast import get_forecast_methods_data
        return get_forecast_methods_data(_SM_ETS_AVAILABLE, _SM_SARIMAX_AVAILABLE, _NF_AVAILABLE, _SF_AVAILABLE, _MLF_AVAILABLE, _CHRONOS_AVAILABLE, _TIMESFM_AVAILABLE, _LAG_LLAMA_AVAILABLE, _ARCH_AVAILABLE)
    except Exception as e:
        return {"error": f"Error listing forecast methods: {e}"}

def _get_volatility_methods_data() -> Dict[str, Any]:
    try:
        direct: List[Dict[str, Any]] = []
        def add_direct(name: str, available: bool, description: str, params: Dict[str, str]):
            direct.append({
                'method': name,
                'available': bool(available),
                'description': description,
                'params': params,
            })
        add_direct('ewma', True, 'RiskMetrics EWMA on close-to-close returns.', {
            'lookback': 'int (default 1500)', 'halflife': 'float?', 'lambda_': 'float (default 0.94)'
        })
        add_direct('parkinson', True, 'Range-based variance (H/L).', {'window': 'int (default 20)'})
        add_direct('gk', True, 'Garman–Klass variance (O/H/L/C).', {'window': 'int (default 20)'})
        add_direct('rs', True, 'Rogers–Satchell variance (O/H/L/C).', {'window': 'int (default 20)'})
        add_direct('yang_zhang', True, 'Yang–Zhang daily variance (O/H/L/C).', {'window': 'int (default 20)'})
        add_direct('rolling_std', True, 'Rolling variance of returns.', {'window': 'int (default 20)'})
        add_direct('garch', _ARCH_AVAILABLE, 'GARCH(1,1) via arch on percent returns.', {
            'fit_bars': 'int (default 2000)', 'mean': 'Zero|Constant', 'dist': 'normal|t'
        })
        add_direct('egarch', _ARCH_AVAILABLE, 'EGARCH(p,q) via arch.', {
            'fit_bars': 'int (default 2000)', 'p': 'int (default 1)', 'q': 'int (default 1)', 'mean': 'Zero|Constant', 'dist': 'normal|t'
        })
        add_direct('gjr_garch', _ARCH_AVAILABLE, 'GJR-GARCH(p,o,q) via arch.', {
            'fit_bars': 'int (default 2000)', 'p': 'int (default 1)', 'o': 'int (default 1)', 'q': 'int (default 1)', 'mean': 'Zero|Constant', 'dist': 'normal|t'
        })

        proxy_models: List[Dict[str, Any]] = []
        def add_proxy(name: str, available: bool, description: str, params: Dict[str, str]):
            proxy_models.append({
                'method': name,
                'available': bool(available),
                'description': description,
                'requires_proxy': True,
                'supported_proxies': ['squared_return','abs_return','log_r2'],
                'params': params,
            })
        add_proxy('arima', _SM_SARIMAX_AVAILABLE, 'ARIMA on volatility proxy (SARIMAX).', {'p':'int','d':'int','q':'int'})
        add_proxy('sarima', _SM_SARIMAX_AVAILABLE, 'Seasonal ARIMA on volatility proxy.', {'p':'int','d':'int','q':'int','P':'int','D':'int','Q':'int'})
        add_proxy('ets', _SM_ETS_AVAILABLE, 'ETS on volatility proxy.', {})
        add_proxy('theta', True, 'Theta-style smoother on volatility proxy.', {'alpha':'float (default 0.2)'})
        add_proxy('mlf_rf', _MLF_AVAILABLE, 'MLForecast RandomForest on volatility proxy.', {'lags':'list[int]','n_estimators':'int'})
        add_proxy('nhits', _NF_AVAILABLE, 'NeuralForecast NHITS on volatility proxy.', {'max_epochs':'int','input_size':'int','batch_size':'int'})

        return {
            'success': True,
            'schema_version': 1,
            'direct_models': direct,
            'proxy_models': proxy_models,
        }
    except Exception as e:
        return {'error': f'Error listing volatility methods: {e}'}

@mcp.tool()
def list_indicators(search_term: Optional[str] = None, category: Optional[CategoryLiteral] = None) -> Dict[str, Any]:  # type: ignore
    """List indicators as CSV with columns: name,category. Optional filters: search_term, category.

    Parameters: search_term?, category?
    """
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
        return _csv_from_rows_util(["name", "category"], rows)
    except Exception as e:
        return {"error": f"Error listing indicators: {e}"}


# Note: category annotation is set at definition time above to be captured in the MCP schema

@mcp.tool()
def describe_indicator(name: IndicatorNameLiteral) -> Dict[str, Any]:  # type: ignore
    """Return detailed indicator information (name, category, params, description).

    Parameters: name
    """
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

    Parameters: search_term?, limit?

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
                group_path = _extract_group_path_util(symbol)
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
                "group": _extract_group_path_util(symbol),
                "description": symbol.description,
            })
        
        # Apply limit
        if limit and limit > 0:
            symbol_list = symbol_list[:limit]
        # Convert to CSV format using proper escaping
        rows = [[s["name"], s["group"], s["description"]] for s in symbol_list]
        return _csv_from_rows_util(["name", "group", "description"], rows)
    except Exception as e:
        return {"error": f"Error getting symbols: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def list_symbol_groups(search_term: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """List group paths as CSV with a single column: group.

    Parameters: search_term?, limit?

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
            group_path = _extract_group_path_util(symbol)
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
        return _csv_from_rows_util(["group"], rows)
    except Exception as e:
        return {"error": f"Error getting symbol groups: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def describe_symbol(symbol: str) -> Dict[str, Any]:
    """Return symbol information as JSON for `symbol`.
       Parameters: symbol
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
        
        from ..utils.utils import to_float_np as __to_float_np
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
    simplify: Optional[SimplifySpec] = None,
    timezone: str = "auto",
) -> Dict[str, Any]:
    """Return historical candles as CSV.
       Parameters: symbol, timeframe, limit, start?, end?, ohlcv?, indicators?, denoise?, simplify?, timezone
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
         - simplify: Optional dict to reduce or transform rows.
             keys:
               - mode: 'select' (default, select points), 'approximate' (aggregate segments),
                       'encode' (transform data), 'segment' (detect turning points), 'symbolic' (SAX transform).
               - method: (for 'select'/'approximate' modes) 'lttb' (default), 'rdp', 'pla', 'apca'.
               - points: Target number of data points for LTTB, RDP, PLA, APCA, and encode modes.
               - ratio: Alternative to points (0.0 to 1.0).
               - For 'rdp': epsilon (tolerance in y-units).
               - For 'pla'/'apca': max_error (in y-units) or 'segments'.
               - For 'encode' mode:
                 - schema: 'envelope' (OHLC -> high/low/o_pos/c_pos) or 'delta' (OHLC -> d_open/d_high/d_low/d_close).
               - For 'segment' mode:
                 - algo: 'zigzag'.
                 - threshold_pct: Reversal threshold (e.g., 0.5 for 0.5%).
               - For 'symbolic' mode:
                 - schema: 'sax'.
                 - paa: Number of PAA segments (defaults from 'points').
       The full list of supported technical indicators can be retrieved from `get_indicators`.
    """
    try:
        # Backward/compat mappings to internal variable names used in implementation
        candles = int(limit)
        start_datetime = start
        end_datetime = end
        ti = indicators
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
            # Normalize TI spec from structured list, JSON string, or compact string for internal processing
            ti_spec = None
            if ti is not None:
                source = ti
                # Accept JSON string input for robustness
                if isinstance(source, str):
                    s = source.strip()
                    if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
                        try:
                            source = json.loads(s)
                        except Exception:
                            source = ti  # leave as original string if parse fails
                if isinstance(source, (list, tuple)):
                    parts = []
                    for item in source:
                        if isinstance(item, dict) and 'name' in item:
                            nm = str(item.get('name'))
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
                    # Already a compact indicator string like "rsi(14),ema(20)"
                    ti_spec = str(source)
            # Determine warmup bars if technical indicators requested
            warmup_bars = _estimate_warmup_bars_util(ti_spec)

            if start_datetime and end_datetime:
                from_date = _parse_start_datetime_util(start_datetime)
                to_date = _parse_start_datetime_util(end_datetime)
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
                from_date = _parse_start_datetime_util(start_datetime)
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
                to_date = _parse_start_datetime_util(end_datetime)
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
        real_volumes = [int(rate["real_volume"]) for rate in rates]
        
        has_tick_volume = len(set(tick_volumes)) > 1 or any(v != 0 for v in tick_volumes)
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
        _use_ctz = _use_client_tz_util(timezone)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df["time"] = df["time"].apply(_format_time_minimal_local_util if _use_ctz else _format_time_minimal_util)
        if 'volume' not in df.columns and 'tick_volume' in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df['volume'] = df['tick_volume']

        # Track denoise metadata if applied
        denoise_apps: List[Dict[str, Any]] = []
        # Optional pre-TI denoising (in-place by default)
        if denoise and str(denoise.get('when', 'pre_ti')).lower() == 'pre_ti':
            _apply_denoise(df, denoise, default_when='pre_ti')
            try:
                dn = dict(denoise)
                denoise_apps.append({
                    'method': str(dn.get('method','none')).lower(),
                    'when': str(dn.get('when','pre_ti')).lower(),
                    'causality': str(dn.get('causality', 'causal')),
                    'keep_original': bool(dn.get('keep_original', False)),
                    'columns': dn.get('columns','close'),
                    'params': dn.get('params') or {},
                })
            except Exception:
                pass

        # Apply technical indicators if requested (dynamic)
        ti_cols: List[str] = []
        if ti_spec:
            ti_cols = _apply_ta_indicators_util(df, ti_spec)
            headers.extend([c for c in ti_cols if c not in headers])
            # Optional: denoise TI columns as well when requested
            if denoise and bool(denoise.get('apply_to_ti') or denoise.get('ti')) and ti_cols:
                dn_ti = dict(denoise)
                dn_ti['columns'] = list(ti_cols)
                dn_ti.setdefault('when', 'post_ti')
                dn_ti.setdefault('keep_original', False)
                _apply_denoise(df, dn_ti, default_when='post_ti')

        # Build final header list when not using OHLCV subset
        if requested is None:
            # headers already includes OHLC and optional extras
            pass

        # Filter out warmup region to return the intended target window only
        if start_datetime and end_datetime:
            # Keep within original [from_date, to_date]
            target_from = _parse_start_datetime_util(start_datetime).timestamp()
            target_to = _parse_start_datetime_util(end_datetime).timestamp()
            df = df.loc[(df['__epoch'] >= target_from) & (df['__epoch'] <= target_to)].copy()
        elif start_datetime:
            target_from = _parse_start_datetime_util(start_datetime).timestamp()
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
                        target_from_dt = _parse_start_datetime_util(start_datetime)
                        target_to_dt = _parse_start_datetime_util(end_datetime)
                        from_date_internal = target_from_dt - timedelta(seconds=seconds_per_bar * warmup_bars_retry)
                        rates = _mt5_copy_rates_range(symbol, mt5_timeframe, from_date_internal, target_to_dt)
                    elif start_datetime:
                        target_from_dt = _parse_start_datetime_util(start_datetime)
                        to_date_dt = target_from_dt + timedelta(seconds=seconds_per_bar * (candles + 2))
                        from_date_internal = target_from_dt - timedelta(seconds=seconds_per_bar * warmup_bars_retry)
                        rates = _mt5_copy_rates_range(symbol, mt5_timeframe, from_date_internal, to_date_dt)
                    elif end_datetime:
                        target_to_dt = _parse_start_datetime_util(end_datetime)
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
                            df['time'] = df['time'].apply(_format_time_minimal_local_util if _use_ctz else _format_time_minimal_util)
                        if 'volume' not in df.columns and 'tick_volume' in df.columns:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                df['volume'] = df['tick_volume']
                        # Optional pre-TI denoising on retried window
                        if denoise and str(denoise.get('when', 'pre_ti')).lower() == 'pre_ti':
                            _apply_denoise(df, denoise, default_when='pre_ti')
                        # Re-apply indicators and re-extend headers
                        ti_cols = _apply_ta_indicators_util(df, ti_spec)
                        headers.extend([c for c in ti_cols if c not in headers])
                        # Optional: denoise TI columns on retried window
                        if denoise and bool(denoise.get('apply_to_ti') or denoise.get('ti')) and ti_cols:
                            dn_ti = dict(denoise)
                            dn_ti['columns'] = list(ti_cols)
                            dn_ti.setdefault('when', 'post_ti')
                            dn_ti.setdefault('keep_original', False)
                            _apply_denoise(df, dn_ti, default_when='post_ti')
                        # Re-trim to target window
                        if start_datetime and end_datetime:
                            target_from = _parse_start_datetime_util(start_datetime).timestamp()
                            target_to = _parse_start_datetime_util(end_datetime).timestamp()
                            df = df[(df['__epoch'] >= target_from) & (df['__epoch'] <= target_to)]
                        elif start_datetime:
                            target_from = _parse_start_datetime_util(start_datetime).timestamp()
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
        if denoise and str(denoise.get('when', 'pre_ti')).lower() == 'post_ti':
            added_dn = _apply_denoise(df, denoise, default_when='post_ti')
            for c in added_dn:
                if c not in headers:
                    headers.append(c)
            try:
                dn = dict(denoise)
                denoise_apps.append({
                    'method': str(dn.get('method','none')).lower(),
                    'when': 'post_ti',
                    'causality': str(dn.get('causality', 'zero_phase')),
                    'keep_original': bool(dn.get('keep_original', True)),
                    'columns': dn.get('columns','close'),
                    'params': dn.get('params') or {},
                    'added_columns': added_dn,
                })
            except Exception:
                pass

        # Ensure headers are unique and exist in df
        headers = [h for h in headers if h in df.columns or h == 'time']

        # Reformat time consistently across rows for display
        if 'time' in headers and len(df) > 0:
            epochs_list = df['__epoch'].tolist()
            if _use_ctz:
                fmt = _time_format_from_epochs_util_local(epochs_list)
                fmt = _maybe_strip_year_util_local(fmt, epochs_list)
                fmt = _style_time_format_util(fmt)
                tz = _resolve_client_tz_util(timezone)
                # Track used tz name and invalid explicit values
                tz_used_name = None
                tz_warning = None
                if isinstance(timezone, str):
                    vlow = timezone.strip().lower()
                    if vlow not in ('auto','client','utc',''):
                        try:
                            import pytz  # type: ignore
                            tz_explicit = pytz.timezone(timezone.strip())
                            tz = tz_explicit
                        except Exception:
                            tz_warning = f"Unknown timezone '{timezone}', falling back to CLIENT_TZ or system"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if tz is not None:
                        tz_used_name = getattr(tz, 'zone', None) or str(tz)
                        df['time'] = df['__epoch'].apply(lambda t: datetime.fromtimestamp(t, tz=dt_timezone.utc).astimezone(tz).strftime(fmt))
                    else:
                        tz_used_name = 'system'
                        df['time'] = df['__epoch'].apply(lambda t: datetime.fromtimestamp(t, tz=dt_timezone.utc).astimezone().strftime(fmt))
                df.__dict__['_tz_used_name'] = tz_used_name
                df.__dict__['_tz_warning'] = tz_warning
            else:
                fmt = _time_format_from_epochs_util(epochs_list)
                fmt = _maybe_strip_year_util(fmt, epochs_list)
                fmt = _style_time_format_util(fmt)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df['time'] = df['__epoch'].apply(lambda t: datetime.utcfromtimestamp(t).strftime(fmt))

        # Optionally reduce number of rows for readability/output size
        original_rows = len(df)
        simplify_eff = None
        if simplify is not None:
            simplify_eff = dict(simplify)
            # Default mode
            simplify_eff['mode'] = str(simplify_eff.get('mode', SIMPLIFY_DEFAULT_MODE)).lower().strip()
            # If no explicit points/ratio provided, default to 10% of requested limit
            has_points = any(k in simplify_eff and simplify_eff[k] is not None for k in ("points","target_points","max_points","ratio"))
            if not has_points:
                try:
                    default_pts = max(3, int(round(int(limit) * SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT)))
                except Exception:
                    default_pts = max(3, int(round(original_rows * SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT)))
                simplify_eff['points'] = default_pts
        df, simplify_meta = _simplify_dataframe_rows_ext(df, headers, simplify_eff if simplify_eff is not None else simplify)
        # If simplify changed representation, respect returned headers
        if simplify_meta is not None and 'headers' in simplify_meta and isinstance(simplify_meta['headers'], list):
            headers = [h for h in simplify_meta['headers'] if isinstance(h, str)]

        # Assemble rows from (possibly reduced) DataFrame for selected headers
        rows = _format_numeric_rows_from_df_util(df, headers)

        # Build CSV via writer for escaping
        payload = _csv_from_rows_util(headers, rows)
        payload.update({
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": len(df),
        })
        if not _use_ctz:
            payload["timezone"] = "UTC"
        if simplify_meta is not None:
            payload["simplified"] = True
            payload["simplify"] = simplify_meta
            payload["simplify"]["timeframe"] = timeframe
            payload["simplify"]["original_candles"] = original_rows
        # Attach denoise applications metadata if any
        if denoise_apps:
            payload['denoise'] = {
                'applied': True,
                'applications': denoise_apps,
            }
        return payload
    except Exception as e:
        return {"error": f"Error getting rates: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def fetch_ticks(
    symbol: str,
    limit: int = 100,
    start: Optional[str] = None,
    end: Optional[str] = None,
    simplify: Optional[SimplifySpec] = None,
    timezone: str = "auto",
) -> Dict[str, Any]:
    """Return latest ticks as CSV with columns: time,bid,ask and optional last,volume,flags.
    Parameters: symbol, limit, start?, end?, simplify?, timezone
    - `limit` limits the number of rows.
    - `start` starts from a flexible date/time; optional `end` enables range.
    - `simplify`: Optional dict to reduce or aggregate rows (select/approximate/resample).
    """
    try:
        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = mt5.symbol_info(symbol)
        _was_visible = bool(_info_before.visible) if _info_before is not None else None
        err = _ensure_symbol_ready(symbol)
        if err:
            return {"error": err}
        
        try:
            # Normalized params only
            effective_limit = int(limit)
            if start:
                from_date = _parse_start_datetime_util(start)
                if not from_date:
                    return {"error": "Invalid date format. Try examples like '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00', '2 days ago'."}
                if end:
                    to_date = _parse_start_datetime_util(end)
                    if not to_date:
                        return {"error": "Invalid 'end' date format. Try '2025-08-29 14:30' or 'yesterday 18:00'."}
                    ticks = None
                    for _ in range(FETCH_RETRY_ATTEMPTS):
                        ticks = _mt5_copy_ticks_range(symbol, from_date, to_date, mt5.COPY_TICKS_ALL)
                        if ticks is not None and len(ticks) > 0:
                            break
                        time.sleep(FETCH_RETRY_DELAY)
                    if ticks is not None and effective_limit and len(ticks) > effective_limit:
                        ticks = ticks[-effective_limit:]
                else:
                    ticks = None
                    for _ in range(FETCH_RETRY_ATTEMPTS):
                        ticks = _mt5_copy_ticks_from(symbol, from_date, effective_limit, mt5.COPY_TICKS_ALL)
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
                if ticks is not None and effective_limit and len(ticks) > effective_limit:
                    ticks = ticks[-effective_limit:]  # Get the last ticks
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
        _use_ctz = _use_client_tz_util(timezone)
        if not _use_ctz:
            fmt = _time_format_from_epochs_util(_epochs)
            fmt = _maybe_strip_year_util(fmt, _epochs)
            fmt = _style_time_format_util(fmt)
        # Build a DataFrame of ticks to support non-select simplify modes
        def _tick_field(t, name: str):
            try:
                # numpy.void structured array element
                return t[name]
            except Exception:
                pass
            try:
                # namedtuple-like from symbol_info_tick
                return getattr(t, name)
            except Exception:
                pass
            try:
                # dict-like
                return t.get(name)
            except Exception:
                return None

        df_ticks = pd.DataFrame({
            "__epoch": _epochs,
            "bid": [float(_tick_field(t, "bid")) for t in ticks],
            "ask": [float(_tick_field(t, "ask")) for t in ticks],
        })
        if has_last:
            df_ticks["last"] = [float(_tick_field(t, "last")) for t in ticks]
        if has_volume:
            df_ticks["volume"] = [float(_tick_field(t, "volume")) for t in ticks]
        if has_flags:
            df_ticks["flags"] = [int(_tick_field(t, "flags")) for t in ticks]
        # Add display time column
        if _use_ctz:
            df_ticks["time"] = [
                _format_time_minimal_local_util(e) for e in _epochs
            ]
        else:
            df_ticks["time"] = [
                datetime.utcfromtimestamp(e).strftime(fmt) for e in _epochs
            ]
        # If simplify mode requests approximation or resampling, use shared path
        original_count = len(df_ticks)
        simplify_eff = None
        if simplify is not None:
            simplify_eff = dict(simplify)
            simplify_eff['mode'] = str(simplify_eff.get('mode', SIMPLIFY_DEFAULT_MODE)).lower().strip()
            has_points = any(k in simplify_eff and simplify_eff[k] is not None for k in ("points","target_points","max_points","ratio"))
            if not has_points:
                try:
                    default_pts = max(3, int(round(int(count) * SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT)))
                except Exception:
                    default_pts = max(3, int(round(original_count * SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT)))
                simplify_eff['points'] = default_pts
        simplify_present = (simplify_eff is not None) or (simplify is not None)
        simplify_used = simplify_eff if simplify_eff is not None else simplify
        _mode = str((simplify_used or {}).get('mode', SIMPLIFY_DEFAULT_MODE)).lower().strip() if simplify_present else SIMPLIFY_DEFAULT_MODE
        if simplify_present and _mode in ('approximate', 'resample'):
            df_out, simplify_meta = _simplify_dataframe_rows_ext(df_ticks, headers, simplify_used)
            rows = _format_numeric_rows_from_df_util(df_out, headers)
            payload = _csv_from_rows_util(headers, rows)
            payload.update({
                "success": True,
                "symbol": symbol,
                "count": len(rows),
            })
            if not _use_ctz:
                payload["timezone"] = "UTC"
            if simplify_meta is not None and original_count > len(rows):
                payload["simplified"] = True
                meta = dict(simplify_meta)
                meta["columns"] = [c for c in ["bid","ask"] + (["last"] if has_last else []) + (["volume"] if has_volume else [])]
                payload["simplify"] = meta
            return payload
        # Optional simplification based on a chosen y-series
        original_count = len(ticks)
        select_indices = list(range(original_count))
        _simp_method_used: Optional[str] = None
        _simp_params_meta: Optional[Dict[str, Any]] = None
        if simplify_present and original_count > 3:
            try:
                # Always represent all available numeric columns (bid/ask/(last)/(volume))
                cols: List[str] = ['bid', 'ask']
                if has_last:
                    cols.append('last')
                if has_volume:
                    cols.append('volume')
                n_out = _choose_simplify_points_util(original_count, simplify_used)
                per = max(3, int(round(n_out / max(1, len(cols)))))
                idx_set: set = set([0, original_count - 1])
                params_accum: Dict[str, Any] = {}
                method_used_overall = None
                for c in cols:
                    series: List[float] = []
                    for t in ticks:
                        v = _tick_field(t, c)
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
                            vals.append(float(_tick_field(t, c)))
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
                            vv = (float(_tick_field(ticks[i], c)) - mins[c]) / ranges[c]
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
                _simp_method_used = method_used_overall or str((simplify_used or {}).get('method', SIMPLIFY_DEFAULT_METHOD)).lower()
                _simp_params_meta = params_accum
            except Exception:
                select_indices = list(range(original_count))

        rows = []
        for i in select_indices:
            tick = ticks[i]
            if _use_ctz:
                time_str = _format_time_minimal_local_util(_epochs[i])
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

        payload = _csv_from_rows_util(headers, rows)
        payload.update({
            "success": True,
            "symbol": symbol,
            "count": len(rows),
        })
        if not _use_ctz:
            payload["timezone"] = "UTC"
        if simplify_present and original_count > len(rows):
            payload["simplified"] = True
            meta = {
                "method": (_simp_method_used or str((simplify_used or {}).get('method', SIMPLIFY_DEFAULT_METHOD)).lower()),
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
            # Normalize points to actual returned rows
            meta["points"] = len(rows)
            payload["simplify"] = meta
        return payload
    except Exception as e:
        return {"error": f"Error getting ticks: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def detect_candlestick_patterns(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    limit: int = 10,
    timezone: str = "auto",
) -> Dict[str, Any]:
    """Detect candlestick patterns and return CSV rows of detections.
    Parameters: symbol, timeframe, limit, timezone

    Inputs:
    - `symbol`: Trading symbol (e.g., "EURUSD").
    - `timeframe`: One of the supported MT5 timeframes (e.g., "M15", "H1").
    - `limit`: Number of most recent candles to analyze.

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
            # Fetch last `limit` bars from now (UTC anchor)
            utc_now = datetime.utcnow()
            rates = _mt5_copy_rates_from(symbol, mt5_timeframe, utc_now, limit)
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
        _use_ctz = _use_client_tz_util(timezone)
        if _use_ctz:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df['time'] = df['time'].apply(_format_time_minimal_local_util)
        else:
            time_fmt = _time_format_from_epochs_util(epochs) if epochs else "%Y-%m-%d %H:%M:%S"
            time_fmt = _maybe_strip_year_util(time_fmt, epochs)
            time_fmt = _style_time_format_util(time_fmt)
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
        payload = _csv_from_rows_util(headers, rows)
        payload.update({
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": int(limit),
        })
        if not _use_ctz:
            payload["timezone"] = "UTC"
        return payload
    except Exception as e:
        return {"error": f"Error detecting candlestick patterns: {str(e)}"}

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
    "nhits",
    "nbeatsx",
    "tft",
    "patchtst",
    "sf_autoarima",
    "sf_theta",
    "sf_autoets",
    "sf_seasonalnaive",
    "mlf_rf",
    "mlf_lightgbm",
    "chronos_bolt",
    "timesfm",
    "lag_llama",
    "ensemble",
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

# Optional Nixtla NeuralForecast availability for listings (lazy import used in execution)
try:
    import importlib.util as _importlib_util  # type: ignore
    _NF_AVAILABLE = _importlib_util.find_spec("neuralforecast") is not None
except Exception:
    _NF_AVAILABLE = False
try:
    import importlib.util as _importlib_util2  # type: ignore
    _SF_AVAILABLE = _importlib_util2.find_spec("statsforecast") is not None
except Exception:
    _SF_AVAILABLE = False
try:
    import importlib.util as _importlib_util3  # type: ignore
    _MLF_AVAILABLE = _importlib_util3.find_spec("mlforecast") is not None
except Exception:
    _MLF_AVAILABLE = False
try:
    import importlib.util as _importlib_util4  # type: ignore
    _LGB_AVAILABLE = _importlib_util4.find_spec("lightgbm") is not None
except Exception:
    _LGB_AVAILABLE = False
try:
    import importlib.util as _importlib_util5  # type: ignore
    _CHRONOS_AVAILABLE = (_importlib_util5.find_spec("chronos") is not None) or (_importlib_util5.find_spec("transformers") is not None)
except Exception:
    _CHRONOS_AVAILABLE = False
try:
    import importlib.util as _importlib_util6  # type: ignore
    _TIMESFM_AVAILABLE = (_importlib_util6.find_spec("timesfm") is not None) or (_importlib_util6.find_spec("transformers") is not None)
except Exception:
    _TIMESFM_AVAILABLE = False
try:
    import importlib.util as _importlib_util7  # type: ignore
    _LAG_LLAMA_AVAILABLE = (_importlib_util7.find_spec("lag_llama") is not None) or (_importlib_util7.find_spec("transformers") is not None)
except Exception:
    _LAG_LLAMA_AVAILABLE = False

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

def _pd_freq_from_timeframe(tf: str) -> str:
    t = str(tf).upper()
    mapping = {
        'M1': '1min', 'M2': '2min', 'M3': '3min', 'M4': '4min', 'M5': '5min',
        'M10': '10min', 'M12': '12min', 'M15': '15min', 'M20': '20min', 'M30': '30min',
        'H1': 'H', 'H2': '2H', 'H3': '3H', 'H4': '4H', 'H6': '6H', 'H8': '8H', 'H12': '12H',
        'D1': 'D', 'W1': 'W', 'MN1': 'MS'
    }
    return mapping.get(t, 'D')


@mcp.tool()
def list_forecast_methods() -> Dict[str, Any]:
    """List available forecast methods and their parameters."""
    return _get_forecast_methods_data_safe()


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
    quantity: Literal['price','return','volatility'] = 'price',  # type: ignore
    target: Literal['price','return'] = 'price',  # type: ignore
    denoise: Optional[DenoiseSpec] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    target_spec: Optional[Dict[str, Any]] = None,
    timezone: str = "auto",
) -> Dict[str, Any]:
    """Fast forecasts for the next `horizon` bars using lightweight methods.

    Delegates to the implementation under `mtdata.forecast.forecast`.
    """
    from ..forecast.forecast import forecast as _impl
    return _impl(
        symbol=symbol,
        timeframe=timeframe,  # type: ignore[arg-type]
        method=method,        # type: ignore[arg-type]
        horizon=horizon,
        lookback=lookback,
        as_of=as_of,
        params=params,
        ci_alpha=ci_alpha,
        quantity=quantity,    # type: ignore[arg-type]
        target=target,        # type: ignore[arg-type]
        denoise=denoise,
        features=features,
        dimred_method=dimred_method,
        dimred_params=dimred_params,
        target_spec=target_spec,
        timezone=timezone,
    )


@mcp.tool()
@_auto_connect_wrapper
def forecast_backtest(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    steps: int = 5,
    spacing: int = 20,
    methods: Optional[List[str]] = None,
    params_per_method: Optional[Dict[str, Any]] = None,
    quantity: Literal['price','return','volatility'] = 'price',  # type: ignore
    target: Literal['price','return'] = 'price',  # type: ignore
    denoise: Optional[DenoiseSpec] = None,
    params: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Rolling-origin backtest over historical anchors using the forecast tool."""
    from ..forecast.backtest import forecast_backtest as _impl
    return _impl(
        symbol=symbol,
        timeframe=timeframe,  # type: ignore[arg-type]
        horizon=horizon,
        steps=steps,
        spacing=spacing,
        methods=methods,
        params_per_method=params_per_method,
        quantity=quantity,    # type: ignore[arg-type]
        target=target,        # type: ignore[arg-type]
        denoise=denoise,
        params=params,
        features=features,
        dimred_method=dimred_method,
        dimred_params=dimred_params,
    )


@mcp.tool()
@_auto_connect_wrapper
def forecast_volatility(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 1,
    method: Literal['ewma','parkinson','gk','rs','yang_zhang','rolling_std','garch','egarch','gjr_garch','arima','sarima','ets','theta'] = 'ewma',  # type: ignore
    proxy: Optional[Literal['squared_return','abs_return','log_r2']] = None,  # type: ignore
    params: Optional[Dict[str, Any]] = None,
    as_of: Optional[str] = None,
    denoise: Optional[DenoiseSpec] = None,
) -> Dict[str, Any]:
    """Forecast volatility over `horizon` bars using direct estimators or proxies."""
    from ..forecast.volatility import forecast_volatility as _impl
    return _impl(
        symbol=symbol,
        timeframe=timeframe,  # type: ignore[arg-type]
        horizon=horizon,
        method=method,        # type: ignore[arg-type]
        proxy=proxy,          # type: ignore[arg-type]
        params=params,
        as_of=as_of,
        denoise=denoise,
    )

@mcp.tool()
def _get_dimred_methods_data() -> Dict[str, Any]:
    try:
        methods = _list_dimred_methods_util()
        return {"success": True, "methods": methods}
    except Exception as e:
        return {"error": f"Error listing dimred methods: {str(e)}"}


@mcp.tool()
def list_capabilities(
    sections: Optional[List[str]] = None,
    include_details: bool = False,
) -> Dict[str, Any]:
    """Consolidated capabilities and feature info across forecasting, volatility, denoise, indicators, and dimred.

    Parameters: sections? (subset of: forecast, volatility, denoise, indicators, dimred, frameworks, pattern_search), include_details?
    """
    try:
        wanted = None
        if sections and isinstance(sections, (list, tuple)):
            wanted = {str(s).lower() for s in sections}
        out: Dict[str, Any] = {"success": True}

        def want(key: str) -> bool:
            return (wanted is None) or (str(key).lower() in wanted)

        if want("frameworks"):
            # Availability flags for optional frameworks
            frameworks = {
                "neuralforecast": bool(_NF_AVAILABLE),
                "statsforecast": bool(_SF_AVAILABLE),
                "mlforecast": bool(_MLF_AVAILABLE),
                "arch": bool(_ARCH_AVAILABLE),
                "chronos_foundation": bool(_CHRONOS_AVAILABLE),
                "timesfm_foundation": bool(_TIMESFM_AVAILABLE),
                "lag_llama_foundation": bool(_LAG_LLAMA_AVAILABLE),
            }
            # Torch/cuda hints
            try:
                import torch  # type: ignore
                frameworks["torch"] = True
                frameworks["cuda_available"] = bool(torch.cuda.is_available())
            except Exception:
                frameworks["torch"] = False
                frameworks["cuda_available"] = False
            out["frameworks"] = frameworks

        if want("forecast"):
            out["forecast_methods"] = _get_forecast_methods_data_safe()

        if want("volatility"):
            out["volatility_methods"] = _get_volatility_methods_data()

        if want("denoise"):
            out["denoise_methods"] = _get_denoise_methods_data_safe()

        if want("indicators"):
            try:
                items = _list_ta_indicators_util()
                if include_details:
                    out["indicators"] = items
                else:
                    cats = {}
                    for it in items:
                        c = it.get('category') or 'Other'
                        cats[c] = cats.get(c, 0) + 1
                    out["indicators_summary"] = {
                        "count": int(len(items)),
                        "categories": cats,
                    }
            except Exception as e:
                out["indicators_error"] = str(e)

        if want("dimred"):
            out["dimred_methods"] = _get_dimred_methods_data()

        if want("pattern_search"):
            # Summarize engines/backends availability
            ps = {
                "engines": {
                    "ckdtree": True,
                },
                "shape_metrics": ["ncc", "affine", "dtw", "softdtw"],
                "dtw_backends": {},
            }
            # ANN engine
            try:
                import hnswlib  # type: ignore
                ps["engines"]["hnsw"] = True
            except Exception:
                ps["engines"]["hnsw"] = False
            # DTW backends
            try:
                import tslearn.metrics as _tsm  # type: ignore
                ps["dtw_backends"]["tslearn"] = True
            except Exception:
                ps["dtw_backends"]["tslearn"] = False
            try:
                import dtaidistance.dtw as _dd  # type: ignore
                ps["dtw_backends"]["dtaidistance"] = True
            except Exception:
                ps["dtw_backends"]["dtaidistance"] = False
            out["pattern_search"] = ps

        return out
    except Exception as e:
        return {"error": f"Error building capabilities: {str(e)}"}


@mcp.tool()
@_auto_connect_wrapper
def pattern_detect_classic(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    lookback: int = 1500,
    denoise: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Detect classic chart patterns (triangles, flags, wedges, H&S, channels, rectangles, etc.).

    - Pulls last `lookback` bars for `symbol`/`timeframe`.
    - Applies optional denoise.
    - Returns a list of patterns with status (completed|forming), confidence, time bounds, and pattern-specific levels.
    """
    try:
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_tf = TIMEFRAME_MAP[timeframe]
        # Ensure symbol is visible
        _info = mt5.symbol_info(symbol)
        _was_visible = bool(_info.visible) if _info is not None else None
        try:
            if _was_visible is False:
                mt5.symbol_select(symbol, True)
        except Exception:
            pass
        # Fetch bars
        utc_now = datetime.utcnow()
        count = max(400, int(lookback) + 2)
        rates = _mt5_copy_rates_from(symbol, mt5_tf, utc_now, count)
        if rates is None or len(rates) < 100:
            return {"error": f"Failed to fetch sufficient bars for {symbol}"}
        df = pd.DataFrame(rates)
        if 'volume' not in df.columns and 'tick_volume' in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df['volume'] = df['tick_volume']
        # Drop forming last bar for stability
        if len(df) >= 2:
            df = df.iloc[:-1]
        if denoise and isinstance(denoise, dict):
            try:
                dn = dict(denoise)
                dn.setdefault('when', 'pre_ti')
                dn.setdefault('columns', ['close'])
                dn.setdefault('keep_original', False)
                _apply_denoise_util(df, dn, default_when='pre_ti')
            except Exception:
                pass
        # Clip to lookback
        if len(df) > int(lookback):
            df = df.iloc[-int(lookback):].copy()

        # Build config
        cfg = _ClassicCfg()
        if isinstance(config, dict):
            for k, v in config.items():
                if hasattr(cfg, k):
                    try:
                        setattr(cfg, k, type(getattr(cfg, k))(v))
                    except Exception:
                        try:
                            setattr(cfg, k, v)
                        except Exception:
                            pass

        # Detect
        pats = _detect_classic_patterns(df, cfg)

        # Serialize
        def _round(x):
            try:
                return float(np.round(float(x), 8))
            except Exception:
                return x
        out_list = []
        for p in pats:
            try:
                # Format times per global time format
                st_epoch = float(p.start_time) if p.start_time is not None else None
                et_epoch = float(p.end_time) if p.end_time is not None else None
                try:
                    start_date = _format_time_minimal_util(st_epoch) if st_epoch is not None else None
                except Exception:
                    start_date = None
                try:
                    end_date = _format_time_minimal_util(et_epoch) if et_epoch is not None else None
                except Exception:
                    end_date = None
                d = {
                    "name": p.name,
                    "status": p.status,
                    "confidence": float(max(0.0, min(1.0, p.confidence))),
                    "start_index": int(p.start_index),
                    "end_index": int(p.end_index),
                    "start_date": start_date,
                    "end_date": end_date,
                    "start_epoch": st_epoch,
                    "end_epoch": et_epoch,
                    "details": {k: _round(v) for k, v in (p.details or {}).items()},
                }
                out_list.append(d)
            except Exception:
                continue

        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "lookback": int(lookback),
            "patterns": out_list,
            # include series for plotting
            "series_close": [float(v) for v in __to_float_np(df.get('close')).tolist()],
            "series_epoch": [float(v) for v in __to_float_np(df.get('time')).tolist()] if 'time' in df.columns else None,
            "series_time": [
                _format_time_minimal_util(float(v)) for v in __to_float_np(df.get('time')).tolist()
            ] if 'time' in df.columns else None,
            "n_patterns": int(len(out_list)),
        }
    except Exception as e:
        return {"error": f"Error detecting classic patterns: {str(e)}"}


@mcp.tool()
@_auto_connect_wrapper
def pattern_search(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    window_size: int = 30,
    top_k: int = 50,
    future_size: int = 8,
    symbols: Optional[List[str]] = None,
    max_symbols: Optional[int] = None,
    max_bars_per_symbol: int = 8000,
    denoise: Optional[Dict[str, Any]] = None,
    scale: str = "minmax",
    metric: str = "euclidean",
    pca_components: Optional[int] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    engine: str = "ckdtree",
    include_values: bool = False,
    min_symbol_correlation: Optional[float] = None,
    corr_lookback: int = 1000,
    compact: bool = True,
    include_anchor_values: bool = False,
    lookback: Optional[int] = None,
    refine_k: Optional[int] = 200,
    shape_metric: Optional[str] = 'ncc',
    allow_lag: int = 5,
    time_scale_span: Optional[float] = 0.1,
    cache_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Unified pattern search: builds an index if needed, then searches.

    Parameters: symbol, timeframe, window_size, top_k, future_size, symbols?, max_symbols?, max_bars_per_symbol?, denoise?, scale?, metric?, pca_components? or dimred_method?/dimred_params?
    - If a compatible index is cached and includes the symbol, reuses it.
    - Else builds an index using `symbols` (or visible symbols capped by `max_symbols`),
      falling back to a minimal single-symbol index if none provided.
    - Provides signal statistics and suppresses raw series unless `include_values=True`.
    - Optional `min_symbol_correlation` filters cross-instrument matches by return correlation over `corr_lookback` bars.
    - Historical depth is controlled by `max_bars_per_symbol`; response includes `bars_per_symbol` and `windows_per_symbol`.
    """
    try:
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        # Cache key for settings
        dn_key = _denoise_cache_key(denoise)
        eff_lookback = int(lookback) if lookback is not None else int(max_bars_per_symbol)
        # Compose cache key including dimension reduction method when provided
        if dimred_method and str(dimred_method).lower() not in ("", "none"):
            dr_desc = f"dr={str(dimred_method).lower()}"
        else:
            dr_desc = f"pca={int(pca_components) if pca_components else 0}"
        cache_key = (str(timeframe), int(window_size), int(future_size),
                     f"dn={dn_key}|sc={str(scale).lower()}|mt={str(metric).lower()}|{dr_desc}|eng={(engine or 'ckdtree').lower()}|lb={eff_lookback}")
        idx = _PATTERN_INDEX_CACHE.get(cache_key)

        need_build = (idx is None) or (symbol not in idx.symbols)
        used_symbols: List[str] = []
        if need_build:
            # Optional disk cache load
            if cache_id:
                try:
                    disk_idx = _load_pattern_index_from_disk(cache_dir, cache_id)
                except Exception:
                    disk_idx = None
                if disk_idx is not None and disk_idx.timeframe == timeframe and disk_idx.window_size == int(window_size) and disk_idx.future_size == int(future_size):
                    idx = disk_idx
                    _PATTERN_INDEX_CACHE[cache_key] = idx
                    need_build = False
            if symbols and isinstance(symbols, (list, tuple)) and len(symbols) > 0:
                used_symbols = [str(s) for s in symbols if str(s)]
            else:
                # Use visible symbols when not specified; fallback to single-symbol index
                all_syms = mt5.symbols_get()
                if all_syms is not None:
                    for s in all_syms:
                        try:
                            if bool(getattr(s, 'visible', False)):
                                used_symbols.append(str(getattr(s, 'name')))
                        except Exception:
                            continue
                    if max_symbols and max_symbols > 0:
                        used_symbols = used_symbols[: int(max_symbols)]
            if not used_symbols:
                used_symbols = [symbol]
            if idx is None:
                idx = _build_pattern_index(
                    symbols=used_symbols,
                    timeframe=str(timeframe),
                    window_size=int(window_size),
                    future_size=int(future_size),
                    max_bars_per_symbol=int(eff_lookback),
                    denoise=denoise,
                    scale=str(scale),
                    metric=str(metric),
                    pca_components=int(pca_components) if pca_components else None,
                    dimred_method=dimred_method,
                    dimred_params=dimred_params,
                    engine=str(engine),
                )
                _PATTERN_INDEX_CACHE[cache_key] = idx
                _save_pattern_index_to_disk(idx, cache_dir, cache_id)

        # Build anchor
        def _fetch_anchor(symbol: str, timeframe: str, window_size: int) -> Tuple[np.ndarray, float]:
            utc_now = datetime.utcnow()
            rates = _mt5_copy_rates_from(symbol, TIMEFRAME_MAP[timeframe], utc_now, int(window_size) + 2)
            if rates is None or len(rates) == 0:
                raise RuntimeError(f"Failed to fetch anchor bars for {symbol}")
            df = pd.DataFrame(rates)
            try:
                df['time'] = df['time'].astype(float).apply(_mt5_epoch_to_utc)
            except Exception:
                pass
            if 'volume' not in df.columns and 'tick_volume' in df.columns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df['volume'] = df['tick_volume']
            if denoise and isinstance(denoise, dict):
                try:
                    dn = dict(denoise)
                    dn.setdefault('when', 'pre_ti')
                    dn.setdefault('columns', ['close'])
                    dn.setdefault('keep_original', False)
                    _apply_denoise_util(df, dn, default_when='pre_ti')
                except Exception:
                    pass
            if len(df) < window_size:
                raise RuntimeError("Not enough bars for anchor window")
            from ..utils.utils import to_float_np
            closes = to_float_np(df['close'])
            try:
                end_epoch = float(df['time'].iloc[-1])
            except Exception:
                end_epoch = float(datetime.utcnow().timestamp())
            return closes[-int(window_size):], end_epoch

        anchor_vals, anchor_end_epoch = _fetch_anchor(symbol, str(timeframe), int(window_size))
        idxs, dists = idx.search(anchor_vals, top_k=int(top_k))

        total_candidates = int(len(idxs))
        matches = []
        changes: List[float] = []
        pct_changes: List[float] = []
        kept_dist: List[float] = []
        kept = 0
        tf_secs = int(TIMEFRAME_SECONDS.get(str(timeframe), 0) or 0)
        anchor_start_epoch = float(anchor_end_epoch) - float(max(0, int(window_size) - 1) * tf_secs)
        kept_intervals_by_sym: Dict[str, List[Tuple[float, float]]] = {}

        for i, d in zip(idxs.tolist(), dists.tolist()):
            m_sym = idx.get_match_symbol(i)
            if min_symbol_correlation is not None and m_sym != symbol:
                try:
                    r_a = idx.get_symbol_returns(symbol, lookback=int(corr_lookback))
                    r_b = idx.get_symbol_returns(m_sym, lookback=int(corr_lookback))
                    if r_a is not None and r_b is not None:
                        n = min(r_a.size, r_b.size)
                        if n > 10:
                            c = float(np.corrcoef(r_a[-n:], r_b[-n:])[0, 1])
                            if not np.isfinite(c) or c < float(min_symbol_correlation):
                                continue
                except Exception:
                    pass
            vals = idx.get_match_values(i, include_future=True)
            times = idx.get_match_times(i, include_future=True)
            if vals.size < (idx.window_size + max(0, idx.future_size)):
                continue
            today_v = float(vals[idx.window_size - 1])
            future_v = float(vals[min(vals.size - 1, idx.window_size + idx.future_size - 1)])
            change = float(future_v - today_v)
            pct = float((future_v - today_v) / today_v) if today_v != 0 else 0.0
            changes.append(change)
            pct_changes.append(pct)
            kept_dist.append(float(d))
            kept += 1
            start_epoch = float(times[0])
            end_epoch = float(times[idx.window_size - 1])
            if m_sym == symbol and end_epoch >= anchor_start_epoch - 1e-6:
                continue
            kept_list = kept_intervals_by_sym.get(m_sym, [])
            cand_start, cand_end = float(start_epoch), float(end_epoch)
            overlap = False
            for ks, ke in kept_list:
                if not (cand_end <= ks - 1e-6 or cand_start >= ke + 1e-6):
                    overlap = True
                    break
            if overlap:
                continue
            start_time = _format_time_minimal_util(start_epoch)
            end_time = _format_time_minimal_util(end_epoch)
            _m = {
                "symbol": m_sym,
                "distance": float(d),
                "start_date": start_time,
                "end_date": end_time,
                "todays_value": today_v,
                "future_value": future_v,
                "change": change,
                "pct_change": pct,
                "time_scale": 1.0,
            }
            if bool(include_values):
                # Resample to current anchor window+future length if needed
                try:
                    target_len = int(window_size) + int(future_size)
                    if target_len > 0 and vals.size != target_len:
                        x = np.linspace(0.0, 1.0, num=vals.size, dtype=float)
                        xi = np.linspace(0.0, 1.0, num=target_len, dtype=float)
                        vals_rs = np.interp(xi, x, vals.astype(float))
                        _m["values"] = [float(v) for v in vals_rs.tolist()]
                    else:
                        _m["values"] = [float(v) for v in vals.tolist()]
                except Exception:
                    _m["values"] = [float(v) for v in vals.tolist()]
            matches.append(_m)
            kept_intervals_by_sym.setdefault(m_sym, []).append((cand_start, cand_end))

        if not matches:
            return {"error": "No matches found"}
        arr = np.array(changes, dtype=float)
        parr = np.array(pct_changes, dtype=float)
        d_arr = np.array(kept_dist, dtype=float)
        pos_ratio = float(np.mean(parr > 0.0)) if parr.size > 0 else 0.0
        mean_change = float(np.mean(arr)) if arr.size else 0.0
        median_change = float(np.median(arr)) if arr.size else 0.0
        std_change = float(np.std(arr, ddof=0)) if arr.size else 0.0
        mean_pct = float(np.mean(parr)) if parr.size else 0.0
        median_pct = float(np.median(parr)) if parr.size else 0.0
        std_pct = float(np.std(parr, ddof=0)) if parr.size else 0.0
        per_bar_mean_change = float(mean_change / max(1, int(future_size)))
        per_bar_mean_pct = float(mean_pct / max(1, int(future_size)))
        eps = 1e-9
        if d_arr.size:
            w = 1.0 / (d_arr + eps)
            w /= np.sum(w)
            w_mean_change = float(np.sum(w * arr))
            w_mean_pct = float(np.sum(w * parr))
        else:
            w_mean_change = mean_change
            w_mean_pct = mean_pct
        forecast_type = "gain" if pos_ratio > 0.5 else "loss"
        forecast_confidence = pos_ratio if pos_ratio > 0.5 else (1.0 - pos_ratio)

        payload: Dict[str, Any] = {
            "success": True,
            "anchor_symbol": symbol,
            "timeframe": timeframe,
            "window_size": int(window_size),
            "future_size": int(future_size),
            "top_k": int(top_k),
            "matches": matches,
            "forecast_type": forecast_type,
            "forecast_confidence": float(forecast_confidence),
            "n_matches": int(kept),
            "n_candidates": int(total_candidates),
            "prob_gain": float(pos_ratio),
            "avg_change": mean_change,
            "avg_pct_change": mean_pct,
            "per_bar_avg_change": per_bar_mean_change,
            "per_bar_avg_pct_change": per_bar_mean_pct,
        }
        if not compact:
            payload.update({
                "median_change": median_change,
                "std_change": std_change,
                "median_pct_change": median_pct,
                "std_pct_change": std_pct,
                "distance_weighted_avg_change": w_mean_change,
                "distance_weighted_avg_pct_change": w_mean_pct,
                "scale": idx.scale,
                "metric": idx.metric,
                "pca_components": idx.pca_components or 0,
                "dimred_method": getattr(idx, 'dimred_method', 'none'),
                "dimred_params": getattr(idx, 'dimred_params', {}),
                "built_symbols": used_symbols if need_build else None,
                "engine": getattr(idx, 'engine', 'ckdtree'),
                "max_bars_per_symbol": int(getattr(idx, 'max_bars_per_symbol', eff_lookback)),
                "bars_per_symbol": getattr(idx, 'bars_per_symbol', lambda: {})(),
                "windows_per_symbol": getattr(idx, 'windows_per_symbol', lambda: {})(),
                "lookback": int(eff_lookback),
            })
        if include_anchor_values:
            payload["anchor_values"] = [float(v) for v in anchor_vals.tolist()]
        return payload
    except Exception as e:
        return {"error": f"Error in pattern search: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def fetch_market_depth(symbol: str, timezone: str = "auto") -> Dict[str, Any]:
    """Return DOM if available; otherwise current bid/ask snapshot for `symbol`.

    Parameters: symbol, timezone
    """
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
                    # spread removed from outputs by request
                    "note": "Full market depth not available, showing current bid/ask"
                }
            }
            try:
                _use_ctz = _use_client_tz_util(timezone)
                if tick.time and _use_ctz:
                    out["data"]["time_display"] = _format_time_minimal_local_util(_mt5_epoch_to_utc(float(tick.time)))
                elif tick.time:
                    out["data"]["time_display"] = _format_time_minimal_util(_mt5_epoch_to_utc(float(tick.time)))
            except Exception:
                pass
            if not _use_ctz:
                out["timezone"] = "UTC"
            return out
    except Exception as e:
        return {"error": f"Error getting market depth: {str(e)}"}

def main():
    """Main entry point for the MCP server"""
    mcp.run()

if __name__ == "__main__":
    main()

# Helper function assignments for backwards compatibility
_list_ta_indicators = _list_ta_indicators_util
_parse_ti_specs = _parse_ti_specs_util
_apply_denoise = _apply_denoise_util

from .schema_attach import attach_schemas_to_tools as _attach_schemas_to_tools_ext

# Attach schemas at import time so external clients can discover them
try:
    _attach_schemas_to_tools_ext(mcp, {
        "DENOISE_METHODS": _DENOISE_METHODS,
        "SIMPLIFY_MODES": _SIMPLIFY_MODES,
        "SIMPLIFY_METHODS": _SIMPLIFY_METHODS,
        "PIVOT_METHODS": _PIVOT_METHODS,
        "FORECAST_METHODS": _FORECAST_METHODS,
        "CATEGORY_CHOICES": _CATEGORY_CHOICES if '_CATEGORY_CHOICES' in globals() else [],
        "INDICATOR_NAME_CHOICES": _INDICATOR_NAME_CHOICES if '_INDICATOR_NAME_CHOICES' in globals() else [],
    })
except Exception:
    pass
