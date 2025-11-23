
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

from .schema import SimplifySpec
from .constants import SIMPLIFY_DEFAULT_METHOD, SIMPLIFY_DEFAULT_MODE, SIMPLIFY_DEFAULT_MIN_POINTS, SIMPLIFY_DEFAULT_MAX_POINTS, SIMPLIFY_DEFAULT_RATIO
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from .schema import SimplifySpec
from ..services.simplification import _simplify_dataframe_rows_ext as _simplify_impl
from ..utils.simplify import (
    _choose_simplify_points, _select_indices_for_timeseries, _lttb_select_indices,
    _rdp_select_indices, _pla_select_indices, _apca_select_indices
)

# Export helper functions that were previously available here
__all__ = [
    '_simplify_dataframe_rows_ext',
    '_choose_simplify_points',
    '_select_indices_for_timeseries',
    '_lttb_select_indices',
    '_rdp_select_indices',
    '_pla_select_indices',
    '_apca_select_indices'
]

def _simplify_dataframe_rows_ext(df: pd.DataFrame, headers: List[str], simplify: SimplifySpec) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Delegate to services.simplification._simplify_dataframe_rows_ext
    """
    return _simplify_impl(df, headers, simplify)

def _default_target_points(total: int) -> int:
    """Default target points when simplify requested without explicit points/ratio."""
    from ..utils.simplify import _default_target_points as _impl
    return _impl(total)


def _choose_simplify_points(total: int, spec: Dict[str, Any]) -> int:
    """Determine target number of points from a simplify spec."""
    return _choose_simplify_points_util(total, spec)


def _point_line_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Delegate to utils implementation."""
    from ..utils.simplify import _point_line_distance as _impl
    return _impl(px, py, x1, y1, x2, y2)


def _rdp_select_indices(x: List[float], y: List[float], epsilon: float) -> List[int]:
    return _rdp_select_indices_util(x, y, epsilon)


def _max_line_error(x: List[float], y: List[float], i0: int, i1: int) -> float:
    """Delegate to utils implementation.""" 
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
