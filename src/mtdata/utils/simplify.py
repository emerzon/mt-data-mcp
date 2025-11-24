"""
Simplification helpers extracted for reuse across server tools.

Contains target-point selection utilities and core selection algorithms.
"""
from typing import Any, Dict, List, Optional, Tuple, Callable
import math

import pandas as pd
import numpy as np

# Local defaults to avoid circular import with core package during initialization.
# Keep in sync with src/mtdata/core/constants.py
SIMPLIFY_DEFAULT_RATIO = 0.25
SIMPLIFY_DEFAULT_MIN_POINTS = 100
SIMPLIFY_DEFAULT_MAX_POINTS = 500


def _default_target_points(total: int) -> int:
    """Default target points when simplify spec lacks explicit size.

    Uses SIMPLIFY_DEFAULT_RATIO bounded by
    [SIMPLIFY_DEFAULT_MIN_POINTS, SIMPLIFY_DEFAULT_MAX_POINTS].
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
                if 0 < r < 1:
                    n = int(max(3, round(total * r)))
            except Exception:
                pass
        if n is None:
            if spec and ("method" in spec or len(spec) > 0):
                return _default_target_points(total)
            return total
        n = max(3, min(int(n), total))
        return n
    except Exception:
        return total


# ---- Core selection algorithms ----
def _lttb_select_indices(x: List[float], y: List[float], n_out: int) -> List[int]:
    """Largest-Triangle-Three-Buckets (LTTB) downsampling.

    Returns the indices of selected points (monotonic, includes first/last).
    """
    try:
        m = len(x)
        if n_out >= m:
            return list(range(m))
        if n_out <= 2 or m <= 2:
            return [0, max(0, m - 1)]
        idxs: List[int] = [0]
        bucket_size = (m - 2) / float(n_out - 2)
        a = 0
        for i in range(0, n_out - 2):
            start = int(math.floor(1 + i * bucket_size))
            end = int(math.floor(1 + (i + 1) * bucket_size))
            end = min(end, m - 1)
            if start >= end:
                start = max(min(start, m - 2), 1)
                end = start + 1

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
        out = []
        last = -1
        for ix in sorted(set(idxs)):
            if ix > last:
                out.append(ix)
                last = ix
        return out
    except Exception:
        return list(range(len(x)))


def _point_line_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Vertical distance from P to line y(x) through (x1,y1)-(x2,y2)."""
    try:
        dx = x2 - x1
        if dx == 0.0:
            return abs(py - (y1 + y2) / 2.0)
        m = (y2 - y1) / dx
        y_on_line = y1 + m * (px - x1)
        return abs(py - y_on_line)
    except Exception:
        return 0.0


def _uniform_segment_indices(n: int, segments: int) -> List[int]:
    """Return indices that split a sequence of length n into `segments` segments."""
    if n <= 2 or segments <= 1:
        return list(range(n))
    segments = max(1, min(int(segments), n - 1))
    idxs = [0]
    for k in range(1, segments):
        idxs.append(int(round(k * (n - 1) / segments)))
    idxs.append(n - 1)
    return sorted(set(idxs))


def _line_deviation_hi(x: List[float], y: List[float]) -> float:
    """Compute a robust upper bound for line-based deviation in [x,y]."""
    n = len(x)
    if n == 0:
        return 1.0
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    dx = x1 - x0
    if dx == 0:
        base = sum(y) / float(n)
        return max(abs(v - base) for v in y) if n > 0 else 1.0
    m = (y1 - y0) / dx
    hi = 0.0
    for i in range(n):
        yline = y0 + m * (x[i] - x0)
        hi = max(hi, abs(y[i] - yline))
    if hi <= 0:
        rng = (max(y) - min(y)) if n > 1 else 1.0
        hi = max(1e-12, rng)
    return float(hi)


def _abs_deviation_hi(y: List[float]) -> float:
    """Upper bound for absolute deviation from the mean for APCA."""
    n = len(y)
    if n == 0:
        return 1.0
    base = sum(y) / float(n)
    hi = max(abs(v - base) for v in y) if n > 0 else 1.0
    if hi <= 0:
        rng = (max(y) - min(y)) if n > 1 else 1.0
        hi = max(1e-12, rng)
    return float(hi)


def _binary_search_param(
    target_points: int,
    lo: float,
    hi: float,
    iter_limit: int,
    compute_indices: Callable[[float], List[int]],
) -> Tuple[List[int], float]:
    """Generic binary search to find a parameter so that indices length ~ target_points.

    Returns (best_indices, best_param). On ties, prefers smaller parameter value.
    """
    target = max(3, int(target_points))
    best_idxs = compute_indices(lo)
    best_p = lo
    best_diff = abs(len(best_idxs) - target)
    for _ in range(max(1, int(iter_limit))):
        mid = (lo + hi) / 2.0
        idxs = compute_indices(mid)
        cnt = len(idxs)
        diff = abs(cnt - target)
        if diff < best_diff or (diff == best_diff and mid < best_p):
            best_idxs = idxs
            best_p = mid
            best_diff = diff
        if cnt > target:
            lo = mid if mid > lo else lo + (hi - lo) * 0.5
        elif cnt < target:
            hi = mid
        else:
            break
        if hi - lo <= 1e-12:
            break
    return best_idxs, float(best_p)


def _rdp_select_indices(x: List[float], y: List[float], epsilon: float) -> List[int]:
    """Douglas–Peucker simplification returning kept indices."""
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


def _finalize_indices(n: int, idxs: List[int]) -> List[int]:
    """Ensure last index included and indices are strictly increasing unique."""
    if not idxs:
        return list(range(n))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    out: List[int] = []
    for i in idxs:
        ii = int(i)
        if not out or ii > out[-1]:
            out.append(ii)
    return out


def _greedy_segment_indices(n: int, check_ok: Callable[[int, int], bool]) -> List[int]:
    """Generic greedy grow-and-split segment index selector using a monotone check.

    Expands [start:end] while check_ok(start,end) holds, then commits the last good end.
    """
    idxs: List[int] = [0]
    start = 0
    while start < n - 1:
        end = start + 1
        last_good = end
        while end < n:
            if check_ok(start, end):
                last_good = end
                end += 1
            else:
                break
        if last_good == start:
            last_good = start + 1
        idxs.append(last_good)
        start = last_good
    return idxs


def _pla_select_indices(
    x: List[float],
    y: List[float],
    max_error: Optional[float] = None,
    segments: Optional[int] = None,
    points: Optional[int] = None,
) -> List[int]:
    """Piecewise Linear Approximation; greedy by max perpendicular error or uniform segments."""
    n = len(x)
    if n <= 2:
        return list(range(n))
    if (max_error is None or max_error <= 0) and (segments or points):
        if points and (segments is None):
            segments = max(1, int(points) - 1)
        return _uniform_segment_indices(n, int(segments or 1))
    if max_error is None or max_error <= 0:
        return list(range(n))
    idxs = _greedy_segment_indices(n, lambda s, e: _max_line_error(x, y, s, e) <= float(max_error))
    return _finalize_indices(n, idxs)


def _apca_select_indices(
    y: List[float],
    max_error: Optional[float] = None,
    segments: Optional[int] = None,
    points: Optional[int] = None,
) -> List[int]:
    """Adaptive Piecewise Constant Approximation; greedy by max absolute deviation or uniform."""
    n = len(y)
    if n <= 2:
        return list(range(n))
    if (max_error is None or max_error <= 0) and (segments or points):
        if points and (segments is None):
            segments = max(1, int(points) - 1)
        return _uniform_segment_indices(n, int(segments or 1))
    if max_error is None or max_error <= 0:
        return list(range(n))
    def _ok(s: int, e: int) -> bool:
        seg = y[s:e + 1]
        mean = sum(seg) / float(len(seg))
        err = max(abs(v - mean) for v in seg)
        return err <= float(max_error)
    idxs = _greedy_segment_indices(n, _ok)
    return _finalize_indices(n, idxs)


def _rdp_autotune_epsilon(x: List[float], y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    """Binary search epsilon so RDP keeps ~target_points. Returns (indices, epsilon)."""
    n = len(x)
    target = max(3, min(int(target_points), n))
    if target >= n:
        return list(range(n)), 0.0
    lo = 0.0
    hi = _line_deviation_hi(x, y)
    return _binary_search_param(target, lo, hi, max_iter, lambda mid: _rdp_select_indices(x, y, mid))


def _pla_autotune_max_error(x: List[float], y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    """Binary search max_error so PLA keeps ~target_points. Returns (indices, max_error)."""
    n = len(x)
    target = max(3, min(int(target_points), n))
    if target >= n:
        return list(range(n)), 0.0
    lo = 0.0
    hi = _line_deviation_hi(x, y)
    return _binary_search_param(target, lo, hi, max_iter, lambda mid: _pla_select_indices(x, y, mid, None, None))


def _apca_autotune_max_error(y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    """Binary search max_error so APCA keeps ~target_points. Returns (indices, max_error)."""
    n = len(y)
    target = max(3, min(int(target_points), n))
    if target >= n:
        return list(range(n)), 0.0
    lo = 0.0
    hi = _abs_deviation_hi(y)
    return _binary_search_param(target, lo, hi, max_iter, lambda mid: _apca_select_indices(y, mid, None, None))


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
        eps = spec.get("epsilon", None) or spec.get("tolerance", None) or spec.get("eps", None)
        try:
            eps_val = float(eps) if eps is not None else None
        except Exception:
            eps_val = None
        if eps_val is not None and eps_val > 0:
            idxs = _rdp_select_indices(x, y, eps_val)
            meta.update({"epsilon": eps_val})
            return idxs, "rdp", meta
        target = spec.get("points") or spec.get("target_points") or spec.get("max_points") or None
        if target is None and spec.get("ratio") is not None:
            try:
                r = float(spec.get("ratio"))
                if 0 < r < 1:
                    target = max(3, int(round(len(x) * r)))
            except Exception:
                pass
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
        n_out = _choose_simplify_points(len(x), spec)
        idxs = _lttb_select_indices(x, y, n_out)
        meta.update({"points": n_out, "fallback": "rdp->lttb"})
        return idxs, "lttb", meta
    if method == "pla":
        max_error = spec.get("max_error", None)
        try:
            me = float(max_error) if max_error is not None else None
        except Exception:
            me = None
        segments = spec.get("segments", None)
        points = spec.get("points", None) or spec.get("target_points", None) or spec.get("max_points", None)
        if me is not None and me > 0:
            idxs = _pla_select_indices(x, y, me, None, None)
            meta.update({"max_error": me})
            return idxs, "pla", meta
        if segments is not None:
            idxs = _pla_select_indices(x, y, None, segments, None)
            meta.update({"segments": segments})
            return idxs, "pla", meta
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
        n_out = _choose_simplify_points(len(x), spec)
        idxs = _lttb_select_indices(x, y, n_out)
        meta.update({"points": n_out, "fallback": "apca->lttb"})
        return idxs, "lttb", meta
    n_out = _choose_simplify_points(len(x), spec)
    idxs = _lttb_select_indices(x, y, n_out)
    meta.update({"points": n_out, "fallback": f"{method}->lttb"})
    return idxs, "lttb", meta


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
        
        # Import constants to avoid circular imports
        from ..core.constants import SIMPLIFY_DEFAULT_METHOD, SIMPLIFY_DEFAULT_MODE
        
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

        # For other modes, delegate to services/simplification for now
        # This maintains compatibility while allowing future consolidation
        from ..services.simplification import _simplify_dataframe_rows_ext as _simplify_ext
        return _simplify_ext(df, headers, simplify)
        
    except Exception:
        return df, None
