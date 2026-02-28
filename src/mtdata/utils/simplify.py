"""
Simplification helpers extracted for reuse across server tools.

Contains target-point selection utilities and core selection algorithms.
"""
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import ruptures as rpt

try:
    from tsdownsample import MinMaxLTTBDownsampler
except Exception:
    MinMaxLTTBDownsampler = None

# Import defaults from core.constants to avoid duplication.
# Use a lazy import to prevent circular imports during initialization.
def _get_simplify_defaults() -> Tuple[float, int, int]:
    """Lazy-load simplify defaults from core.constants to avoid circular imports."""
    try:
        from ..core.constants import (
            SIMPLIFY_DEFAULT_RATIO,
            SIMPLIFY_DEFAULT_MIN_POINTS,
            SIMPLIFY_DEFAULT_MAX_POINTS,
        )
        return (SIMPLIFY_DEFAULT_RATIO, SIMPLIFY_DEFAULT_MIN_POINTS, SIMPLIFY_DEFAULT_MAX_POINTS)
    except ImportError:
        # Fallback if core.constants is not available (e.g., during isolated testing)
        return (0.25, 100, 500)


def _default_target_points(total: int) -> int:
    """Default target points when simplify spec lacks explicit size.

    Uses SIMPLIFY_DEFAULT_RATIO bounded by
    [SIMPLIFY_DEFAULT_MIN_POINTS, SIMPLIFY_DEFAULT_MAX_POINTS].
    """
    try:
        ratio, min_pts, max_pts = _get_simplify_defaults()
        t = int(round(total * ratio))
        t = max(min_pts, min(max_pts, t))
        return max(3, min(t, total))
    except Exception:
        return max(3, min(100, total))


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


_LTTB_DOWNSAMPLER = MinMaxLTTBDownsampler() if MinMaxLTTBDownsampler is not None else None


def _fallback_lttb_indices(y: np.ndarray, n_out: int) -> List[int]:
    """Pure-Python min/max bucket fallback when tsdownsample is unavailable."""
    m = int(len(y))
    if n_out >= m:
        return list(range(m))
    if n_out <= 2 or m <= 2:
        return [0, max(0, m - 1)]

    interior_target = max(0, int(n_out) - 2)
    if interior_target <= 0:
        return [0, m - 1]

    bucket_count = max(1, int(np.ceil(interior_target / 2.0)))
    edges = np.linspace(1, m - 1, num=bucket_count + 1, dtype=int)
    selected: List[int] = []

    for i in range(bucket_count):
        start = int(edges[i])
        stop = int(edges[i + 1])
        if stop <= start:
            stop = min(start + 1, m - 1)
        if start >= m - 1:
            break

        bucket = y[start:stop]
        if bucket.size == 0:
            continue

        base = np.arange(start, stop, dtype=int)
        lo = int(base[int(np.argmin(bucket))])
        hi = int(base[int(np.argmax(bucket))])
        if lo == hi:
            selected.append(lo)
        elif lo < hi:
            selected.extend([lo, hi])
        else:
            selected.extend([hi, lo])

    selected = sorted(set(i for i in selected if 0 < i < (m - 1)))
    if len(selected) < interior_target:
        fill = np.linspace(1, m - 2, num=interior_target, dtype=int).tolist()
        selected = sorted(set(selected + fill))
    if len(selected) > interior_target:
        keep = np.linspace(0, len(selected) - 1, num=interior_target, dtype=int)
        selected = [selected[int(i)] for i in keep]

    return _finalize_indices(m, [0, *selected, m - 1])


def _finalize_indices(n: int, idxs: List[int]) -> List[int]:
    """Ensure first/last indices exist and output is unique/increasing."""
    if n <= 0:
        return []
    if not idxs:
        return list(range(n))
    out = sorted(set(int(i) for i in idxs if 0 <= int(i) < n))
    if not out:
        return [0, n - 1] if n > 1 else [0]
    if out[0] != 0:
        out.insert(0, 0)
    if out[-1] != n - 1:
        out.append(n - 1)
    return out


def _segment_endpoints_to_indices(n: int, bkps: List[int]) -> List[int]:
    """Convert ruptures breakpoints (segment end positions) into row indices."""
    idxs = [0]
    for b in bkps:
        bi = int(b)
        if 0 < bi < n:
            idxs.append(bi - 1)
    idxs.append(n - 1)
    return _finalize_indices(n, idxs)


def _n_bkps_from_segments_points(n: int, segments: Optional[int], points: Optional[int]) -> Optional[int]:
    seg = segments
    if seg is None and points is not None:
        try:
            seg = max(1, int(points) - 1)
        except Exception:
            seg = None
    if seg is None:
        return None
    seg_i = max(1, min(int(seg), n - 1))
    return max(0, seg_i - 1)


def _lttb_select_indices(x: List[float], y: List[float], n_out: int) -> List[int]:
    """Downsampling using tsdownsample when available, else a Python fallback."""
    m = len(x)
    if n_out >= m:
        return list(range(m))
    if n_out <= 2 or m <= 2:
        return [0, max(0, m - 1)]
    yy = np.asarray(y, dtype=float)
    if _LTTB_DOWNSAMPLER is None:
        return _fallback_lttb_indices(yy, int(n_out))
    xx = np.asarray(x, dtype=float)
    idxs = _LTTB_DOWNSAMPLER.downsample(xx, yy, n_out=int(n_out))
    return _finalize_indices(m, np.asarray(idxs, dtype=int).tolist())


def _point_line_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Vertical distance from P to line y(x) through (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    if dx == 0.0:
        return abs(py - (y1 + y2) / 2.0)
    m = (y2 - y1) / dx
    y_on_line = y1 + m * (px - x1)
    return abs(py - y_on_line)


def _rdp_keep_mask(x: List[float], y: List[float], epsilon: float) -> np.ndarray:
    """Iterative Douglas-Peucker mask to avoid an external rdp dependency."""
    n = len(x)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    keep[0] = True
    keep[-1] = True

    stack: List[Tuple[int, int]] = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue

        x0, y0 = x[i0], y[i0]
        x1, y1 = x[i1], y[i1]
        split_idx = -1
        split_dist = -1.0

        for i in range(i0 + 1, i1):
            dist = _point_line_distance(x[i], y[i], x0, y0, x1, y1)
            if dist > split_dist:
                split_idx = i
                split_dist = dist

        if split_idx >= 0 and split_dist > float(epsilon):
            keep[split_idx] = True
            stack.append((i0, split_idx))
            stack.append((split_idx, i1))

    return keep


def _rdp_select_indices(x: List[float], y: List[float], epsilon: float) -> List[int]:
    """Douglas-Peucker simplification returning kept indices."""
    n = len(x)
    if n <= 2 or epsilon <= 0:
        return list(range(n))
    mask = _rdp_keep_mask(list(map(float, x)), list(map(float, y)), float(epsilon))
    idxs = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(int).tolist()
    return _finalize_indices(n, idxs)


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


def _pla_select_indices(
    x: List[float],
    y: List[float],
    max_error: Optional[float] = None,
    segments: Optional[int] = None,
    points: Optional[int] = None,
) -> List[int]:
    """Piecewise-linear selection using ruptures."""
    n = len(x)
    if n <= 2:
        return list(range(n))
    signal = np.column_stack([np.asarray(y, dtype=float), np.asarray(x, dtype=float)])
    algo = rpt.BottomUp(model="linear", min_size=2, jump=1).fit(signal)
    n_bkps = _n_bkps_from_segments_points(n, segments, points)
    if max_error is not None and float(max_error) > 0:
        bkps = algo.predict(epsilon=float(max_error))
    elif n_bkps is not None:
        bkps = algo.predict(n_bkps=int(n_bkps))
    else:
        return list(range(n))
    return _segment_endpoints_to_indices(n, [int(b) for b in bkps])


def _apca_select_indices(
    y: List[float],
    max_error: Optional[float] = None,
    segments: Optional[int] = None,
    points: Optional[int] = None,
) -> List[int]:
    """Piecewise-constant selection using ruptures."""
    n = len(y)
    if n <= 2:
        return list(range(n))
    signal = np.asarray(y, dtype=float).reshape(-1, 1)
    algo = rpt.BottomUp(model="l2", min_size=2, jump=1).fit(signal)
    n_bkps = _n_bkps_from_segments_points(n, segments, points)
    if max_error is not None and float(max_error) > 0:
        bkps = algo.predict(epsilon=float(max_error))
    elif n_bkps is not None:
        bkps = algo.predict(n_bkps=int(n_bkps))
    else:
        return list(range(n))
    return _segment_endpoints_to_indices(n, [int(b) for b in bkps])


def _rdp_autotune_epsilon(x: List[float], y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    """Binary-search epsilon for rdp to keep ~target_points."""
    n = len(x)
    target = max(3, min(int(target_points), n))
    if target >= n:
        return list(range(n)), 0.0
    y_arr = np.asarray(y, dtype=float)
    hi = float(np.ptp(y_arr)) if y_arr.size else 1.0
    if not np.isfinite(hi) or hi <= 0:
        hi = 1.0
    lo = 0.0
    best = _rdp_select_indices(x, y, lo)
    best_eps = lo
    best_diff = abs(len(best) - target)
    for _ in range(max(1, int(max_iter))):
        mid = (lo + hi) / 2.0
        idxs = _rdp_select_indices(x, y, mid)
        diff = abs(len(idxs) - target)
        if diff < best_diff:
            best, best_eps, best_diff = idxs, mid, diff
        if len(idxs) > target:
            lo = mid
        elif len(idxs) < target:
            hi = mid
        else:
            best, best_eps = idxs, mid
            break
        if abs(hi - lo) <= 1e-12:
            break
    return best, float(best_eps)


def _pla_autotune_max_error(x: List[float], y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    """Approximate target points for PLA via segment count."""
    n = len(x)
    target = max(3, min(int(target_points), n))
    if target >= n:
        return list(range(n)), 0.0
    idxs = _pla_select_indices(x, y, max_error=None, segments=max(1, target - 1), points=None)
    return idxs, 0.0


def _apca_autotune_max_error(y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    """Approximate target points for APCA via segment count."""
    n = len(y)
    target = max(3, min(int(target_points), n))
    if target >= n:
        return list(range(n)), 0.0
    idxs = _apca_select_indices(y, max_error=None, segments=max(1, target - 1), points=None)
    return idxs, 0.0


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
        if _LTTB_DOWNSAMPLER is None:
            meta["implementation"] = "python-fallback"
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
        if _LTTB_DOWNSAMPLER is None:
            meta["implementation"] = "python-fallback"
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
        if _LTTB_DOWNSAMPLER is None:
            meta["implementation"] = "python-fallback"
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
        if _LTTB_DOWNSAMPLER is None:
            meta["implementation"] = "python-fallback"
        return idxs, "lttb", meta
    n_out = _choose_simplify_points(len(x), spec)
    idxs = _lttb_select_indices(x, y, n_out)
    meta.update({"points": n_out, "fallback": f"{method}->lttb"})
    if _LTTB_DOWNSAMPLER is None:
        meta["implementation"] = "python-fallback"
    return idxs, "lttb", meta


def _handle_select_mode(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    original_count = len(df)
    if original_count <= 2:
        return df, None

    n_out = _choose_simplify_points(original_count, spec)
    if n_out >= original_count:
        return df, None

    series = None
    if 'close' in df.columns:
        series = df['close'].values
    elif len(headers) > 1:
        for h in headers:
            if h != 'time' and h in df.columns:
                try:
                    series = df[h].astype(float).values
                    break
                except Exception:
                    pass
    if series is None:
        return df, None

    epochs = df['__epoch'].values if '__epoch' in df.columns else np.arange(original_count)
    idxs, method, params = _select_indices_for_timeseries(epochs, series, spec)
    simplified_df = df.iloc[idxs].copy()
    meta: Dict[str, Any] = {
        'mode': 'select',
        'method': method,
        'original_rows': int(original_count),
        'returned_rows': int(len(simplified_df)),
        'points': int(len(simplified_df)),
    }
    if params:
        meta.update(params)
    return simplified_df, meta


def _handle_resample_mode(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    rule = spec.get('rule') or spec.get('interval')
    if not rule:
        return df, {'error': 'Missing rule for resample'}
    try:
        if 'time' in df.columns:
            if '__epoch' in df.columns:
                df = df.set_index(pd.to_datetime(df['__epoch'], unit='s'))
            else:
                df = df.set_index(pd.to_datetime(df['time']))

        agg_map: Dict[str, str] = {}
        for h in headers:
            if h in ['open']:
                agg_map[h] = 'first'
            elif h in ['high']:
                agg_map[h] = 'max'
            elif h in ['low']:
                agg_map[h] = 'min'
            elif h in ['close']:
                agg_map[h] = 'last'
            elif h in ['tick_volume', 'real_volume', 'volume']:
                agg_map[h] = 'sum'
            else:
                agg_map[h] = 'last'

        resampled = df.resample(rule).agg(agg_map).dropna()
        return resampled, {'mode': 'resample', 'rule': rule, 'rows': int(len(resampled))}
    except Exception as exc:
        return df, {'error': f'Resample failed: {exc}'}


def _handle_encode_mode(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    value_col = str(spec.get('value_col') or '').strip()
    if not value_col or value_col not in df.columns:
        if 'close' in df.columns:
            value_col = 'close'
        else:
            value_col = next(
                (
                    h for h in headers
                    if h in df.columns and h != 'time' and pd.api.types.is_numeric_dtype(df[h])
                ),
                '',
            )
    if not value_col:
        return df, {'mode': 'encode', 'error': 'No numeric column available for encoding'}

    vals = pd.to_numeric(df[value_col], errors='coerce').to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 0:
        return df, {'mode': 'encode', 'error': 'No finite values to encode'}

    schema = str(spec.get('schema', 'delta')).lower().strip()
    if schema not in ('delta', 'envelope'):
        schema = 'delta'

    if schema == 'envelope':
        encoded = (
            f"start={float(vals[0]):.6g}|end={float(vals[-1]):.6g}|"
            f"min={float(np.min(vals)):.6g}|max={float(np.max(vals)):.6g}"
        )
    else:
        scale = spec.get('scale', 1.0)
        try:
            scale_f = float(scale)
        except Exception:
            scale_f = 1.0
        scale_f = scale_f if abs(scale_f) > 1e-12 else 1.0
        diffs = np.diff(vals, prepend=vals[0])
        q = np.round(diffs / scale_f).astype(int)
        if bool(spec.get('as_chars', False)):
            zero_char = str(spec.get('zero_char', '0'))[:1] or '0'
            encoded = ''.join('+' if d > 0 else '-' if d < 0 else zero_char for d in q.tolist())
        else:
            encoded = ','.join(str(int(v)) for v in q.tolist())

    out_df = pd.DataFrame([{'encoding': encoded}])
    meta = {
        'mode': 'encode',
        'schema': schema,
        'value_col': value_col,
        'headers': ['encoding'],
        'original_rows': int(len(df)),
        'returned_rows': 1,
        'points': 1,
    }
    return out_df, meta


def _handle_segment_mode(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    value_col = str(spec.get('value_col') or '').strip()
    if not value_col or value_col not in df.columns:
        value_col = 'close' if 'close' in df.columns else ''
    if not value_col:
        return _handle_select_mode(df, headers, spec)

    vals = pd.to_numeric(df[value_col], errors='coerce').to_numpy(dtype=float)
    if vals.size <= 2:
        return df, {'mode': 'segment', 'algo': 'zigzag', 'points': int(len(df))}

    try:
        threshold_pct = float(spec.get('threshold_pct', 0.005))
    except Exception:
        threshold_pct = 0.005
    threshold_pct = max(0.0, threshold_pct)
    if threshold_pct <= 0.0:
        return _handle_select_mode(df, headers, spec)

    idxs: List[int] = [0]
    anchor_val = float(vals[0])
    trend = 0
    for i in range(1, len(vals)):
        cur = float(vals[i])
        denom = max(abs(anchor_val), 1e-12)
        move = (cur - anchor_val) / denom
        if trend >= 0 and cur >= anchor_val:
            anchor_val = cur
            if idxs:
                idxs[-1] = i
            continue
        if trend <= 0 and cur <= anchor_val:
            anchor_val = cur
            if idxs:
                idxs[-1] = i
            continue
        if abs(move) >= threshold_pct:
            trend = 1 if move > 0 else -1
            idxs.append(i)
            anchor_val = cur

    if idxs[-1] != len(vals) - 1:
        idxs.append(len(vals) - 1)
    idxs = sorted(set(int(i) for i in idxs if 0 <= int(i) < len(df)))
    if len(idxs) < 2:
        idxs = [0, len(df) - 1]

    out_df = df.iloc[idxs].copy()
    meta = {
        'mode': 'segment',
        'algo': 'zigzag',
        'threshold_pct': float(threshold_pct),
        'value_col': value_col,
        'original_rows': int(len(df)),
        'returned_rows': int(len(out_df)),
        'points': int(len(out_df)),
    }
    return out_df, meta


def _handle_symbolic_mode(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    value_col = str(spec.get('value_col') or '').strip()
    if not value_col or value_col not in df.columns:
        value_col = 'close' if 'close' in df.columns else ''
    if not value_col:
        return df, {'mode': 'symbolic', 'error': 'No numeric column available for symbolic mode'}

    vals = pd.to_numeric(df[value_col], errors='coerce').to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 0:
        return df, {'mode': 'symbolic', 'error': 'No finite values to symbolize'}

    try:
        paa = int(spec.get('paa', 8))
    except Exception:
        paa = 8
    paa = max(1, min(paa, int(vals.size)))

    alphabet = str(spec.get('alphabet') or 'abcdefghijklmnopqrstuvwxyz')
    alphabet = ''.join(dict.fromkeys(ch for ch in alphabet if ch.strip()))
    if len(alphabet) < 2:
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
    bins_n = min(26, len(alphabet))
    alphabet = alphabet[:bins_n]

    x = vals.copy()
    if bool(spec.get('znorm', True)):
        mu = float(np.mean(x))
        sigma = float(np.std(x))
        if sigma > 1e-12:
            x = (x - mu) / sigma
        else:
            x = x - mu

    chunks = np.array_split(x, paa)
    paa_vals = np.array([float(np.mean(c)) if len(c) else 0.0 for c in chunks], dtype=float)
    quantiles = np.linspace(0.0, 1.0, bins_n + 1)
    edges = np.quantile(paa_vals, quantiles)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    ids = np.searchsorted(edges[1:-1], paa_vals, side='right')
    symbols = ''.join(alphabet[int(i)] for i in ids.tolist())

    out_df = pd.DataFrame([{'symbolic': symbols}])
    meta = {
        'mode': 'symbolic',
        'schema': 'sax',
        'value_col': value_col,
        'paa': int(paa),
        'alphabet': alphabet,
        'headers': ['symbolic'],
        'original_rows': int(len(df)),
        'returned_rows': 1,
        'points': 1,
    }
    return out_df, meta


def _simplify_dataframe_rows_ext(
    df: pd.DataFrame,
    headers: List[str],
    simplify: Optional[Dict[str, Any]],
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Extended simplify dispatcher shared by core and service adapters."""
    if df.empty:
        return df, None

    from ..core.constants import SIMPLIFY_DEFAULT_MODE

    spec = dict(simplify) if simplify else {}
    mode = str(spec.get('mode', SIMPLIFY_DEFAULT_MODE)).lower().strip() or SIMPLIFY_DEFAULT_MODE

    if mode == 'resample':
        return _handle_resample_mode(df, headers, spec)
    if mode == 'encode':
        return _handle_encode_mode(df, headers, spec)
    if mode == 'segment':
        return _handle_segment_mode(df, headers, spec)
    if mode == 'symbolic':
        return _handle_symbolic_mode(df, headers, spec)
    return _handle_select_mode(df, headers, spec)


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
        spec_eff = dict(simplify)
        spec_eff["mode"] = mode

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
        n_out = _choose_simplify_points(total, spec_eff)
        
        if mode == "resample" and "__epoch" in df.columns:
            bs = spec_eff.get("bucket_seconds")
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

        return _simplify_dataframe_rows_ext(df, headers, spec_eff)
        
    except Exception:
        return df, None
