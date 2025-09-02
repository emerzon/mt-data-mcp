from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from ..core.constants import (
    SIMPLIFY_DEFAULT_MAX_POINTS,
    SIMPLIFY_DEFAULT_MIN_POINTS,
    SIMPLIFY_DEFAULT_RATIO,
)


def _default_simplify_points(total: int) -> int:
    if total <= SIMPLIFY_DEFAULT_MIN_POINTS:
        return total
    t = int(round(total * SIMPLIFY_DEFAULT_RATIO))
    return max(min(t, SIMPLIFY_DEFAULT_MAX_POINTS), 3)


def _choose_simplify_points(total: int, spec: Dict[str, Any]) -> int:
    if not spec:
        return _default_simplify_points(total)
    if isinstance(spec.get("points"), int) and spec["points"] >= 3:
        return int(spec["points"])
    ratio = spec.get("ratio")
    if isinstance(ratio, (int, float)) and ratio > 0 and ratio < 1:
        n = int(max(3, round(total * ratio)))
        return min(n, total)
    return _default_simplify_points(total)


def _lttb_select_indices(x: List[float], y: List[float], n_out: int) -> List[int]:
    n = len(x)
    if n_out >= n:
        return list(range(n))
    if n_out <= 2:
        return [0, n - 1]
    bucket_size = (n - 2) / (n_out - 2)
    a = 0
    idxs = [0]
    for i in range(1, n_out - 1):
        start = int(round((i - 1) * bucket_size)) + 1
        end = int(round(i * bucket_size)) + 1
        end = min(end, n - 1)
        start = min(start, end)
        avg_range_start = int(round(i * bucket_size)) + 1
        avg_range_end = int(round((i + 1) * bucket_size)) + 1
        avg_range_end = min(avg_range_end, n)
        avg_x = np.mean(x[avg_range_start:avg_range_end]) if avg_range_start < avg_range_end else x[end]
        avg_y = np.mean(y[avg_range_start:avg_range_end]) if avg_range_start < avg_range_end else y[end]
        best = None
        best_area = -1
        for j in range(start, end):
            area = abs((x[a] - avg_x) * (y[j] - y[a]) - (x[a] - x[j]) * (avg_y - y[a]))
            if area > best_area:
                best_area = area
                best = j
        if best is None:
            best = start
        idxs.append(best)
        a = best
    idxs.append(n - 1)
    return idxs


def _ramer_douglas_peucker_indices(x: List[float], y: List[float], n_out: int) -> List[int]:
    n = len(x)
    if n_out >= n:
        return list(range(n))
    if n_out <= 2:
        return [0, n - 1]
    keep = {0, n - 1}

    def perp_dist(i, a, b):
        x1, y1 = x[a], y[a]
        x2, y2 = x[b], y[b]
        x0, y0 = x[i], y[i]
        if x1 == x2 and y1 == y2:
            return np.hypot(x0 - x1, y0 - y1)
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = np.hypot(y2 - y1, x2 - x1)
        return num / den

    segs: List[Tuple[int, int]] = [(0, n - 1)]
    while len(keep) < n_out and segs:
        a, b = segs.pop(0)
        # find furthest point
        dmax = -1
        idx = None
        for i in range(a + 1, b):
            d = perp_dist(i, a, b)
            if d > dmax:
                dmax = d
                idx = i
        if idx is None:
            continue
        keep.add(idx)
        segs.append((a, idx))
        segs.append((idx, b))
        segs.sort(key=lambda t: t[1] - t[0], reverse=True)
    return sorted(keep)


def _uniform_select_indices(n: int, n_out: int) -> List[int]:
    if n_out >= n:
        return list(range(n))
    if n_out <= 2:
        return [0, n - 1]
    idxs = [0]
    segments = n_out - 1
    for k in range(1, segments):
        idxs.append(int(round(k * (n - 1) / segments)))
    idxs.append(n - 1)
    return idxs


def _peak_valley_select_indices(x: List[float], y: List[float], n_out: int) -> List[int]:
    n = len(x)
    if n_out >= n:
        return list(range(n))
    if n_out <= 2:
        return [0, n - 1]
    idxs = [0]
    segments = n_out - 1
    for k in range(1, segments):
        idxs.append(int(round(k * (n - 1) / segments)))
    idxs.append(n - 1)
    return idxs


def _select_indices(x: List[float], y: List[float], spec: Dict[str, Any]) -> List[int]:
    n_out = _choose_simplify_points(len(x), spec)
    method = str(spec.get("method", "lttb")).lower().strip()
    if method == "lttb":
        return _lttb_select_indices(x, y, n_out)
    if method in ("rdp", "douglas-peucker"):
        return _ramer_douglas_peucker_indices(x, y, n_out)
    if method == "uniform":
        return _uniform_select_indices(len(x), n_out)
    if method in ("pv", "peak-valley"):
        return _peak_valley_select_indices(x, y, n_out)
    return _lttb_select_indices(x, y, n_out)


def _simplify_dataframe_rows(df: pd.DataFrame, headers: List[str], simplify: Optional[Dict[str, Any]]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    if not simplify:
        return df, None
    total = len(df)
    if total <= 3:
        return df, None
    method = str(simplify.get("method", "lttb")).lower().strip()
    n_out = _choose_simplify_points(total, simplify)

    # choose representative indices on price-like column
    y_col = None
    for c in ("close", "C", "price", "mid", "bid", "ask"):
        if c in df.columns:
            y_col = c
            break
    if y_col is None:
        y_col = [c for c in df.columns if c != "time" and pd.api.types.is_numeric_dtype(df[c])][0]

    x = list(range(total))
    y = df[y_col].astype(float).tolist()
    idxs = _select_indices(x, y, simplify)
    idxs = sorted(set(idxs))
    idxs = [i for i in idxs if 0 <= i < total]
    idxs = sorted(set([0, total - 1] + idxs))
    out = df.iloc[idxs].reset_index(drop=True)

    meta = {
        "method": (method or "lttb"),
        "points": len(out),
        "original_points": total,
    }
    return out, meta

