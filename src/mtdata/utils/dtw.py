from __future__ import annotations

from math import sqrt
from typing import Optional

import numpy as np


def dtw_distance_fallback(
    a: np.ndarray,
    b: np.ndarray,
    *,
    sakoe_chiba_radius: Optional[int] = None,
) -> float:
    """Compute a simple 1D DTW distance without optional third-party backends."""
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)

    if x.size == 0 and y.size == 0:
        return 0.0
    if x.size == 0 or y.size == 0:
        return float("inf")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return float("inf")

    radius: Optional[int] = None
    if sakoe_chiba_radius is not None:
        radius = max(int(sakoe_chiba_radius), abs(int(x.size) - int(y.size)))

    prev = np.full(y.size + 1, np.inf, dtype=float)
    prev[0] = 0.0

    for i in range(1, x.size + 1):
        curr = np.full(y.size + 1, np.inf, dtype=float)
        if radius is None:
            j_start = 1
            j_end = y.size
        else:
            j_start = max(1, i - radius)
            j_end = min(y.size, i + radius)
            if j_start > j_end:
                prev = curr
                continue

        xi = float(x[i - 1])
        for j in range(j_start, j_end + 1):
            diff = xi - float(y[j - 1])
            cost = diff * diff
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr

    distance = float(prev[y.size])
    if not np.isfinite(distance):
        return float("inf")
    return float(sqrt(max(distance, 0.0)))
