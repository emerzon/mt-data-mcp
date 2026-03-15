from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class PatternResultBase:
    confidence: float
    start_index: int
    end_index: int
    start_time: Optional[float]
    end_time: Optional[float]

    @staticmethod
    def resolve_time(times: Any, index: int) -> Optional[float]:
        try:
            idx = int(index)
        except (TypeError, ValueError):
            return None
        if idx < 0:
            return None
        try:
            arr = np.asarray(times, dtype=float)
        except (TypeError, ValueError):
            return None
        if arr.ndim == 0 or arr.size <= idx:
            return None
        value = float(arr[idx])
        return value if np.isfinite(value) else None


def interval_overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Return the inclusive overlap ratio between two index intervals."""
    lo = max(int(a_start), int(b_start))
    hi = min(int(a_end), int(b_end))
    inter = max(0, hi - lo + 1)
    union = max(int(a_end), int(b_end)) - min(int(a_start), int(b_start)) + 1
    if union <= 0:
        return 0.0
    return float(inter) / float(union)
