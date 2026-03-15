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
