"""Microstructure denoisers: Jacod-style pre-averaging."""
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..base import _series_like, register_filter


def _preaverage_increments(
    increments: np.ndarray,
    window: int,
    causality: str,
) -> np.ndarray:
    count = len(increments)
    if count == 0:
        return increments.astype(float, copy=True)
    kn = max(2, int(window))
    values = np.asarray(increments, dtype=float)
    if causality == "zero_phase":
        half = kn // 2
        width = 2 * half + 1
        kernel = np.minimum(np.arange(1, width + 1, dtype=float), np.arange(width, 0, -1, dtype=float))
        kernel /= float(np.sum(kernel))
        padded = np.pad(values, (half, half), mode="edge")
        return np.lib.stride_tricks.sliding_window_view(padded, width) @ kernel

    kernel = np.arange(1, kn + 1, dtype=float)
    padded = np.pad(values, (kn - 1, 0), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, kn)
    finite = np.isfinite(windows)
    filled = np.where(finite, windows, 0.0)
    weights = np.where(finite, kernel, 0.0)
    denom = np.sum(weights, axis=1)
    numerators = np.sum(filled * weights, axis=1)
    return np.divide(numerators, denom, out=np.zeros(count, dtype=float), where=denom > 0.0)


def _preaverage_1d(
    x: np.ndarray,
    window: int,
    causality: str,
    space: str,
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    n = len(values)
    if n < 2:
        return values.copy()
    if space == "log":
        if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
            raise ValueError("preaverage space='log' requires strictly positive finite values")
        working = np.log(values)
    else:
        working = values
    increments = np.diff(working)
    filtered = _preaverage_increments(increments, window=window, causality=causality)
    reconstructed = np.empty(n, dtype=float)
    reconstructed[0] = working[0]
    reconstructed[1:] = working[0] + np.cumsum(filtered)
    if space == "log":
        return np.exp(reconstructed)
    return reconstructed


@register_filter("preaverage")
def _denoise_preaverage_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    window = max(2, int(params.get("window", 10)))
    space = str(params.get("space", "level")).strip().lower()
    if space not in {"level", "log"}:
        raise ValueError("preaverage space must be 'level' or 'log'")
    y = _preaverage_1d(x, window=window, causality=causality, space=space)
    return _series_like(s, y)
