"""Shared lightweight market-regime heuristics."""

from typing import Any, Dict, Optional

import numpy as np


def infer_market_regime(
    prices: Any,
    *,
    window_bars: int = 160,
    efficiency_threshold: float = 0.35,
    trend_strength_threshold: float = 1.25,
    min_bars: int = 20,
) -> Optional[Dict[str, Any]]:
    """Classify a recent price path as trending, ranging, or transition."""
    try:
        values = np.asarray(prices, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    values = values[np.isfinite(values)]
    if values.size < min_bars:
        return None

    effective_window = min(max(min_bars, int(window_bars)), int(values.size))
    segment = values[-effective_window:]
    diffs = np.diff(segment)
    path_length = float(np.sum(np.abs(diffs))) if diffs.size else 0.0
    move = float(segment[-1] - segment[0])
    base_price = float(segment[0]) if abs(float(segment[0])) > 1e-9 else 1e-9
    trend_strength = float(abs(move) / max(float(np.nanstd(segment)), 1e-9))
    efficiency_ratio = float(abs(move) / max(path_length, 1e-9))
    ranging_threshold = max(0.1, 0.55 * float(efficiency_threshold))

    if (
        efficiency_ratio >= float(efficiency_threshold)
        and trend_strength >= float(trend_strength_threshold)
    ):
        state = "trending"
    elif efficiency_ratio <= ranging_threshold:
        state = "ranging"
    else:
        state = "transition"

    direction = "neutral"
    if move > 1e-9:
        direction = "bullish"
    elif move < -1e-9:
        direction = "bearish"

    return {
        "state": state,
        "direction": direction,
        "window_bars": int(effective_window),
        "trend_strength": trend_strength,
        "efficiency_ratio": efficiency_ratio,
        "window_move_pct": (move / base_price) * 100.0,
    }
