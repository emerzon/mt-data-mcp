from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.utils import to_float_np
from .common import PatternResultBase


@dataclass
class FractalDetectorConfig:
    left_bars: int = 2
    right_bars: int = 2
    breakout_basis: str = "close"
    min_prominence_pct: float = 0.0
    confidence_prominence_cap_pct: float = 1.0


@dataclass
class FractalPatternResult(PatternResultBase):
    name: str
    status: str  # "forming" | "completed"
    direction: str
    price: float
    details: Dict[str, Any] = field(default_factory=dict)


def validate_fractal_detector_config(
    cfg: FractalDetectorConfig,
) -> list[str]:
    warnings: list[str] = []
    if int(cfg.left_bars) < 1:
        warnings.append(f"left_bars must be >= 1, got {cfg.left_bars}")
    if int(cfg.right_bars) < 1:
        warnings.append(f"right_bars must be >= 1, got {cfg.right_bars}")
    if float(cfg.min_prominence_pct) < 0.0:
        warnings.append(
            f"min_prominence_pct must be >= 0.0, got {cfg.min_prominence_pct}"
        )
    if float(cfg.confidence_prominence_cap_pct) <= 0.0:
        warnings.append(
            "confidence_prominence_cap_pct must be > 0.0, "
            f"got {cfg.confidence_prominence_cap_pct}"
        )
    breakout_basis = str(cfg.breakout_basis or "").strip().lower()
    if breakout_basis not in {"close", "high_low"}:
        warnings.append(
            "breakout_basis must be 'close' or 'high_low', "
            f"got {cfg.breakout_basis!r}"
        )
    return warnings


def _is_bearish_fractal(
    center: float,
    left: np.ndarray,
    right: np.ndarray,
    left_bars: int,
    right_bars: int,
) -> bool:
    if left.size != left_bars or right.size != right_bars:
        return False
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return False
    return bool(np.all(center > left) and np.all(center >= right))


def _is_bullish_fractal(
    center: float,
    left: np.ndarray,
    right: np.ndarray,
    left_bars: int,
    right_bars: int,
) -> bool:
    if left.size != left_bars or right.size != right_bars:
        return False
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return False
    return bool(np.all(center < left) and np.all(center <= right))


def _fractal_prominence_pct(
    *,
    direction: str,
    level_price: float,
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> float:
    reference = max(abs(float(level_price)), 1e-12)
    if str(direction).lower() == "bearish":
        shoulder = max(float(np.max(left_values)), float(np.max(right_values)))
        prominence = max(0.0, float(level_price) - shoulder)
    else:
        shoulder = min(float(np.min(left_values)), float(np.min(right_values)))
        prominence = max(0.0, shoulder - float(level_price))
    return float((prominence / reference) * 100.0)


def _breakout_basis_series(
    *,
    direction: str,
    breakout_basis: str,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> tuple[np.ndarray, str]:
    direction_key = str(direction).strip().lower()
    basis_key = str(breakout_basis or "close").strip().lower()
    if direction_key == "bearish":
        if basis_key == "high_low":
            return highs, "bullish"
        return closes, "bullish"
    if basis_key == "high_low":
        return lows, "bearish"
    return closes, "bearish"


def _find_breakout(
    *,
    direction: str,
    level_price: float,
    confirmation_index: int,
    breakout_basis: str,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> tuple[Optional[int], Optional[float], Optional[str]]:
    series, breakout_direction = _breakout_basis_series(
        direction=direction,
        breakout_basis=breakout_basis,
        highs=highs,
        lows=lows,
        closes=closes,
    )
    future = series[int(confirmation_index) + 1:]
    if future.size <= 0:
        return None, None, None

    if str(direction).strip().lower() == "bearish":
        breakout_hits = np.flatnonzero(future > float(level_price))
    else:
        breakout_hits = np.flatnonzero(future < float(level_price))
    if breakout_hits.size <= 0:
        return None, None, None

    breakout_offset = int(breakout_hits[0])
    breakout_index = int(confirmation_index) + 1 + breakout_offset
    breakout_price = float(future[breakout_offset])
    return breakout_index, breakout_price, breakout_direction


def _confidence_from_prominence(
    prominence_pct: float,
    cfg: FractalDetectorConfig,
) -> float:
    cap = max(float(cfg.confidence_prominence_cap_pct), 1e-12)
    prominence_norm = min(max(float(prominence_pct), 0.0) / cap, 1.0)
    return float(max(0.0, min(1.0, 0.55 + (0.35 * prominence_norm))))


def _build_fractal_result(
    *,
    direction: str,
    center_index: int,
    confirmation_index: int,
    level_price: float,
    prominence_pct: float,
    breakout_basis: str,
    breakout_index: Optional[int],
    breakout_price: Optional[float],
    breakout_direction: Optional[str],
    times: np.ndarray,
    n_bars: int,
    cfg: FractalDetectorConfig,
) -> FractalPatternResult:
    fractal_time = PatternResultBase.resolve_time(times, int(center_index))
    confirmation_time = PatternResultBase.resolve_time(times, int(confirmation_index))
    is_broken = breakout_index is not None and breakout_direction is not None
    level_state = "broken" if is_broken else "active"
    status = "completed" if is_broken else "forming"
    bias = (
        str(breakout_direction)
        if is_broken and breakout_direction not in (None, "")
        else str(direction)
    )
    end_index = int(breakout_index) if is_broken else int(confirmation_index)
    breakout_time = (
        PatternResultBase.resolve_time(times, int(breakout_index))
        if is_broken
        else None
    )
    end_time = breakout_time if is_broken else confirmation_time
    details: Dict[str, Any] = {
        "pattern_family": "fractal",
        "bias": bias,
        "fractal_direction": str(direction),
        "level_role": "support" if str(direction) == "bullish" else "resistance",
        "level_state": level_state,
        "fractal_index": int(center_index),
        "confirmation_index": int(confirmation_index),
        "confirmation_bars": int(cfg.right_bars),
        "bars_since_confirmation": int(max((n_bars - 1) - int(confirmation_index), 0)),
        "prominence_pct": float(np.round(float(prominence_pct), 8)),
        "breakout_basis": str(breakout_basis),
    }
    if fractal_time is not None:
        details["fractal_time"] = float(fractal_time)
    if confirmation_time is not None:
        details["confirmation_time"] = float(confirmation_time)
    if is_broken:
        details["breakout_direction"] = str(breakout_direction)
        details["breakout_index"] = int(breakout_index)
        details["breakout_bars_after_confirmation"] = int(
            int(breakout_index) - int(confirmation_index)
        )
        if breakout_price is not None and np.isfinite(float(breakout_price)):
            details["breakout_price"] = float(breakout_price)
        if breakout_time is not None:
            details["breakout_time"] = float(breakout_time)
    return FractalPatternResult(
        confidence=_confidence_from_prominence(prominence_pct, cfg),
        start_index=int(center_index),
        end_index=end_index,
        start_time=fractal_time,
        end_time=end_time,
        name=f"{str(direction).title()} Fractal",
        status=status,
        direction=str(direction),
        price=float(level_price),
        details=details,
    )


def detect_fractal_patterns(
    df: pd.DataFrame,
    cfg: Optional[FractalDetectorConfig] = None,
) -> List[FractalPatternResult]:
    if cfg is None:
        cfg = FractalDetectorConfig()
    if not isinstance(df, pd.DataFrame) or "close" not in df.columns:
        return []

    try:
        left_bars = max(1, int(cfg.left_bars))
        right_bars = max(1, int(cfg.right_bars))
    except Exception:
        return []
    breakout_basis = str(cfg.breakout_basis or "close").strip().lower()
    if breakout_basis not in {"close", "high_low"}:
        breakout_basis = "close"

    highs = to_float_np(df["high"]) if "high" in df.columns else to_float_np(df["close"])
    lows = to_float_np(df["low"]) if "low" in df.columns else to_float_np(df["close"])
    closes = to_float_np(df["close"])
    if highs.size != closes.size:
        highs = closes
    if lows.size != closes.size:
        lows = closes
    n_bars = int(closes.size)
    if n_bars < int(left_bars + right_bars + 1):
        return []

    if "time" in df.columns:
        times = to_float_np(df["time"])
        if times.size != n_bars:
            times = np.arange(n_bars, dtype=float)
    else:
        times = np.arange(n_bars, dtype=float)

    results: List[FractalPatternResult] = []
    min_prominence_pct = max(0.0, float(cfg.min_prominence_pct))

    for center_index in range(left_bars, n_bars - right_bars):
        high = float(highs[center_index])
        low = float(lows[center_index])
        if not (np.isfinite(high) and np.isfinite(low)):
            continue

        left_highs = highs[center_index - left_bars:center_index]
        right_highs = highs[center_index + 1:center_index + 1 + right_bars]
        left_lows = lows[center_index - left_bars:center_index]
        right_lows = lows[center_index + 1:center_index + 1 + right_bars]

        if _is_bearish_fractal(high, left_highs, right_highs, left_bars, right_bars):
            prominence_pct = _fractal_prominence_pct(
                direction="bearish",
                level_price=high,
                left_values=left_highs,
                right_values=right_highs,
            )
            if prominence_pct >= min_prominence_pct:
                confirmation_index = int(center_index + right_bars)
                breakout_index, breakout_price, breakout_direction = _find_breakout(
                    direction="bearish",
                    level_price=high,
                    confirmation_index=confirmation_index,
                    breakout_basis=breakout_basis,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                )
                results.append(
                    _build_fractal_result(
                        direction="bearish",
                        center_index=center_index,
                        confirmation_index=confirmation_index,
                        level_price=high,
                        prominence_pct=prominence_pct,
                        breakout_basis=breakout_basis,
                        breakout_index=breakout_index,
                        breakout_price=breakout_price,
                        breakout_direction=breakout_direction,
                        times=times,
                        n_bars=n_bars,
                        cfg=cfg,
                    )
                )

        if _is_bullish_fractal(low, left_lows, right_lows, left_bars, right_bars):
            prominence_pct = _fractal_prominence_pct(
                direction="bullish",
                level_price=low,
                left_values=left_lows,
                right_values=right_lows,
            )
            if prominence_pct >= min_prominence_pct:
                confirmation_index = int(center_index + right_bars)
                breakout_index, breakout_price, breakout_direction = _find_breakout(
                    direction="bullish",
                    level_price=low,
                    confirmation_index=confirmation_index,
                    breakout_basis=breakout_basis,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                )
                results.append(
                    _build_fractal_result(
                        direction="bullish",
                        center_index=center_index,
                        confirmation_index=confirmation_index,
                        level_price=low,
                        prominence_pct=prominence_pct,
                        breakout_basis=breakout_basis,
                        breakout_index=breakout_index,
                        breakout_price=breakout_price,
                        breakout_direction=breakout_direction,
                        times=times,
                        n_bars=n_bars,
                        cfg=cfg,
                    )
                )

    results.sort(key=lambda item: (int(item.end_index), float(item.confidence)), reverse=True)
    return results
