import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..shared.constants import TIMEFRAME_SECONDS
from ..shared.symbols import is_probably_crypto_symbol


def _is_probably_crypto_symbol(symbol: Any) -> bool:
    return is_probably_crypto_symbol(symbol)


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


def data_quality_warnings(
    df: Any,
    *,
    symbol: Optional[str] = None,
    timeframe_seconds: Optional[float] = None,
) -> list[str]:
    warnings: list[str] = []
    if not isinstance(df, pd.DataFrame) or len(df) < 3:
        return warnings

    close_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
    if close_col is not None:
        try:
            close = pd.to_numeric(df[close_col], errors="coerce").to_numpy(dtype=float, copy=False)
        except Exception:
            close = np.asarray([], dtype=float)
        close = close[np.isfinite(close)]
        if close.size >= 3:
            steps = np.abs(np.diff(close))
            if steps.size > 0:
                zero_share = float(np.mean(steps <= 1e-12))
                if zero_share >= 0.6:
                    warnings.append("Data quality warning: repeated close prices dominate the sample.")
                elif float(np.nanmax(steps)) <= 1e-12:
                    warnings.append("Data quality warning: close prices are nearly flat across the sample.")

    if timeframe_seconds is not None and float(timeframe_seconds) > 0:
        time_col = "time" if "time" in df.columns else ("Time" if "Time" in df.columns else None)
        if time_col is not None:
            try:
                times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float, copy=False)
            except Exception:
                times = np.asarray([], dtype=float)
            times = times[np.isfinite(times)]
            if times.size >= 3:
                gaps = np.diff(times)
                if gaps.size > 0 and float(np.nanmax(gaps)) > (1.5 * float(timeframe_seconds)):
                    warnings.append("Data quality warning: detected time gaps larger than 1.5 bar intervals.")

    volume_col = None
    for candidate in ("real_volume", "volume", "tick_volume", "Volume"):
        if candidate in df.columns:
            volume_col = candidate
            break
    if volume_col is not None:
        try:
            volume = pd.to_numeric(df[volume_col], errors="coerce").to_numpy(dtype=float, copy=False)
        except Exception:
            volume = np.asarray([], dtype=float)
        volume = volume[np.isfinite(volume)]
        if volume.size >= 5:
            zero_share = float(np.mean(volume <= 0))
            if zero_share >= 0.6:
                if _is_probably_crypto_symbol(symbol):
                    warnings.append(
                        "Data quality warning: zero-volume bars dominate the sample "
                        "(common for crypto low-volume periods)."
                    )
                else:
                    warnings.append("Data quality warning: zero-volume bars dominate the sample.")

    return warnings


def should_drop_last_live_bar(
    df: pd.DataFrame,
    timeframe: str,
    *,
    now_utc: Optional[datetime] = None,
    current_time_epoch: Optional[float] = None,
) -> bool:
    """Return True when the last bar is still forming or cannot be validated."""
    if len(df) < 2:
        return False
    seconds_per_bar = float(TIMEFRAME_SECONDS.get(timeframe, 0) or 0)
    if seconds_per_bar <= 0 or "time" not in df.columns:
        return True
    try:
        last_open = float(pd.to_numeric(df["time"], errors="coerce").iloc[-1])
    except Exception:
        return True
    if not math.isfinite(last_open):
        return True
    if current_time_epoch is not None:
        try:
            current_ts = float(current_time_epoch)
        except Exception:
            current_ts = float("nan")
        if not math.isfinite(current_ts):
            current_ts = float((now_utc or datetime.now(timezone.utc)).timestamp())
    else:
        current_ts = float((now_utc or datetime.now(timezone.utc)).timestamp())
    elapsed = current_ts - last_open
    if elapsed < 0:
        return True
    return elapsed < seconds_per_bar
