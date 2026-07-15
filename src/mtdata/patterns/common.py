import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ..shared.constants import TIMEFRAME_SECONDS
from ..shared.symbols import is_probably_crypto_symbol, is_probably_forex_symbol


def compute_atr_sma(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """SMA of true range over ``period`` (min_periods = max(2, period//2))."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n = min(h.size, l.size, c.size)
    if n <= 0:
        return np.asarray([], dtype=float)
    h = h[:n]
    l = l[:n]
    c = c[:n]
    prev_c = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    win = max(2, int(period))
    try:
        return (
            pd.Series(tr)
            .rolling(win, min_periods=max(2, win // 2))
            .mean()
            .to_numpy(dtype=float)
        )
    except (TypeError, ValueError):
        return tr.astype(float)


def fallback_local_extrema(
    src: np.ndarray,
    min_dist: int,
    order: int,
    *,
    prefer_high: bool,
) -> np.ndarray:
    """Find local extrema with a sliding window when primary peak detection undershoots.

    Plateau runs are collapsed to their midpoint so flat tops/bottoms yield a
    single representative index. Candidates closer than ``min_dist`` keep the
    more extreme value.
    """
    values = np.asarray(src, dtype=float)
    n = int(values.size)
    if n < (2 * order + 1):
        return np.asarray([], dtype=int)
    candidates: List[int] = []
    for idx in range(order, n - order):
        center = float(values[idx])
        if not np.isfinite(center):
            continue
        window = values[idx - order : idx + order + 1]
        if not np.all(np.isfinite(window)):
            continue
        plateau_tol = max(1e-12, abs(center) * 1e-12)
        plateau_left = idx
        while (
            plateau_left > 0
            and np.isfinite(values[plateau_left - 1])
            and np.isclose(
                values[plateau_left - 1],
                center,
                rtol=0.0,
                atol=plateau_tol,
            )
        ):
            plateau_left -= 1
        plateau_right = idx
        while (
            plateau_right < (n - 1)
            and np.isfinite(values[plateau_right + 1])
            and np.isclose(
                values[plateau_right + 1],
                center,
                rtol=0.0,
                atol=plateau_tol,
            )
        ):
            plateau_right += 1
        if plateau_left != plateau_right:
            if int((plateau_left + plateau_right) // 2) != int(idx):
                continue
        if prefer_high:
            if center < float(np.max(window)):
                continue
        elif center > float(np.min(window)):
            continue
        candidates.append(int(idx))
    if not candidates:
        return np.asarray([], dtype=int)
    reduced: List[int] = []
    for idx in candidates:
        if not reduced or (idx - reduced[-1]) >= int(min_dist):
            reduced.append(int(idx))
            continue
        prev_idx = int(reduced[-1])
        prev_val = float(values[prev_idx])
        curr_val = float(values[idx])
        better = idx if (curr_val > prev_val if prefer_high else curr_val < prev_val) else prev_idx
        reduced[-1] = int(better)
    return np.asarray(reduced, dtype=int)


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


def _crosses_weekend(start_epoch: float, end_epoch: float) -> bool:
    try:
        start_dt = datetime.fromtimestamp(float(start_epoch), tz=timezone.utc)
        end_dt = datetime.fromtimestamp(float(end_epoch), tz=timezone.utc)
    except Exception:
        return False
    if end_dt <= start_dt:
        return False
    current = start_dt
    while current <= end_dt:
        if current.weekday() >= 5:
            return True
        current = (current + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    return end_dt.weekday() >= 5


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
                threshold = 1.5 * float(timeframe_seconds)
                if gaps.size > 0 and float(np.nanmax(gaps)) > threshold:
                    expected_weekend_gaps = 0
                    unexpected_gaps = 0
                    is_fx = is_probably_forex_symbol(symbol)
                    is_crypto = is_probably_crypto_symbol(symbol)
                    for idx, gap in enumerate(gaps):
                        if not np.isfinite(gap) or float(gap) <= threshold:
                            continue
                        start_epoch = float(times[idx])
                        end_epoch = float(times[idx + 1])
                        if is_fx and not is_crypto and _crosses_weekend(
                            start_epoch, end_epoch
                        ):
                            expected_weekend_gaps += 1
                        else:
                            unexpected_gaps += 1
                    if unexpected_gaps > 0:
                        suffix = ""
                        if expected_weekend_gaps:
                            suffix = (
                                f" ({expected_weekend_gaps} expected weekend/session "
                                "gap(s) suppressed)."
                            )
                        warnings.append(
                            "Data quality warning: detected "
                            f"{unexpected_gaps} unexpected time gap(s) larger than "
                            f"1.5 bar intervals.{suffix}"
                        )

    volume_col = None
    volume_series: Optional[np.ndarray] = None
    fallback_volume_col = None
    fallback_volume_series: Optional[np.ndarray] = None
    for candidate in ("real_volume", "volume", "tick_volume", "Volume"):
        if candidate in df.columns:
            try:
                candidate_volume = pd.to_numeric(
                    df[candidate], errors="coerce"
                ).to_numpy(dtype=float, copy=False)
            except Exception:
                candidate_volume = np.asarray([], dtype=float)
            candidate_volume = candidate_volume[np.isfinite(candidate_volume)]
            if candidate_volume.size < 5:
                continue
            if fallback_volume_series is None:
                fallback_volume_col = candidate
                fallback_volume_series = candidate_volume
            if np.any(candidate_volume > 0):
                volume_col = candidate
                volume_series = candidate_volume
                break
    if volume_series is None:
        volume_col = fallback_volume_col
        volume_series = fallback_volume_series
    if volume_col is not None and volume_series is not None:
        if volume_series.size >= 5:
            zero_share = float(np.mean(volume_series <= 0))
            if zero_share >= 0.6:
                if is_probably_crypto_symbol(symbol):
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
