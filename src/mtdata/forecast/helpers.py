"""Helper functions for forecast orchestration."""
from typing import List

from .common import (
    default_seasonality as _default_seasonality_common,
    next_times_from_last as _next_times_from_last_common,
    pd_freq_from_timeframe as _pd_freq_from_timeframe_common,
)


def default_seasonality_period(timeframe: str) -> int:
    """Return default seasonality period for a given timeframe."""
    return _default_seasonality_common(timeframe)


def next_times_from_last(last_epoch: float, tf_secs: int, horizon: int) -> List[float]:
    """Generate future timestamps from last bar."""
    return _next_times_from_last_common(last_epoch, tf_secs, horizon)


def pd_freq_from_timeframe(tf: str) -> str:
    """Convert MT5 timeframe to pandas frequency string."""
    return _pd_freq_from_timeframe_common(tf)
