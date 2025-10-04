"""Helper functions for forecast orchestration."""
from typing import List


def default_seasonality_period(timeframe: str) -> int:
    """Return default seasonality period for a given timeframe."""
    tf_map = {
        'M1': 60, 'M5': 288, 'M15': 96, 'M30': 48,
        'H1': 24, 'H4': 6, 'D1': 5, 'W1': 4, 'MN1': 12
    }
    return tf_map.get(timeframe, 24)


def next_times_from_last(last_epoch: float, tf_secs: int, horizon: int) -> List[float]:
    """Generate future timestamps from last bar."""
    return [last_epoch + (i + 1) * tf_secs for i in range(horizon)]


def pd_freq_from_timeframe(tf: str) -> str:
    """Convert MT5 timeframe to pandas frequency string."""
    freq_map = {
        'M1': '1min', 'M5': '5min', 'M15': '15min', 'M30': '30min',
        'H1': '1H', 'H4': '4H', 'D1': '1D', 'W1': '1W', 'MN1': '1MS'
    }
    return freq_map.get(tf, '1H')
