import logging
import time
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import MetaTrader5 as mt5

from ..core.config import mt5_config
from ..core.constants import DATA_READY_TIMEOUT, DATA_POLL_INTERVAL

logger = logging.getLogger(__name__)

_SYMBOL_INFO_TTL_SECONDS = 5


@lru_cache(maxsize=256)
def _cached_symbol_info(symbol: str, ttl_bucket: int):
    return mt5.symbol_info(symbol)


def get_symbol_info_cached(symbol: str, ttl_seconds: int = _SYMBOL_INFO_TTL_SECONDS):
    """Fetch symbol info with a short-lived cache to reduce repeated MT5 calls."""
    try:
        ttl = int(ttl_seconds)
        if ttl <= 0:
            return mt5.symbol_info(symbol)
        bucket = int(time.time() / ttl)
    except Exception:
        bucket = int(time.time())
    return _cached_symbol_info(symbol, bucket)


def clear_symbol_info_cache() -> None:
    """Clear the cached symbol info entries."""
    _cached_symbol_info.cache_clear()


def _mt5_epoch_to_utc(epoch_seconds: float) -> float:
    """Convert MT5-reported epoch seconds to UTC.

    If MT5_SERVER_TZ is set, interpret the epoch as server-local and convert to UTC with DST awareness.
    Else, subtract configured static offset minutes.
    """
    try:
        tz = mt5_config.get_server_tz()
        if tz is not None:
            base = datetime(1970, 1, 1)
            dt_local_naive = base + timedelta(seconds=float(epoch_seconds))
            try:
                dt_local = tz.localize(dt_local_naive, is_dst=None)
            except Exception:
                dt_local = dt_local_naive.replace(tzinfo=tz)
            return dt_local.astimezone(timezone.utc).timestamp()
        off = int(mt5_config.get_time_offset_seconds())
        return float(epoch_seconds) - float(off)
    except Exception:
        return float(epoch_seconds)


def _to_server_naive_dt(dt: datetime) -> datetime:
    """Convert a UTC-naive datetime to server-local naive datetime if server TZ configured."""
    try:
        tz = mt5_config.get_server_tz()
        if tz is None:
            return dt
        aware_utc = dt.replace(tzinfo=timezone.utc)
        aware_srv = aware_utc.astimezone(tz)
        return aware_srv.replace(tzinfo=None)
    except Exception:
        return dt


def _normalize_times_in_struct(arr: Any):
    try:
        if arr is None:
            return arr
        names = getattr(getattr(arr, 'dtype', None), 'names', None)
        if not names or 'time' not in names:
            return arr
        for i in range(len(arr)):
            try:
                arr[i]['time'] = _mt5_epoch_to_utc(float(arr[i]['time']))
            except Exception:
                continue
        return arr
    except Exception:
        return arr


def _mt5_copy_rates_from(symbol: str, timeframe, to_dt_utc: datetime, count: int):
    dt_srv = _to_server_naive_dt(to_dt_utc)
    data = mt5.copy_rates_from(symbol, timeframe, dt_srv, count)
    return _normalize_times_in_struct(data)


def _mt5_copy_rates_range(symbol: str, timeframe, from_dt_utc: datetime, to_dt_utc: datetime):
    dt_from = _to_server_naive_dt(from_dt_utc)
    dt_to = _to_server_naive_dt(to_dt_utc)
    data = mt5.copy_rates_range(symbol, timeframe, dt_from, dt_to)
    return _normalize_times_in_struct(data)


def _mt5_copy_ticks_from(symbol: str, from_dt_utc: datetime, count: int, flags: int):
    dt_from = _to_server_naive_dt(from_dt_utc)
    data = mt5.copy_ticks_from(symbol, dt_from, count, flags)
    return _normalize_times_in_struct(data)


def _mt5_copy_ticks_range(symbol: str, from_dt_utc: datetime, to_dt_utc: datetime, flags: int):
    dt_from = _to_server_naive_dt(from_dt_utc)
    dt_to = _to_server_naive_dt(to_dt_utc)
    data = mt5.copy_ticks_range(symbol, dt_from, dt_to, flags)
    return _normalize_times_in_struct(data)


class MT5Connection:
    def __init__(self):
        self.connected = False

    def _ensure_connection(self) -> bool:
        if self.is_connected():
            return True
        try:
            if mt5_config.has_credentials():
                login = mt5_config.get_login()
                password = mt5_config.get_password()
                server = mt5_config.get_server()
                if not mt5.initialize(login=login, password=password, server=server):
                    logger.warning(f"Failed to initialize MT5 with credentials: {mt5.last_error()}")
                    if not mt5.initialize():
                        logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
                        return False
                else:
                    logger.info(f"Connected to MT5 with account {login}")
            else:
                if not mt5.initialize():
                    logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
                    return False
                else:
                    logger.info("Connected to MT5 using terminal's current login")
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False

    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MetaTrader5")

    def is_connected(self) -> bool:
        if not self.connected:
            return False
        terminal_info = mt5.terminal_info()
        return terminal_info is not None and terminal_info.connected


mt5_connection = MT5Connection()


def _auto_connect_wrapper(func):
    """Decorator to ensure MT5 connection before tool execution"""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not mt5_connection._ensure_connection():
            return {"error": "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."}
        return func(*args, **kwargs)
    return wrapper


def _ensure_symbol_ready(symbol: str) -> Optional[str]:
    """Ensure a symbol is selected and its tick data is initialized.

    Returns an error string if selection or data readiness fails, else None.
    """
    try:
        info_before = mt5.symbol_info(symbol)
        was_visible = bool(info_before.visible) if info_before is not None else None
        if not mt5.symbol_select(symbol, True):
            return f"Failed to select symbol {symbol}: {mt5.last_error()}"
        # If we just made it visible, wait briefly for fresh tick data
        if was_visible is False:
            deadline = time.time() + DATA_READY_TIMEOUT
            while time.time() < deadline:
                tick = mt5.symbol_info_tick(symbol)
                if tick and (getattr(tick, 'time', 0) or getattr(tick, 'bid', 0) or getattr(tick, 'ask', 0)):
                    break
                time.sleep(DATA_POLL_INTERVAL)
        # Final check
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return f"Failed to refresh {symbol} data: {mt5.last_error()}"
        return None
    except Exception as e:
        return f"Error ensuring symbol readiness: {e}"
