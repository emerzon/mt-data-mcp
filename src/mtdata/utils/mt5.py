import logging
import time
from contextlib import contextmanager
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Iterator, Tuple

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


def _rates_to_df(rates: Any):
    """Convert raw MT5 rates into a DataFrame with UTC epoch seconds in 'time'."""
    import pandas as pd

    df = pd.DataFrame(rates)
    try:
        if 'time' in df.columns:
            df['time'] = df['time'].astype(float).apply(_mt5_epoch_to_utc)
    except Exception:
        pass
    return df


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


def _mt5_copy_rates_from_pos(symbol: str, timeframe, start_pos: int, count: int):
    data = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
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


class MT5Service:
    """Thin wrapper to group MT5 connection state for easier testing/injection."""

    def __init__(self, connection: Optional[MT5Connection] = None):
        self.connection = connection or MT5Connection()

    def ensure_connected(self) -> bool:
        return self.connection._ensure_connection()

    def disconnect(self) -> None:
        self.connection.disconnect()


mt5_service = MT5Service(mt5_connection)


def _auto_connect_wrapper(func=None, *, service: Optional[MT5Service] = None):
    """Decorator to ensure MT5 connection before tool execution.

    Supports both:
    - ``@_auto_connect_wrapper`` (uses the global ``mt5_service``)
    - ``@_auto_connect_wrapper(service=MT5Service(...))`` for tests/injection
    """
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            svc = service or mt5_service
            if not svc.ensure_connected():
                return {"error": "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."}
            return fn(*args, **kwargs)

        return wrapper

    return decorator(func) if callable(func) else decorator


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


@contextmanager
def _symbol_ready_guard(
    symbol: str,
    info_before: Optional[Any] = None,
) -> Iterator[Tuple[Optional[str], Optional[Any]]]:
    """Ensure symbol readiness and restore original visibility on exit."""
    info = info_before if info_before is not None else mt5.symbol_info(symbol)
    was_visible = bool(info.visible) if info is not None else None
    err = _ensure_symbol_ready(symbol)
    try:
        yield err, info
    finally:
        if was_visible is False:
            try:
                mt5.symbol_select(symbol, False)
            except Exception:
                pass


def estimate_server_offset(symbol: str = "EURUSD", samples: int = 5) -> int:
    """Estimate server offset from UTC in seconds by comparing tick time to local UTC time.
    
    Returns 0 if failed.
    """
    try:
        if not mt5_connection._ensure_connection():
            return 0
        
        # Ensure symbol is ready
        if not mt5.symbol_select(symbol, True):
            # Try a fallback if EURUSD not found
            for s in ["GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"]:
                if mt5.symbol_select(s, True):
                    symbol = s
                    break
        
        deltas = []
        for _ in range(samples):
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                # MT5 tick.time is epoch seconds (server time)
                # We compare to time.time() (system local epoch -> UTC)
                # If server is UTC+2, tick.time will be ~ (now + 7200)
                diff = float(tick.time) - time.time()
                deltas.append(diff)
            time.sleep(0.2)
            
        if not deltas:
            return 0
            
        # Median
        deltas.sort()
        med = deltas[len(deltas) // 2]
        
        # Round to nearest 15 minutes (900s) to be safe/clean
        offset = int(round(med / 900.0) * 900)
        return offset
    except Exception as e:
        logger.error(f"Failed to estimate server offset: {e}")
        return 0
