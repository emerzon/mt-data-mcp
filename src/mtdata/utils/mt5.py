"""MT5 connectivity, time alignment, and low-level data helpers.

Time Alignment Contract
-----------------------
All MT5 timestamps pass through a single normalisation chain:

1. **Outbound** (UTC → server-local): ``_to_server_naive_dt()`` converts
   UTC datetimes to the broker's server-local representation before each
   MT5 API call.  It uses either a ``pytz`` timezone (``MT5_SERVER_TZ``)
   or a static offset (``MT5_TIME_OFFSET_MINUTES``).

2. **Inbound** (server-local → UTC): ``_normalize_times_in_struct()``
   converts every ``time`` field in the returned structured arrays back
   to UTC.  When a server timezone is configured it delegates per-element
   to ``_mt5_epoch_to_utc()`` (DST-aware); otherwise it subtracts the
   static offset in bulk (fast path).

3. **Diagnostic** (optional): ``inspect_mt5_time_alignment()`` samples
   the latest tick and bar to infer the actual broker offset, compares it
   to the configured offset, and reports ``ok | misaligned | stale``.
   Results are TTL-cached via ``get_cached_mt5_time_alignment()``.

Configuration priority (``MT5Config.get_time_offset_seconds``):
  static ``MT5_TIME_OFFSET_MINUTES`` > dynamic ``MT5_SERVER_TZ`` > 0

Every ``_mt5_copy_*`` wrapper in this module applies steps 1 + 2 so
callers always receive UTC-normalised data.  The higher-level data
service may apply an additional auto-correction shift
(``_shift_rate_times``) for live data when diagnostic alignment detects
a mismatch, bounded to [30 min, 18 h].
"""

import importlib
import logging
import math
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional, Tuple

from ..bootstrap.settings import mt5_config

logger = logging.getLogger(__name__)

try:
    from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError
except Exception:  # pragma: no cover - pytz is optional at import time
    class AmbiguousTimeError(Exception):
        """Fallback when pytz is unavailable."""

    class NonExistentTimeError(Exception):
        """Fallback when pytz is unavailable."""

_SYMBOL_INFO_TTL_SECONDS = 5
_SYMBOL_INFO_TTL_MAX_SECONDS = 3600.0
_MT5_CONNECTION_FAILURE_MESSAGE = "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."


class MT5ConnectionError(RuntimeError):
    """Raised when the MT5 adapter cannot establish a usable connection."""


def _data_ready_timing() -> tuple[float, float]:
    """Load symbol-readiness timing constants lazily to avoid import cycles."""
    try:
        from ..shared.constants import DATA_POLL_INTERVAL, DATA_READY_TIMEOUT

        return float(DATA_READY_TIMEOUT), float(DATA_POLL_INTERVAL)
    except Exception:
        return 3.0, 0.2


def _load_mt5_module() -> Any:
    """Resolve the live MetaTrader5 module on demand.

    Tests frequently replace ``sys.modules['MetaTrader5']`` after imports, so the
    adapter cannot hold a stale module reference.
    """
    return importlib.import_module("MetaTrader5")


# Reentrant lock serialising all MT5 COM calls so that concurrent
# asyncio.to_thread workers (MCP tool dispatch) cannot deadlock the
# single-threaded COM apartment used by the MetaTrader5 library.
_mt5_lock = threading.RLock()


class MT5Adapter:
    """Thin dynamic adapter around MetaTrader5.

    Every public method acquires ``_mt5_lock`` so that only one thread
    interacts with the underlying COM bridge at a time.  This prevents
    the deadlocks observed when ``asyncio.to_thread`` dispatches
    concurrent MCP tool calls from different worker threads.
    """

    def _module(self) -> Any:
        return _load_mt5_module()

    def __dir__(self) -> list[str]:
        try:
            return sorted(set(object.__dir__(self)) | set(dir(self._module())))
        except Exception:
            return list(object.__dir__(self))

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def initialize(self, *args, **kwargs):
        with _mt5_lock:
            return self._module().initialize(*args, **kwargs)

    def shutdown(self):
        with _mt5_lock:
            return self._module().shutdown()

    def last_error(self):
        with _mt5_lock:
            return self._module().last_error()

    def symbol_info(self, symbol):
        with _mt5_lock:
            return self._module().symbol_info(symbol)

    def symbol_select(self, symbol, visible=True):
        with _mt5_lock:
            return self._module().symbol_select(symbol, visible)

    def symbol_info_tick(self, symbol):
        with _mt5_lock:
            return self._module().symbol_info_tick(symbol)

    def order_send(self, request):
        with _mt5_lock:
            return self._module().order_send(request)

    def positions_get(self, **kwargs):
        with _mt5_lock:
            return self._module().positions_get(**kwargs)

    def orders_get(self, **kwargs):
        with _mt5_lock:
            return self._module().orders_get(**kwargs)

    def history_orders_get(self, dt_from, dt_to, **kwargs):
        with _mt5_lock:
            return self._module().history_orders_get(dt_from, dt_to, **kwargs)

    def history_deals_get(self, dt_from, dt_to, **kwargs):
        with _mt5_lock:
            return self._module().history_deals_get(dt_from, dt_to, **kwargs)

    def account_info(self):
        with _mt5_lock:
            return self._module().account_info()

    def terminal_info(self):
        with _mt5_lock:
            return self._module().terminal_info()

    def copy_rates_from(self, symbol, timeframe, dt_from, count):
        with _mt5_lock:
            return self._module().copy_rates_from(symbol, timeframe, dt_from, count)

    def copy_rates_range(self, symbol, timeframe, dt_from, dt_to):
        with _mt5_lock:
            return self._module().copy_rates_range(symbol, timeframe, dt_from, dt_to)

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        with _mt5_lock:
            return self._module().copy_rates_from_pos(symbol, timeframe, start_pos, count)

    def copy_ticks_from(self, symbol, dt_from, count, flags):
        with _mt5_lock:
            return self._module().copy_ticks_from(symbol, dt_from, count, flags)

    def copy_ticks_range(self, symbol, dt_from, dt_to, flags):
        with _mt5_lock:
            return self._module().copy_ticks_range(symbol, dt_from, dt_to, flags)

    def market_book_add(self, symbol):
        with _mt5_lock:
            return self._module().market_book_add(symbol)

    def market_book_get(self, symbol):
        with _mt5_lock:
            return self._module().market_book_get(symbol)

    def market_book_release(self, symbol):
        with _mt5_lock:
            return self._module().market_book_release(symbol)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._module(), name)
        if callable(attr):
            def _locked(*a, **kw):
                with _mt5_lock:
                    return attr(*a, **kw)
            return _locked
        return attr


mt5_adapter = MT5Adapter()
mt5 = mt5_adapter


@lru_cache(maxsize=256)
def _cached_symbol_info(symbol: str, ttl_bucket: int):
    return mt5.symbol_info(symbol)


def _symbol_info_ttl_bucket(ttl_seconds: float) -> Optional[int]:
    try:
        ttl = float(ttl_seconds)
    except Exception:
        return None
    if not math.isfinite(ttl) or ttl <= 0.0:
        return None
    ttl = min(ttl, _SYMBOL_INFO_TTL_MAX_SECONDS)
    ttl_ns = max(int(math.ceil(ttl * 1_000_000_000.0)), 1)
    return time.monotonic_ns() // ttl_ns


def get_symbol_info_cached(symbol: str, ttl_seconds: float = _SYMBOL_INFO_TTL_SECONDS):
    """Fetch symbol info with a short-lived cache to reduce repeated MT5 calls."""
    bucket = _symbol_info_ttl_bucket(ttl_seconds)
    if bucket is None:
        return mt5.symbol_info(symbol)
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
            except AmbiguousTimeError:
                logger.warning(
                    "Ambiguous MT5 server-local time %s in %s; resolving with standard-time offset.",
                    dt_local_naive,
                    getattr(tz, "zone", tz),
                )
                dt_local = tz.localize(dt_local_naive, is_dst=False)
            except NonExistentTimeError:
                logger.warning(
                    "Non-existent MT5 server-local time %s in %s; shifting to the next valid local instant.",
                    dt_local_naive,
                    getattr(tz, "zone", tz),
                )
                dt_local = tz.localize(dt_local_naive + timedelta(hours=1), is_dst=False)
            return dt_local.astimezone(timezone.utc).timestamp()
        off = int(mt5_config.get_time_offset_seconds())
        return float(epoch_seconds) - float(off)
    except Exception as exc:
        logger.warning(
            "Failed to convert MT5 epoch %s to UTC; leaving raw value unchanged: %s",
            epoch_seconds,
            exc,
        )
        return float(epoch_seconds)


_DEFAULT_MT5_EPOCH_TO_UTC = _mt5_epoch_to_utc


def _rates_to_df(rates: Any):
    """Convert MT5 rates into a DataFrame.

    Low-level MT5 copy helpers already normalize timestamps to UTC before
    returning structured arrays, so this function should avoid re-normalizing
    the same values a second time.
    """
    import pandas as pd

    return pd.DataFrame(rates)


def _to_server_naive_dt(dt: datetime) -> datetime:
    """Convert a UTC-naive datetime to server-local naive datetime."""
    try:
        tz = mt5_config.get_server_tz()
        if tz is None:
            offset_seconds = int(mt5_config.get_time_offset_seconds())
            if offset_seconds:
                return dt + timedelta(seconds=offset_seconds)
            return dt
        aware_utc = dt.replace(tzinfo=timezone.utc)
        aware_srv = aware_utc.astimezone(tz)
        return aware_srv.replace(tzinfo=None)
    except Exception as exc:
        logger.warning(
            "Failed to convert UTC datetime %s to MT5 server-local time; using original datetime: %s",
            dt,
            exc,
        )
        return dt


def _normalize_times_in_struct(arr: Any):
    """Convert all time fields in a structured array to UTC."""
    try:
        if arr is None:
            return arr
        names = getattr(getattr(arr, "dtype", None), "names", None)
        if not names:
            return arr

        # Identify all fields that look like timestamps
        time_fields = [
            n
            for n in names
            if n
            in (
                "time",
                "time_msc",
                "time_setup",
                "time_setup_msc",
                "time_done",
                "time_done_msc",
                "time_update",
                "time_update_msc",
                "time_expiration",
                "expiration",
            )
        ]
        if not time_fields:
            return arr

        out = arr
        flags = getattr(arr, "flags", None)
        if flags is not None and not bool(getattr(flags, "writeable", True)):
            out = arr.copy()

        # Optimization: if using default function and no TZ (static offset),
        # use vector subtraction.
        if _mt5_epoch_to_utc is _DEFAULT_MT5_EPOCH_TO_UTC:
            tz = mt5_config.get_server_tz()
            if tz is None:
                offset_seconds = int(mt5_config.get_time_offset_seconds())
                if offset_seconds:
                    for field in time_fields:
                        try:
                            shift = (
                                float(offset_seconds) * 1000.0
                                if field.endswith("_msc")
                                else float(offset_seconds)
                            )
                            out[field] = out[field] - shift
                        except Exception as exc:
                            logger.warning(
                                "Failed to normalize MT5 timestamp field %s with static offset; leaving raw values unchanged: %s",
                                field,
                                exc,
                            )
                            continue
                return out

        # Fallback to per-element conversion (handles DST correctly)
        for i in range(len(out)):
            for field in time_fields:
                try:
                    val = float(out[i][field])
                    if val <= 0:
                        continue
                    if field.endswith("_msc"):
                        out[i][field] = _mt5_epoch_to_utc(val / 1000.0) * 1000.0
                    else:
                        out[i][field] = _mt5_epoch_to_utc(val)
                except Exception as exc:
                    logger.warning(
                        "Failed to normalize MT5 timestamp at index %s field %s; leaving raw value unchanged: %s",
                        i,
                        field,
                        exc,
                    )
                    continue
        return out
    except Exception as exc:
        logger.warning(
            "Failed to normalize MT5 timestamps in structured array; leaving values unchanged: %s",
            exc,
        )
        return arr


def _normalize_object_times(obj: Any) -> Any:
    """Normalize timestamp attributes on an MT5 object to UTC.
    Returns a copy (SimpleNamespace) if the original was likely immutable.
    """
    if obj is None:
        return None

    # Common MT5 timestamp attributes
    time_attrs = (
        "time",
        "time_msc",
        "time_setup",
        "time_setup_msc",
        "time_done",
        "time_done_msc",
        "time_update",
        "time_update_msc",
    )

    # Check if any attributes exist
    has_any = False
    for attr in time_attrs:
        if hasattr(obj, attr):
            has_any = True
            break
    if not has_any:
        return obj

    # MT5 library objects are often read-only.
    # We'll convert to a SimpleNamespace for safe modification.
    from types import SimpleNamespace

    try:
        # Try to use _asdict() if it's a namedtuple-like object
        if hasattr(obj, "_asdict"):
            data = obj._asdict()
        else:
            data = {
                attr: getattr(obj, attr)
                for attr in dir(obj)
                if not attr.startswith("_") and not callable(getattr(obj, attr))
            }

        modified = False
        for attr in time_attrs:
            if attr in data and data[attr]:
                try:
                    val = float(data[attr])
                    if val > 0:
                        if attr.endswith("_msc"):
                            data[attr] = _mt5_epoch_to_utc(val / 1000.0) * 1000.0
                        else:
                            data[attr] = _mt5_epoch_to_utc(val)
                        modified = True
                except Exception:
                    continue

        if not modified:
            return obj

        return SimpleNamespace(**data)
    except Exception as exc:
        logger.debug("Failed to normalize object timestamps: %s", exc)
        return obj


# ---------------------------------------------------------------------------
# Read-operation retry & rate-budget helpers
# ---------------------------------------------------------------------------

_MT5_READ_MAX_RETRIES = 2        # 3 total attempts for read-only calls
_MT5_READ_BASE_DELAY = 0.5      # exponential backoff base (seconds)
_MT5_READ_MIN_SPACING = 0.05    # minimum seconds between consecutive reads
_mt5_last_read_ts: float = 0.0  # monotonic timestamp of last read call


def _enforce_read_spacing() -> None:
    """Sleep if needed to honour minimum spacing between MT5 read calls."""
    global _mt5_last_read_ts
    now = time.monotonic()
    gap = _MT5_READ_MIN_SPACING - (now - _mt5_last_read_ts)
    if gap > 0:
        time.sleep(gap)
    _mt5_last_read_ts = time.monotonic()


def _mt5_read_with_retry(fn, *args, max_retries: int = _MT5_READ_MAX_RETRIES):
    """Execute a read-only MT5 operation with bounded retry and backoff.

    Only for **idempotent** read calls (``copy_rates_*``, ``copy_ticks_*``).
    Write operations (``order_send``, ``symbol_select``, market-book lifecycle)
    must **never** use this helper — duplicate execution would be dangerous.

    Returns the first non-``None`` result, or ``None`` if all attempts fail.
    """
    for attempt in range(max_retries + 1):
        _enforce_read_spacing()
        result = fn(*args)
        if result is not None:
            return result
        if attempt < max_retries:
            delay = _MT5_READ_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "MT5 read returned None (attempt %d/%d), retrying in %.1fs",
                attempt + 1, max_retries + 1, delay,
            )
            time.sleep(delay)
    logger.warning(
        "MT5 read exhausted %d attempt(s) — returning None",
        max_retries + 1,
    )
    return None


def _mt5_copy_rates_from(symbol: str, timeframe, to_dt_utc: datetime, count: int):
    dt_srv = _to_server_naive_dt(to_dt_utc)
    data = _mt5_read_with_retry(mt5.copy_rates_from, symbol, timeframe, dt_srv, count)
    return _normalize_times_in_struct(data)


def _mt5_copy_rates_range(symbol: str, timeframe, from_dt_utc: datetime, to_dt_utc: datetime):
    dt_from = _to_server_naive_dt(from_dt_utc)
    dt_to = _to_server_naive_dt(to_dt_utc)
    data = _mt5_read_with_retry(mt5.copy_rates_range, symbol, timeframe, dt_from, dt_to)
    return _normalize_times_in_struct(data)


def _mt5_copy_ticks_from(symbol: str, from_dt_utc: datetime, count: int, flags: int):
    dt_from = _to_server_naive_dt(from_dt_utc)
    data = _mt5_read_with_retry(mt5.copy_ticks_from, symbol, dt_from, count, flags)
    return _normalize_times_in_struct(data)


def _mt5_copy_rates_from_pos(symbol: str, timeframe, start_pos: int, count: int):
    data = _mt5_read_with_retry(mt5.copy_rates_from_pos, symbol, timeframe, start_pos, count)
    return _normalize_times_in_struct(data)


def _mt5_copy_ticks_range(symbol: str, from_dt_utc: datetime, to_dt_utc: datetime, flags: int):
    dt_from = _to_server_naive_dt(from_dt_utc)
    dt_to = _to_server_naive_dt(to_dt_utc)
    data = _mt5_read_with_retry(mt5.copy_ticks_range, symbol, dt_from, dt_to, flags)
    return _normalize_times_in_struct(data)


class MT5Connection:
    def __init__(self):
        self.connected = False
        self._connection_identity: Optional[tuple[Optional[int], Optional[str]]] = None

    def _read_connection_identity(self) -> tuple[Optional[int], Optional[str]]:
        login: Optional[int] = None
        server: Optional[str] = None
        try:
            account_info = mt5.account_info()
        except Exception:
            account_info = None
        if account_info is not None:
            try:
                login_value = getattr(account_info, "login", None)
                if login_value is not None:
                    login = int(login_value)
            except Exception:
                login = None
            try:
                server_value = getattr(account_info, "server", None)
                if server_value:
                    server = str(server_value)
            except Exception:
                server = None
        if server is None:
            try:
                terminal_info = mt5.terminal_info()
            except Exception:
                terminal_info = None
            if terminal_info is not None:
                try:
                    server_value = getattr(terminal_info, "server", None)
                    if server_value:
                        server = str(server_value)
                except Exception:
                    server = None
        return login, server

    def _refresh_connection_identity(self) -> None:
        current_identity = self._read_connection_identity()
        if self._connection_identity != current_identity:
            clear_symbol_info_cache()
            clear_mt5_time_alignment_cache()
            self._connection_identity = current_identity

    def _ensure_connection(self) -> bool:
        if self.is_connected():
            self._refresh_connection_identity()
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
                    logger.debug(f"Connected to MT5 with account {login}")
            else:
                if not mt5.initialize():
                    logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
                    return False
                else:
                    logger.debug("Connected to MT5 using terminal's current login")
            self.connected = True
            self._refresh_connection_identity()
            return True
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False

    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self._connection_identity = None
            clear_symbol_info_cache()
            clear_mt5_time_alignment_cache()
            logger.debug("Disconnected from MetaTrader5")

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


def ensure_mt5_connection_or_raise(*, service: Optional[MT5Service] = None) -> None:
    """Ensure MT5 is connected or raise a typed adapter error."""
    svc = service or mt5_service
    try:
        connected = bool(svc.ensure_connected())
    except MT5ConnectionError:
        raise
    except Exception as exc:
        raise MT5ConnectionError(_MT5_CONNECTION_FAILURE_MESSAGE) from exc
    if not connected:
        raise MT5ConnectionError(_MT5_CONNECTION_FAILURE_MESSAGE)


def _ensure_symbol_ready(symbol: str) -> Optional[str]:
    """Ensure a symbol is selected and its tick data is initialized.

    Returns an error string if selection or data readiness fails, else None.
    """
    try:
        data_ready_timeout, data_poll_interval = _data_ready_timing()
        info_before = mt5.symbol_info(symbol)
        was_visible = bool(info_before.visible) if info_before is not None else None
        if not mt5.symbol_select(symbol, True):
            return f"Failed to select symbol {symbol}: {mt5.last_error()}"
        # If we just made it visible, wait briefly for fresh tick data
        if was_visible is False:
            deadline = time.time() + data_ready_timeout
            while time.time() < deadline:
                tick = mt5.symbol_info_tick(symbol)
                if tick and (getattr(tick, 'time', 0) or getattr(tick, 'bid', 0) or getattr(tick, 'ask', 0)):
                    break
                time.sleep(data_poll_interval)
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
        ensure_mt5_connection_or_raise()
        
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


def _epoch_to_utc_iso(epoch_seconds: Optional[float]) -> Optional[str]:
    try:
        if epoch_seconds is None:
            return None
        return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _round_seconds(value: float, bucket_seconds: int = 900) -> int:
    try:
        bucket = int(bucket_seconds)
        if bucket <= 0:
            return int(round(float(value)))
        return int(round(float(value) / float(bucket)) * float(bucket))
    except Exception:
        return int(round(float(value)))


def inspect_mt5_time_alignment(
    symbol: str = "EURUSD",
    probe_timeframe: str = "M1",
    *,
    tick_offset_bucket_seconds: int = 900,
    max_plausible_offset_seconds: int = 18 * 3600,
    max_future_seconds: int = 90,
    max_tick_age_seconds: int = 180,
    stale_bar_tolerance: int = 3,
) -> Dict[str, Any]:
    """Inspect broker time alignment using raw ticks and the latest converted bar times.

    The check is intentionally best-effort:
    - infer the broker offset from the raw latest tick time
    - compare that inferred offset to the configured server offset/TZ
    - fetch the latest bars for ``probe_timeframe`` and verify the converted bar open
      is not in the future relative to UTC and is not implausibly stale
    """
    out: Dict[str, Any] = {
        "symbol": str(symbol),
        "probe_timeframe": str(probe_timeframe),
        "status": "unavailable",
    }

    try:
        ensure_mt5_connection_or_raise()
    except Exception as exc:
        out["reason"] = "connection_failed"
        out["error"] = str(exc)
        return out

    try:
        from ..shared.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
    except Exception as exc:
        out["reason"] = "timeframe_constants_unavailable"
        out["error"] = str(exc)
        return out

    tf_name = str(probe_timeframe or "M1").upper()
    mt5_tf = TIMEFRAME_MAP.get(tf_name)
    tf_secs = TIMEFRAME_SECONDS.get(tf_name)
    if mt5_tf is None or not tf_secs:
        out["reason"] = "unsupported_timeframe"
        out["error"] = f"Unsupported probe timeframe: {tf_name}"
        return out

    try:
        err = _ensure_symbol_ready(symbol)
        if err:
            out["reason"] = "symbol_not_ready"
            out["error"] = str(err)
            return out
    except Exception as exc:
        out["reason"] = "symbol_ready_check_failed"
        out["error"] = str(exc)
        return out

    now_utc_epoch = float(time.time())
    out["now_utc_epoch"] = now_utc_epoch
    out["now_utc_time"] = _epoch_to_utc_iso(now_utc_epoch)

    try:
        configured_offset_seconds = int(mt5_config.get_time_offset_seconds())
    except Exception:
        configured_offset_seconds = 0
    out["configured_offset_seconds"] = configured_offset_seconds
    if getattr(mt5_config, "server_tz_name", None):
        out["configured_server_tz"] = str(mt5_config.server_tz_name)

    raw_tick_epoch: Optional[float] = None
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None:
            raw_tick_epoch = float(getattr(tick, "time", 0.0) or 0.0)
    except Exception:
        raw_tick_epoch = None

    inferred_offset_seconds: Optional[int] = None
    raw_tick_delta_seconds: Optional[float] = None
    tick_utc_epoch: Optional[float] = None
    tick_age_seconds: Optional[float] = None
    offset_inference_reliable = False
    if raw_tick_epoch and raw_tick_epoch > 0:
        raw_tick_delta_seconds = float(raw_tick_epoch - now_utc_epoch)
        inferred_offset_seconds = _round_seconds(raw_tick_delta_seconds, tick_offset_bucket_seconds)
        tick_utc_epoch = float(_mt5_epoch_to_utc(raw_tick_epoch))
        tick_age_seconds = float(now_utc_epoch - tick_utc_epoch)
        offset_inference_reliable = abs(float(raw_tick_delta_seconds)) <= float(max_plausible_offset_seconds)
        out["raw_tick_epoch"] = raw_tick_epoch
        out["raw_tick_time"] = _epoch_to_utc_iso(raw_tick_epoch)
        out["raw_tick_delta_seconds"] = raw_tick_delta_seconds
        out["inferred_offset_seconds"] = inferred_offset_seconds
        out["tick_utc_epoch"] = tick_utc_epoch
        out["tick_utc_time"] = _epoch_to_utc_iso(tick_utc_epoch)
        out["tick_age_seconds"] = tick_age_seconds
        out["offset_inference_reliable"] = offset_inference_reliable
        out["offset_mismatch_seconds"] = int(inferred_offset_seconds - configured_offset_seconds)

    current_bar_open_epoch: Optional[float] = None
    last_closed_bar_open_epoch: Optional[float] = None
    try:
        rates = _mt5_copy_rates_from_pos(symbol, mt5_tf, 0, 3)
        if rates is None or len(rates) < 2:
            out["reason"] = "insufficient_bar_samples"
            out["error"] = f"Not enough {tf_name} bars returned for broker-time sanity check"
            return out
        import pandas as pd

        # _mt5_copy_rates_from_pos() already normalizes MT5 epochs to UTC.
        df = pd.DataFrame(rates)
        if "time" not in df.columns or len(df) < 2:
            out["reason"] = "missing_bar_times"
            out["error"] = f"{tf_name} rates did not include usable time values"
            return out
        bar_times = sorted(float(t) for t in df["time"].tolist())
        current_bar_open_epoch = float(bar_times[-1])
        last_closed_bar_open_epoch = float(bar_times[-2])
    except Exception as exc:
        out["reason"] = "bar_fetch_failed"
        out["error"] = str(exc)
        return out

    expected_current_bar_open_epoch = math.floor(now_utc_epoch / float(tf_secs)) * float(tf_secs)
    expected_last_closed_bar_open_epoch = expected_current_bar_open_epoch - float(tf_secs)
    current_bar_delta_seconds = float(current_bar_open_epoch - expected_current_bar_open_epoch)
    last_closed_bar_delta_seconds = float(last_closed_bar_open_epoch - expected_last_closed_bar_open_epoch)

    out.update(
        {
            "current_bar_open_utc_epoch": current_bar_open_epoch,
            "current_bar_open_utc_time": _epoch_to_utc_iso(current_bar_open_epoch),
            "expected_current_bar_open_utc_epoch": expected_current_bar_open_epoch,
            "expected_current_bar_open_utc_time": _epoch_to_utc_iso(expected_current_bar_open_epoch),
            "current_bar_delta_seconds": current_bar_delta_seconds,
            "last_closed_bar_open_utc_epoch": last_closed_bar_open_epoch,
            "last_closed_bar_open_utc_time": _epoch_to_utc_iso(last_closed_bar_open_epoch),
            "expected_last_closed_bar_open_utc_epoch": expected_last_closed_bar_open_epoch,
            "expected_last_closed_bar_open_utc_time": _epoch_to_utc_iso(expected_last_closed_bar_open_epoch),
            "last_closed_bar_delta_seconds": last_closed_bar_delta_seconds,
        }
    )

    stale_threshold_seconds = max(int(stale_bar_tolerance) * int(tf_secs), int(max_future_seconds))
    tick_stale = tick_age_seconds is not None and tick_age_seconds > float(max_tick_age_seconds)
    future_bar = current_bar_delta_seconds > float(max_future_seconds)
    stale_bar = current_bar_delta_seconds < -float(stale_threshold_seconds)
    offset_mismatch = (
        offset_inference_reliable
        and inferred_offset_seconds is not None
        and abs(int(inferred_offset_seconds) - int(configured_offset_seconds)) >= int(tick_offset_bucket_seconds)
    )
    tick_not_live_like = raw_tick_delta_seconds is not None and not offset_inference_reliable

    if future_bar or offset_mismatch:
        parts = []
        if offset_mismatch and inferred_offset_seconds is not None:
            parts.append(
                f"inferred broker offset is {inferred_offset_seconds}s but configuration resolves to {configured_offset_seconds}s"
            )
        if future_bar:
            parts.append(
                f"latest converted {tf_name} bar opens {int(round(current_bar_delta_seconds))}s in the future"
            )
        out["status"] = "misaligned"
        out["reason"] = "timezone_mismatch"
        out["warning"] = "MT5 broker-time sanity check failed: " + "; ".join(parts)
        return out

    if tick_not_live_like or tick_stale or stale_bar:
        parts = []
        if tick_not_live_like and raw_tick_delta_seconds is not None:
            parts.append(
                f"latest tick delta vs UTC is {int(round(raw_tick_delta_seconds))}s, which is not a plausible live broker offset"
            )
        if tick_stale and tick_age_seconds is not None:
            parts.append(f"latest tick is {int(round(tick_age_seconds))}s old after UTC normalization")
        if stale_bar:
            parts.append(
                f"latest converted {tf_name} bar lags expected current bar by {int(round(-current_bar_delta_seconds))}s"
            )
        out["status"] = "stale"
        out["reason"] = "market_data_stale"
        out["warning"] = "MT5 broker-time sanity check could not confirm live alignment: " + "; ".join(parts)
        return out

    out["status"] = "ok"
    out["reason"] = None
    return out


@lru_cache(maxsize=256)
def _cached_mt5_time_alignment(
    symbol: str,
    probe_timeframe: str,
    ttl_bucket: int,
    tick_offset_bucket_seconds: int,
    max_plausible_offset_seconds: int,
    max_future_seconds: int,
    max_tick_age_seconds: int,
    stale_bar_tolerance: int,
) -> Dict[str, Any]:
    return inspect_mt5_time_alignment(
        symbol=symbol,
        probe_timeframe=probe_timeframe,
        tick_offset_bucket_seconds=tick_offset_bucket_seconds,
        max_plausible_offset_seconds=max_plausible_offset_seconds,
        max_future_seconds=max_future_seconds,
        max_tick_age_seconds=max_tick_age_seconds,
        stale_bar_tolerance=stale_bar_tolerance,
    )


def clear_mt5_time_alignment_cache() -> None:
    """Clear cached broker/server time-alignment diagnostics."""
    _cached_mt5_time_alignment.cache_clear()


def get_cached_mt5_time_alignment(
    symbol: str = "EURUSD",
    probe_timeframe: str = "M1",
    *,
    ttl_seconds: int = 60,
    tick_offset_bucket_seconds: int = 900,
    max_plausible_offset_seconds: int = 18 * 3600,
    max_future_seconds: int = 90,
    max_tick_age_seconds: int = 180,
    stale_bar_tolerance: int = 3,
) -> Dict[str, Any]:
    """Return broker/server time-alignment diagnostics with an optional TTL cache."""
    try:
        ttl = int(ttl_seconds)
    except Exception:
        ttl = 60
    if ttl <= 0:
        return inspect_mt5_time_alignment(
            symbol=symbol,
            probe_timeframe=probe_timeframe,
            tick_offset_bucket_seconds=tick_offset_bucket_seconds,
            max_plausible_offset_seconds=max_plausible_offset_seconds,
            max_future_seconds=max_future_seconds,
            max_tick_age_seconds=max_tick_age_seconds,
            stale_bar_tolerance=stale_bar_tolerance,
        )
    bucket = int(time.time() / ttl)
    cached = _cached_mt5_time_alignment(
        str(symbol),
        str(probe_timeframe),
        bucket,
        int(tick_offset_bucket_seconds),
        int(max_plausible_offset_seconds),
        int(max_future_seconds),
        int(max_tick_age_seconds),
        int(stale_bar_tolerance),
    )
    return dict(cached)
