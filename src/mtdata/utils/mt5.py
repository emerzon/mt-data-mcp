"""MT5 connectivity, UTC timestamp, and low-level data helpers.

The MetaTrader5 Python API accepts UTC datetimes and returns Unix timestamps
in UTC. Broker timezone settings are therefore never applied to request
bounds or returned epochs; they are reserved for session/calendar semantics.
"""

import importlib
import logging
import math
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional, Tuple

from ..bootstrap.settings import mt5_config

logger = logging.getLogger(__name__)

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
            return _normalize_object_times(self._module().symbol_info_tick(symbol))

    def order_send(self, request):
        with _mt5_lock:
            return self._module().order_send(request)

    def positions_get(self, **kwargs):
        with _mt5_lock:
            return _normalize_object_time_rows(self._module().positions_get(**kwargs))

    def orders_get(self, **kwargs):
        with _mt5_lock:
            return _normalize_object_time_rows(self._module().orders_get(**kwargs))

    def history_orders_get(self, dt_from, dt_to, **kwargs):
        with _mt5_lock:
            return _normalize_object_time_rows(
                self._module().history_orders_get(dt_from, dt_to, **kwargs)
            )

    def history_deals_get(self, dt_from, dt_to, **kwargs):
        with _mt5_lock:
            return _normalize_object_time_rows(
                self._module().history_deals_get(dt_from, dt_to, **kwargs)
            )

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


def _raw_mt5_module() -> Any:
    if isinstance(mt5, MT5Adapter):
        return _load_mt5_module()
    return mt5


def _raw_symbol_info_tick(symbol: str) -> Any:
    with _mt5_lock:
        return _raw_mt5_module().symbol_info_tick(symbol)


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
    """Return an MT5 epoch unchanged because MT5 epochs are already UTC."""
    return float(epoch_seconds)


def _broker_timezone_note(
    *,
    server_tz_name: Optional[str],
    offset_seconds: Optional[int] = None,
) -> str:
    session_config = server_tz_name or (
        f"UTC offset {offset_seconds} seconds" if offset_seconds is not None else "UTC"
    )
    return (
        "MT5 request bounds and returned epochs use native UTC; broker session/calendar "
        f"calculations use {session_config}."
    )


def describe_mt5_time_normalization() -> Dict[str, Any]:
    """Describe MT5's native UTC timestamp contract and session configuration."""
    metadata: Dict[str, Any] = {
        "raw_time_basis": "mt5_utc_epoch",
        "time_basis": "utc",
        "time_normalization": "mt5_utc_native",
    }
    server_tz_name = str(getattr(mt5_config, "server_tz_name", "") or "").strip() or None
    try:
        static_offset_minutes = int(getattr(mt5_config, "time_offset_minutes", 0) or 0)
    except Exception:
        static_offset_minutes = 0

    if server_tz_name:
        metadata["broker_server_tz"] = server_tz_name
    elif static_offset_minutes:
        metadata["session_utc_offset_seconds"] = static_offset_minutes * 60
    metadata["timezone_note"] = _broker_timezone_note(
        server_tz_name=server_tz_name,
        offset_seconds=static_offset_minutes * 60 if static_offset_minutes else None,
    )
    return metadata


def _rates_to_df(rates: Any):
    """Convert MT5 rates into a DataFrame.

    Low-level MT5 copy helpers preserve the API's native UTC epochs, so this
    function must not apply broker timezone offsets.
    """
    import pandas as pd

    return pd.DataFrame(rates)


def _to_server_query_dt(dt: datetime) -> datetime:
    """Return the absolute request instant as a UTC-aware datetime."""
    return _to_utc_history_query_dt(dt)


def _to_utc_history_query_dt(dt: datetime) -> datetime:
    """Convert a datetime to a UTC-aware instant for MT5 history_* queries."""
    from .utils import _utc_epoch_seconds

    return datetime.fromtimestamp(_utc_epoch_seconds(dt), tz=timezone.utc)


def _to_mt5_history_epoch_seconds(dt: datetime, *, config: Any = None) -> float:
    """Convert an absolute UTC instant to MT5's native UTC epoch axis."""
    from .utils import _utc_epoch_seconds

    return float(_utc_epoch_seconds(dt))


def _normalize_times_in_struct(arr: Any):
    """Return MT5 structured rows unchanged; their time fields are already UTC."""
    return arr


def _normalize_object_times(obj: Any) -> Any:
    """Return an MT5 object unchanged; its timestamp attributes are already UTC."""
    return obj


def _normalize_object_time_rows(rows: Any) -> Any:
    """Return MT5 object-row collections unchanged; their epochs are UTC."""
    return rows


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
        with _mt5_lock:
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
    dt_srv = _to_server_query_dt(to_dt_utc)
    data = _mt5_read_with_retry(mt5.copy_rates_from, symbol, timeframe, dt_srv, count)
    return _normalize_times_in_struct(data)


def _mt5_copy_rates_range(symbol: str, timeframe, from_dt_utc: datetime, to_dt_utc: datetime):
    dt_from = _to_server_query_dt(from_dt_utc)
    dt_to = _to_server_query_dt(to_dt_utc)
    data = _mt5_read_with_retry(mt5.copy_rates_range, symbol, timeframe, dt_from, dt_to)
    return _normalize_times_in_struct(data)


def _mt5_copy_ticks_from(symbol: str, from_dt_utc: datetime, count: int, flags: int):
    dt_from = _to_server_query_dt(from_dt_utc)
    data = _mt5_read_with_retry(mt5.copy_ticks_from, symbol, dt_from, count, flags)
    return _normalize_times_in_struct(data)


def _mt5_copy_rates_from_pos(symbol: str, timeframe, start_pos: int, count: int):
    data = _mt5_read_with_retry(mt5.copy_rates_from_pos, symbol, timeframe, start_pos, count)
    return _normalize_times_in_struct(data)


def _mt5_copy_ticks_range(symbol: str, from_dt_utc: datetime, to_dt_utc: datetime, flags: int):
    dt_from = _to_server_query_dt(from_dt_utc)
    dt_to = _to_server_query_dt(to_dt_utc)
    data = _mt5_read_with_retry(mt5.copy_ticks_range, symbol, dt_from, dt_to, flags)
    return _normalize_times_in_struct(data)


class MT5Connection:
    def __init__(self):
        self._lock = threading.RLock()
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
        with self._lock:
            if self.is_connected():
                self._refresh_connection_identity()
                return True
            try:
                if mt5_config.has_credentials():
                    login = mt5_config.get_login()
                    password = mt5_config.get_password()
                    server = mt5_config.get_server()
                    if not mt5.initialize(login=login, password=password, server=server):
                        logger.error(
                            "Failed to initialize MT5 with configured credentials: "
                            f"{mt5.last_error()}"
                        )
                        return False
                    connected_login, _connected_server = self._read_connection_identity()
                    if connected_login is None or int(connected_login) != int(login):
                        logger.error(
                            "Connected MT5 account does not match configured login "
                            f"{login}; connected_login={connected_login}."
                        )
                        try:
                            mt5.shutdown()
                        except Exception:
                            pass
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
        with self._lock:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                self._connection_identity = None
                clear_symbol_info_cache()
                clear_mt5_time_alignment_cache()
                logger.debug("Disconnected from MetaTrader5")

    def is_connected(self) -> bool:
        with self._lock:
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


def _compact_symbol_name(value: Any) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def resolve_broker_symbol_name(symbol: str) -> str:
    query = str(symbol or "").strip()
    if not query:
        return query
    try:
        names = [
            str(getattr(info, "name", "") or "").strip()
            for info in (mt5.symbols_get() or [])
        ]
    except Exception:
        return query
    names = [name for name in names if name]
    case_matches = [name for name in names if name.casefold() == query.casefold()]
    if len(case_matches) == 1:
        return case_matches[0]
    query_compact = _compact_symbol_name(query)
    compact_matches = [
        name
        for name in names
        if query_compact and _compact_symbol_name(name) == query_compact
    ]
    if len(compact_matches) == 1:
        return compact_matches[0]
    return query


def _symbol_name_suggestions(symbol: str, *, limit: int = 5) -> list[str]:
    query = str(symbol or "").strip()
    if not query:
        return []
    query_upper = query.upper()
    query_compact = _compact_symbol_name(query)
    try:
        symbols = list(mt5.symbols_get() or [])
    except Exception:
        return []

    ranked: list[tuple[tuple[int, str], str]] = []
    seen: set[str] = set()
    for info in symbols:
        name = str(getattr(info, "name", "") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        name_upper = name.upper()
        name_compact = _compact_symbol_name(name)
        description = str(getattr(info, "description", "") or "").upper()
        if name_upper == query_upper:
            score = 0
        elif name_upper.startswith(query_upper):
            score = 1
        elif query_compact and name_compact.startswith(query_compact):
            score = 2
        elif query_upper in name_upper:
            score = 3
        elif query_upper in description:
            score = 4
        else:
            continue
        ranked.append(((score, name_upper), name))
    ranked.sort(key=lambda item: item[0])
    return [name for _key, name in ranked[: max(1, int(limit))]]


def _symbol_suggestion_suffix(symbol: str) -> str:
    suggestions = _symbol_name_suggestions(symbol)
    if not suggestions:
        return ""
    return " Closest broker symbols: " + ", ".join(suggestions) + "."


def _ensure_symbol_ready(symbol: str) -> Optional[str]:
    """Ensure a symbol is selected and its tick data is initialized.

    Returns an error string if selection or data readiness fails, else None.
    """
    try:
        data_ready_timeout, data_poll_interval = _data_ready_timing()
        info_before = mt5.symbol_info(symbol)
        was_visible = bool(info_before.visible) if info_before is not None else None
        if not mt5.symbol_select(symbol, True):
            if info_before is None:
                return (
                    f"Symbol '{symbol}' was not found in MT5. "
                    f"Use symbols_list(search_term='{symbol}') to find broker-specific names and suffixes."
                    f"{_symbol_suggestion_suffix(symbol)}"
                )
            return (
                f"Symbol '{symbol}' exists but could not be selected in MT5. "
                f"MT5 error: {mt5.last_error()}"
            )
        # If we just made it visible, wait briefly for fresh tick data
        if was_visible is False:
            deadline = time.time() + data_ready_timeout
            while time.time() < deadline:
                tick = _raw_symbol_info_tick(symbol)
                if tick and (getattr(tick, 'time', 0) or getattr(tick, 'bid', 0) or getattr(tick, 'ask', 0)):
                    break
                time.sleep(data_poll_interval)
        # Final check
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return (
                f"Symbol '{symbol}' was selected but no tick data is available. "
                f"The market may be closed or the broker may not be streaming this symbol. "
                f"MT5 error: {mt5.last_error()}"
            )
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
    """Inspect freshness and plausibility of native-UTC MT5 ticks and bars.

    The check is intentionally best-effort:
    - compare the latest tick's UTC epoch to the current UTC instant
    - fetch the latest bars for ``probe_timeframe`` and verify the bar open
      is not in the future relative to UTC and is not implausibly stale

    Legacy offset-related keyword arguments remain accepted so callers do not
    break, but broker offsets are not inferred or applied to UTC epochs.
    """
    from .time import format_epoch_utc

    out: Dict[str, Any] = {
        "symbol": str(symbol),
        "probe_timeframe": str(probe_timeframe),
        "status": "unavailable",
        "timestamp_contract": "mt5_utc_native",
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
    out["now_utc_time"] = format_epoch_utc(now_utc_epoch)

    raw_tick_epoch: Optional[float] = None
    try:
        tick = _raw_symbol_info_tick(symbol)
        if tick is not None:
            raw_tick_epoch = float(getattr(tick, "time", 0.0) or 0.0)
    except Exception:
        raw_tick_epoch = None

    tick_age_seconds: Optional[float] = None
    if raw_tick_epoch and raw_tick_epoch > 0:
        tick_age_seconds = float(now_utc_epoch - raw_tick_epoch)
        out["raw_tick_epoch"] = raw_tick_epoch
        out["raw_tick_time"] = format_epoch_utc(raw_tick_epoch)
        out["tick_age_seconds"] = tick_age_seconds

    current_bar_open_epoch: Optional[float] = None
    last_closed_bar_open_epoch: Optional[float] = None
    try:
        rates = _mt5_copy_rates_from_pos(symbol, mt5_tf, 0, 3)
        if rates is None or len(rates) < 2:
            out["reason"] = "insufficient_bar_samples"
            out["error"] = f"Not enough {tf_name} bars returned for MT5 UTC freshness check"
            return out
        import pandas as pd

        # MT5 rate epochs are native UTC.
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
            "current_bar_open_utc_time": format_epoch_utc(current_bar_open_epoch),
            "expected_current_bar_open_utc_epoch": expected_current_bar_open_epoch,
            "expected_current_bar_open_utc_time": format_epoch_utc(expected_current_bar_open_epoch),
            "current_bar_delta_seconds": current_bar_delta_seconds,
            "last_closed_bar_open_utc_epoch": last_closed_bar_open_epoch,
            "last_closed_bar_open_utc_time": format_epoch_utc(last_closed_bar_open_epoch),
            "expected_last_closed_bar_open_utc_epoch": expected_last_closed_bar_open_epoch,
            "expected_last_closed_bar_open_utc_time": format_epoch_utc(expected_last_closed_bar_open_epoch),
            "last_closed_bar_delta_seconds": last_closed_bar_delta_seconds,
        }
    )

    stale_threshold_seconds = max(int(stale_bar_tolerance) * int(tf_secs), int(max_future_seconds))
    tick_stale = tick_age_seconds is not None and tick_age_seconds > float(max_tick_age_seconds)
    tick_future = tick_age_seconds is not None and tick_age_seconds < -float(max_future_seconds)
    future_bar = current_bar_delta_seconds > float(max_future_seconds)
    stale_bar = current_bar_delta_seconds < -float(stale_threshold_seconds)

    if future_bar or tick_future:
        parts = []
        if tick_future and tick_age_seconds is not None:
            parts.append(f"latest tick is {int(round(-tick_age_seconds))}s in the future")
        if future_bar:
            parts.append(
                f"latest {tf_name} bar opens {int(round(current_bar_delta_seconds))}s in the future"
            )
        out["status"] = "misaligned"
        out["reason"] = "timestamp_in_future"
        out["warning"] = "MT5 UTC timestamp sanity check failed: " + "; ".join(parts)
        return out

    if tick_stale or stale_bar:
        parts = []
        if tick_stale and tick_age_seconds is not None:
            parts.append(f"latest tick is {int(round(tick_age_seconds))}s old")
        if stale_bar:
            parts.append(
                f"latest {tf_name} bar lags expected current bar by {int(round(-current_bar_delta_seconds))}s"
            )
        out["status"] = "stale"
        out["reason"] = "market_data_stale"
        out["warning"] = "MT5 UTC freshness check found stale data: " + "; ".join(parts)
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
