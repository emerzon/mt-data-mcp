import inspect
import logging
import statistics
from typing import Any, Dict, List, Optional, Union

from ...services.data_service import fetch_candles, fetch_ticks
from ...utils.mt5 import ensure_mt5_connection_or_raise
from ...utils.utils import _coerce_finite_float
from .._mcp_instance import mcp
from ..execution_logging import run_logged_operation
from ..mt5_gateway import get_mt5_gateway
from ..pivot import pivot_compute_points, support_resistance_levels
from ..schema import TimeframeLiteral
from .requests import (
    DataFetchCandlesRequest,
    DataFetchTicksRequest,
    WaitEventRequest,
)
from .use_cases import (
    run_data_fetch_candles,
    run_data_fetch_ticks,
    run_wait_event,
)
from .wait_events import _WAIT_EVENT_IDENTITY_FIELDS

# Explicitly define what should be exported for '*' imports
__all__ = ['data_fetch_candles', 'data_fetch_ticks', 'wait_event']

logger = logging.getLogger(__name__)

WaitEventPublicWatchSpec = Union[str, Dict[str, Any]]

_WAIT_EVENT_PUBLIC_TYPE_ALIASES = {
    "price_level_touch": "price_touch_level",
}

_WAIT_EVENT_PUBLIC_WATCH_TYPES = frozenset(
    {
        "order_created",
        "order_filled",
        "order_cancelled",
        "position_opened",
        "position_closed",
        "tp_hit",
        "sl_hit",
        "price_change",
        "volume_spike",
        "tick_count_spike",
        "spread_spike",
        "tick_count_drought",
        "range_expansion",
        "price_touch_level",
        "price_break_level",
        "price_enter_zone",
        "pending_near_fill",
        "stop_threat",
    }
)


def _normalize_public_wait_event_type(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return _WAIT_EVENT_PUBLIC_TYPE_ALIASES.get(text, text)


def _build_default_wait_event_watchers(
    *,
    instrument: str,
    timeframe: TimeframeLiteral,
    watch_tick_count_spike: bool,
) -> List[Dict[str, Any]]:
    watch_for: List[Dict[str, Any]] = [
        {"type": "order_created", "symbol": instrument},
        {"type": "order_filled", "symbol": instrument},
        {"type": "order_cancelled", "symbol": instrument},
        {"type": "position_opened", "symbol": instrument},
        {"type": "position_closed", "symbol": instrument},
        {"type": "tp_hit", "symbol": instrument},
        {"type": "sl_hit", "symbol": instrument},
        {"type": "pending_near_fill", "symbol": instrument},
        {"type": "stop_threat", "symbol": instrument},
        {"type": "price_change", "symbol": instrument},
        {"type": "volume_spike", "symbol": instrument},
        {"type": "spread_spike", "symbol": instrument},
        {"type": "tick_count_drought", "symbol": instrument},
        {"type": "range_expansion", "symbol": instrument},
    ]
    if watch_tick_count_spike:
        watch_for.append({"type": "tick_count_spike", "symbol": instrument})
    watch_for.extend(
        _support_resistance_watchers(instrument=instrument, timeframe=timeframe)
    )
    watch_for.extend(_pivot_zone_watchers(instrument=instrument, timeframe=timeframe))
    return _dedupe_wait_event_watchers(watch_for)


def _expand_public_wait_event_watchers(
    watch_for: Optional[List[WaitEventPublicWatchSpec]],
    *,
    instrument: str,
    timeframe: TimeframeLiteral,
) -> Optional[List[Any]]:
    if watch_for is None:
        return None
    if not any(isinstance(item, str) for item in watch_for):
        return list(watch_for)

    default_watchers = _build_default_wait_event_watchers(
        instrument=instrument,
        timeframe=timeframe,
        watch_tick_count_spike=True,
    )
    watchers_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for item in default_watchers:
        event_type = _normalize_public_wait_event_type(item.get("type"))
        if not event_type:
            continue
        watchers_by_type.setdefault(event_type, []).append(dict(item))

    resolved: List[Any] = []
    for item in watch_for:
        if not isinstance(item, str):
            resolved.append(item)
            continue

        event_type = _normalize_public_wait_event_type(item)
        if event_type not in _WAIT_EVENT_PUBLIC_WATCH_TYPES:
            supported = ", ".join(sorted(_WAIT_EVENT_PUBLIC_WATCH_TYPES))
            raise ValueError(
                f"Unsupported wait_event watch_for shorthand {item!r}. "
                f"Use one of: {supported}, or pass a full event object."
            )

        matches = watchers_by_type.get(event_type, [])
        if not matches:
            raise ValueError(
                f"Could not resolve default wait_event watchers for {event_type!r}. "
                "Pass a full event object with explicit parameters instead."
            )
        resolved.extend(dict(match) for match in matches)

    if all(isinstance(item, dict) for item in resolved):
        return _dedupe_wait_event_watchers([dict(item) for item in resolved])
    return resolved


def _support_resistance_watchers(
    *,
    instrument: str,
    timeframe: TimeframeLiteral,
) -> List[Dict[str, Any]]:
    try:
        raw_tool = getattr(support_resistance_levels, "__wrapped__", support_resistance_levels)
        payload = raw_tool(symbol=instrument, timeframe=timeframe, detail="compact")
    except Exception:
        return []
    if not isinstance(payload, dict) or payload.get("error"):
        return []
    levels = payload.get("levels")
    if not isinstance(levels, list):
        return []
    watch_for: List[Dict[str, Any]] = []
    for level in levels:
        if not isinstance(level, dict):
            continue
        level_value = _coerce_finite_float(level.get("value"))
        if level_value is None:
            continue
        level_type = str(level.get("type") or "").strip().lower()
        direction = "either"
        if level_type == "support":
            direction = "down"
        elif level_type == "resistance":
            direction = "up"
        watch_for.append(
            {
                "type": "price_touch_level",
                "symbol": instrument,
                "level": level_value,
                "direction": "either",
            }
        )
        watch_for.append(
            {
                "type": "price_break_level",
                "symbol": instrument,
                "level": level_value,
                "direction": direction,
            }
        )
    return watch_for


def _pivot_zone_watchers(*, instrument: str, timeframe: TimeframeLiteral) -> List[Dict[str, Any]]:
    try:
        raw_tool = getattr(pivot_compute_points, "__wrapped__", pivot_compute_points)
        payload = raw_tool(symbol=instrument, timeframe=_default_wait_event_pivot_timeframe(timeframe))
    except Exception:
        return []
    if not isinstance(payload, dict) or payload.get("error"):
        return []
    levels = _extract_pivot_levels(payload)
    if len(levels) < 2:
        return []
    watch_for: List[Dict[str, Any]] = []
    for idx in range(len(levels) - 1):
        lower = levels[idx]["value"]
        upper = levels[idx + 1]["value"]
        if upper <= lower:
            continue
        watch_for.append(
            {
                "type": "price_enter_zone",
                "symbol": instrument,
                "lower": lower,
                "upper": upper,
                "direction": "either",
            }
        )
    return watch_for


def _default_wait_event_pivot_timeframe(timeframe: TimeframeLiteral) -> TimeframeLiteral:
    normalized = str(timeframe or "M1").upper().strip()
    if normalized in {"D1", "W1", "MN1"}:
        return normalized  # type: ignore[return-value]
    return "D1"


def _extract_pivot_levels(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("levels")
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    seen_values: set[float] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("level") or "").strip().upper()
        if not label:
            continue
        values = [
            numeric
            for numeric in (_coerce_finite_float(value) for key, value in row.items() if key != "level")
            if numeric is not None
        ]
        if not values:
            continue
        price = round(float(statistics.median(values)), 10)
        if price in seen_values:
            continue
        seen_values.add(price)
        out.append({"label": label, "value": price})
    out.sort(key=lambda item: float(item["value"]))
    return out


def _dedupe_wait_event_watchers(watch_for: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in watch_for:
        key = (
            str(item.get("type") or ""),
            str(item.get("symbol") or "").upper(),
            item.get("order_ticket"),
            item.get("position_ticket"),
            item.get("magic"),
            item.get("side"),
            item.get("direction"),
            item.get("level"),
            item.get("lower"),
            item.get("upper"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(item))
    return out


def _compact_wait_event_public_result(
    result: Dict[str, Any],
    *,
    explicit_watch_for: bool,
    explicit_end_on: bool,
    verbose: bool = False,
) -> Dict[str, Any]:
    out = dict(result)
    out.pop("max_wait_seconds", None)

    criteria_in = out.get("criteria")
    criteria = dict(criteria_in) if isinstance(criteria_in, dict) else None
    if criteria is not None:
        criteria["watch_for_inferred"] = not explicit_watch_for
        criteria["end_on_inferred"] = not explicit_end_on

    if verbose:
        if criteria is not None:
            out["criteria"] = criteria
        return out

    for key in (
        "matched",
        "event",
        "criteria",
        "started_at_utc",
        "elapsed_seconds",
        "polls",
        "poll_interval_seconds",
        "sleep_seconds",
        "slept",
        "slept_seconds",
        "remaining_seconds",
    ):
        out.pop(key, None)

    boundary_event = out.get("boundary_event")
    if isinstance(boundary_event, dict):
        compact_boundary = {
            key: boundary_event.get(key)
            for key in ("type", "timeframe")
            if boundary_event.get(key) is not None
        }
        out["boundary_event"] = compact_boundary or None

    matched_event = out.get("matched_event")
    if isinstance(matched_event, dict):
        compact_matched: Dict[str, Any] = {}
        event_type = matched_event.get("type")
        if event_type is not None:
            compact_matched["type"] = event_type
        for field_name in _WAIT_EVENT_IDENTITY_FIELDS:
            value = matched_event.get(field_name)
            if value is not None:
                compact_matched[field_name] = value
        observed = matched_event.get("observed")
        if isinstance(observed, dict) and observed:
            compact_matched["observed"] = dict(observed)
        out["matched_event"] = compact_matched or None

    return out


def _apply_legacy_wait_event_positional_args(
    args: tuple[Any, ...],
    *,
    symbol: Optional[str],
    instrument: Optional[str],
    timeframe: TimeframeLiteral,
    watch_tick_count_spike: bool,
    watch_for: Optional[List[WaitEventPublicWatchSpec]],
    end_on: Optional[List[Dict[str, Any]]],
    verbose: bool,
) -> tuple[Any, ...]:
    if not args:
        return symbol, instrument, timeframe, watch_tick_count_spike, watch_for, end_on, verbose

    legacy_names = (
        "symbol",
        "timeframe",
        "watch_tick_count_spike",
        "watch_for",
        "end_on",
        "verbose",
    )
    if len(args) > len(legacy_names):
        raise TypeError(
            f"wait_event() takes from 0 to {len(legacy_names)} positional arguments but {len(args)} were given"
        )

    duplicate_checks = (
        ("symbol", len(args) >= 1 and symbol is not None),
        ("timeframe", len(args) >= 2 and timeframe != "M1"),
        ("watch_tick_count_spike", len(args) >= 3 and watch_tick_count_spike is not True),
        ("watch_for", len(args) >= 4 and watch_for is not None),
        ("end_on", len(args) >= 5 and end_on is not None),
        ("verbose", len(args) >= 6 and verbose is not False),
    )
    for name, duplicated in duplicate_checks:
        if duplicated:
            raise TypeError(f"wait_event() got multiple values for argument '{name}'")

    legacy_values = dict(zip(legacy_names, args))
    return (
        legacy_values.get("symbol", symbol),
        instrument,
        legacy_values.get("timeframe", timeframe),
        legacy_values.get("watch_tick_count_spike", watch_tick_count_spike),
        legacy_values.get("watch_for", watch_for),
        legacy_values.get("end_on", end_on),
        legacy_values.get("verbose", verbose),
    )


_WAIT_EVENT_PUBLIC_SIGNATURE = inspect.Signature(
    parameters=[
        inspect.Parameter(
            "symbol",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
            annotation=Optional[str],
        ),
        inspect.Parameter(
            "instrument",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
            annotation=Optional[str],
        ),
        inspect.Parameter(
            "timeframe",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default="M1",
            annotation=TimeframeLiteral,
        ),
        inspect.Parameter(
            "watch_tick_count_spike",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=True,
            annotation=bool,
        ),
        inspect.Parameter(
            "watch_for",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
            annotation=Optional[List[WaitEventPublicWatchSpec]],
        ),
        inspect.Parameter(
            "end_on",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
            annotation=Optional[List[Dict[str, Any]]],
        ),
        inspect.Parameter(
            "verbose",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=False,
            annotation=bool,
        ),
    ],
    return_annotation=Dict[str, Any],
)


@mcp.tool()
def data_fetch_candles(
    request: DataFetchCandlesRequest,
) -> Dict[str, Any]:
    """Fetch historical candle data with optional technical indicators and denoising.
    
    **REQUIRED**: symbol parameter must be provided (e.g., "EURUSD", "BTCUSD")
    
    Features:
    ---------
    - OHLCV data as tabular rows
    - Technical indicators (RSI, MACD, EMA, SMA, etc.)
    - Data denoising and smoothing
    - Data simplification for large datasets
    - Defaults to closed candles only; set include_incomplete=true to keep the latest forming candle
    - Set allow_stale=true to return the latest available closed bars even when freshness checks would normally fail
    - Includes metadata: last_candle_open (true if last candle is still forming)
    
    Parameters:
    -----------
    symbol : str (REQUIRED)
        Trading symbol (e.g., "EURUSD", "GBPUSD", "BTCUSD")
    
    timeframe : str, optional (default="H1")
        Candle timeframe: "M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"

    detail : {"compact", "full"}, optional
        Response detail level. `compact` (default) strips runtime diagnostics,
        while `full` preserves the existing `meta` diagnostics block.
    
    limit : int, optional (default=200)
        Maximum number of candles to return
    
    start : str, optional
        Start time (dateparser)

    end : str, optional
        End time (dateparser)
    
    ohlcv : str, optional
        Fields to include: "close", "ohlc", "ohlcv", "all"
    
    indicators : list, optional
        Technical indicators list, e.g., [{"name": "rsi", "params": [14]}]
        Or compact string: "rsi(14),ema(20),macd(12,26,9)"
    
    denoise : dict, optional
        Denoising configuration to smooth price data
    
    simplify : dict, optional
        Data reduction options for large datasets

    include_incomplete : bool, optional
        Keep the latest forming candle instead of trimming it. Defaults to false.

    allow_stale : bool, optional
        Return the latest available closed bars even if they fall outside the normal freshness window. Defaults to false.
    
    Returns:
    --------
    dict
        - success: bool
        - symbol: str
        - timeframe: str
        - candles: int (number of candles returned)
        - last_candle_open: bool (true if last candle is still forming)
        - data: list[dict] (tabular candle rows)
    
    Examples:
    ---------
    # Get last 200 H1 candles
    data_fetch_candles(symbol="EURUSD")
    
    # Get 100 M15 candles with RSI indicator
    data_fetch_candles(
        symbol="EURUSD",
        timeframe="M15",
        limit=100,
        indicators="rsi(14)"
    )
    
    # Get date range with multiple indicators
    data_fetch_candles(
        symbol="GBPUSD",
        start="2025-11-01",
        end="2025-11-30",
        indicators="rsi(14),ema(20),macd(12,26,9)"
    )
    """
    return run_logged_operation(
        logger,
        operation="data_fetch_candles",
        symbol=request.symbol,
        timeframe=request.timeframe,
        detail=request.detail,
        limit=request.limit,
        func=lambda: run_data_fetch_candles(
            request,
            gateway=get_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
            fetch_candles_impl=fetch_candles,
        ),
    )

@mcp.tool()
def data_fetch_ticks(
    request: DataFetchTicksRequest,
) -> Dict[str, Any]:
    """Fetch tick data for a symbol.

    By default (`format="summary"`), returns a compact set of descriptive stats
    over the fetched ticks (bid/ask/mid/spread, plus last and volume; volume uses real
    volume when available, otherwise tick_volume).

    Shared output-contract aliases are also accepted: `format="compact"`
    maps to `summary`, and `format="full"` maps to `stats`.

    Use `format="stats"` for a more detailed stats payload.
    Use `format="rows"` to return raw tick rows as structured data.
    `simplify` only applies to row output.
    """
    return run_logged_operation(
        logger,
        operation="data_fetch_ticks",
        symbol=request.symbol,
        limit=request.limit,
        format=request.format,
        func=lambda: run_data_fetch_ticks(
            request,
            gateway=get_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
            fetch_ticks_impl=fetch_ticks,
        ),
    )


@mcp.tool()
def wait_event(
    *args: Any,
    symbol: Optional[str] = None,
    instrument: Optional[str] = None,
    timeframe: TimeframeLiteral = "M1",
    watch_tick_count_spike: bool = True,
    watch_for: Optional[List[WaitEventPublicWatchSpec]] = None,
    end_on: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Wait for watch events on a symbol until the next timeframe boundary.

    If `watch_for` is omitted, the public default watches the full event set:
    order/position lifecycle events, pending/stop proximity, volatility/activity
    events, support/resistance touch and break levels, and pivot-based zone
    entry events. Support/resistance defaults come from
    `support_resistance_levels(symbol, timeframe="auto")`; pivot zones default
    to adjacent daily pivot bands for intraday waits and same-timeframe pivots
    for daily-or-higher waits.

    Boundary waits belong in `end_on` as `{"type": "candle_close", ...}`.
    `watch_for` is for market/account events, and the canonical price-level
    event name is `price_touch_level`.

    For compatibility, `watch_for` may also be a list of event-name strings
    such as `"order_created"` or `"price_touch_level"`; these expand into the
    corresponding built-in public watcher defaults for the current symbol.

    Advanced callers can pass explicit `watch_for` and `end_on` event specs to
    use the richer wait-event engine directly. When explicit `watch_for` is
    provided, `watch_tick_count_spike` no longer alters the watcher list.
    Set `verbose=true` to include polling/timing details and the full criteria
    echo in the response.
    """
    (
        symbol,
        instrument,
        timeframe,
        watch_tick_count_spike,
        watch_for,
        end_on,
        verbose,
    ) = _apply_legacy_wait_event_positional_args(
        args,
        symbol=symbol,
        instrument=instrument,
        timeframe=timeframe,
        watch_tick_count_spike=watch_tick_count_spike,
        watch_for=watch_for,
        end_on=end_on,
        verbose=verbose,
    )
    symbol_value = str(symbol or "").strip()
    instrument_value = str(instrument or "").strip()
    resolved_symbol: Optional[str]
    symbol_error: Optional[str] = None
    if symbol_value and instrument_value and symbol_value != instrument_value:
        resolved_symbol = None
        symbol_error = "Provide either symbol or instrument, not both with different values."
    else:
        resolved_symbol = symbol_value or instrument_value or None
        if resolved_symbol is None:
            symbol_error = "A symbol is required."

    def _run() -> Dict[str, Any]:
        if symbol_error is not None:
            return {"error": symbol_error}
        assert resolved_symbol is not None
        explicit_watch_for = watch_for is not None
        explicit_end_on = end_on is not None
        request_kwargs: Dict[str, Any] = {
            "symbol": resolved_symbol,
            "timeframe": timeframe,
        }
        if end_on is not None:
            request_kwargs["end_on"] = list(end_on)
        resolved_watch_for = (
            _expand_public_wait_event_watchers(
                watch_for,
                instrument=resolved_symbol,
                timeframe=timeframe,
            )
            if explicit_watch_for
            else _build_default_wait_event_watchers(
                instrument=resolved_symbol,
                timeframe=timeframe,
                watch_tick_count_spike=watch_tick_count_spike,
            )
        )
        request = WaitEventRequest(
            **request_kwargs,
            watch_for=resolved_watch_for,
        )
        result = run_wait_event(
            request,
            gateway=get_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
        )
        if isinstance(result, dict):
            result = _compact_wait_event_public_result(
                result,
                explicit_watch_for=explicit_watch_for,
                explicit_end_on=explicit_end_on,
                verbose=verbose,
            )
        return result

    return run_logged_operation(
        logger,
        operation="wait_event",
        symbol=resolved_symbol,
        instrument_alias_used=bool(not symbol_value and instrument_value),
        timeframe=timeframe,
        watch_tick_count_spike=watch_tick_count_spike,
        verbose=verbose,
        explicit_watch_for=watch_for is not None,
        end_on_count=len(end_on or []),
        func=_run,
    )


wait_event.__signature__ = _WAIT_EVENT_PUBLIC_SIGNATURE
_raw_wait_event = getattr(wait_event, "__wrapped__", None)
if callable(_raw_wait_event):
    _raw_wait_event.__signature__ = _WAIT_EVENT_PUBLIC_SIGNATURE
