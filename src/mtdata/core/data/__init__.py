import logging
import statistics
from typing import Any, Dict, List, Optional

from ...services.data_service import fetch_candles, fetch_ticks
from ...shared.schema import CompactFullDetailLiteral, TimeframeLiteral
from ...utils.mt5 import ensure_mt5_connection_or_raise
from ...utils.utils import _coerce_finite_float
from .._mcp_instance import mcp
from ..execution_logging import run_logged_operation
from ..mt5_gateway import create_mt5_gateway
from ..pivot import pivot_compute_points, support_resistance_levels
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

_COMPACT_WAIT_EVENT_SPEC_FIELDS = (
    "type",
    "symbol",
    "timeframe",
    "ticket",
    "order_ticket",
    "position_ticket",
    "magic",
    "side",
    "direction",
    "level",
    "lower",
    "upper",
    "distance",
    "price_source",
    "threshold_mode",
    "threshold_value",
)


def _build_default_wait_event_watchers(
    *,
    symbol: str,
    timeframe: TimeframeLiteral,
    watch_tick_count_spike: bool,
) -> List[Dict[str, Any]]:
    watch_for: List[Dict[str, Any]] = [
        {"type": "order_created", "symbol": symbol},
        {"type": "order_filled", "symbol": symbol},
        {"type": "order_cancelled", "symbol": symbol},
        {"type": "position_opened", "symbol": symbol},
        {"type": "position_closed", "symbol": symbol},
        {"type": "tp_hit", "symbol": symbol},
        {"type": "sl_hit", "symbol": symbol},
        {"type": "pending_near_fill", "symbol": symbol},
        {"type": "stop_threat", "symbol": symbol},
        {"type": "price_change", "symbol": symbol},
        {"type": "volume_spike", "symbol": symbol},
        {"type": "spread_spike", "symbol": symbol},
        {"type": "tick_count_drought", "symbol": symbol},
        {"type": "range_expansion", "symbol": symbol},
    ]
    if watch_tick_count_spike:
        watch_for.append({"type": "tick_count_spike", "symbol": symbol})
    watch_for.extend(_support_resistance_watchers(symbol=symbol, timeframe=timeframe))
    watch_for.extend(_pivot_zone_watchers(symbol=symbol, timeframe=timeframe))
    return _dedupe_wait_event_watchers(watch_for)


def _support_resistance_watchers(
    *,
    symbol: str,
    timeframe: TimeframeLiteral,
) -> List[Dict[str, Any]]:
    try:
        raw_tool = getattr(support_resistance_levels, "__wrapped__", support_resistance_levels)
        payload = raw_tool(symbol=symbol, timeframe=timeframe, detail="compact")
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
                "symbol": symbol,
                "level": level_value,
                "direction": "either",
            }
        )
        watch_for.append(
            {
                "type": "price_break_level",
                "symbol": symbol,
                "level": level_value,
                "direction": direction,
            }
        )
    return watch_for


def _pivot_zone_watchers(*, symbol: str, timeframe: TimeframeLiteral) -> List[Dict[str, Any]]:
    try:
        raw_tool = getattr(pivot_compute_points, "__wrapped__", pivot_compute_points)
        payload = raw_tool(symbol=symbol, timeframe=_default_wait_event_pivot_timeframe(timeframe))
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
                "symbol": symbol,
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


def _compact_wait_event_spec(spec: Any) -> Dict[str, Any]:
    if hasattr(spec, "model_dump"):
        raw = spec.model_dump(exclude_none=True)
    elif isinstance(spec, dict):
        raw = dict(spec)
    else:
        raw = {"type": str(spec)}
    return {
        field_name: raw.get(field_name)
        for field_name in _COMPACT_WAIT_EVENT_SPEC_FIELDS
        if raw.get(field_name) is not None
    }


def _compact_wait_event_specs(specs: Any, *, inferred: bool) -> Dict[str, Any]:
    source = list(specs or []) if isinstance(specs, list) else []
    items = [_compact_wait_event_spec(item) for item in source]
    event_types = sorted(
        {
            str(item.get("type"))
            for item in items
            if item.get("type") not in (None, "")
        }
    )
    out: Dict[str, Any] = {
        "inferred": bool(inferred),
        "count": len(items),
        "types": event_types,
    }
    if not inferred or len(items) <= 8:
        out["items"] = items
    return out


def _compact_wait_event_public_result(
    result: Dict[str, Any],
    *,
    explicit_watch_for: bool,
    explicit_end_on: bool,
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    out = dict(result)
    out.pop("max_wait_seconds", None)

    criteria_in = out.get("criteria")
    criteria = dict(criteria_in) if isinstance(criteria_in, dict) else None
    if criteria is not None:
        criteria["watch_for_inferred"] = not explicit_watch_for
        criteria["end_on_inferred"] = not explicit_end_on

    if str(detail or "compact").strip().lower() == "full":
        if criteria is not None:
            out["criteria"] = criteria
        return out

    for key in (
        "matched",
        "event",
        "criteria",
        "timeframe",
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
        closed_candle = boundary_event.get("closed_candle")
        if isinstance(closed_candle, dict) and closed_candle:
            compact_boundary["closed_candle"] = dict(closed_candle)
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


@mcp.tool()
def data_fetch_candles(
    request: DataFetchCandlesRequest,
) -> Dict[str, Any]:
    """Fetch historical candle data with optional technical indicators and denoising.
    
    **REQUIRED**: symbol parameter must be provided (e.g., "EURUSD", "BTCUSD")
    
    Features:
    ---------
    - OHLCV data as tabular rows
    - Optional historical candle spread column via include_spread=true
    - Technical indicators (RSI, MACD, EMA, SMA, etc.)
    - Data denoising and smoothing
    - Data simplification for large datasets
    - Defaults to closed candles only; set include_incomplete=true to keep the latest forming candle
    - Set allow_stale=true to return the latest available closed bars even when freshness checks would normally fail
    - Includes metadata for forming-candle handling (for example has_forming_candle and incomplete_candles_skipped)
    
    Parameters:
    -----------
    symbol : str (REQUIRED)
        Trading symbol (e.g., "EURUSD", "GBPUSD", "BTCUSD")
    
    timeframe : str, optional (default="H1")
        Candle timeframe: "M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"

    detail : {"compact", "standard", "full"}, optional
        Response detail level. `compact` (default) returns rows plus concise
        freshness when available. `standard` also includes latency and policy
        freshness signals. `full` preserves the debug `meta` diagnostics block.
    
    limit : int, optional (default=200)
        Maximum number of candles to return
    
    start : str, optional
        Start time (dateparser)

    end : str, optional
        End time (dateparser)
    
    ohlcv : str, optional
        Candle fields to include. Use "all", "ohlcv", "ohlc", "close"/"price",
        compact letters from o/h/l/c/v, or comma-separated field names such as
        "open,high,low,close,volume".

    include_spread : bool, optional
        Append the historical MT5 candle spread column to each returned row.
        Defaults to false because many symbols/timeframes return missing or zero
        historical spread and the extra column increases every row.
    
    indicators : list, optional
        Technical indicators list, e.g., [{"name": "rsi", "params": [14]}]
        Or compact string: "rsi(14),ema(20),macd(12,26,9)"
    
    denoise : dict, optional
        Denoising configuration to smooth price data
    
    simplify : dict, optional
        Data reduction options for large datasets. Use a dict such as
        {"method": "lttb", "points": 100} or {"ratio": 0.25}. Passing
        true/"on"/"default" enables default simplification; false/"off"
        disables it.

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
        - has_forming_candle: bool (true when the latest available candle is still forming)
        - forming_candle_status: str ("included", "skipped", "detected", or "none")
        - forming_candle_included: bool (true when the forming candle is present in data)
        - forming_candle_skipped: bool (true when a forming candle was detected but trimmed)
        - incomplete_candles_skipped: int (number of forming candles trimmed because include_incomplete=false)
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

    # Opt in to historical candle spread output
    data_fetch_candles(symbol="EURUSD", include_spread=True)
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
            gateway=create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
            fetch_candles_impl=fetch_candles,
        ),
    )

@mcp.tool()
def data_fetch_ticks(
    request: DataFetchTicksRequest,
) -> Dict[str, Any]:
    """Fetch tick data for a symbol.

    By default (`detail="compact"`), returns a compact set of descriptive stats
    over the fetched ticks (bid/ask/mid/spread, plus last and volume; volume uses real
    volume when available, otherwise tick_volume).

    Use `detail="stats"` for a more detailed stats payload.
    Use `detail="rows"` to return raw tick rows as structured data.
    `simplify` only applies to row output. Use a dict such as
    {"method": "lttb", "points": 100} or pass true/"on"/"default" for
    default simplification; false/"off" disables it.
    """
    return run_logged_operation(
        logger,
        operation="data_fetch_ticks",
        symbol=request.symbol,
        limit=request.limit,
        detail=request.detail,
        func=lambda: run_data_fetch_ticks(
            request,
            gateway=create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
            fetch_ticks_impl=fetch_ticks,
        ),
    )


@mcp.tool()
def wait_event(
    symbol: Optional[str] = None,
    timeframe: TimeframeLiteral = "M1",
    watch_tick_count_spike: bool = True,
    watch_for: Optional[List[Dict[str, Any]]] = None,
    end_on: Optional[List[Dict[str, Any]]] = None,
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    """Wait for watch events on a symbol until the next timeframe boundary.

    If `watch_for` is omitted, the public default watches the full event set:
    order/position lifecycle events, pending/stop proximity, volatility/activity
    events, support/resistance touch and break levels, and pivot-based zone
    entry events. Support/resistance defaults come from
    `support_resistance_levels(symbol, timeframe="auto")`; pivot zones default
    to adjacent daily pivot bands for intraday waits and same-timeframe pivots
    for daily-or-higher waits.

    `symbol` is required when `watch_for` is omitted and the tool is inferring
    its default watcher set. For boundary-only waits, pass `watch_for=[]` and
    rely on `timeframe` or explicit `end_on` candle-close events.

    Boundary waits belong in `end_on` as `{"type": "candle_close", ...}`.
    `watch_for` is for explicit market/account event objects only; pass
    candle-close boundary objects in `end_on` instead.
    When a candle boundary is reached and a symbol is known, `boundary_event`
    includes a best-effort `closed_candle` snapshot with OHLCV and basic
    range/body/wick stats for the candle that just closed.

    Example: `end_on=[{"type": "candle_close", "timeframe": "H1"}]` or
    `watch_for=[{"type": "order_filled", "symbol": "EURUSD"}]`.

    Advanced callers can pass explicit `watch_for` and `end_on` event specs to
    use the richer wait-event engine directly. When explicit `watch_for` is
    provided, `watch_tick_count_spike` no longer alters the watcher list.
    Set `detail="full"` to include polling/timing details and the full criteria
    echo in the response.
    """
    symbol_value = str(symbol or "").strip() or None
    explicit_watch_for = watch_for is not None
    explicit_end_on = end_on is not None
    symbol_error: Optional[str] = None
    if symbol_value is None and not explicit_watch_for:
        symbol_error = "symbol is required when watch_for is omitted."

    def _run() -> Dict[str, Any]:
        if symbol_error is not None:
            return {"error": symbol_error}
        request_kwargs: Dict[str, Any] = {
            "timeframe": timeframe,
        }
        if symbol_value is not None:
            request_kwargs["symbol"] = symbol_value
        if end_on is not None:
            request_kwargs["end_on"] = list(end_on)
        else:
            request_kwargs["end_on"] = [
                {"type": "candle_close", "timeframe": timeframe},
            ]
        resolved_watch_for = (
            list(watch_for)
            if explicit_watch_for
            else _build_default_wait_event_watchers(
                symbol=symbol_value,
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
            gateway=create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
        )
        if isinstance(result, dict):
            result = _compact_wait_event_public_result(
                result,
                explicit_watch_for=explicit_watch_for,
                explicit_end_on=explicit_end_on,
                detail=detail,
            )
        return result

    return run_logged_operation(
        logger,
        operation="wait_event",
        symbol=symbol_value,
        timeframe=timeframe,
        watch_tick_count_spike=watch_tick_count_spike,
        detail=detail,
        explicit_watch_for=explicit_watch_for,
        end_on_count=len(end_on or []),
        func=_run,
    )
