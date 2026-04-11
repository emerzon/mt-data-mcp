
import logging
import math
import os
import time
from typing import Any, Dict

from ..utils.mt5 import (
    MT5ConnectionError,
    _mt5_epoch_to_utc,
    ensure_mt5_connection_or_raise,
    mt5,
)
from ..utils.utils import (
    _format_time_minimal,
    _format_time_minimal_local,
    _use_client_tz,
)
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .mt5_gateway import get_mt5_gateway

logger = logging.getLogger(__name__)
_MARKET_DEPTH_ENABLE_ENV = "MTDATA_ENABLE_MARKET_DEPTH_FETCH"


def _market_depth_fetch_enabled() -> bool:
    raw = os.getenv(_MARKET_DEPTH_ENABLE_ENV)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _format_mt5_error_text(error: Any) -> str:
    if isinstance(error, tuple) and len(error) >= 2:
        code = error[0]
        message = error[1]
        if message not in (None, ""):
            return f"{code}: {message}"
        return str(code)
    return str(error)


def _describe_symbol_select_error(symbol: str, error: Any) -> str:
    text = str(_format_mt5_error_text(error or "")).strip()
    lowered = text.lower()
    if any(marker in lowered for marker in ("call failed", "unknown symbol", "not found", "invalid symbol")):
        return f"Symbol '{symbol}' was not found or is not available in MT5."
    if text:
        return f"Failed to select symbol {symbol}: {text}"
    return f"Failed to select symbol {symbol}."


def _market_depth_disabled_payload() -> Dict[str, Any]:
    return {
        "error": (
            "market_depth_fetch is disabled. "
            f"Set {_MARKET_DEPTH_ENABLE_ENV}=1 to enable it."
        ),
        "recommended_alternative": "market_ticker",
    }


def _unregister_market_depth_fetch_tool() -> None:
    try:
        from . import _mcp_tools

        registry = getattr(_mcp_tools, "_TOOL_REGISTRY", None)
        if isinstance(registry, dict):
            registry.pop("market_depth_fetch", None)
        object_registry = getattr(_mcp_tools, "_TOOL_OBJECT_REGISTRY", None)
        if isinstance(object_registry, dict):
            object_registry.pop("market_depth_fetch", None)
    except Exception:
        pass


def _market_depth_fetch_impl(symbol: str, spread: bool = False, compact: bool = False) -> Dict[str, Any]:  # noqa: C901
    """Return DOM if available; otherwise current bid/ask snapshot for `symbol`.

    Parameters: symbol
    """
    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            mt5_gateway = get_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            started = time.perf_counter()
            if not mt5_gateway.symbol_select(symbol, True):
                return {"error": _describe_symbol_select_error(symbol, mt5_gateway.last_error())}

            symbol_info = mt5_gateway.symbol_info(symbol)
            if symbol_info is None:
                return {"error": f"Symbol {symbol} not found"}

            digits = max(0, int(getattr(symbol_info, "digits", 0) or 0))
            point = float(getattr(symbol_info, "point", 0.0) or 0.0)
            tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0) or 0.0)
            tick_value = float(getattr(symbol_info, "trade_tick_value", 0.0) or 0.0)

            def _compute_spread_metrics(bid: Any, ask: Any) -> Dict[str, Any] | None:
                try:
                    bid_f = float(bid)
                    ask_f = float(ask)
                except Exception:
                    return None
                if not (math.isfinite(bid_f) and math.isfinite(ask_f)):
                    return None
                spread_abs = float(ask_f - bid_f)
                if spread_abs < 0:
                    return None
                mid = (ask_f + bid_f) / 2.0
                spread_points = (spread_abs / point) if point > 0 else None
                spread_pct = ((spread_abs / mid) * 100.0) if mid > 0 else None
                spread_usd = None
                if tick_size > 0 and tick_value > 0:
                    spread_usd = (spread_abs / tick_size) * tick_value
                return {
                    "spread": spread_abs,
                    "spread_points": spread_points,
                    "spread_pct": spread_pct,
                    "spread_usd": spread_usd,
                    "pricing_basis": "per_1_lot_estimate" if spread_usd is not None else "quote_only",
                }

            def _price_display(value: Any) -> Any:
                if value is None:
                    return None
                try:
                    val = float(value)
                    if digits > 0:
                        return f"{val:.{digits}f}"
                    return str(val)
                except Exception:
                    return None

            book_subscription_active = False
            try:
                book_subscription_active = bool(mt5_gateway.market_book_add(symbol))
            except Exception:
                book_subscription_active = False
            try:
                depth = mt5_gateway.market_book_get(symbol)
            finally:
                if book_subscription_active:
                    try:
                        mt5_gateway.market_book_release(symbol)
                    except Exception:
                        pass

            if depth is not None and len(depth) > 0:
                buy_orders = []
                sell_orders = []

                for level in depth:
                    order_data = {
                        "price": float(level["price"]),
                        "price_display": _price_display(level["price"]),
                        "volume": float(level["volume"]),
                        "volume_real": float(level["volume_real"]),
                    }

                    if int(level["type"]) == 0:
                        buy_orders.append(order_data)
                    else:
                        sell_orders.append(order_data)

                out = {
                    "success": True,
                    "symbol": symbol,
                    "type": "full_depth",
                    "price_precision": digits,
                    "capabilities": {
                        "dom_available": True,
                        "depth_source": "market_book_get",
                        "spread_overlay_requested": bool(spread),
                    },
                    "data": {
                        "buy_orders": buy_orders,
                        "sell_orders": sell_orders,
                        "depth_levels": {
                            "buy": int(len(buy_orders)),
                            "sell": int(len(sell_orders)),
                            "total": int(len(buy_orders) + len(sell_orders)),
                        },
                    },
                }
                if spread and buy_orders and sell_orders:
                    best_bid = max(float(row.get("price")) for row in buy_orders if row.get("price") is not None)
                    best_ask = min(float(row.get("price")) for row in sell_orders if row.get("price") is not None)
                    spread_metrics = _compute_spread_metrics(best_bid, best_ask)
                    if spread_metrics is not None:
                        out["data"]["best_bid"] = best_bid
                        out["data"]["best_ask"] = best_ask
                        out["data"].update(spread_metrics)
                        out["capabilities"]["spread_overlay_applied"] = True
                out["query_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
                return out

            tick = mt5_gateway.symbol_info_tick(symbol)
            if tick is None:
                return {"error": f"Failed to get tick data for {symbol}"}
            if compact:
                return {
                    "error": f"DOM not available for {symbol}. Use market_ticker for bid/ask snapshot instead.",
                    "recommended_alternative": "market_ticker",
                }

            out = {
                "success": True,
                "symbol": symbol,
                "type": "tick_data",
                "price_precision": digits,
                "capabilities": {
                    "dom_available": False,
                    "depth_source": "symbol_info_tick",
                    "spread_overlay_requested": bool(spread),
                    "fallback_reason": "market_book_get returned no levels",
                },
                "data": {
                    "bid": float(tick.bid) if tick.bid else None,
                    "ask": float(tick.ask) if tick.ask else None,
                    "last": float(tick.last) if tick.last else None,
                    "volume": int(tick.volume) if tick.volume else None,
                    "time": int(_mt5_epoch_to_utc(float(tick.time))) if tick.time else None,
                    "note": "Full market depth not available, showing current bid/ask snapshot.",
                    "recommended_alternative": "market_ticker",
                },
            }
            if spread:
                spread_metrics = _compute_spread_metrics(
                    out["data"].get("bid"),
                    out["data"].get("ask"),
                )
                if spread_metrics is not None:
                    out["data"].update(spread_metrics)
                    out["capabilities"]["spread_overlay_applied"] = True
            _use_ctz = _use_client_tz()
            if tick.time and _use_ctz:
                out["data"]["time_display"] = _format_time_minimal_local(_mt5_epoch_to_utc(float(tick.time)))
            elif tick.time:
                out["data"]["time_display"] = _format_time_minimal(_mt5_epoch_to_utc(float(tick.time)))
            if not _use_ctz:
                out["timezone"] = "UTC"
            out["query_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            return out
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error getting market depth: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="market_depth_fetch",
        symbol=symbol,
        spread=spread,
        compact=compact,
        func=_run,
    )


def _market_depth_fetch_disabled(symbol: str, spread: bool = False, compact: bool = False) -> Dict[str, Any]:
    """Return DOM if available; otherwise current bid/ask snapshot for `symbol`.

    Parameters: symbol
    """
    if _market_depth_fetch_enabled():
        return _market_depth_fetch_impl(symbol, spread=spread, compact=compact)
    return run_logged_operation(
        logger,
        operation="market_depth_fetch",
        symbol=symbol,
        spread=spread,
        compact=compact,
        func=_market_depth_disabled_payload,
    )


if _market_depth_fetch_enabled():
    market_depth_fetch = mcp.tool()(_market_depth_fetch_impl)
else:
    _unregister_market_depth_fetch_tool()
    market_depth_fetch = _market_depth_fetch_disabled


@mcp.tool()
def market_ticker(symbol: str) -> Dict[str, Any]:
    """Return a lightweight ticker snapshot with bid/ask/spread for `symbol`.

    Parameters: symbol
    """
    def _run() -> Dict[str, Any]:
        try:
            mt5_gateway = get_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            started = time.perf_counter()
            if not mt5_gateway.symbol_select(symbol, True):
                return {"error": _describe_symbol_select_error(symbol, mt5_gateway.last_error())}

            symbol_info = mt5_gateway.symbol_info(symbol)
            if symbol_info is None:
                return {"error": f"Symbol {symbol} not found"}

            tick = mt5_gateway.symbol_info_tick(symbol)
            if tick is None:
                return {"error": f"Failed to get tick data for {symbol}"}

            digits = max(0, int(getattr(symbol_info, "digits", 0) or 0))
            point = float(getattr(symbol_info, "point", 0.0) or 0.0)
            tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0) or 0.0)
            tick_value = float(getattr(symbol_info, "trade_tick_value", 0.0) or 0.0)

            bid = float(tick.bid) if tick.bid else None
            ask = float(tick.ask) if tick.ask else None
            last = float(tick.last) if tick.last else None

            spread_abs = None
            spread_points = None
            spread_pct = None
            spread_usd = None
            if bid is not None and ask is not None and ask >= bid:
                spread_abs = float(ask - bid)
                mid = (ask + bid) / 2.0
                spread_points = (spread_abs / point) if point > 0 else None
                spread_pct = ((spread_abs / mid) * 100.0) if mid > 0 else None
                if tick_size > 0 and tick_value > 0:
                    spread_usd = (spread_abs / tick_size) * tick_value

            tick_time = int(_mt5_epoch_to_utc(float(tick.time))) if tick.time else None
            _use_ctz = _use_client_tz()

            out: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "type": "ticker",
                "price_precision": digits,
                "bid": bid,
                "ask": ask,
                "last": last,
                "spread": spread_abs,
                "spread_points": spread_points,
                "spread_pct": spread_pct,
                "spread_usd": spread_usd,
                "time": tick_time,
            }
            if tick_time is not None:
                if _use_ctz:
                    out["time_display"] = _format_time_minimal_local(float(tick_time))
                else:
                    out["time_display"] = _format_time_minimal(float(tick_time))
            age_seconds = None
            if tick_time is not None:
                try:
                    age_seconds = max(0.0, float(time.time()) - float(tick_time))
                except Exception:
                    age_seconds = None
            out["diagnostics"] = {
                "source": "mt5.symbol_info_tick",
                "cache_used": False,
                "data_freshness_seconds": age_seconds,
                "query_latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
            if not _use_ctz:
                out["timezone"] = "UTC"
            return out
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error getting ticker snapshot: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="market_ticker",
        symbol=symbol,
        func=_run,
    )
