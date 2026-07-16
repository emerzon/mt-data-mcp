
import logging
import math
import os
import time
from typing import Any, Dict, Literal, Optional

from ..shared.market_units import forex_points_per_pip
from ..shared.schema import DetailLiteral
from ..utils.coercion import round_finite
from ..utils.freshness import (
    QUOTE_STALE_SECONDS,
    format_age_seconds,
)
from ..utils.market_metadata import (
    FRESHNESS_ANCHOR_WALL_CLOCK,
    FRESHNESS_METRIC_LAST_TICK_AGE,
    build_tick_freshness_context,
)
from ..utils.mt5 import (
    MT5ConnectionError,
    describe_mt5_time_normalization,
    ensure_mt5_connection_or_raise,
    mt5,
    resolve_broker_symbol_name,
    symbol_price_currency,
    symbol_price_digits,
    symbol_price_point,
)
from ..utils.symbol import match_symbol_infos
from ..utils.time import (
    _format_time_second_explicit,
    _format_time_second_explicit_local,
    _resolve_client_tz,
    _use_client_tz,
)
from ._mcp_instance import mcp
from ._mcp_tools import unregister_tool
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway
from .output_contract import ensure_common_meta, normalize_output_verbosity_detail
from .runtime_metadata import display_timezone_label

logger = logging.getLogger(__name__)
_MARKET_DEPTH_ENABLE_ENV = "MTDATA_ENABLE_MARKET_DEPTH_FETCH"
_MARKET_TICKER_STALE_SECONDS = QUOTE_STALE_SECONDS
_MARKET_DEPTH_INITIAL_SNAPSHOT_ATTEMPTS = 20
_MARKET_DEPTH_INITIAL_SNAPSHOT_INTERVAL_SECONDS = 0.01
_MARKET_DEPTH_BOOK_UNITS = {
    "volume": "book_volume",
    "volume_real": "book_volume_real",
}
_MARKET_DEPTH_TICK_UNITS = {"volume": "mt5_tick_volume"}
def _round_market_ticker_value(value: Any, *, digits: int) -> Any:
    return round_finite(value, digits, on_invalid="passthrough")


def _market_ticker_age_seconds(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return round(max(0.0, numeric), 1)


def _market_ticker_age_display(seconds: Any) -> Optional[str]:
    return format_age_seconds(seconds)


def _market_ticker_stale_warning(payload: Dict[str, Any], tick_time: Any) -> str:
    if payload.get("timestamp_in_future"):
        return str(payload.get("timestamp_warning"))
    return (
        "Tick data may be stale; last tick time is "
        f"{payload.get('time_display') or tick_time}."
    )


def _market_ticker_points_per_pip(
    symbol_info: Any,
    *,
    symbol: str,
    point: float,
    digits: int,
) -> Optional[float]:
    return forex_points_per_pip(
        symbol,
        path=str(getattr(symbol_info, "path", "") or ""),
        point=point,
        digits=digits,
    )


def _market_depth_fetch_enabled() -> bool:
    raw = os.getenv(_MARKET_DEPTH_ENABLE_ENV)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _market_depth_level_field(level: Any, *names: str) -> Any:
    for name in names:
        if isinstance(level, dict) and name in level:
            return level.get(name)
        try:
            return level[name]  # type: ignore[index]
        except Exception:
            pass
        try:
            return getattr(level, name)
        except Exception:
            pass
    return None


def _compact_market_ticker_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in (
        "success",
        "symbol",
        "type",
        "price_precision",
        "price_currency",
        "bid",
        "ask",
        "mid",
        "market_status",
        "freshness",
        "freshness_state",
        "data_age_seconds",
        "usable_for_live_trading",
        "live_max_age_seconds",
        "stale_after_seconds",
        "market_status_reason",
        "time",
        "time_epoch",
        "timezone",
    ):
        if key == "freshness":
            # The full quote builder classifies live/recent/delayed/stale using
            # execution thresholds.  Keep that label rather than reducing it to
            # a binary fresh/stale view in compact output.
            value = payload.get(key)
        elif key == "time":
            value = payload.get("time_display") or payload.get("time")
        else:
            value = payload.get(key)
        if value is not None:
            out[key] = value
    market_state = out.pop("market_status", None)
    if market_state is not None:
        out["market_state"] = market_state
    for key in ("spread", "spread_points", "spread_pips", "spread_pct"):
        if payload.get(key) is not None:
            out[key] = payload[key]
    return out


def _positive_market_ticker_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


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
    if lowered in {"success", "1: success"}:
        return f"Symbol '{symbol}' was not found or is not available in MT5."
    if any(marker in lowered for marker in ("call failed", "unknown symbol", "not found", "invalid symbol")):
        return f"Symbol '{symbol}' was not found or is not available in MT5."
    if text:
        return f"Failed to select symbol {symbol}: {text}"
    return f"Failed to select symbol {symbol}."


def _market_ticker_error(
    message: Any,
    *,
    code: str,
    remediation: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = build_error_payload(
        message,
        code=code,
        operation="market_ticker",
        details=details,
    )
    if remediation:
        payload["remediation"] = remediation
    return payload


def _market_ticker_symbol_suggestions(
    mt5_gateway: Any,
    query: str,
    *,
    limit: int = 5,
) -> list[Dict[str, str]]:
    text = str(query or "").strip()
    if not text:
        return []
    try:
        symbols = list(mt5_gateway.symbols_get() or [])
    except Exception:
        return []
    matches = match_symbol_infos(symbols, text, limit=limit)
    suggestions: list[Dict[str, str]] = []
    for info in matches:
        suggestion = {"symbol": str(getattr(info, "name", "") or "")}
        description = str(getattr(info, "description", "") or "").strip()
        path = str(getattr(info, "path", "") or "").strip()
        if description:
            suggestion["description"] = description
        if path:
            suggestion["group"] = path
        suggestions.append(suggestion)
    return suggestions


def _market_depth_disabled_payload() -> Dict[str, Any]:
    return {
        "success": False,
        "error": (
            "market_depth_fetch is disabled. "
            f"Set {_MARKET_DEPTH_ENABLE_ENV}=1 to enable it."
        ),
        "error_code": "feature_disabled",
        "feature": "market_depth_fetch",
        "env_var": _MARKET_DEPTH_ENABLE_ENV,
        "enable_instructions": f"Set {_MARKET_DEPTH_ENABLE_ENV}=1 to enable market_depth_fetch.",
        "why_disabled": "Market depth requires broker DOM support and is disabled by default.",
        "recommended_alternative": "market_ticker",
    }


def _market_depth_fetch_impl(symbol: str, spread: bool = False, require_dom: bool = False) -> Dict[str, Any]:  # noqa: C901
    """Return DOM if available; otherwise current bid/ask snapshot for `symbol`.

    Parameters: symbol
    """
    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            mt5_gateway = create_mt5_gateway(
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

            digits = symbol_price_digits(symbol_info)
            point = symbol_price_point(symbol_info) or 0.0
            tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0) or 0.0)
            tick_value = float(getattr(symbol_info, "trade_tick_value", 0.0) or 0.0)
            price_currency = symbol_price_currency(symbol_info)

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
                spread_cost_per_lot = None
                if tick_size > 0 and tick_value > 0:
                    spread_cost_per_lot = (spread_abs / tick_size) * tick_value
                return {
                    "spread": spread_abs,
                    "spread_points": spread_points,
                    "spread_pct": spread_pct,
                    "spread_cost_per_lot": spread_cost_per_lot,
                    "pricing_basis": "per_1_lot_estimate" if spread_cost_per_lot is not None else "quote_only",
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
                depth = None
                attempts = (
                    _MARKET_DEPTH_INITIAL_SNAPSHOT_ATTEMPTS
                    if book_subscription_active
                    else 1
                )
                for attempt in range(attempts):
                    depth = mt5_gateway.market_book_get(symbol)
                    if depth is not None and len(depth) > 0:
                        break
                    if attempt + 1 < attempts:
                        time.sleep(_MARKET_DEPTH_INITIAL_SNAPSHOT_INTERVAL_SECONDS)
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
                    try:
                        price = float(_market_depth_level_field(level, "price"))
                        volume = float(_market_depth_level_field(level, "volume"))
                        volume_real = float(
                            _market_depth_level_field(
                                level,
                                "volume_real",
                                "volume_dbl",
                            )
                        )
                        level_type = int(_market_depth_level_field(level, "type"))
                    except (KeyError, TypeError, ValueError):
                        continue
                    order_data = {
                        "price": price,
                        "price_display": _price_display(price),
                        "volume": volume,
                        "volume_real": volume_real,
                    }

                    if level_type == 0:
                        buy_orders.append(order_data)
                    else:
                        sell_orders.append(order_data)

                out = {
                    "success": True,
                    "symbol": symbol,
                    "type": "full_depth",
                    "price_precision": digits,
                    "price_currency": price_currency,
                    "capabilities": {
                        "dom_available": True,
                        "depth_status": "available",
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
                    "units": dict(_MARKET_DEPTH_BOOK_UNITS),
                }
                if spread and buy_orders and sell_orders:
                    valid_buy_prices = [
                        float(row.get("price"))
                        for row in buy_orders
                        if row.get("price") is not None
                    ]
                    valid_sell_prices = [
                        float(row.get("price"))
                        for row in sell_orders
                        if row.get("price") is not None
                    ]
                    if valid_buy_prices and valid_sell_prices:
                        best_bid = max(valid_buy_prices)
                        best_ask = min(valid_sell_prices)
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
            if require_dom:
                return {
                    "error": f"DOM not available for {symbol}. Use market_ticker for bid/ask snapshot instead.",
                    "recommended_alternative": "market_ticker",
                }

            out = {
                "success": True,
                "symbol": symbol,
                "type": "quote_fallback",
                "depth_status": "unavailable",
                "price_precision": digits,
                "price_currency": price_currency,
                "recommended_alternative": "market_ticker",
                "capabilities": {
                    "dom_available": False,
                    "depth_status": "unavailable",
                    "depth_source": "symbol_info_tick",
                    "spread_overlay_requested": bool(spread),
                    "fallback_reason": "market_book_get returned no levels",
                },
                "data": {
                    "bid": float(tick.bid) if tick.bid else None,
                    "ask": float(tick.ask) if tick.ask else None,
                    "last": float(tick.last) if tick.last else None,
                    "volume": int(tick.volume) if tick.volume else None,
                    "note": "Full market depth not available, showing current bid/ask snapshot.",
                },
                "units": dict(_MARKET_DEPTH_TICK_UNITS),
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
                out["data"]["time"] = _format_time_second_explicit_local(float(tick.time))
                out["data"]["time_epoch"] = int(float(tick.time))
            elif tick.time:
                out["data"]["time"] = _format_time_second_explicit(float(tick.time))
                out["data"]["time_epoch"] = int(float(tick.time))
            out["timezone"] = display_timezone_label(
                use_client_tz=_use_ctz,
                resolve_client_tz=_resolve_client_tz,
            )
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
        require_dom=require_dom,
        func=_run,
    )


def market_depth_fetch(symbol: str, spread: bool = False, require_dom: bool = False) -> Dict[str, Any]:
    """Return DOM when `MTDATA_ENABLE_MARKET_DEPTH_FETCH=1`; otherwise disabled.

    When enabled, returns broker DOM if available and otherwise falls back to
    the current bid/ask snapshot unless `require_dom=True`.

    Parameters: symbol
    """
    if not _market_depth_fetch_enabled():
        return run_logged_operation(
            logger,
            operation="market_depth_fetch",
            symbol=symbol,
            spread=spread,
            require_dom=require_dom,
            func=_market_depth_disabled_payload,
        )
    return _market_depth_fetch_impl(symbol, spread=spread, require_dom=require_dom)


if _market_depth_fetch_enabled():
    market_depth_fetch = mcp.tool()(market_depth_fetch)
else:
    unregister_tool("market_depth_fetch", mcp_obj=mcp)


@mcp.tool()
def market_ticker(  # noqa: C901
    symbol: str,
    detail: DetailLiteral = "compact",
    price_field: Optional[Literal["bid", "ask", "mid", "last", "spread"]] = None,
) -> Dict[str, Any]:
    """Return a lightweight quote snapshot with bid/ask/spread for `symbol`.

    Parameters: symbol
    Use `detail="compact"` to keep only the most operational bid/ask/spread fields.
    Set `price_field` to bid, ask, mid, last, or spread for a simple price result.
    """
    detail_mode = normalize_output_verbosity_detail(detail, default="compact")

    def _run() -> Dict[str, Any]:  # noqa: C901
        def _finalize(payload: Dict[str, Any]) -> Dict[str, Any]:
            return ensure_common_meta(payload, tool_name="market_ticker")

        try:
            mt5_gateway = create_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            resolved_symbol = resolve_broker_symbol_name(symbol)
            started = time.perf_counter()
            if not mt5_gateway.symbol_select(resolved_symbol, True):
                suggestions = _market_ticker_symbol_suggestions(mt5_gateway, symbol)
                return _finalize(
                    _market_ticker_error(
                        _describe_symbol_select_error(resolved_symbol, mt5_gateway.last_error()),
                        code="symbol_not_found",
                        remediation=(
                            f"Verify the broker symbol name with symbols_list(search_term='{symbol}') "
                            "or symbols_top_markets, then retry with an available MT5 symbol."
                        ),
                        details={"symbol": symbol, "did_you_mean": suggestions},
                    )
                )

            symbol_info = mt5_gateway.symbol_info(resolved_symbol)
            if symbol_info is None:
                suggestions = _market_ticker_symbol_suggestions(mt5_gateway, symbol)
                return _finalize(
                    _market_ticker_error(
                        f"Symbol {resolved_symbol} not found",
                        code="symbol_not_found",
                        remediation=(
                            f"Verify the broker symbol name with symbols_list(search_term='{symbol}') "
                            "or symbols_describe."
                        ),
                        details={"symbol": symbol, "did_you_mean": suggestions},
                    )
                )

            tick = mt5_gateway.symbol_info_tick(resolved_symbol)
            if tick is None:
                return _finalize(
                    _market_ticker_error(
                        f"Failed to get tick data for {resolved_symbol}",
                        code="market_ticker_tick_unavailable",
                        remediation=(
                            "Ensure the symbol is visible in Market Watch and that the market is open "
                            "or recent ticks are available from the broker."
                        ),
                    )
                )

            digits = symbol_price_digits(symbol_info)
            point = symbol_price_point(symbol_info) or 0.0
            tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0) or 0.0)
            tick_value = float(getattr(symbol_info, "trade_tick_value", 0.0) or 0.0)
            spread_cost_currency = symbol_price_currency(symbol_info)
            price_currency = spread_cost_currency
            contract_size = _positive_market_ticker_float(
                getattr(symbol_info, "trade_contract_size", None)
            )

            bid = float(tick.bid) if tick.bid else None
            ask = float(tick.ask) if tick.ask else None
            last = float(tick.last) if tick.last else None
            tick_time = int(float(tick.time)) if tick.time else None
            if bid is None and ask is None and last is None and tick_time is None:
                return _finalize(
                    _market_ticker_error(
                        (
                            f"No usable quote data for {resolved_symbol}. "
                            f"Use symbols_list(search_term='{symbol}') to find broker-specific names and suffixes."
                        ),
                        code="market_ticker_quote_unavailable",
                        remediation=(
                            "Verify the broker symbol name and ensure the symbol has a recent tick "
                            "in Market Watch."
                        ),
                    )
                )
            tick_volume = getattr(tick, "volume", None)
            if tick_volume is not None:
                try:
                    tick_volume = int(tick_volume)
                except (TypeError, ValueError, OverflowError):
                    tick_volume = None

            spread_abs = None
            spread_points = None
            spread_pct = None
            spread_pips = None
            mid = None
            spread_cost_per_lot = None
            pricing_basis = "quote_only"
            if bid is not None and ask is not None and ask >= bid:
                spread_abs = float(ask - bid)
                mid = (ask + bid) / 2.0
                spread_points = (spread_abs / point) if point > 0 else None
                points_per_pip = _market_ticker_points_per_pip(
                    symbol_info,
                    symbol=resolved_symbol,
                    point=point,
                    digits=digits,
                )
                spread_pips = (
                    spread_points / points_per_pip
                    if spread_points is not None
                    and points_per_pip is not None
                    and points_per_pip > 0
                    else None
                )
                spread_pct = ((spread_abs / mid) * 100.0) if mid > 0 else None
                spread_abs = _round_market_ticker_value(spread_abs, digits=digits)
                spread_points = _round_market_ticker_value(spread_points, digits=4)
                spread_pips = _round_market_ticker_value(spread_pips, digits=4)
                spread_pct = _round_market_ticker_value(spread_pct, digits=6)
                if tick_size > 0 and tick_value > 0:
                    spread_cost_per_lot = _round_market_ticker_value(
                        (float(spread_abs) / tick_size) * tick_value,
                        digits=6,
                    )
                    pricing_basis = "per_1_lot_estimate"

            _use_ctz = _use_client_tz()

            out: Dict[str, Any] = {
                "success": True,
                "symbol": resolved_symbol,
                "type": "quote",
                "price_precision": digits,
                "price_currency": price_currency,
                "bid": _round_market_ticker_value(bid, digits=digits),
                "ask": _round_market_ticker_value(ask, digits=digits),
                "mid": _round_market_ticker_value(mid, digits=digits),
                "last": _round_market_ticker_value(last, digits=digits),
                "spread": spread_abs,
                "spread_points": spread_points,
                "spread_pips": spread_pips,
                "spread_pct": spread_pct,
                "spread_cost_per_lot": spread_cost_per_lot,
                "pricing_basis": pricing_basis,
                "units": {
                    "bid": "price",
                    "ask": "price",
                    "mid": "price",
                    "last": "price",
                    "spread": "price",
                    "spread_points": "broker_points",
                    "spread_pct": "percentage_points (1.0 = 1%)",
                    "spread_cost_per_lot": "currency_per_lot_estimate",
                },
            }
            time_normalization = describe_mt5_time_normalization(
                symbol=resolved_symbol
            )
            out.update(time_normalization)
            if contract_size is not None:
                out["contract_size"] = _round_market_ticker_value(contract_size, digits=6)
                out["lot_definition"] = "1 broker lot equals contract_size contract units."
                out["units"]["contract_size"] = "contract_units_per_lot"
                out["units"]["lot"] = "broker_lot"
                if pricing_basis == "per_1_lot_estimate":
                    out["pricing_basis_units"] = "broker_lot"
            if spread_pips is not None:
                out["units"]["spread_pips"] = "pips"
            if tick_volume not in (None, 0):
                out["tick_volume"] = tick_volume
            if spread_cost_per_lot is not None and spread_cost_currency:
                out["spread_cost_currency"] = spread_cost_currency
            if tick_time is not None:
                out["time_epoch"] = float(tick_time)
                if _use_ctz:
                    out["time"] = _format_time_second_explicit_local(float(tick_time))
                else:
                    out["time"] = _format_time_second_explicit(float(tick_time))
            age_seconds = None
            now_epoch = None
            if tick_time is not None:
                try:
                    now_epoch = float(time.time())
                    age_seconds = max(0.0, now_epoch - float(tick_time))
                except Exception:
                    age_seconds = None
            if age_seconds is not None:
                freshness_context = build_tick_freshness_context(
                    resolved_symbol,
                    tick_epoch=tick_time,
                    now_epoch=now_epoch,
                    item="tick",
                    stale_after_seconds=_MARKET_TICKER_STALE_SECONDS,
                    age_rounder=_market_ticker_age_seconds,
                )
                rounded_age_seconds = freshness_context.get("data_age_seconds")
                out.update(freshness_context)
                age_display = _market_ticker_age_display(rounded_age_seconds)
                if age_display is not None:
                    out["data_age"] = age_display
                if out["data_stale"]:
                    out["warning"] = _market_ticker_stale_warning(out, tick_time)
            diagnostics = {
                "source": "mt5.symbol_info_tick",
                "cache_used": False,
                "data_freshness_seconds": _market_ticker_age_seconds(age_seconds),
                "data_freshness_anchor": FRESHNESS_ANCHOR_WALL_CLOCK,
                "data_freshness_metric": FRESHNESS_METRIC_LAST_TICK_AGE,
                "timestamp_mode": time_normalization.get("timestamp_mode"),
                "time_normalization": time_normalization.get("time_normalization"),
                "query_latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
            meta = out.get("meta")
            if not isinstance(meta, dict):
                meta = {}
            meta["diagnostics"] = dict(diagnostics)
            out["meta"] = meta
            out["timezone"] = display_timezone_label(
                use_client_tz=_use_ctz,
                resolve_client_tz=_resolve_client_tz,
            )
            if price_field is not None:
                field_value = str(price_field or "").strip().lower()
                price_values = {
                    "bid": out.get("bid"),
                    "ask": out.get("ask"),
                    "mid": _round_market_ticker_value((bid + ask) / 2.0, digits=digits)
                    if bid is not None and ask is not None
                    else None,
                    "last": out.get("last"),
                    "spread": out.get("spread"),
                }
                if field_value not in price_values:
                    return _finalize(
                        _market_ticker_error(
                            "price_field must be one of: bid, ask, mid, last, spread.",
                            code="market_ticker_invalid_price_field",
                            remediation="Choose one of bid, ask, mid, last, or spread.",
                        )
                    )
                price = price_values.get(field_value)
                if price is None:
                    return _finalize(
                        _market_ticker_error(
                            f"{field_value} price is unavailable for {resolved_symbol}.",
                            code="market_ticker_price_unavailable",
                            remediation=(
                                "Use bid, ask, mid, or spread when the broker does not publish "
                                "the requested price field for this symbol."
                            ),
                        )
                    )
                simple: Dict[str, Any] = {
                    "success": True,
                    "symbol": resolved_symbol,
                    "type": "price",
                    "field": field_value,
                    "price": price,
                    "price_precision": digits,
                    "price_currency": price_currency,
                }
                for key in (
                    "time",
                    "time_epoch",
                    "timezone",
                    "data_age_seconds",
                    "data_age_anchor",
                    "data_age_metric",
                    "data_age",
                    "stale_after_seconds",
                    "data_stale",
                    "freshness_basis",
                    "market_status",
                    "market_status_reason",
                    "market_status_source",
                    "freshness_policy_relaxed",
                    "note",
                    "warning",
                ):
                    if out.get(key) is not None:
                        simple[key] = out.get(key)
                return _finalize(simple)
            if detail_mode == "compact":
                out = _compact_market_ticker_payload(out)
            return _finalize(out)
        except MT5ConnectionError as exc:
            return _finalize(
                _market_ticker_error(
                    str(exc),
                    code="market_ticker_mt5_connection",
                    remediation="Ensure MetaTrader 5 is running, logged in, and reachable.",
                )
            )
        except Exception as exc:
            return _finalize(
                _market_ticker_error(
                    f"Error getting quote snapshot: {str(exc)}",
                    code="market_ticker_error",
                    remediation="Retry after checking the MT5 terminal and symbol availability.",
                )
            )

    return run_logged_operation(
        logger,
        operation="market_ticker",
        symbol=symbol,
        detail=detail_mode,
        price_field=price_field,
        func=_run,
    )
