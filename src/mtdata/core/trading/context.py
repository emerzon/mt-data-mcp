"""Trading session context utilities."""

import logging
import math
from typing import Any, Dict, Optional

from ...shared.constants import BROKER_VOLUME_UNIT
from .._mcp_instance import mcp
from ..execution_logging import run_logged_operation
from ..market_depth import market_ticker
from ..market_status import _check_symbol_market_status
from ..output_contract import ensure_common_meta
from .account import trade_account_info
from .positions import trade_get_open, trade_get_pending
from .requests import (
    TradeGetOpenRequest,
    TradeGetPendingRequest,
    TradeSessionContextRequest,
)

logger = logging.getLogger(__name__)


def _sanitize_trade_session_section_error(
    section: Any,
    *,
    label: str,
    include_count: bool = False,
) -> tuple[Any, bool]:
    if not isinstance(section, dict):
        return section, False
    if section.get("error") in (None, ""):
        return section, False

    sanitized: Dict[str, Any] = {"error": f"Unable to fetch {label}."}
    if include_count:
        sanitized["count"] = 0
    return sanitized, True


def _strip_nested_envelope(section: Any) -> Any:
    """Remove redundant envelope fields (success, meta) from nested sections."""
    if not isinstance(section, dict):
        return section
    # Keep all data fields, remove redundant envelope fields
    return {k: v for k, v in section.items() if k not in ("success", "meta")}


def _trade_session_section_count(section: Any) -> Optional[int]:
    if isinstance(section, dict):
        count = section.get("count")
        if count not in (None, ""):
            try:
                return max(0, int(count))
            except Exception:
                return None
        items = section.get("items")
        if isinstance(items, list):
            return len(items)
    if isinstance(section, list):
        return len(section)
    return None


_TRADE_SESSION_PRICE_KEYS = {
    "price",
    "price_open",
    "price_current",
    "price_stoplimit",
    "trigger_price",
    "entry_price",
    "open_price",
    "current_price",
    "sl",
    "tp",
    "Price",
    "Open Price",
    "Current Price",
    "Stoplimit Price",
    "SL",
    "TP",
}


def _price_precision_from_quote(quote: Any) -> int:
    if isinstance(quote, dict):
        for key in ("price_precision", "digits"):
            try:
                return max(0, int(quote.get(key)))
            except Exception:
                continue
    return 6


def _round_trade_session_price(value: Any, *, digits: int) -> Any:
    try:
        numeric = float(value)
    except Exception:
        return value
    if not math.isfinite(numeric):
        return value
    return float(round(numeric, max(0, int(digits))))


def _round_trade_session_prices(value: Any, *, digits: int, key: Optional[str] = None) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _round_trade_session_prices(
                item_value,
                digits=digits,
                key=str(item_key),
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [
            _round_trade_session_prices(item, digits=digits, key=key)
            for item in value
        ]
    if key in _TRADE_SESSION_PRICE_KEYS:
        return _round_trade_session_price(value, digits=digits)
    return value


def _normalize_nested_quote_time(quote: Dict[str, Any], *, compact: bool) -> Dict[str, Any]:
    normalized = dict(quote)
    raw_time = normalized.get("time_epoch")
    if raw_time in (None, "") and isinstance(normalized.get("time"), (int, float)):
        raw_time = normalized.get("time")
    display_time = normalized.get("time_display")
    if display_time in (None, "") and isinstance(normalized.get("time"), str):
        display_time = normalized.get("time")

    normalized.pop("time_display", None)
    if display_time not in (None, ""):
        normalized["time"] = display_time
    if compact:
        normalized.pop("time_epoch", None)
    elif raw_time not in (None, ""):
        normalized["time_epoch"] = raw_time
    return normalized


def _build_trade_ready(
    account: Any,
    quote: Any,
    tradability: Any = None,
) -> Dict[str, Any]:
    blockers: list[str] = []
    margin_free = None
    margin_level = None
    margin_utilization_pct = None
    if not isinstance(account, dict) or account.get("error") not in (None, ""):
        blockers.append("account_unavailable")
    else:
        margin_free = account.get("margin_free")
        margin_level = account.get("margin_level")
        try:
            margin = float(account.get("margin"))
            equity = float(account.get("equity"))
            if math.isfinite(margin) and math.isfinite(equity) and equity > 0:
                margin_utilization_pct = round((margin / equity) * 100.0, 2)
        except Exception:
            margin_utilization_pct = None
        if account.get("execution_ready") is False:
            blockers.append("account_execution_not_ready")
        execution_blockers = account.get("execution_blockers")
        if isinstance(execution_blockers, list):
            blockers.extend(str(item) for item in execution_blockers if item not in (None, ""))
        try:
            if margin_free is not None and float(margin_free) <= 0:
                blockers.append("no_free_margin")
        except Exception:
            pass

    if not isinstance(quote, dict) or quote.get("error") not in (None, ""):
        blockers.append("quote_unavailable")
    elif quote.get("usable_for_live_trading") is False:
        blockers.append("quote_not_live")
    elif bool(quote.get("data_stale")):
        blockers.append("quote_stale")

    can_open_new_positions = None
    if isinstance(tradability, dict) and tradability.get("error") in (None, ""):
        can_open_new_positions = tradability.get("can_open_new_positions")
        if can_open_new_positions is False:
            blockers.append("market_not_open_for_new_positions")

    deduped_blockers = list(dict.fromkeys(blockers))
    margin_sufficient = None
    try:
        if margin_free is not None:
            margin_sufficient = float(margin_free) > 0
    except Exception:
        margin_sufficient = None
    result = {
        "execution_preconditions_met": not deduped_blockers,
        "any_blockers": bool(deduped_blockers),
        "blockers": deduped_blockers,
        "margin_sufficient_for_min_lot": margin_sufficient,
        "readiness_scope": "connectivity_account_quote_and_symbol_not_portfolio_risk_approval",
        "portfolio_risk_assessed": False,
    }
    if margin_level not in (None, ""):
        result["margin_level"] = margin_level
    if margin_utilization_pct is not None:
        result["margin_utilization_pct"] = margin_utilization_pct
    if can_open_new_positions is not None:
        result["can_open_new_positions"] = can_open_new_positions
    return result


def _trade_session_tradability(symbol: str) -> Dict[str, Any]:
    try:
        result = _check_symbol_market_status(
            symbol,
            detail="compact",
            timezone_display="utc",
        )
    except Exception:
        return {}
    if not isinstance(result, dict) or result.get("error"):
        return {}
    return {
        key: result[key]
        for key in (
            "status",
            "reason",
            "is_tradable",
            "can_open_new_positions",
            "trade_mode_allows_opening",
            "tick_freshness",
        )
        if key in result
    }


def _build_quote_quality(quote: Any) -> Dict[str, Any]:
    if not isinstance(quote, dict) or quote.get("error") not in (None, ""):
        return {
            "status": "unavailable",
            "is_live": False,
            "warning": "quote_unavailable",
        }
    age_seconds = quote.get("data_age_seconds")
    stale = bool(quote.get("data_stale"))
    execution_usable = quote.get("usable_for_live_trading") is True
    status = "stale" if stale else "live" if execution_usable else "recent"
    out: Dict[str, Any] = {
        "status": status,
        "is_live": execution_usable,
        "data_stale": stale,
    }
    if age_seconds not in (None, ""):
        out["age_seconds"] = age_seconds
    for key in (
        "freshness",
        "freshness_state",
        "usable_for_live_trading",
        "live_max_age_seconds",
        "market_status",
        "timezone",
        "time",
    ):
        value = quote.get(key)
        if value not in (None, ""):
            out[key] = value
    warning = quote.get("stale_warning") or quote.get("warning")
    if warning not in (None, ""):
        out["warning"] = warning
    return out


def _compact_trade_session_items(
    section: Any,
    *,
    field_map: tuple[tuple[str, ...], ...],
) -> Optional[list[Dict[str, Any]]]:
    if not isinstance(section, dict):
        return None
    items = section.get("items")
    if not isinstance(items, list) or not items:
        return None

    rows: list[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compact: Dict[str, Any] = {}
        for out_key, *input_keys in field_map:
            for input_key in input_keys:
                if input_key in item and item.get(input_key) not in (None, ""):
                    compact[out_key] = item.get(input_key)
                    break
        if compact:
            rows.append(compact)
    return rows or None


def _compact_trade_session_context_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        key: payload.get(key)
        for key in (
            "success",
            "symbol",
            "state",
            "state_scope",
            "portfolio_positions_count",
            "other_positions_count",
            "partial_failure",
            "trade_ready",
            "quote_quality",
            "market_status",
            "market_status_reason",
            "is_tradable",
            "can_open_new_positions",
        )
        if payload.get(key) not in (None, "")
    }

    account = payload.get("account")
    if isinstance(account, dict):
        if account.get("error") not in (None, ""):
            compact["account"] = {"error": account.get("error")}
        else:
            account_summary = {
                key: account.get(key)
                for key in (
                    "equity",
                    "profit",
                    "balance",
                    "margin",
                    "margin_free",
                    "margin_level",
                    "currency",
                    "leverage",
                    "account_type",
                    "is_demo",
                    "is_live",
                    "server",
                )
                if account.get(key) not in (None, "")
            }
            if account_summary:
                compact["account"] = account_summary

    quote = payload.get("quote")
    if isinstance(quote, dict):
        if quote.get("error") not in (None, ""):
            compact["quote"] = {"error": quote.get("error")}
        else:
            quote_summary = {
                key: quote.get(key)
                for key in (
                    "bid",
                    "ask",
                    "mid",
                    "last",
                    "price_currency",
                    "price_precision",
                    "spread",
                    "spread_points",
                    "spread_pips",
                    "spread_pct",
                    "spread_cost_per_lot",
                    "spread_cost_currency",
                    "freshness",
                    "market_status",
                    "time",
                    "time_display",
                    "time_epoch",
                    "timezone",
                    "data_age_seconds",
                    "data_age",
                    "data_stale",
                    "freshness_state",
                    "usable_for_live_trading",
                    "usable_for_live_trading_basis",
                    "live_max_age_seconds",
                    "stale_warning",
                    "warning",
                )
                if quote.get(key) not in (None, "")
            }
            if quote_summary:
                normalized_quote = _normalize_nested_quote_time(
                    quote_summary,
                    compact=True,
                )
                compact["quote"] = normalized_quote

    open_positions = payload.get("open_positions")
    volume_units: Dict[str, str] = {}
    if isinstance(open_positions, dict):
        if open_positions.get("error") not in (None, ""):
            open_error = {"error": open_positions.get("error")}
            if open_positions.get("count") not in (None, ""):
                open_error["count"] = open_positions.get("count")
            compact["open_positions"] = open_error
        else:
            compact_rows = _compact_trade_session_items(
                open_positions,
                field_map=(
                    ("symbol", "symbol", "Symbol"),
                    ("ticket", "ticket", "Ticket"),
                    ("time", "time", "Time"),
                    ("type", "type", "Type"),
                    ("volume", "volume", "Volume"),
                    ("price_open", "price_open", "open_price", "Open Price"),
                    (
                        "price_current",
                        "price_current",
                        "current_price",
                        "Current Price",
                    ),
                    ("price_current_basis", "price_current_basis"),
                    ("sl", "sl", "SL"),
                    ("tp", "tp", "TP"),
                    ("profit", "profit", "Profit"),
                    ("comment", "comment", "Comments"),
                    ("magic", "magic", "Magic"),
                    ("timezone", "timezone", "Timezone"),
                ),
            )
            if compact_rows:
                compact["open_positions"] = compact_rows
            else:
                compact["open_positions"] = []
            compact["open_positions_count"] = int(open_positions.get("count") or 0)
            if compact["open_positions_count"] > 0:
                volume_units["volume"] = BROKER_VOLUME_UNIT

    pending_orders = payload.get("pending_orders")
    if isinstance(pending_orders, dict):
        if pending_orders.get("error") not in (None, ""):
            pending_error = {"error": pending_orders.get("error")}
            if pending_orders.get("count") not in (None, ""):
                pending_error["count"] = pending_orders.get("count")
            compact["pending_orders"] = pending_error
        else:
            compact_rows = _compact_trade_session_items(
                pending_orders,
                field_map=(
                    ("symbol", "symbol", "Symbol"),
                    ("ticket", "ticket", "Ticket"),
                    ("time", "time", "Time"),
                    ("expiration", "expiration", "Expiration"),
                    ("type", "type", "Type"),
                    ("order_type", "order_type", "type", "Type"),
                    ("side", "side", "Side"),
                    ("volume", "volume", "Volume"),
                    ("price_open", "price_open", "open_price", "Open Price"),
                    (
                        "trigger_price",
                        "trigger_price",
                        "price_open",
                        "open_price",
                        "Open Price",
                    ),
                    (
                        "entry_price",
                        "entry_price",
                        "price_open",
                        "open_price",
                        "Open Price",
                    ),
                    (
                        "price_current",
                        "price_current",
                        "current_price",
                        "Current Price",
                    ),
                    ("price_current_basis", "price_current_basis"),
                    ("sl", "sl", "SL"),
                    ("tp", "tp", "TP"),
                    ("comment", "comment", "Comments"),
                    ("magic", "magic", "Magic"),
                    ("timezone", "timezone", "Timezone"),
                ),
            )
            if compact_rows:
                compact["pending_orders"] = compact_rows
            else:
                compact["pending_orders"] = []
            compact["pending_orders_count"] = int(pending_orders.get("count") or 0)
            if compact["pending_orders_count"] > 0:
                volume_units["volume"] = BROKER_VOLUME_UNIT

    if volume_units:
        compact["units"] = volume_units

    return compact


@mcp.tool()
def trade_session_context(request: TradeSessionContextRequest) -> Dict[str, Any]:
    """Get a consolidated session context including account info, open positions, pending orders, quote, and computed state for a symbol.

    Use this for a fast execution snapshot before deciding what to do next. It
    intentionally summarizes account/quote/order state and is not the
    authoritative risk calculator. Use `trade_risk_analyze` for stop-loss
    exposure and position sizing, or `trade_var_cvar_calculate` for portfolio
    VaR/CVaR.

    Parameters: symbol, detail, include_account
    """

    def _run() -> Dict[str, Any]:
        # Un-wrap original functions if necessary to bypass double-logging or async mcp wrappers
        acc_func = getattr(trade_account_info, "__wrapped__", trade_account_info)
        quote_func = getattr(market_ticker, "__wrapped__", market_ticker)
        open_func = getattr(trade_get_open, "__wrapped__", trade_get_open)
        pending_func = getattr(trade_get_pending, "__wrapped__", trade_get_pending)

        account_res = acc_func() if request.include_account else None
        quote_res = quote_func(symbol=request.symbol, detail=request.detail)
        tradability = _trade_session_tradability(request.symbol)

        open_req = TradeGetOpenRequest(symbol=request.symbol)
        open_res = open_func(request=open_req)

        portfolio_open_res = None
        try:
            portfolio_open_res = open_func(request=TradeGetOpenRequest())
        except Exception:
            portfolio_open_res = None

        pending_req = TradeGetPendingRequest(symbol=request.symbol)
        pending_res = pending_func(request=pending_req)

        if request.include_account:
            account_res, account_failed = _sanitize_trade_session_section_error(
                account_res,
                label="account context",
            )
        else:
            account_failed = False
        quote_res, quote_failed = _sanitize_trade_session_section_error(
            quote_res,
            label="quote data",
        )
        open_res, open_failed = _sanitize_trade_session_section_error(
            open_res,
            label="open positions",
            include_count=True,
        )
        pending_res, pending_failed = _sanitize_trade_session_section_error(
            pending_res,
            label="pending orders",
            include_count=True,
        )
        partial_failure = any(
            (account_failed, quote_failed, open_failed, pending_failed)
        )

        # Determine internal book state
        has_open = bool(open_res.get("success", False) and open_res.get("count", 0) > 0)
        has_pending = bool(pending_res.get("success", False) and pending_res.get("count", 0) > 0)

        if has_open and has_pending:
            state = "mixed"
        elif has_open:
            state = "open_position"
        elif has_pending:
            state = "pending_only"
        else:
            state = "flat"

        symbol_positions_count = _trade_session_section_count(open_res)
        portfolio_positions_count = _trade_session_section_count(portfolio_open_res)
        other_positions_count = None
        if (
            portfolio_positions_count is not None
            and symbol_positions_count is not None
            and portfolio_positions_count > symbol_positions_count
        ):
            other_positions_count = portfolio_positions_count - symbol_positions_count

        payload = {
            "success": True,
            "symbol": request.symbol,
            "state": state,
            "state_scope": "symbol",
            "open_positions": open_res,
            "pending_orders": pending_res,
            "quote": quote_res,
            "quote_quality": _build_quote_quality(quote_res),
        }
        if tradability:
            payload["market_status"] = tradability.get("status")
            payload["market_status_reason"] = tradability.get("reason")
            payload["is_tradable"] = tradability.get("is_tradable")
            payload["can_open_new_positions"] = tradability.get(
                "can_open_new_positions"
            )
        if other_positions_count is not None:
            payload["portfolio_positions_count"] = portfolio_positions_count
            payload["other_positions_count"] = other_positions_count
        if request.include_account:
            payload["account"] = account_res
            payload["trade_ready"] = _build_trade_ready(
                account_res,
                quote_res,
                tradability,
            )
        if partial_failure:
            payload["partial_failure"] = True
        if request.detail == "compact":
            payload = _compact_trade_session_context_payload(payload)
        else:
            # For full detail, strip redundant envelope fields from nested sections
            if request.include_account:
                payload["account"] = _strip_nested_envelope(payload["account"])
            payload["open_positions"] = _strip_nested_envelope(payload["open_positions"])
            payload["pending_orders"] = _strip_nested_envelope(payload["pending_orders"])
            payload["quote"] = _strip_nested_envelope(payload["quote"])
            price_digits = _price_precision_from_quote(payload.get("quote"))
            payload["open_positions"] = _round_trade_session_prices(
                payload["open_positions"],
                digits=price_digits,
            )
            payload["pending_orders"] = _round_trade_session_prices(
                payload["pending_orders"],
                digits=price_digits,
            )
            if isinstance(payload["quote"], dict):
                payload["quote"] = _normalize_nested_quote_time(
                    payload["quote"],
                    compact=False,
                )
        return ensure_common_meta(payload, tool_name="trade_session_context")

    return run_logged_operation(
        logger,
        operation="trade_session_context",
        symbol=request.symbol,
        detail=request.detail,
        include_account=request.include_account,
        func=_run,
    )
