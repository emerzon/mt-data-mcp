"""Trading session context utilities."""

import logging
import math
from typing import Any, Dict, Optional

from .._mcp_instance import mcp
from ..execution_logging import run_logged_operation
from ..market_depth import market_ticker
from ..output_contract import ensure_common_meta
from .account import trade_account_info
from .positions import trade_get_open, trade_get_pending
from .requests import TradeGetOpenRequest, TradeGetPendingRequest, TradeSessionContextRequest

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


_TRADE_SESSION_PRICE_KEYS = {
    "price",
    "price_open",
    "price_current",
    "price_stoplimit",
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


def _price_precision_from_ticker(ticker: Any) -> int:
    if isinstance(ticker, dict):
        for key in ("price_precision", "digits"):
            try:
                return max(0, int(ticker.get(key)))
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


def _normalize_nested_ticker_time(ticker: Dict[str, Any], *, compact: bool) -> Dict[str, Any]:
    normalized = dict(ticker)
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


def _compact_trade_session_items(
    section: Any,
    *,
    field_map: tuple[tuple[str, str], ...],
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
        compact = {
            out_key: item.get(in_key)
            for out_key, in_key in field_map
            if in_key in item and item.get(in_key) not in (None, "")
        }
        if compact:
            rows.append(compact)
    return rows or None


def _compact_trade_session_context_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        key: payload.get(key)
        for key in ("success", "symbol", "state", "partial_failure")
        if payload.get(key) not in (None, "")
    }

    account = payload.get("account")
    if isinstance(account, dict):
        if account.get("error") not in (None, ""):
            compact["account"] = {"error": account.get("error")}
        else:
            account_summary = {
                key: account.get(key)
                for key in ("equity", "margin_free")
                if account.get(key) not in (None, "")
            }
            if account.get("execution_ready") is False:
                account_summary["execution_ready"] = False
            if account_summary:
                compact["account"] = account_summary

    ticker = payload.get("ticker")
    if isinstance(ticker, dict):
        if ticker.get("error") not in (None, ""):
            compact["ticker"] = {"error": ticker.get("error")}
        else:
            ticker_summary = {
                key: ticker.get(key)
                for key in (
                    "bid",
                    "ask",
                    "last",
                    "spread",
                    "spread_points",
                    "spread_pips",
                    "spread_pct",
                    "spread_usd",
                    "time",
                    "time_display",
                    "time_epoch",
                    "timezone",
                )
                if ticker.get(key) not in (None, "")
            }
            if ticker_summary:
                compact["ticker"] = _normalize_nested_ticker_time(
                    ticker_summary,
                    compact=True,
                )

    open_positions = payload.get("open_positions")
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
                    ("symbol", "Symbol"),
                    ("ticket", "Ticket"),
                    ("time", "Time"),
                    ("type", "Type"),
                    ("volume", "Volume"),
                    ("open_price", "Open Price"),
                    ("current_price", "Current Price"),
                    ("sl", "SL"),
                    ("tp", "TP"),
                    ("profit", "Profit"),
                    ("comment", "Comments"),
                    ("magic", "Magic"),
                ),
            )
            if compact_rows:
                compact["open_positions"] = compact_rows
            else:
                compact["positions"] = int(open_positions.get("count") or 0)

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
                    ("symbol", "Symbol"),
                    ("ticket", "Ticket"),
                    ("time", "Time"),
                    ("expiration", "Expiration"),
                    ("type", "Type"),
                    ("volume", "Volume"),
                    ("open_price", "Open Price"),
                    ("current_price", "Current Price"),
                    ("sl", "SL"),
                    ("tp", "TP"),
                    ("comment", "Comments"),
                    ("magic", "Magic"),
                ),
            )
            if compact_rows:
                compact["pending_orders"] = compact_rows
            else:
                compact["pending"] = int(pending_orders.get("count") or 0)

    return compact


@mcp.tool()
def trade_session_context(request: TradeSessionContextRequest) -> Dict[str, Any]:
    """Get a consolidated session context including account info, open positions, pending orders, ticker, and computed state for a symbol.

    Parameters: symbol, detail, include_account
    """

    def _run() -> Dict[str, Any]:
        # Un-wrap original functions if necessary to bypass double-logging or async mcp wrappers
        acc_func = getattr(trade_account_info, "__wrapped__", trade_account_info)
        ticker_func = getattr(market_ticker, "__wrapped__", market_ticker)
        open_func = getattr(trade_get_open, "__wrapped__", trade_get_open)
        pending_func = getattr(trade_get_pending, "__wrapped__", trade_get_pending)

        account_res = acc_func() if request.include_account else None
        ticker_res = ticker_func(symbol=request.symbol, detail=request.detail)

        open_req = TradeGetOpenRequest(symbol=request.symbol)
        open_res = open_func(request=open_req)

        pending_req = TradeGetPendingRequest(symbol=request.symbol)
        pending_res = pending_func(request=pending_req)

        if request.include_account:
            account_res, account_failed = _sanitize_trade_session_section_error(
                account_res,
                label="account context",
            )
        else:
            account_failed = False
        ticker_res, ticker_failed = _sanitize_trade_session_section_error(
            ticker_res,
            label="ticker data",
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
            (account_failed, ticker_failed, open_failed, pending_failed)
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

        payload = {
            "success": True,
            "symbol": request.symbol,
            "state": state,
            "open_positions": open_res,
            "pending_orders": pending_res,
            "ticker": ticker_res,
        }
        if request.include_account:
            payload["account"] = account_res
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
            payload["ticker"] = _strip_nested_envelope(payload["ticker"])
            price_digits = _price_precision_from_ticker(payload.get("ticker"))
            payload["open_positions"] = _round_trade_session_prices(
                payload["open_positions"],
                digits=price_digits,
            )
            payload["pending_orders"] = _round_trade_session_prices(
                payload["pending_orders"],
                digits=price_digits,
            )
            if isinstance(payload["ticker"], dict):
                payload["ticker"] = _normalize_nested_ticker_time(
                    payload["ticker"],
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
