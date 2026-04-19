"""Trading session context utilities."""

import logging
from typing import Any, Dict, Optional

from .._mcp_instance import mcp
from ..execution_logging import run_logged_operation
from ..market_depth import market_ticker
from ..output_contract import ensure_common_meta
from .account import trade_account_info
from .positions import trade_get_open, trade_get_pending
from .requests import TradeGetOpenRequest, TradeGetPendingRequest, TradeSessionContextRequest

logger = logging.getLogger(__name__)


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
        for key in ("success", "symbol", "state")
        if payload.get(key) not in (None, "")
    }

    account = payload.get("account")
    if isinstance(account, dict):
        if account.get("error") not in (None, ""):
            compact["account"] = {"error": account.get("error")}
        else:
            account_summary = {
                key: account.get(key)
                for key in ("balance", "equity", "margin_level")
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
                    "spread_pct",
                    "spread_usd",
                    "time",
                    "time_display",
                    "timezone",
                )
                if ticker.get(key) not in (None, "")
            }
            if ticker_summary:
                compact["ticker"] = ticker_summary

    open_positions = payload.get("open_positions")
    if isinstance(open_positions, dict):
        if open_positions.get("error") not in (None, ""):
            compact["open_positions"] = {"error": open_positions.get("error")}
        else:
            compact_rows = _compact_trade_session_items(
                open_positions,
                field_map=(
                    ("ticket", "Ticket"),
                    ("time", "Time"),
                    ("type", "Type"),
                    ("volume", "Volume"),
                    ("open_price", "Open Price"),
                    ("current_price", "Current Price"),
                    ("sl", "SL"),
                    ("tp", "TP"),
                    ("profit", "Profit"),
                ),
            )
            if compact_rows:
                compact["open_positions"] = compact_rows

    pending_orders = payload.get("pending_orders")
    if isinstance(pending_orders, dict):
        if pending_orders.get("error") not in (None, ""):
            compact["pending_orders"] = {"error": pending_orders.get("error")}
        else:
            compact_rows = _compact_trade_session_items(
                pending_orders,
                field_map=(
                    ("ticket", "Ticket"),
                    ("time", "Time"),
                    ("expiration", "Expiration"),
                    ("type", "Type"),
                    ("volume", "Volume"),
                    ("open_price", "Open Price"),
                    ("current_price", "Current Price"),
                    ("sl", "SL"),
                    ("tp", "TP"),
                ),
            )
            if compact_rows:
                compact["pending_orders"] = compact_rows

    return compact


@mcp.tool()
def trade_session_context(request: TradeSessionContextRequest) -> Dict[str, Any]:
    """Get a consolidated session context including account info, open positions, pending orders, ticker, and computed state for a symbol.

    Parameters: symbol, detail
    """

    def _run() -> Dict[str, Any]:
        # Un-wrap original functions if necessary to bypass double-logging or async mcp wrappers
        acc_func = getattr(trade_account_info, "__wrapped__", trade_account_info)
        ticker_func = getattr(market_ticker, "__wrapped__", market_ticker)
        open_func = getattr(trade_get_open, "__wrapped__", trade_get_open)
        pending_func = getattr(trade_get_pending, "__wrapped__", trade_get_pending)

        account_res = acc_func()
        ticker_res = ticker_func(symbol=request.symbol, detail=request.detail)

        open_req = TradeGetOpenRequest(symbol=request.symbol)
        open_res = open_func(request=open_req)

        pending_req = TradeGetPendingRequest(symbol=request.symbol)
        pending_res = pending_func(request=pending_req)

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
            "account": account_res,
            "open_positions": open_res,
            "pending_orders": pending_res,
            "ticker": ticker_res,
        }
        if request.detail == "compact":
            payload = _compact_trade_session_context_payload(payload)
        return ensure_common_meta(payload, tool_name="trade_session_context")

    return run_logged_operation(
        logger,
        operation="trade_session_context",
        symbol=request.symbol,
        detail=request.detail,
        func=_run,
    )
