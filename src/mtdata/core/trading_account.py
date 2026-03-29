"""Trading account and history views."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from ._mcp_instance import mcp
from . import trading_comments, trading_validation
from .config import mt5_config
from .execution_logging import run_logged_operation
from .trading_gateway import MT5TradingGateway, create_trading_gateway
from .trading_requests import TradeHistoryRequest
from .trading_use_cases import run_trade_history
from ..utils.mt5 import MT5ConnectionError, _mt5_epoch_to_utc, ensure_mt5_connection_or_raise, mt5_adapter
from ..utils.mt5_enums import decode_mt5_enum_label
from ..utils.utils import (
    _format_time_minimal,
    _format_time_minimal_local,
    _normalize_limit,
    _parse_start_datetime,
    _use_client_tz,
)

logger = logging.getLogger(__name__)


def _run_trade_history_request(request: TradeHistoryRequest) -> Any:
    return run_trade_history(
        request,
        gateway=create_trading_gateway(
            include_trade_preflight=True,
            adapter=mt5_adapter,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        ),
        use_client_tz=_use_client_tz,
        format_time_minimal=_format_time_minimal,
        format_time_minimal_local=_format_time_minimal_local,
        mt5_epoch_to_utc=_mt5_epoch_to_utc,
        parse_start_datetime=_parse_start_datetime,
        normalize_limit=_normalize_limit,
        comment_row_metadata=trading_comments._comment_row_metadata,
        normalize_ticket_filter=trading_validation._normalize_ticket_filter,
        normalize_minutes_back=trading_validation._normalize_minutes_back,
        decode_mt5_enum_label=decode_mt5_enum_label,
        mt5_config=mt5_config,
    )


def lookup_trade_ticket_history(ticket: Any) -> Optional[Dict[str, Any]]:
    ticket_text = str(ticket).strip()
    if not ticket_text:
        return None

    lookback_minutes = 60 * 24 * 7

    def _latest_row(result: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(result, list) or not result:
            return None
        rows = [row for row in result if isinstance(row, dict)]
        return rows[-1] if rows else None

    def _time_label(row: Dict[str, Any], *keys: str) -> Optional[str]:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    deals = _run_trade_history_request(
        TradeHistoryRequest(
            history_kind="deals",
            position_ticket=ticket,
            minutes_back=lookback_minutes,
            limit=20,
        )
    )
    deal_row = _latest_row(deals)
    if deal_row is not None:
        symbol = str(deal_row.get("symbol") or "").strip()
        type_label = str(deal_row.get("type") or "position").strip()
        time_label = _time_label(deal_row, "time", "time_done", "time_setup")
        reason_label = str(deal_row.get("reason") or "").strip()
        message = f"Ticket {ticket_text} was a {type_label} position that has already been closed"
        if symbol:
            message += f" on {symbol}"
        if time_label:
            message += f" at {time_label}"
        if reason_label:
            message += f" ({reason_label})"
        message += ". No action taken."
        return {
            "message": message,
            "no_action": True,
            "checked_scopes": ["positions", "pending_orders", "history_deals"],
        }

    orders = _run_trade_history_request(
        TradeHistoryRequest(
            history_kind="orders",
            order_ticket=ticket,
            minutes_back=lookback_minutes,
            limit=20,
        )
    )
    order_row = _latest_row(orders)
    if order_row is not None:
        symbol = str(order_row.get("symbol") or "").strip()
        type_label = str(order_row.get("type") or "order").strip()
        state_label = str(order_row.get("state") or "completed").strip()
        time_label = _time_label(order_row, "time_done", "time_setup", "time")
        message = f"Ticket {ticket_text} was a {type_label} order that was {state_label}"
        if symbol:
            message += f" on {symbol}"
        if time_label:
            message += f" at {time_label}"
        message += ". No action taken."
        return {
            "message": message,
            "no_action": True,
            "checked_scopes": ["positions", "pending_orders", "history_orders"],
        }

    return None


@mcp.tool()
def trade_account_info() -> dict:
    """Get account information (balance, equity, profit, margin level, free margin, account type, leverage, currency)."""
    def _run() -> dict:
        mt5 = create_trading_gateway(
            include_trade_preflight=True,
            adapter=mt5_adapter,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )

        try:
            mt5.ensure_connection()
        except MT5ConnectionError as exc:
            return {"error": str(exc)}

        info = mt5.account_info()
        if info is None:
            return {"error": "Failed to get account info"}
        preflight = mt5.build_trade_preflight(account_info=info)
        margin_level: Optional[float] = getattr(info, "margin_level", None)
        margin_level_note: Optional[str] = None
        try:
            margin_val = float(getattr(info, "margin", 0.0) or 0.0)
            ml_val = float(getattr(info, "margin_level", 0.0) or 0.0)
            if margin_val <= 0.0 and ml_val <= 0.0:
                margin_level = None
                margin_level_note = "N/A (no open margin/positions)"
            elif not math.isfinite(ml_val):
                margin_level = None
            else:
                margin_level = round(float(ml_val), 2)
        except Exception:
            pass

        payload = {
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "margin_level": margin_level,
            "currency": info.currency,
            "leverage": info.leverage,
            "trade_allowed": info.trade_allowed,
            "trade_expert": info.trade_expert,
            "server": preflight.get("server"),
            "company": preflight.get("company"),
            "trade_mode": preflight.get("trade_mode"),
            "terminal_trade_allowed": preflight.get("terminal_trade_allowed"),
            "terminal_tradeapi_disabled": preflight.get("terminal_tradeapi_disabled"),
            "terminal_connected": preflight.get("terminal_connected"),
            "auto_trading_enabled": preflight.get("auto_trading_enabled"),
            "execution_ready": preflight.get("execution_ready"),
            "execution_ready_strict": preflight.get("execution_ready_strict"),
            "execution_hard_blockers": preflight.get("execution_hard_blockers"),
            "execution_soft_blockers": preflight.get("execution_soft_blockers"),
            "execution_blockers": preflight.get("execution_blockers"),
        }
        if margin_level_note:
            payload["margin_level_note"] = margin_level_note
        return payload

    return run_logged_operation(
        logger,
        operation="trade_account_info",
        func=_run,
    )


@mcp.tool()
def trade_history(request: TradeHistoryRequest) -> List[Dict[str, Any]]:
    """Get deal or order history as tabular data."""
    return run_logged_operation(
        logger,
        operation="trade_history",
        history_kind=request.history_kind,
        symbol=request.symbol,
        limit=request.limit,
        func=lambda: _run_trade_history_request(request),
    )
