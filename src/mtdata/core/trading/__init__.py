"""Trading functions for MetaTrader integration."""

import logging

from .._mcp_instance import mcp
from ..execution_logging import run_logged_operation
from . import time, validation
from .account import (
    lookup_trade_ticket_history,
    trade_account_info,
    trade_history,
    trade_journal_analyze,
)
from .context import trade_session_context
from .execution import (
    _cancel_pending,
    _close_positions,
    _modify_pending_order,
    _modify_position,
)
from .orders import _place_market_order, _place_pending_order
from .positions import trade_get_open, trade_get_pending
from .requests import TradeCloseRequest, TradeModifyRequest, TradePlaceRequest
from .risk import trade_risk_analyze
from .use_cases import run_trade_close, run_trade_modify, run_trade_place

logger = logging.getLogger(__name__)


@mcp.tool()
def trade_place(request: TradePlaceRequest) -> dict:
    """Place a market or pending order.

    Required inputs: symbol, volume, order_type.
    - BUY/SELL: market by default; treated as pending when `price`/`expiration` is provided.
    - BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP: pending (requires `price`).
    - Also accepts ORDER_TYPE_* aliases and MT5 numeric constants 0..5 for order_type.
    - dry_run: validate routing and preview the order without sending it to MT5.
    - require_sl_tp: for market orders, require both SL and TP inputs before order
      submission, and fail when a filled order cannot apply TP/SL.
      Defaults to True for safer automation behavior.
    - auto_close_on_sl_tp_fail: if TP/SL application fails on a filled market order,
      attempt to immediately close the unprotected position.
    """
    return run_logged_operation(
        logger,
        operation="trade_place",
        symbol=request.symbol,
        order_type=request.order_type,
        volume=request.volume,
        func=lambda: run_trade_place(
            request,
            normalize_order_type_input=validation._normalize_order_type_input,
            normalize_pending_expiration=time._normalize_pending_expiration,
            prevalidate_trade_place_market_input=validation._prevalidate_trade_place_market_input,
            place_market_order=_place_market_order,
            place_pending_order=_place_pending_order,
            close_positions=_close_positions,
            safe_int_ticket=validation._safe_int_ticket,
        ),
    )


@mcp.tool()
def trade_modify(request: TradeModifyRequest) -> dict:
    """Modify an open position or pending order by ticket."""
    return run_logged_operation(
        logger,
        operation="trade_modify",
        ticket=request.ticket,
        func=lambda: run_trade_modify(
            request,
            normalize_pending_expiration=time._normalize_pending_expiration,
            modify_pending_order=_modify_pending_order,
            modify_position=_modify_position,
        ),
    )


@mcp.tool()
def trade_close(request: TradeCloseRequest) -> dict:
    """Close positions or cancel pending orders.

    `ticket` closes a specific position or pending order.
    Any bulk close requires `close_all=true`.
    Set `volume` only to partially close a specific open position by ticket.
    """
    return run_logged_operation(
        logger,
        operation="trade_close",
        ticket=request.ticket,
        symbol=request.symbol,
        func=lambda: run_trade_close(
            request,
            close_positions=_close_positions,
            cancel_pending=_cancel_pending,
            lookup_ticket_history=lookup_trade_ticket_history,
        ),
    )
