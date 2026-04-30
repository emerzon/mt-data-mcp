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
    _resolve_close_dry_run_target,
)
from .orders import _place_market_order, _place_pending_order
from .positions import trade_get_open, trade_get_pending
from .requests import TradeCloseRequest, TradeModifyRequest, TradePlaceRequest
from .risk import trade_risk_analyze, trade_var_cvar_calculate
from .use_cases import run_trade_close, run_trade_modify, run_trade_place

logger = logging.getLogger(__name__)


@mcp.tool()
def trade_place(request: TradePlaceRequest) -> dict:
    """Place a market or pending order.

    Required inputs: symbol, volume, order_type.
    - BUY/SELL: market orders; omit `price`.
    - BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP: pending (requires `price`).
    - dry_run: validate routing and preview the order without sending it to MT5.
      Use `preview_detail="compact"|"full"` to control preview depth.
    - require_sl_tp: for market orders, require both SL and TP inputs before order
      submission. Requested SL/TP levels are sent atomically with the order.
      Defaults to True for safer automation behavior.
    - auto_close_on_sl_tp_fail: retained for defensive handling of legacy injected
      order helpers that report a filled market order without TP/SL protection.
    - Environment guardrails can block orders before MT5 submission based on
      configured symbol policies, volume caps, or wallet/account risk limits.
    - idempotency_key: optional in-process dedupe key for retry-safe clients.
      Reusing a key with the same payload replays the prior outcome instead of
      sending another order; changed payloads require a new key.
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
    """Modify an open position or pending order by ticket.

    Risk-increasing pending-order changes can be blocked by configured trade
    guardrails, while close/reduce flows remain allowed.
    Optional idempotency_key values suppress duplicate in-process retries for
    the same payload.
    """
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
    Set `dry_run=true` to preview routing and validation without sending a close/cancel request.
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
            resolve_close_target=_resolve_close_dry_run_target,
        ),
    )
