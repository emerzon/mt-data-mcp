"""Trading functions for MetaTrader integration."""

import logging
import time

from ._mcp_instance import mcp
from . import trading_time, trading_validation
from .execution_logging import infer_result_success, log_operation_finish, log_operation_start
from .trading_account import trade_account_info, trade_history
from .trading_execution import _cancel_pending, _close_positions, _modify_pending_order, _modify_position
from .trading_orders import _place_market_order, _place_pending_order
from .trading_positions import trade_get_open, trade_get_pending
from .trading_requests import TradeCloseRequest, TradeModifyRequest, TradePlaceRequest
from .trading_risk import trade_risk_analyze
from .trading_use_cases import run_trade_close, run_trade_modify, run_trade_place

logger = logging.getLogger(__name__)


@mcp.tool()
def trade_place(request: TradePlaceRequest) -> dict:
    """Place a market or pending order.

    Required inputs: symbol, volume, order_type.
    - BUY/SELL: market by default; treated as pending when `price`/`expiration` is provided.
    - BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP: pending (requires `price`).
    - Also accepts ORDER_TYPE_* aliases and MT5 numeric constants 0..5 for order_type.
    - require_sl_tp: for market orders, require both SL and TP inputs before order
      submission, and fail when a filled order cannot apply TP/SL.
      Defaults to True for safer automation behavior.
    - auto_close_on_sl_tp_fail: if TP/SL application fails on a filled market order,
      attempt to immediately close the unprotected position.
    """
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="trade_place",
        symbol=request.symbol,
        order_type=request.order_type,
        volume=request.volume,
    )
    result = run_trade_place(
        request,
        normalize_order_type_input=trading_validation._normalize_order_type_input,
        normalize_pending_expiration=trading_time._normalize_pending_expiration,
        prevalidate_trade_place_market_input=trading_validation._prevalidate_trade_place_market_input,
        place_market_order=_place_market_order,
        place_pending_order=_place_pending_order,
        close_positions=_close_positions,
        safe_int_ticket=trading_validation._safe_int_ticket,
    )
    log_operation_finish(
        logger,
        operation="trade_place",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        order_type=request.order_type,
        volume=request.volume,
    )
    return result


@mcp.tool()
def trade_modify(request: TradeModifyRequest) -> dict:
    """Modify an open position or pending order by ticket."""
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="trade_modify",
        ticket=request.ticket,
    )
    result = run_trade_modify(
        request,
        normalize_pending_expiration=trading_time._normalize_pending_expiration,
        modify_pending_order=_modify_pending_order,
        modify_position=_modify_position,
    )
    log_operation_finish(
        logger,
        operation="trade_modify",
        started_at=started_at,
        success=infer_result_success(result),
        ticket=request.ticket,
    )
    return result


@mcp.tool()
def trade_close(request: TradeCloseRequest) -> dict:
    """Close positions or cancel pending orders."""
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="trade_close",
        ticket=request.ticket,
        symbol=request.symbol,
    )
    result = run_trade_close(
        request,
        close_positions=_close_positions,
        cancel_pending=_cancel_pending,
    )
    log_operation_finish(
        logger,
        operation="trade_close",
        started_at=started_at,
        success=infer_result_success(result),
        ticket=request.ticket,
        symbol=request.symbol,
    )
    return result
