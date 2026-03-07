"""Trading functions for MetaTrader integration."""

from typing import Optional, Union, List, Dict, Any

from ._mcp_instance import mcp
from . import trading_time, trading_validation
from .trading_account import trade_account_info, trade_history
from .trading_common import _build_trade_preflight, _retcode_name
from .trading_execution import _cancel_pending, _close_positions, _modify_pending_order, _modify_position
from .trading_orders import _place_market_order, _place_pending_order
from .trading_positions import _resolve_open_position, trade_get_open, trade_get_pending
from .trading_risk import trade_risk_analyze
from .trading_time import ExpirationValue
from .trading_validation import MarketOrderTypeInput, OrderTypeInput


@mcp.tool()
def trade_place(
    symbol: Optional[str] = None,
    volume: Optional[float] = None,
    order_type: Optional[OrderTypeInput] = None,
    price: Optional[Union[int, float]] = None,
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    expiration: Optional[ExpirationValue] = None,
    comment: Optional[str] = None,
    deviation: int = 20,
    require_sl_tp: bool = True,
    auto_close_on_sl_tp_fail: bool = False,
) -> dict:
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

    missing: List[str] = []
    symbol_norm = str(symbol).strip() if symbol is not None else ""
    if not symbol_norm:
        missing.append("symbol")
    if volume is None:
        missing.append("volume")
    if order_type is None or (isinstance(order_type, str) and not order_type.strip()):
        missing.append("order_type")
    if missing:
        return {
            "error": (
                f"Missing required field(s): {', '.join(missing)}. "
                "Required: symbol, volume, order_type."
            ),
            "required": ["symbol", "volume", "order_type"],
            "hint": (
                "Example: symbol='BTCUSD', volume=0.03, "
                "order_type='BUY_LIMIT' (or ORDER_TYPE_BUY_LIMIT or 2)."
            ),
        }

    t, order_type_error = trading_validation._normalize_order_type_input(order_type)
    if order_type_error:
        return {"error": order_type_error}
    explicit_pending_types = {
        "BUY_LIMIT",
        "BUY_STOP",
        "SELL_LIMIT",
        "SELL_STOP",
    }
    market_side_types = {"BUY", "SELL"}
    supported_order_types = explicit_pending_types.union(market_side_types)
    if t not in supported_order_types:
        return {
            "error": (
                f"Unsupported order_type '{order_type}'. "
                "Use BUY/SELL or BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP."
            )
        }

    price_provided = price not in (None, 0)
    try:
        _, expiration_provided = trading_time._normalize_pending_expiration(expiration)
    except (TypeError, ValueError) as ex:
        return {"error": str(ex)}

    is_pending = (t in explicit_pending_types) or price_provided or expiration_provided
    if bool(require_sl_tp) and not is_pending:
        missing_protection: List[str] = []
        if stop_loss in (None, 0):
            missing_protection.append("stop_loss")
        if take_profit in (None, 0):
            missing_protection.append("take_profit")
        if missing_protection:
            prevalidation_error = trading_validation._prevalidate_trade_place_market_input(symbol_norm, volume)
            if prevalidation_error is not None:
                return prevalidation_error
            return {
                "error": (
                    "require_sl_tp=True requires both stop_loss and take_profit for market orders. "
                    "Refusing to place an unprotected position."
                ),
                "require_sl_tp": True,
                "missing": missing_protection,
                "hint": (
                    "Provide both --stop-loss and --take-profit, "
                    "or explicitly set --require-sl-tp false."
                ),
            }

    if not is_pending:
        result = _place_market_order(
            symbol=symbol_norm,
            volume=float(volume),
            order_type=t,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
            deviation=deviation,
        )
        if isinstance(result, dict):
            sl_tp_requested = bool(result.get("sl_tp_requested"))
            sl_tp_failed = str(result.get("sl_tp_apply_status") or "").lower() == "failed"
            if sl_tp_requested and sl_tp_failed:
                warnings_out: List[str] = list(result.get("warnings") or [])
                pos_ticket = result.get("position_ticket")
                if pos_ticket is not None:
                    critical = (
                        "CRITICAL: Order executed without applied TP/SL protection. "
                        f"Run trade_modify {pos_ticket} now, or close the position."
                    )
                else:
                    critical = (
                        "CRITICAL: Order executed without applied TP/SL protection. "
                        "Run trade_modify now, or close the position."
                    )
                if critical not in warnings_out:
                    warnings_out.append(critical)
                if warnings_out:
                    result["warnings"] = warnings_out
                if bool(auto_close_on_sl_tp_fail):
                    close_ticket = trading_validation._safe_int_ticket(pos_ticket)
                    if close_ticket is None:
                        auto_close_result: Dict[str, Any] = {
                            "error": "Auto-close skipped: position_ticket unavailable."
                        }
                    else:
                        auto_close_result = _close_positions(
                            ticket=close_ticket,
                            comment="AUTO-CLOSE: TP/SL apply failed",
                            deviation=deviation,
                        )
                    result["auto_close_on_sl_tp_fail"] = True
                    result["auto_close_result"] = auto_close_result

                    auto_close_ok = False
                    if isinstance(auto_close_result, dict) and "error" not in auto_close_result:
                        if auto_close_result.get("retcode") is not None:
                            auto_close_ok = True
                        else:
                            try:
                                auto_close_ok = int(auto_close_result.get("closed_count", 0)) > 0
                            except Exception:
                                auto_close_ok = False
                    if auto_close_ok:
                        result["protection_status"] = "auto_closed_after_sl_tp_fail"
                    else:
                        warnings_out = list(result.get("warnings") or [])
                        auto_close_warning = (
                            "AUTO-CLOSE FAILED: position remains unprotected; close immediately."
                        )
                        if auto_close_warning not in warnings_out:
                            warnings_out.append(auto_close_warning)
                        result["warnings"] = warnings_out

            if bool(require_sl_tp) and sl_tp_requested and sl_tp_failed and "error" not in result:
                result["error"] = "Order was executed, but TP/SL protection could not be applied."
                result["require_sl_tp"] = bool(require_sl_tp)
                result["protection_status"] = result.get("protection_status") or "unprotected_position"
        return result
    if price is None:
        return {"error": "price is required for pending orders."}
    return _place_pending_order(
        symbol=symbol_norm,
        volume=float(volume),
        order_type=t,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        expiration=expiration,
        comment=comment,
        deviation=deviation,
    )


@mcp.tool()
def trade_modify(
    ticket: Union[int, str],
    price: Optional[Union[int, float]] = None,
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    expiration: Optional[ExpirationValue] = None,
    comment: Optional[str] = None,
) -> dict:
    """Modify an open position or pending order by ticket."""
    price_val = None if price in (None, 0) else price
    try:
        _, expiration_specified = trading_time._normalize_pending_expiration(expiration)
    except (TypeError, ValueError) as ex:
        return {"error": str(ex)}

    if price_val is not None or expiration_specified:
        result = _modify_pending_order(
            ticket=ticket,
            price=price_val,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expiration=expiration,
            comment=comment,
        )
        if result.get("error") == f"Pending order {ticket} not found":
            return {
                "error": (
                    f"Pending order {ticket} not found. "
                    "Note: price/expiration only apply to pending orders."
                ),
                "checked_scopes": ["pending_orders"],
            }
        return result

    position_result = _modify_position(
        ticket=ticket,
        stop_loss=stop_loss,
        take_profit=take_profit,
        comment=comment,
    )
    if position_result.get("success"):
        return position_result
    if position_result.get("error") == f"Position {ticket} not found":
        pending_result = _modify_pending_order(
            ticket=ticket,
            price=None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expiration=None,
            comment=comment,
        )
        if pending_result.get("error") == f"Pending order {ticket} not found":
            return {
                "error": f"Ticket {ticket} not found as position or pending order.",
                "checked_scopes": ["positions", "pending_orders"],
            }
        return pending_result
    return position_result


@mcp.tool()
def trade_close(
    ticket: Optional[Union[int, str]] = None,
    symbol: Optional[str] = None,
    profit_only: bool = False,
    loss_only: bool = False,
    comment: Optional[str] = None,
    deviation: int = 20,
) -> dict:
    """Close positions or cancel pending orders."""

    def _with_no_action(payload: Optional[Dict[str, Any]] = None, *, message: Optional[str] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(payload or {})
        if message and not str(out.get("message", "")).strip():
            out["message"] = message
        out["no_action"] = True
        return out

    if profit_only or loss_only:
        result = _close_positions(
            ticket=ticket,
            symbol=symbol,
            profit_only=profit_only,
            loss_only=loss_only,
            comment=comment,
            deviation=deviation,
        )
        if isinstance(result, dict):
            msg = str(result.get("message", "")).strip().lower()
            if msg.startswith("no open positions") or msg == "no positions matched criteria":
                return _with_no_action(result)
        return result

    if ticket is not None:
        position_result = _close_positions(
            ticket=ticket,
            symbol=symbol,
            profit_only=False,
            loss_only=False,
            comment=comment,
            deviation=deviation,
        )
        if isinstance(position_result, dict) and position_result.get("error") == f"Position {ticket} not found":
            pending_result = _cancel_pending(ticket=ticket, symbol=symbol, comment=comment)
            if isinstance(pending_result, dict) and pending_result.get("error") == f"Pending order {ticket} not found":
                return {
                    "error": f"Ticket {ticket} not found as position or pending order.",
                    "checked_scopes": ["positions", "pending_orders"],
                }
            return pending_result
        return position_result

    if symbol is not None:
        position_result = _close_positions(
            symbol=symbol,
            profit_only=False,
            loss_only=False,
            comment=comment,
            deviation=deviation,
        )
        if isinstance(position_result, dict):
            msg = str(position_result.get("message", "")).strip().lower()
            if msg.startswith("no open positions for "):
                pending_result = _cancel_pending(symbol=symbol, comment=comment)
                if isinstance(pending_result, dict):
                    pending_msg = str(pending_result.get("message", "")).strip().lower()
                    if pending_msg.startswith("no pending orders for "):
                        return _with_no_action(message=f"No open positions or pending orders for {symbol}")
                return pending_result
        return position_result

    position_result = _close_positions(
        profit_only=False,
        loss_only=False,
        comment=comment,
        deviation=deviation,
    )
    if isinstance(position_result, dict):
        msg = str(position_result.get("message", "")).strip().lower()
        if msg == "no open positions":
            pending_result = _cancel_pending(comment=comment)
            if isinstance(pending_result, dict) and str(pending_result.get("message", "")).strip().lower() == "no pending orders":
                return _with_no_action(message="No open positions or pending orders")
            return pending_result
    return position_result
