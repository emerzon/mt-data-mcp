from __future__ import annotations

from typing import Any, Dict, List, Optional

from .trading_requests import TradeCloseRequest, TradeModifyRequest, TradePlaceRequest


def run_trade_place(
    request: TradePlaceRequest,
    *,
    normalize_order_type_input: Any,
    normalize_pending_expiration: Any,
    prevalidate_trade_place_market_input: Any,
    place_market_order: Any,
    place_pending_order: Any,
    close_positions: Any,
    safe_int_ticket: Any,
) -> Dict[str, Any]:
    missing: List[str] = []
    symbol_norm = str(request.symbol).strip() if request.symbol is not None else ""
    if not symbol_norm:
        missing.append("symbol")
    if request.volume is None:
        missing.append("volume")
    if request.order_type is None or (
        isinstance(request.order_type, str) and not request.order_type.strip()
    ):
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

    order_type_norm, order_type_error = normalize_order_type_input(request.order_type)
    if order_type_error:
        return {"error": order_type_error}
    explicit_pending_types = {"BUY_LIMIT", "BUY_STOP", "SELL_LIMIT", "SELL_STOP"}
    market_side_types = {"BUY", "SELL"}
    supported_order_types = explicit_pending_types.union(market_side_types)
    if order_type_norm not in supported_order_types:
        return {
            "error": (
                f"Unsupported order_type '{request.order_type}'. "
                "Use BUY/SELL or BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP."
            )
        }

    price_provided = request.price not in (None, 0)
    try:
        _, expiration_provided = normalize_pending_expiration(request.expiration)
    except (TypeError, ValueError) as ex:
        return {"error": str(ex)}

    is_pending = (
        order_type_norm in explicit_pending_types
        or price_provided
        or expiration_provided
    )
    if bool(request.require_sl_tp) and not is_pending:
        missing_protection: List[str] = []
        if request.stop_loss in (None, 0):
            missing_protection.append("stop_loss")
        if request.take_profit in (None, 0):
            missing_protection.append("take_profit")
        if missing_protection:
            prevalidation_error = prevalidate_trade_place_market_input(
                symbol_norm,
                request.volume,
            )
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
        result = place_market_order(
            symbol=symbol_norm,
            volume=float(request.volume),
            order_type=order_type_norm,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            comment=request.comment,
            deviation=request.deviation,
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
                if bool(request.auto_close_on_sl_tp_fail):
                    close_ticket = safe_int_ticket(pos_ticket)
                    if close_ticket is None:
                        auto_close_result: Dict[str, Any] = {
                            "error": "Auto-close skipped: position_ticket unavailable."
                        }
                    else:
                        auto_close_result = close_positions(
                            ticket=close_ticket,
                            comment="AUTO-CLOSE: TP/SL apply failed",
                            deviation=request.deviation,
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

            if (
                bool(request.require_sl_tp)
                and sl_tp_requested
                and sl_tp_failed
                and "error" not in result
            ):
                result["error"] = "Order was executed, but TP/SL protection could not be applied."
                result["require_sl_tp"] = bool(request.require_sl_tp)
                result["protection_status"] = (
                    result.get("protection_status") or "unprotected_position"
                )
        return result
    if request.price is None:
        return {"error": "price is required for pending orders."}
    return place_pending_order(
        symbol=symbol_norm,
        volume=float(request.volume),
        order_type=order_type_norm,
        price=request.price,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
        expiration=request.expiration,
        comment=request.comment,
        deviation=request.deviation,
    )


def run_trade_modify(
    request: TradeModifyRequest,
    *,
    normalize_pending_expiration: Any,
    modify_pending_order: Any,
    modify_position: Any,
) -> Dict[str, Any]:
    price_val = None if request.price in (None, 0) else request.price
    try:
        _, expiration_specified = normalize_pending_expiration(request.expiration)
    except (TypeError, ValueError) as ex:
        return {"error": str(ex)}

    if price_val is not None or expiration_specified:
        result = modify_pending_order(
            ticket=request.ticket,
            price=price_val,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            expiration=request.expiration,
            comment=request.comment,
        )
        if result.get("error") == f"Pending order {request.ticket} not found":
            return {
                "error": (
                    f"Pending order {request.ticket} not found. "
                    "Note: price/expiration only apply to pending orders."
                ),
                "checked_scopes": ["pending_orders"],
            }
        return result

    position_result = modify_position(
        ticket=request.ticket,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
        comment=request.comment,
    )
    if position_result.get("success"):
        return position_result
    if position_result.get("error") == f"Position {request.ticket} not found":
        pending_result = modify_pending_order(
            ticket=request.ticket,
            price=None,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            expiration=None,
            comment=request.comment,
        )
        if pending_result.get("error") == f"Pending order {request.ticket} not found":
            return {
                "error": f"Ticket {request.ticket} not found as position or pending order.",
                "checked_scopes": ["positions", "pending_orders"],
            }
        return pending_result
    return position_result


def run_trade_close(
    request: TradeCloseRequest,
    *,
    close_positions: Any,
    cancel_pending: Any,
) -> Dict[str, Any]:
    def _with_no_action(
        payload: Optional[Dict[str, Any]] = None,
        *,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(payload or {})
        if message and not str(out.get("message", "")).strip():
            out["message"] = message
        out["no_action"] = True
        return out

    if request.profit_only or request.loss_only:
        result = close_positions(
            ticket=request.ticket,
            symbol=request.symbol,
            profit_only=request.profit_only,
            loss_only=request.loss_only,
            comment=request.comment,
            deviation=request.deviation,
        )
        if isinstance(result, dict):
            msg = str(result.get("message", "")).strip().lower()
            if msg.startswith("no open positions") or msg == "no positions matched criteria":
                return _with_no_action(result)
        return result

    if request.ticket is not None:
        position_result = close_positions(
            ticket=request.ticket,
            symbol=request.symbol,
            profit_only=False,
            loss_only=False,
            comment=request.comment,
            deviation=request.deviation,
        )
        if (
            isinstance(position_result, dict)
            and position_result.get("error") == f"Position {request.ticket} not found"
        ):
            pending_result = cancel_pending(
                ticket=request.ticket,
                symbol=request.symbol,
                comment=request.comment,
            )
            if (
                isinstance(pending_result, dict)
                and pending_result.get("error") == f"Pending order {request.ticket} not found"
            ):
                return {
                    "error": f"Ticket {request.ticket} not found as position or pending order.",
                    "checked_scopes": ["positions", "pending_orders"],
                }
            return pending_result
        return position_result

    if request.symbol is not None:
        position_result = close_positions(
            symbol=request.symbol,
            profit_only=False,
            loss_only=False,
            comment=request.comment,
            deviation=request.deviation,
        )
        if isinstance(position_result, dict):
            msg = str(position_result.get("message", "")).strip().lower()
            if msg.startswith("no open positions for "):
                pending_result = cancel_pending(
                    symbol=request.symbol,
                    comment=request.comment,
                )
                if isinstance(pending_result, dict):
                    pending_msg = str(pending_result.get("message", "")).strip().lower()
                    if pending_msg.startswith("no pending orders for "):
                        return _with_no_action(
                            message=f"No open positions or pending orders for {request.symbol}"
                        )
                return pending_result
        return position_result

    position_result = close_positions(
        profit_only=False,
        loss_only=False,
        comment=request.comment,
        deviation=request.deviation,
    )
    if isinstance(position_result, dict):
        msg = str(position_result.get("message", "")).strip().lower()
        if msg == "no open positions":
            pending_result = cancel_pending(comment=request.comment)
            if (
                isinstance(pending_result, dict)
                and str(pending_result.get("message", "")).strip().lower() == "no pending orders"
            ):
                return _with_no_action(message="No open positions or pending orders")
            return pending_result
    return position_result
