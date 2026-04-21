"""Trade modification and closure workflows for MetaTrader integration."""

import math
import time as _stdlib_time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

from ...utils.mt5 import _mt5_epoch_to_utc, _to_server_naive_dt
from ..config import trade_guardrails_config
from . import comments, time, validation
from .gateway import MT5TradingGateway, create_trading_gateway, trading_connection_error
from .positions import _resolve_open_position, _resolve_pending_order
from .safety import evaluate_trade_guardrails, pending_order_risk_increased
from .time import ExpirationValue


def _resolve_position_side(position: Any, mt5: Any) -> Optional[str]:
    return validation._resolve_position_side(position, mt5)


def _normalize_protection_level(value: Optional[float], *, tol: float) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric) or math.isclose(numeric, 0.0, abs_tol=tol):
        return None
    return numeric


def _protection_levels_match(lhs: Optional[float], rhs: Optional[float], *, tol: float) -> bool:
    if lhs is None or rhs is None:
        return lhs is None and rhs is None
    return math.isclose(float(lhs), float(rhs), abs_tol=tol)


def _protection_level_tolerance(*, point: float) -> float:
    if math.isfinite(point) and point > 0.0:
        return point * 0.1
    return 1e-9


def _unexpected_operation_error(
    operation: str,
    exc: Exception,
    *,
    mt5: Any,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "error": f"Unexpected error while {operation}",
        "error_type": type(exc).__name__,
        "error_detail": str(exc),
        "traceback": traceback.format_exc(limit=5).strip(),
    }
    last_error = validation._safe_last_error(mt5)
    if last_error is not None:
        payload["last_error"] = last_error
    if context:
        payload["context"] = context
    return payload


def _deal_history_sort_key(row: Any) -> float:
    for field in ("time_msc", "time", "time_update_msc", "time_update"):
        try:
            raw_value = getattr(row, field, None)
            if raw_value is None:
                continue
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0.0:
                continue
            if field.endswith("_msc"):
                value /= 1000.0
            return value
        except Exception:
            continue
    return 0.0


def _resolve_closed_deal_from_history(
    mt5: Any,
    *,
    result: Any,
    position: Any,
    closed_at_utc: datetime,
) -> Optional[Any]:
    deal_ticket = validation._safe_int_ticket(getattr(result, "deal", None))
    order_ticket = validation._safe_int_ticket(getattr(result, "order", None))
    position_ticket = validation._safe_int_ticket(getattr(position, "ticket", None))
    if deal_ticket is None and order_ticket is None and position_ticket is None:
        return None
    try:
        rows = mt5.history_deals_get(
            _to_server_naive_dt(closed_at_utc - timedelta(minutes=5)),
            _to_server_naive_dt(closed_at_utc + timedelta(minutes=1)),
        )
    except Exception:
        return None
    if not rows:
        return None

    exact_deal_matches: List[Any] = []
    order_matches: List[Any] = []
    position_matches: List[Any] = []
    for row in rows:
        row_ticket = validation._safe_int_ticket(getattr(row, "ticket", None))
        row_order = validation._safe_int_ticket(getattr(row, "order", None))
        row_position_candidates = {
            validation._safe_int_ticket(getattr(row, "position_id", None)),
            validation._safe_int_ticket(getattr(row, "position_by_id", None)),
            validation._safe_int_ticket(getattr(row, "position", None)),
        }
        row_position_candidates.discard(None)
        if deal_ticket is not None and row_ticket == deal_ticket:
            exact_deal_matches.append(row)
            continue
        if order_ticket is not None and row_order == order_ticket:
            order_matches.append(row)
            continue
        if position_ticket is not None and position_ticket in row_position_candidates:
            position_matches.append(row)

    for matches in (exact_deal_matches, order_matches, position_matches):
        if matches:
            return max(matches, key=_deal_history_sort_key)
    return None


def _count_done_results(mt5: Any, results: List[Dict[str, Any]]) -> int:
    success_count = 0
    done_codes = validation._trade_done_codes(mt5)
    for item in results:
        if validation._retcode_is_done(mt5, item.get("retcode"), done_codes):
            success_count += 1
    return success_count


def _modify_position(
    ticket: Union[int, str],
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    comment: Optional[str] = None,
    gateway: Optional[MT5TradingGateway] = None,
) -> dict:
    """Internal helper to modify a position by ticket."""
    mt5 = create_trading_gateway(
        gateway=gateway,
        include_retcode_name=True,
    )

    connection_error = trading_connection_error(mt5)
    if connection_error is not None:
        return connection_error

    def _modify_position():
        try:
            ticket_id = int(ticket)
            position, resolved_ticket, ticket_resolution = _resolve_open_position(
                mt5,
                ticket_candidates=[ticket_id],
                require_exact_ticket_match=True,
                allow_alternate_ticket_match=True,
            )
            if position is None or resolved_ticket is None:
                out = {"error": f"Position {ticket} not found", "checked_scopes": ["positions"]}
                if isinstance(ticket_resolution, dict):
                    out["ticket_resolution"] = ticket_resolution
                return out

            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info is None:
                return {"error": f"Failed to get symbol info for {position.symbol}"}

            price_inputs, price_inputs_error = validation._normalize_trade_price_inputs(
                symbol_info=symbol_info,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            if price_inputs_error is not None:
                return {"error": price_inputs_error}
            point = float(price_inputs["point"])
            digits = int(price_inputs["digits"])
            requested_sl = price_inputs["stop_loss"]
            requested_tp = price_inputs["take_profit"]
            explicit_remove_sl = bool(price_inputs["explicit_remove_stop_loss"])
            explicit_remove_tp = bool(price_inputs["explicit_remove_take_profit"])

            # Normalize SL/TP values
            existing_sl = validation._normalize_price_for_symbol(
                getattr(position, "sl", None),
                point=point,
                digits=digits,
            )
            existing_tp = validation._normalize_price_for_symbol(
                getattr(position, "tp", None),
                point=point,
                digits=digits,
            )
            norm_sl = (
                0.0
                if explicit_remove_sl
                else (
                    float(requested_sl)
                    if stop_loss is not None
                    else float(existing_sl or 0.0)
                )
            )
            norm_tp = (
                0.0
                if explicit_remove_tp
                else (
                    float(requested_tp)
                    if take_profit is not None
                    else float(existing_tp or 0.0)
                )
            )
            validate_sl = (
                float(norm_sl)
                if not explicit_remove_sl and float(norm_sl) > 0.0
                else None
            )
            validate_tp = (
                float(norm_tp)
                if not explicit_remove_tp and float(norm_tp) > 0.0
                else None
            )

            price_tol = _protection_level_tolerance(point=point)
            current_sl = _normalize_protection_level(existing_sl, tol=price_tol)
            current_tp = _normalize_protection_level(existing_tp, tol=price_tol)
            desired_sl = _normalize_protection_level(norm_sl, tol=price_tol)
            desired_tp = _normalize_protection_level(norm_tp, tol=price_tol)

            if _protection_levels_match(current_sl, desired_sl, tol=price_tol) and _protection_levels_match(current_tp, desired_tp, tol=price_tol):
                no_change_code = validation._safe_int_attr(mt5, "TRADE_RETCODE_NO_CHANGES", 10025)
                return {
                    "success": True,
                    "retcode": no_change_code,
                    "retcode_name": mt5.retcode_name(no_change_code),
                    "comment": "No changes",
                    "request_id": 0,
                    "position_ticket": resolved_ticket,
                    "ticket_requested": ticket_id,
                    "ticket_resolution": ticket_resolution,
                    "applied_sl": desired_sl,
                    "applied_tp": desired_tp,
                    "no_change": True,
                    "message": "Requested SL/TP already match the live position.",
                }

            side = _resolve_position_side(position, mt5)
            if side is None:
                return {"error": "Unable to determine position side for protection validation."}
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return {"error": f"Failed to get current price for {position.symbol}"}
            live_protection_error = validation._validate_live_protection_levels(
                symbol_info=symbol_info,
                tick=tick,
                side=side,
                stop_loss=validate_sl,
                take_profit=validate_tp,
            )
            if live_protection_error is not None:
                return live_protection_error

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": resolved_ticket,
                "sl": norm_sl,
                "tp": norm_tp,
                "comment": comments._normalize_trade_comment(comment, default="MCP modify position"),
            }
            request_magic = validation._safe_int_ticket(getattr(position, "magic", None))
            if request_magic is not None:
                request["magic"] = request_magic

            result, comment_fallback, last_error = comments._send_order_with_comment_fallback(
                mt5,
                request,
            )
            if result is None:
                return {"error": "Failed to modify position", "last_error": last_error}

            result_retcode = getattr(result, "retcode", None)
            no_change_code = validation._safe_int_attr(mt5, "TRADE_RETCODE_NO_CHANGES", 10025)

            if result_retcode == no_change_code:
                refreshed_position = None
                try:
                    refreshed_rows = mt5.positions_get(ticket=resolved_ticket)
                except Exception:
                    refreshed_rows = None
                if refreshed_rows:
                    try:
                        refreshed_position = list(refreshed_rows)[0]
                    except Exception:
                        refreshed_position = None
                refreshed_sl = validation._normalize_price_for_symbol(
                    getattr(refreshed_position, "sl", None) if refreshed_position is not None else getattr(position, "sl", None),
                    point=point,
                    digits=digits,
                )
                refreshed_tp = validation._normalize_price_for_symbol(
                    getattr(refreshed_position, "tp", None) if refreshed_position is not None else getattr(position, "tp", None),
                    point=point,
                    digits=digits,
                )
                final_sl = _normalize_protection_level(refreshed_sl, tol=price_tol)
                final_tp = _normalize_protection_level(refreshed_tp, tol=price_tol)
                if _protection_levels_match(final_sl, desired_sl, tol=price_tol) and _protection_levels_match(final_tp, desired_tp, tol=price_tol):
                    out = {
                        "success": True,
                        "retcode": result_retcode,
                        "retcode_name": mt5.retcode_name(result_retcode),
                        "deal": result.deal,
                        "order": result.order,
                        "comment": result.comment,
                        "request_id": result.request_id,
                        "position_ticket": resolved_ticket,
                        "ticket_requested": ticket_id,
                        "ticket_resolution": ticket_resolution,
                        "applied_sl": final_sl,
                        "applied_tp": final_tp,
                        "no_change": True,
                        "message": "Requested SL/TP already match the live position.",
                    }
                    if isinstance(comment_fallback, dict):
                        out["comment_fallback"] = comment_fallback
                    return out

                out = {
                    "error": "Broker reported no changes, but live SL/TP do not match the requested values.",
                    "retcode": result_retcode,
                    "retcode_name": mt5.retcode_name(result_retcode),
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "position_ticket": resolved_ticket,
                    "ticket_requested": ticket_id,
                    "ticket_resolution": ticket_resolution,
                    "desired_sl": desired_sl,
                    "desired_tp": desired_tp,
                    "actual_sl": final_sl,
                    "actual_tp": final_tp,
                    "no_change": True,
                    "last_error": last_error,
                }
                if isinstance(comment_fallback, dict):
                    out["comment_fallback"] = comment_fallback
                return out

            if result_retcode != mt5.TRADE_RETCODE_DONE:
                out = {
                    "error": "Failed to modify position",
                    "retcode": result_retcode,
                    "retcode_name": mt5.retcode_name(result_retcode),
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "last_error": last_error,
                }
                if isinstance(comment_fallback, dict):
                    out["comment_fallback"] = comment_fallback
                return out

            out = {
                "success": True,
                "retcode": result_retcode,
                "retcode_name": mt5.retcode_name(result_retcode),
                "deal": result.deal,
                "order": result.order,
                "comment": result.comment,
                "request_id": result.request_id,
                "position_ticket": resolved_ticket,
                "ticket_requested": ticket_id,
                "ticket_resolution": ticket_resolution,
                "applied_sl": desired_sl,
                "applied_tp": desired_tp,
            }
            if isinstance(comment_fallback, dict):
                out["comment_fallback"] = comment_fallback
            return out

        except Exception as e:
            return _unexpected_operation_error(
                "modifying position",
                e,
                mt5=mt5,
                context={"ticket": ticket},
            )

    return _modify_position()


def _modify_pending_order(
    ticket: Union[int, str],
    price: Optional[Union[int, float]] = None,
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    expiration: Optional[ExpirationValue] = None,
    comment: Optional[str] = None,
    gateway: Optional[MT5TradingGateway] = None,
) -> dict:
    """Internal helper to modify a pending order by ticket."""
    mt5 = create_trading_gateway(
        gateway=gateway,
        include_retcode_name=True,
    )

    connection_error = trading_connection_error(mt5)
    if connection_error is not None:
        return connection_error

    def _modify_pending_order():
        try:
            ticket_id = int(ticket)
            order, resolved_ticket, ticket_resolution = _resolve_pending_order(
                mt5,
                ticket_candidates=[ticket_id],
                require_exact_ticket_match=True,
            )
            if order is None:
                out = {"error": f"Pending order {ticket} not found", "checked_scopes": ["pending_orders"]}
                if isinstance(ticket_resolution, dict):
                    out["ticket_resolution"] = ticket_resolution
                return out
            normalized_expiration, expiration_specified = time._normalize_pending_expiration(expiration)
            symbol_info = mt5.symbol_info(order.symbol)
            if symbol_info is None:
                return {"error": f"Failed to get symbol info for {order.symbol}"}
            tick = mt5.symbol_info_tick(order.symbol)
            if tick is None:
                return {"error": f"Failed to get current price for {order.symbol}"}
            price_inputs, price_inputs_error = validation._normalize_trade_price_inputs(
                symbol_info=symbol_info,
                price=price if price is not None else getattr(order, "price_open", None),
                require_price=True,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            if price_inputs_error is not None:
                return {"error": price_inputs_error}
            point = float(price_inputs["point"])
            digits = int(price_inputs["digits"])
            normalized_price = float(price_inputs["price"])
            requested_sl = price_inputs["stop_loss"]
            requested_tp = price_inputs["take_profit"]
            explicit_remove_sl = bool(price_inputs["explicit_remove_stop_loss"])
            explicit_remove_tp = bool(price_inputs["explicit_remove_take_profit"])

            existing_sl = validation._normalize_price_for_symbol(
                getattr(order, "sl", None),
                point=point,
                digits=digits,
            )
            existing_tp = validation._normalize_price_for_symbol(
                getattr(order, "tp", None),
                point=point,
                digits=digits,
            )
            request_sl = (
                0.0
                if explicit_remove_sl
                else (
                    float(requested_sl)
                    if stop_loss is not None
                    else float(existing_sl or 0.0)
                )
            )
            request_tp = (
                0.0
                if explicit_remove_tp
                else (
                    float(requested_tp)
                    if take_profit is not None
                    else float(existing_tp or 0.0)
                )
            )

            order_type_value = validation._safe_int_attr(order, "type", -1)
            pending_level_error = validation._validate_pending_order_levels(
                symbol_info=symbol_info,
                tick=tick,
                order_type_value=order_type_value,
                price=float(normalized_price),
                stop_loss=None if request_sl == 0.0 else float(request_sl),
                take_profit=None if request_tp == 0.0 else float(request_tp),
                mt5=mt5,
            )
            if pending_level_error is not None:
                return pending_level_error

            order_volume = (
                validation._safe_float_attr(order, "volume_current")
                or validation._safe_float_attr(order, "volume_initial")
                or validation._safe_float_attr(order, "volume")
            )
            existing_entry_price = validation._safe_float_attr(order, "price_open")
            candidate_stop_loss = None if request_sl == 0.0 else float(request_sl)
            current_stop_loss = existing_sl
            side = "BUY" if int(order_type_value) in {
                validation._safe_int_attr(mt5, "ORDER_TYPE_BUY_LIMIT", 2),
                validation._safe_int_attr(mt5, "ORDER_TYPE_BUY_STOP", 4),
            } else "SELL"
            if (
                trade_guardrails_config.is_enabled()
                and order_volume is not None
                and pending_order_risk_increased(
                    symbol_info=symbol_info,
                    side=side,
                    volume=float(order_volume),
                    existing_entry_price=existing_entry_price,
                    existing_stop_loss=current_stop_loss,
                    candidate_entry_price=float(normalized_price),
                    candidate_stop_loss=candidate_stop_loss,
                )
            ):
                try:
                    account_info = mt5.account_info()
                except Exception:
                    account_info = None
                try:
                    positions = mt5.positions_get()
                except Exception:
                    positions = None
                guardrail_block = evaluate_trade_guardrails(
                    trade_guardrails_config,
                    symbol=order.symbol,
                    volume=float(order_volume),
                    stop_loss=candidate_stop_loss,
                    deviation=None,
                    side=side,
                    entry_price=float(normalized_price),
                    account_info=account_info,
                    existing_positions=list(positions or []),
                    symbol_info=symbol_info,
                    symbol_info_resolver=mt5.symbol_info,
                    enforce_symbol_rules=False,
                )
                if guardrail_block is not None:
                    guardrail_block["pending_order_ticket"] = resolved_ticket
                    guardrail_block["ticket_requested"] = ticket_id
                    guardrail_block["ticket_resolution"] = ticket_resolution
                    return guardrail_block

            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": ticket_id,
                "symbol": order.symbol,
                "volume": validation._safe_float_attr(order, "volume"),
                "type": order_type_value,
                "price": float(normalized_price),
                "sl": request_sl,
                "tp": request_tp,
                "comment": comments._normalize_trade_comment(comment, default="MCP modify pending order"),
            }
            request_magic = validation._safe_int_ticket(getattr(order, "magic", None))
            if request_magic is not None:
                request["magic"] = request_magic

            if expiration_specified:
                if normalized_expiration is None:
                    request["type_time"] = mt5.ORDER_TIME_GTC
                else:
                    request["type_time"] = mt5.ORDER_TIME_SPECIFIED
                    request["expiration"] = normalized_expiration
            else:
                current_type_time = getattr(order, "type_time", None)
                current_expiration = getattr(order, "time_expiration", None)
                if current_type_time is not None:
                    request["type_time"] = current_type_time
                    if current_type_time == mt5.ORDER_TIME_SPECIFIED and current_expiration:
                        try:
                            request["expiration"] = int(current_expiration)
                        except Exception:
                            if isinstance(current_expiration, datetime):
                                server_dt = time._to_server_time_naive(current_expiration)
                                request["expiration"] = time._server_time_naive_to_mt5_timestamp(server_dt)

            result, comment_fallback, last_error = comments._send_order_with_comment_fallback(
                mt5,
                request,
            )
            if result is None:
                out = {"error": "Failed to modify pending order", "last_error": last_error}
                if isinstance(comment_fallback, dict):
                    out["comment_fallback"] = comment_fallback
                return out

            if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
                out = {
                    "error": "Failed to modify pending order",
                    "retcode": result.retcode,
                    "retcode_name": mt5.retcode_name(result.retcode),
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "last_error": last_error,
                }
                if isinstance(comment_fallback, dict):
                    out["comment_fallback"] = comment_fallback
                return out

            out = {
                "success": True,
                "retcode": result.retcode,
                "retcode_name": mt5.retcode_name(result.retcode),
                "deal": result.deal,
                "order": result.order,
                "comment": result.comment,
                "request_id": result.request_id,
                "applied_price": request.get("price"),
                "applied_sl": request.get("sl"),
                "applied_tp": request.get("tp"),
                "applied_expiration": request.get("expiration"),
                "pending_order_ticket": resolved_ticket,
                "ticket_requested": ticket_id,
                "ticket_resolution": ticket_resolution,
            }
            if isinstance(comment_fallback, dict):
                out["comment_fallback"] = comment_fallback
                if comment_fallback.get("used"):
                    out["warnings"] = [
                        "Broker rejected the comment field; pending order was retried with a minimal MT5-safe comment."
                    ]
            return out

        except Exception as e:
            return _unexpected_operation_error(
                "modifying pending order",
                e,
                mt5=mt5,
                context={"ticket": ticket},
            )

    return _modify_pending_order()


_CLOSE_ABORT_CONSECUTIVE_FAILURES = 3


def _execute_single_close(
    mt5: Any,
    position: Any,
    *,
    requested_volume: Optional[float],
    position_volume_before: Optional[float],
    remaining_volume_estimate: Optional[float],
    deviation: int,
    comment: Optional[str],
    fill_modes: List[Any],
) -> Dict[str, Any]:
    """Execute a close for a single position with fill-mode retry.

    Returns a result dict suitable for inclusion in the bulk-close results list.
    Handles tick refresh on requote/price-change and comment fallback.
    """
    position_side = _resolve_position_side(position, mt5)
    if position_side is None:
        return {
            "ticket": position.ticket,
            "error": "Unable to determine position side for close request.",
        }
    is_buy_position = position_side == "BUY"

    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        tick_error = validation._safe_last_error(mt5)
        return {
            "ticket": position.ticket,
            "error": f"Failed to get tick data for {position.symbol}",
            "last_error": tick_error,
            "attempts": [
                {
                    "error": f"Failed to get tick data for {position.symbol}",
                    "last_error": tick_error,
                }
            ],
        }

    close_type_buy = getattr(mt5, "ORDER_TYPE_BUY", 0)
    close_type_sell = getattr(mt5, "ORDER_TYPE_SELL", 1)
    result = None
    request = None
    last_send_error = None
    close_comment_fallback = None
    attempts: List[Dict[str, Any]] = []

    price_changed_codes = {
        validation._safe_int_attr(mt5, "TRADE_RETCODE_PRICE_CHANGED", 10020),
        validation._safe_int_attr(mt5, "TRADE_RETCODE_REQUOTE", 10004),
    }
    for fill_mode in fill_modes:
        price_retry_count = 0
        while True:
            close_price = (
                validation._safe_float_attr(tick, "bid")
                if is_buy_position
                else validation._safe_float_attr(tick, "ask")
            )
            close_type = close_type_sell if is_buy_position else close_type_buy
            close_comment = comments._normalize_close_trade_comment(comment, default="MCP close")

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": position.ticket,
                "symbol": position.symbol,
                "volume": requested_volume if requested_volume is not None else position.volume,
                "type": close_type,
                "price": close_price,
                "deviation": deviation,
                "comment": close_comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": int(fill_mode),
            }
            request_magic = validation._safe_int_ticket(getattr(position, "magic", None))
            if request_magic is not None:
                request["magic"] = request_magic

            result, attempt_comment_fallback, last_send_error = comments._send_order_with_comment_fallback(
                mt5,
                request,
            )
            if isinstance(attempt_comment_fallback, dict):
                close_comment_fallback = attempt_comment_fallback
            if result is None:
                attempts.append(
                    {
                        "type_filling": int(fill_mode),
                        "error": "Failed to send close order",
                        "last_error": last_send_error,
                    }
                )
                if isinstance(attempt_comment_fallback, dict):
                    attempts[-1]["comment_fallback"] = attempt_comment_fallback
                _stdlib_time.sleep(0.15)
                break

            retcode_val = getattr(result, "retcode", None)
            attempt: Dict[str, Any] = {
                "type_filling": int(fill_mode),
                "retcode": retcode_val,
                "retcode_name": mt5.retcode_name(retcode_val),
                "comment": getattr(result, "comment", None),
            }
            if price_retry_count > 0:
                attempt["price_retry"] = int(price_retry_count)
            attempts.append(attempt)
            if isinstance(attempt_comment_fallback, dict):
                attempts[-1]["comment_fallback"] = attempt_comment_fallback
            if validation._retcode_is_done(mt5, retcode_val):
                break
            if retcode_val not in price_changed_codes or price_retry_count >= 3:
                _stdlib_time.sleep(0.15)
                break
            refreshed_tick = mt5.symbol_info_tick(position.symbol)
            if refreshed_tick is None:
                _stdlib_time.sleep(0.15)
                break
            tick = refreshed_tick
            price_retry_count += 1
            _stdlib_time.sleep(0.15)
        if validation._retcode_is_done(mt5, getattr(result, "retcode", None) if result is not None else None):
            break

    close_ok = (
        validation._retcode_is_done(mt5, getattr(result, "retcode", None))
        if result is not None
        else False
    )

    if not close_ok:
        tick_failures = [
            a for a in attempts if "tick data" in str(a.get("error", "")).lower()
        ]
        error_msg = (
            f"Failed to get tick data for {position.symbol}"
            if attempts and len(tick_failures) == len(attempts)
            else "Failed to send close order"
        )
        fail_result: Dict[str, Any] = {
            "ticket": position.ticket,
            "error": error_msg,
            "attempts": attempts,
            "last_error": last_send_error,
        }
        if isinstance(close_comment_fallback, dict):
            fail_result["comment_fallback"] = close_comment_fallback
        return fail_result

    # Build success result with PnL metadata
    open_price = getattr(position, "price_open", None)
    try:
        open_price = float(open_price) if open_price is not None else None
    except Exception:
        open_price = None
    close_exec_price = getattr(result, "price", None)
    try:
        close_exec_price = float(close_exec_price) if close_exec_price is not None else None
    except Exception:
        close_exec_price = None
    close_observed_at_utc = datetime.now(timezone.utc)
    open_epoch = getattr(position, "time", None)
    try:
        open_epoch_utc = _mt5_epoch_to_utc(float(open_epoch)) if open_epoch is not None else None
    except Exception:
        open_epoch_utc = None
    duration_seconds = None
    if open_epoch_utc is not None:
        try:
            duration_seconds = int(
                max(0.0, close_observed_at_utc.timestamp() - float(open_epoch_utc))
            )
        except Exception:
            duration_seconds = None

    realized_pnl = getattr(result, "profit", None)
    try:
        realized_pnl = float(realized_pnl) if realized_pnl is not None else None
    except Exception:
        realized_pnl = None
    if realized_pnl is None:
        history_deal = _resolve_closed_deal_from_history(
            mt5,
            result=result,
            position=position,
            closed_at_utc=close_observed_at_utc,
        )
        if history_deal is not None:
            try:
                realized_pnl = float(getattr(history_deal, "profit", None))
            except Exception:
                realized_pnl = None

    pnl_price_delta = None
    if open_price is not None and close_exec_price is not None:
        try:
            if is_buy_position:
                pnl_price_delta = close_exec_price - open_price
            else:
                pnl_price_delta = open_price - close_exec_price
        except Exception:
            pnl_price_delta = None

    res_dict: Dict[str, Any] = {
        "ticket": position.ticket,
        "retcode": result.retcode,
        "retcode_name": mt5.retcode_name(result.retcode),
        "deal": result.deal,
        "order": result.order,
        "volume": result.volume,
        "price": result.price,
        "comment": result.comment,
        "open_price": open_price,
        "close_price": close_exec_price,
        "pnl": realized_pnl,
        "pnl_price_delta": pnl_price_delta,
        "duration_seconds": duration_seconds,
        "requested_volume": requested_volume if requested_volume is not None else position_volume_before,
        "position_volume_before": position_volume_before,
        "position_volume_remaining_estimate": remaining_volume_estimate,
        "attempts": attempts,
    }
    if isinstance(close_comment_fallback, dict):
        res_dict["comment_fallback"] = close_comment_fallback
        if close_comment_fallback.get("used"):
            res_dict["warnings"] = [
                "Broker rejected the comment field; close order was retried with a minimal MT5-safe comment."
            ]
    return res_dict


def _sort_close_positions(
    positions: List[Any],
    priority: Optional[str],
) -> List[Any]:
    """Sort positions for close ordering. Returns a new list.

    Supported priorities:
    - None: discovery order (no change)
    - "loss_first": most negative PnL first (close losers first)
    - "profit_first": most positive PnL first (take profits first)
    - "largest_first": largest notional volume first
    """
    if not priority or not positions:
        return list(positions)

    def _safe_profit(pos: Any) -> float:
        return validation._safe_float_attr(pos, "profit")

    def _safe_volume(pos: Any) -> float:
        return validation._safe_float_attr(pos, "volume")

    if priority == "loss_first":
        return sorted(positions, key=_safe_profit)
    elif priority == "profit_first":
        return sorted(positions, key=_safe_profit, reverse=True)
    elif priority == "largest_first":
        return sorted(positions, key=_safe_volume, reverse=True)
    return list(positions)


def _close_positions(  # noqa: C901
    ticket: Optional[Union[int, str]] = None,
    symbol: Optional[str] = None,
    volume: Optional[Union[int, float]] = None,
    profit_only: bool = False,
    loss_only: bool = False,
    close_priority: Optional[str] = None,
    comment: Optional[str] = None,
    deviation: int = 20,
    gateway: Optional[MT5TradingGateway] = None,
) -> dict:
    """Internal helper to close open positions."""
    mt5 = create_trading_gateway(
        gateway=gateway,
        include_retcode_name=True,
    )

    connection_error = trading_connection_error(mt5)
    if connection_error is not None:
        return connection_error

    def _close_positions():  # noqa: C901
        try:
            if profit_only and loss_only:
                return {"error": "profit_only and loss_only cannot both be true."}

            # 1. Fetch positions based on criteria
            requested_ticket = None
            resolved_ticket = None
            ticket_resolution = None
            if ticket is not None:
                t_int = int(ticket)
                requested_ticket = t_int
                position, resolved_ticket, ticket_resolution = _resolve_open_position(
                    mt5,
                    ticket_candidates=[t_int],
                    require_exact_ticket_match=True,
                )
                if position is None:
                    out = {"error": f"Position {ticket} not found", "checked_scopes": ["positions"]}
                    if isinstance(ticket_resolution, dict):
                        out["ticket_resolution"] = ticket_resolution
                    return out
                positions = [position]
            elif symbol is not None:
                positions = mt5.positions_get(symbol=symbol)
                if positions is None or len(positions) == 0:
                    return {"message": f"No open positions for {symbol}"}
            else:
                positions = mt5.positions_get()
                if positions is None or len(positions) == 0:
                    return {"message": "No open positions"}

            # 2. Filter positions
            to_close = []
            for pos in positions:
                position_profit = validation._safe_float_attr(pos, "profit")
                if profit_only and position_profit <= 0.0:
                    continue
                if loss_only and position_profit >= 0.0:
                    continue
                to_close.append(pos)

            if not to_close:
                return {"message": "No positions matched criteria"}

            to_close = _sort_close_positions(to_close, close_priority)

            deviation_validated, deviation_error = validation._validate_deviation(deviation)
            if deviation_error:
                return {"error": deviation_error}

            # 3. Close positions
            results = []
            consecutive_failures = 0
            for position in to_close:
                # Abort early if consecutive transport/broker failures suggest
                # a connection problem rather than per-position issues.
                if consecutive_failures >= _CLOSE_ABORT_CONSECUTIVE_FAILURES:
                    results.append({
                        "ticket": getattr(position, "ticket", None),
                        "error": (
                            f"Skipped: {consecutive_failures} consecutive close failures "
                            "suggest a broker/transport problem."
                        ),
                        "aborted": True,
                    })
                    continue

                # Re-read position to check it still exists
                try:
                    fresh_positions = mt5.positions_get(ticket=position.ticket)
                except Exception:
                    fresh_positions = None
                if not fresh_positions:
                    results.append(
                        {
                            "ticket": position.ticket,
                            "error": "Position no longer open.",
                        }
                    )
                    # Disappearing position is not a transport failure
                    continue
                position = fresh_positions[0]

                symbol_info = None
                try:
                    symbol_info = mt5.symbol_info(position.symbol)
                except Exception:
                    symbol_info = None

                # Volume validation for partial closes
                requested_volume = None
                position_volume_before = None
                remaining_volume_estimate = None
                position_volume_before = validation._safe_float_attr(position, "volume") or None
                if volume is not None:
                    if symbol_info is None:
                        results.append({
                            "ticket": position.ticket,
                            "error": f"Failed to get symbol info for {position.symbol}",
                        })
                        continue
                    requested_volume, requested_volume_error = validation._validate_volume(
                        volume,
                        symbol_info,
                    )
                    if requested_volume_error:
                        results.append({
                            "ticket": position.ticket,
                            "error": requested_volume_error,
                            "requested_volume": volume,
                        })
                        continue
                    if (
                        position_volume_before is None
                        or not math.isfinite(position_volume_before)
                        or position_volume_before <= 0
                    ):
                        results.append({
                            "ticket": position.ticket,
                            "error": "Open position volume is invalid for partial close.",
                            "requested_volume": requested_volume,
                        })
                        continue
                    if requested_volume > (position_volume_before + 1e-12):
                        results.append({
                            "ticket": position.ticket,
                            "error": (
                                f"volume must be <= open position volume ({position_volume_before:g})"
                            ),
                            "requested_volume": requested_volume,
                            "position_volume": position_volume_before,
                        })
                        continue
                    remaining_volume_estimate = max(0.0, position_volume_before - requested_volume)
                    if remaining_volume_estimate > 1e-12:
                        _, remaining_error = validation._validate_volume(
                            remaining_volume_estimate,
                            symbol_info,
                        )
                        if remaining_error:
                            results.append({
                                "ticket": position.ticket,
                                "error": (
                                    "remaining position volume would be invalid after partial close: "
                                    f"{remaining_error}"
                                ),
                                "requested_volume": requested_volume,
                                "position_volume": position_volume_before,
                                "remaining_volume": remaining_volume_estimate,
                            })
                            continue

                fill_modes = validation._candidate_fill_modes(mt5, symbol_info)
                close_result = _execute_single_close(
                    mt5,
                    position,
                    requested_volume=requested_volume,
                    position_volume_before=position_volume_before,
                    remaining_volume_estimate=remaining_volume_estimate,
                    deviation=deviation_validated,
                    comment=comment,
                    fill_modes=fill_modes,
                )
                if requested_ticket is not None and isinstance(close_result, dict):
                    close_result.setdefault("ticket_requested", requested_ticket)
                    if resolved_ticket is not None:
                        close_result.setdefault("ticket_resolved", resolved_ticket)
                    if isinstance(ticket_resolution, dict):
                        close_result.setdefault("ticket_resolution", ticket_resolution)
                results.append(close_result)

                # Track consecutive failures for abort policy
                if close_result.get("error"):
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0

            # If only one position was targeted by ticket, return single result
            if ticket is not None and len(results) == 1:
                return results[0]
            success_count = _count_done_results(mt5, results)
            bulk_result: Dict[str, Any] = {
                "closed_count": success_count,
                "attempted_count": len(results),
                "results": results,
            }
            if close_priority:
                bulk_result["close_priority"] = close_priority
            return bulk_result

        except Exception as e:
            if "results" not in locals():
                return {"error": str(e)}

            error_result: Dict[str, Any] = {"error": str(e)}
            current_position = locals().get("position")
            current_ticket = validation._safe_int_ticket(getattr(current_position, "ticket", None))
            if current_ticket is None and current_position is not None:
                current_ticket = getattr(current_position, "ticket", None)
            if current_ticket is not None:
                error_result["ticket"] = current_ticket

            last_error = validation._safe_last_error(mt5)
            if last_error is not None:
                error_result["last_error"] = last_error

            if current_ticket is None or not any(
                isinstance(result_item, dict) and result_item.get("ticket") == current_ticket
                for result_item in results
            ):
                results.append(error_result)

            if ticket is not None and len(results) == 1:
                return results[0]

            return {
                "closed_count": _count_done_results(mt5, results),
                "attempted_count": len(results),
                "partial_failure": True,
                "error": str(e),
                "results": results,
            }

    return _close_positions()


def _cancel_pending(
    ticket: Optional[Union[int, str]] = None,
    symbol: Optional[str] = None,
    comment: Optional[str] = None,
    gateway: Optional[MT5TradingGateway] = None,
) -> dict:
    """Internal helper to cancel pending orders."""
    mt5 = create_trading_gateway(
        gateway=gateway,
        include_retcode_name=True,
    )

    connection_error = trading_connection_error(mt5)
    if connection_error is not None:
        return connection_error

    def _cancel_pending():
        try:
            # 1. Fetch orders based on criteria
            if ticket is not None:
                t_int = int(ticket)
                requested_ticket = t_int
                order, resolved_ticket, ticket_resolution = _resolve_pending_order(
                    mt5,
                    ticket_candidates=[t_int],
                    require_exact_ticket_match=True,
                )
                if order is None:
                    out = {"error": f"Pending order {ticket} not found", "checked_scopes": ["pending_orders"]}
                    if isinstance(ticket_resolution, dict):
                        out["ticket_resolution"] = ticket_resolution
                    return out
                orders = [order]
            elif symbol is not None:
                orders = mt5.orders_get(symbol=symbol)
                if orders is None or len(orders) == 0:
                    return {"message": f"No pending orders for {symbol}"}
            else:
                orders = mt5.orders_get()
                if orders is None or len(orders) == 0:
                    return {"message": "No pending orders"}

            # 2. Cancel orders
            results = []
            for order in orders:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                    "comment": comments._normalize_trade_comment(comment, default="MCP cancel pending order"),
                }
                request_magic = validation._safe_int_ticket(getattr(order, "magic", None))
                if request_magic is not None:
                    request["magic"] = request_magic

                result, comment_fallback, last_error = comments._send_order_with_comment_fallback(
                    mt5,
                    request,
                )
                if result is None:
                    result_entry = {
                        "ticket": order.ticket,
                        "error": "Failed to send cancel order",
                        "last_error": last_error,
                    }
                    if isinstance(comment_fallback, dict):
                        result_entry["comment_fallback"] = comment_fallback
                    results.append(result_entry)
                else:
                    result_entry = {
                        "ticket": order.ticket,
                        "retcode": result.retcode,
                        "retcode_name": mt5.retcode_name(result.retcode),
                        "deal": result.deal,
                        "order": result.order,
                        "comment": result.comment,
                    }
                    if isinstance(comment_fallback, dict):
                        result_entry["comment_fallback"] = comment_fallback
                    if ticket is not None:
                        result_entry.setdefault("ticket_requested", requested_ticket)
                        if resolved_ticket is not None:
                            result_entry.setdefault("ticket_resolved", resolved_ticket)
                        if isinstance(ticket_resolution, dict):
                            result_entry.setdefault("ticket_resolution", ticket_resolution)
                    results.append(result_entry)

            # If only one order was targeted by ticket, return single result
            if ticket is not None and len(results) == 1:
                return results[0]
            success_count = _count_done_results(mt5, results)
            return {"cancelled_count": success_count, "attempted_count": len(results), "results": results}

        except Exception as e:
            return {"error": str(e)}

    return _cancel_pending()
