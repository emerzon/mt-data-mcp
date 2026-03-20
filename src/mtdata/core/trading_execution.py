"""Trade modification and closure workflows for MetaTrader integration."""

import math
import time
import traceback
from datetime import datetime, timezone
from typing import Optional, Union, List, Dict, Any

from . import trading_comments, trading_time, trading_validation
from .trading_gateway import MT5TradingGateway, create_trading_gateway, trading_connection_error
from .trading_positions import _resolve_open_position
from .trading_time import ExpirationValue
from ..utils.mt5 import _mt5_epoch_to_utc
from ..utils.mt5 import ensure_mt5_connection_or_raise


def _safe_last_error(mt5: Any) -> Any:
    try:
        if hasattr(mt5, "last_error"):
            return mt5.last_error()
    except Exception:
        return None
    return None


def _zero_price_requested(value: Optional[Union[int, float]]) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except Exception:
        return False
    return math.isfinite(numeric) and math.isclose(numeric, 0.0, abs_tol=1e-9)


def _resolve_position_side(position: Any, mt5: Any) -> Optional[str]:
    position_type_buy = trading_validation._safe_int_attr(
        mt5,
        "POSITION_TYPE_BUY",
        trading_validation._safe_int_attr(mt5, "ORDER_TYPE_BUY", 0),
    )
    position_type_sell = trading_validation._safe_int_attr(
        mt5,
        "POSITION_TYPE_SELL",
        trading_validation._safe_int_attr(mt5, "ORDER_TYPE_SELL", 1),
    )
    try:
        position_type = int(getattr(position, "type"))
    except Exception:
        return None
    if position_type == int(position_type_buy):
        return "BUY"
    if position_type == int(position_type_sell):
        return "SELL"
    return None


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
    last_error = _safe_last_error(mt5)
    if last_error is not None:
        payload["last_error"] = last_error
    if context:
        payload["context"] = context
    return payload


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
            )
            if position is None or resolved_ticket is None:
                return {"error": f"Position {ticket} not found", "checked_scopes": ["positions"]}

            # Get symbol info for price normalization
            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info is None:
                return {"error": f"Failed to get symbol info for {position.symbol}"}

            point = float(symbol_info.point or 0.0) if hasattr(symbol_info, "point") else 0.0
            digits = trading_validation._safe_int_attr(symbol_info, "digits", 5)

            explicit_remove_sl = _zero_price_requested(stop_loss)
            explicit_remove_tp = _zero_price_requested(take_profit)
            requested_sl = (
                None
                if stop_loss is None or explicit_remove_sl
                else trading_validation._normalize_price_for_symbol(stop_loss, point=point, digits=digits)
            )
            requested_tp = (
                None
                if take_profit is None or explicit_remove_tp
                else trading_validation._normalize_price_for_symbol(take_profit, point=point, digits=digits)
            )
            if stop_loss is not None and not explicit_remove_sl and requested_sl is None:
                return {"error": "stop_loss must be a non-zero finite price after symbol normalization."}
            if take_profit is not None and not explicit_remove_tp and requested_tp is None:
                return {"error": "take_profit must be a non-zero finite price after symbol normalization."}

            # Normalize SL/TP values
            existing_sl = trading_validation._normalize_price_for_symbol(
                getattr(position, "sl", None),
                point=point,
                digits=digits,
            )
            existing_tp = trading_validation._normalize_price_for_symbol(
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
            validate_sl = None if stop_loss is None or explicit_remove_sl else float(requested_sl)
            validate_tp = None if take_profit is None or explicit_remove_tp else float(requested_tp)

            side = _resolve_position_side(position, mt5)
            if side is None:
                return {"error": "Unable to determine position side for protection validation."}
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return {"error": f"Failed to get current price for {position.symbol}"}
            live_protection_error = trading_validation._validate_live_protection_levels(
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
                "comment": trading_comments._normalize_trade_comment(comment, default="MCP modify position"),
            }
            request_magic = trading_validation._safe_int_ticket(getattr(position, "magic", None))
            if request_magic is not None:
                request["magic"] = request_magic

            result = mt5.order_send(request)
            if result is None:
                # surface the MT5 terminal error for debugging
                try:
                    last_err = mt5.last_error()
                except Exception:
                    last_err = None
                return {"error": "Failed to modify position", "last_error": last_err}

            if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
                return {
                    "error": "Failed to modify position",
                    "retcode": result.retcode,
                    "retcode_name": mt5.retcode_name(result.retcode),
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "last_error": mt5.last_error() if hasattr(mt5, "last_error") else None,
                }

            return {
                "success": True,
                "retcode": result.retcode,
                "retcode_name": mt5.retcode_name(result.retcode),
                "deal": result.deal,
                "order": result.order,
                "comment": result.comment,
                "request_id": result.request_id,
                "position_ticket": resolved_ticket,
                "ticket_requested": ticket_id,
                "ticket_resolution": ticket_resolution,
                "applied_sl": None if float(norm_sl) == 0.0 else float(norm_sl),
                "applied_tp": None if float(norm_tp) == 0.0 else float(norm_tp),
            }

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
            orders = mt5.orders_get(ticket=ticket_id)
            if orders is None or len(orders) == 0:
                return {"error": f"Pending order {ticket} not found", "checked_scopes": ["pending_orders"]}

            order = orders[0]
            normalized_expiration, expiration_specified = trading_time._normalize_pending_expiration(expiration)
            symbol_info = mt5.symbol_info(order.symbol)
            if symbol_info is None:
                return {"error": f"Failed to get symbol info for {order.symbol}"}
            tick = mt5.symbol_info_tick(order.symbol)
            if tick is None:
                return {"error": f"Failed to get current price for {order.symbol}"}
            point = float(getattr(symbol_info, "point", 0.0) or 0.0)
            digits = trading_validation._safe_int_attr(symbol_info, "digits", 5)

            explicit_remove_sl = _zero_price_requested(stop_loss)
            explicit_remove_tp = _zero_price_requested(take_profit)
            normalized_price = trading_validation._normalize_price_for_symbol(
                price if price is not None else getattr(order, "price_open", None),
                point=point,
                digits=digits,
            )
            if normalized_price is None:
                return {"error": "price must be a non-zero finite number after symbol normalization."}

            requested_sl = (
                None
                if stop_loss is None or explicit_remove_sl
                else trading_validation._normalize_price_for_symbol(stop_loss, point=point, digits=digits)
            )
            requested_tp = (
                None
                if take_profit is None or explicit_remove_tp
                else trading_validation._normalize_price_for_symbol(take_profit, point=point, digits=digits)
            )
            if stop_loss is not None and not explicit_remove_sl and requested_sl is None:
                return {"error": "stop_loss must be a non-zero finite price after symbol normalization."}
            if take_profit is not None and not explicit_remove_tp and requested_tp is None:
                return {"error": "take_profit must be a non-zero finite price after symbol normalization."}

            existing_sl = trading_validation._normalize_price_for_symbol(
                getattr(order, "sl", None),
                point=point,
                digits=digits,
            )
            existing_tp = trading_validation._normalize_price_for_symbol(
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

            order_type_value = trading_validation._safe_int_attr(order, "type", -1)
            pending_level_error = trading_validation._validate_pending_order_levels(
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

            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": ticket_id,
                "symbol": order.symbol,
                "volume": float(getattr(order, "volume", 0.0) or 0.0),
                "type": order_type_value,
                "price": float(normalized_price),
                "sl": request_sl,
                "tp": request_tp,
                "comment": trading_comments._normalize_trade_comment(comment, default="MCP modify pending order"),
            }
            request_magic = trading_validation._safe_int_ticket(getattr(order, "magic", None))
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
                                server_dt = trading_time._to_server_time_naive(current_expiration)
                                request["expiration"] = trading_time._server_time_naive_to_mt5_timestamp(server_dt)

            result = mt5.order_send(request)
            if result is None:
                try:
                    last_err = mt5.last_error()
                except Exception:
                    last_err = None
                return {"error": "Failed to modify pending order", "last_error": last_err}

            if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
                return {
                    "error": "Failed to modify pending order",
                    "retcode": result.retcode,
                    "retcode_name": mt5.retcode_name(result.retcode),
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "last_error": mt5.last_error() if hasattr(mt5, "last_error") else None,
                }

            return {
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
            }

        except Exception as e:
            return _unexpected_operation_error(
                "modifying pending order",
                e,
                mt5=mt5,
                context={"ticket": ticket},
            )

    return _modify_pending_order()


def _close_positions(
    ticket: Optional[Union[int, str]] = None,
    symbol: Optional[str] = None,
    volume: Optional[Union[int, float]] = None,
    profit_only: bool = False,
    loss_only: bool = False,
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

    def _close_positions():
        try:
            if profit_only and loss_only:
                return {"error": "profit_only and loss_only cannot both be true."}

            # 1. Fetch positions based on criteria
            if ticket is not None:
                t_int = int(ticket)
                positions = mt5.positions_get(ticket=t_int)
                if positions is None or len(positions) == 0:
                    return {"error": f"Position {ticket} not found", "checked_scopes": ["positions"]}
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
                position_profit: Optional[float]
                try:
                    position_profit = float(getattr(pos, "profit", 0.0) or 0.0)
                except Exception:
                    position_profit = None
                if position_profit is None and (profit_only or loss_only):
                    continue
                if profit_only and position_profit <= 0.0:
                    continue
                if loss_only and position_profit >= 0.0:
                    continue
                to_close.append(pos)

            if not to_close:
                return {"message": "No positions matched criteria"}

            deviation_validated, deviation_error = trading_validation._validate_deviation(deviation)
            if deviation_error:
                return {"error": deviation_error}

            # 3. Close positions
            results = []
            for position in to_close:
                requested_volume = None
                position_volume_before = None
                remaining_volume_estimate = None
                try:
                    position_volume_before = float(getattr(position, "volume", 0.0) or 0.0)
                except Exception:
                    position_volume_before = None
                if volume is not None:
                    symbol_info = mt5.symbol_info(position.symbol)
                    if symbol_info is None:
                        results.append({
                            "ticket": position.ticket,
                            "error": f"Failed to get symbol info for {position.symbol}",
                        })
                        continue
                    requested_volume, requested_volume_error = trading_validation._validate_volume(
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
                        _, remaining_error = trading_validation._validate_volume(
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

                fill_modes: List[int] = []
                for fill_attr in ("ORDER_FILLING_IOC", "ORDER_FILLING_FOK", "ORDER_FILLING_RETURN"):
                    if hasattr(mt5, fill_attr):
                        try:
                            fill_val = int(getattr(mt5, fill_attr))
                        except Exception:
                            continue
                        if fill_val not in fill_modes:
                            fill_modes.append(fill_val)
                if not fill_modes:
                    fill_modes = [1, 0, 2]

                result = None
                request = None
                attempts: List[Dict[str, Any]] = []
                position_type_buy = getattr(mt5, "POSITION_TYPE_BUY", getattr(mt5, "ORDER_TYPE_BUY", 0))
                close_type_buy = getattr(mt5, "ORDER_TYPE_BUY", 0)
                close_type_sell = getattr(mt5, "ORDER_TYPE_SELL", 1)
                done_codes = {
                    int(getattr(mt5, "TRADE_RETCODE_DONE", 10009)),
                    int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)),
                }

                for fill_mode in fill_modes:
                    tick = mt5.symbol_info_tick(position.symbol)
                    if tick is None:
                        tick_error = None
                        try:
                            tick_error = mt5.last_error() if hasattr(mt5, "last_error") else None
                        except Exception:
                            tick_error = None
                        attempts.append(
                            {
                                "type_filling": int(fill_mode),
                                "error": f"Failed to get tick data for {position.symbol}",
                                "last_error": tick_error,
                            }
                        )
                        continue

                    try:
                        is_buy_position = int(getattr(position, "type", position_type_buy)) == int(position_type_buy)
                    except Exception:
                        is_buy_position = True
                    close_price = float(getattr(tick, "bid", 0.0) or 0.0) if is_buy_position else float(
                        getattr(tick, "ask", 0.0) or 0.0
                    )
                    close_type = close_type_sell if is_buy_position else close_type_buy
                    close_comment = trading_comments._normalize_trade_comment(comment, default="MCP close")
                    # Some brokers reject edge-length comments during close-deal requests.
                    if len(close_comment) > 24:
                        close_comment = close_comment[:24]

                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": position.ticket,
                        "symbol": position.symbol,
                        "volume": requested_volume if requested_volume is not None else position.volume,
                        "type": close_type,
                        "price": close_price,
                        "deviation": deviation_validated,
                        "magic": 234000,
                        "comment": close_comment,
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": int(fill_mode),
                    }

                    result = mt5.order_send(request)
                    if result is None:
                        send_error = None
                        try:
                            send_error = mt5.last_error() if hasattr(mt5, "last_error") else None
                        except Exception:
                            send_error = None
                        send_error_text = str(send_error).lower() if send_error is not None else ""
                        # Retry with a minimal/no comment when broker rejects the comment field.
                        if "invalid" in send_error_text and "comment" in send_error_text:
                            alt_requests: List[Dict[str, Any]] = []
                            req_short = dict(request)
                            req_short["comment"] = "MCP"
                            alt_requests.append(req_short)
                            req_nocomment = dict(request)
                            req_nocomment.pop("comment", None)
                            alt_requests.append(req_nocomment)
                            recovered = False
                            for alt_req in alt_requests:
                                alt_res = mt5.order_send(alt_req)
                                if alt_res is None:
                                    alt_err = None
                                    try:
                                        alt_err = mt5.last_error() if hasattr(mt5, "last_error") else None
                                    except Exception:
                                        alt_err = None
                                    attempts.append(
                                        {
                                            "type_filling": int(fill_mode),
                                            "error": "Failed to send close order",
                                            "last_error": alt_err,
                                            "comment_fallback": True,
                                        }
                                    )
                                    continue
                                attempts.append(
                                    {
                                        "type_filling": int(fill_mode),
                                        "retcode": getattr(alt_res, "retcode", None),
                                        "retcode_name": mt5.retcode_name(getattr(alt_res, "retcode", None)),
                                        "comment": getattr(alt_res, "comment", None),
                                        "comment_fallback": True,
                                    }
                                )
                                result = alt_res
                                request = alt_req
                                recovered = True
                                break
                            if recovered:
                                retcode_val = getattr(result, "retcode", None)
                                try:
                                    if retcode_val is not None and int(retcode_val) in done_codes:
                                        break
                                except Exception:
                                    pass
                                time.sleep(0.15)
                                continue
                        attempts.append(
                            {
                                "type_filling": int(fill_mode),
                                "error": "Failed to send close order",
                                "last_error": send_error,
                            }
                        )
                        time.sleep(0.15)
                        continue

                    retcode_val = getattr(result, "retcode", None)
                    attempts.append(
                        {
                            "type_filling": int(fill_mode),
                            "retcode": retcode_val,
                            "retcode_name": mt5.retcode_name(retcode_val),
                            "comment": getattr(result, "comment", None),
                        }
                    )
                    try:
                        if retcode_val is not None and int(retcode_val) in done_codes:
                            break
                    except Exception:
                        pass
                    time.sleep(0.15)

                close_ok = False
                if result is not None:
                    try:
                        close_ok = int(getattr(result, "retcode", -1)) in done_codes
                    except Exception:
                        close_ok = False

                if not close_ok:
                    last_error = None
                    try:
                        last_error = mt5.last_error() if hasattr(mt5, "last_error") else None
                    except Exception:
                        last_error = None
                    tick_failures = [
                        a for a in attempts if "tick data" in str(a.get("error", "")).lower()
                    ]
                    if attempts and len(tick_failures) == len(attempts):
                        error_msg = f"Failed to get tick data for {position.symbol}"
                    else:
                        error_msg = "Failed to send close order"
                    results.append(
                        {
                            "ticket": position.ticket,
                            "error": error_msg,
                            "attempts": attempts,
                            "last_error": last_error,
                        }
                    )
                    continue

                if result is not None:
                    open_price = getattr(position, "price_open", None)
                    try:
                        open_price = float(open_price) if open_price is not None else None
                    except Exception:
                        open_price = None
                    close_exec_price = getattr(result, "price", close_price)
                    try:
                        close_exec_price = float(close_exec_price) if close_exec_price is not None else None
                    except Exception:
                        close_exec_price = None
                    open_epoch = getattr(position, "time", None)
                    try:
                        open_epoch_utc = _mt5_epoch_to_utc(float(open_epoch)) if open_epoch is not None else None
                    except Exception:
                        open_epoch_utc = None
                    duration_seconds = None
                    if open_epoch_utc is not None:
                        try:
                            duration_seconds = int(
                                max(0.0, datetime.now(timezone.utc).timestamp() - float(open_epoch_utc))
                            )
                        except Exception:
                            duration_seconds = None

                    realized_pnl = getattr(result, "profit", None)
                    try:
                        realized_pnl = float(realized_pnl) if realized_pnl is not None else None
                    except Exception:
                        realized_pnl = None
                    if realized_pnl is None:
                        try:
                            realized_pnl = float(getattr(position, "profit", None))
                        except Exception:
                            realized_pnl = None

                    pnl_price_delta = None
                    if open_price is not None and close_exec_price is not None:
                        try:
                            if int(position.type) == int(position_type_buy):
                                pnl_price_delta = close_exec_price - open_price
                            else:
                                pnl_price_delta = open_price - close_exec_price
                        except Exception:
                            pnl_price_delta = None

                    res_dict = {
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
                    results.append(res_dict)

            # If only one position was targeted by ticket, return single result
            if ticket is not None and len(results) == 1:
                return results[0]
            success_count = 0
            done_codes = {
                int(getattr(mt5, "TRADE_RETCODE_DONE", 10009)),
                int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)),
            }
            for item in results:
                try:
                    if int(item.get("retcode")) in done_codes:
                        success_count += 1
                except Exception:
                    continue
            return {"closed_count": success_count, "attempted_count": len(results), "results": results}

        except Exception as e:
            return {"error": str(e)}

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
                orders = mt5.orders_get(ticket=t_int)
                if orders is None or len(orders) == 0:
                    return {"error": f"Pending order {ticket} not found", "checked_scopes": ["pending_orders"]}
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
                    "magic": 234000,
                    "comment": trading_comments._normalize_trade_comment(comment, default="MCP cancel pending order"),
                }

                result = mt5.order_send(request)
                if result is None:
                    results.append({"ticket": order.ticket, "error": "Failed to send cancel order"})
                else:
                    results.append({
                        "ticket": order.ticket,
                        "retcode": result.retcode,
                        "retcode_name": mt5.retcode_name(result.retcode),
                        "deal": result.deal,
                        "order": result.order,
                        "comment": result.comment,
                    })

            # If only one order was targeted by ticket, return single result
            if ticket is not None and len(results) == 1:
                return results[0]
            success_count = 0
            done_codes = {
                int(getattr(mt5, "TRADE_RETCODE_DONE", 10009)),
                int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)),
            }
            for item in results:
                try:
                    if int(item.get("retcode")) in done_codes:
                        success_count += 1
                except Exception:
                    continue
            return {"cancelled_count": success_count, "attempted_count": len(results), "results": results}

        except Exception as e:
            return {"error": str(e)}

    return _cancel_pending()
