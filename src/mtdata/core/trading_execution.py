"""Trade modification and closure workflows for MetaTrader integration."""

import math
import time
from datetime import datetime, timezone
from typing import Optional, Union, List, Dict, Any

from . import trading_comments, trading_time, trading_validation
from .trading_gateway import MT5TradingGateway, create_trading_gateway, trading_connection_error
from .trading_positions import _resolve_open_position
from .trading_time import ExpirationValue
from ..utils.mt5 import _mt5_epoch_to_utc
from ..utils.mt5 import ensure_mt5_connection_or_raise


def _safe_int_attr(obj: Any, name: str, default: int) -> int:
    try:
        value = getattr(obj, name)
    except Exception:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        try:
            fv = float(value)
        except Exception:
            return default
        if not math.isfinite(fv) or not fv.is_integer():
            return default
        return int(fv)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            fv = float(text)
        except Exception:
            return default
        if not math.isfinite(fv) or not fv.is_integer():
            return default
        return int(fv)
    return default


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
            digits = int(symbol_info.digits) if hasattr(symbol_info, "digits") else 5

            def _normalize_price(val: Optional[Union[int, float]]) -> Optional[float]:
                """Normalize price to symbol precision."""
                try:
                    if val is None or val == 0:
                        return None
                    v = float(val)
                    if not math.isfinite(v):
                        return None
                    if point and point > 0:
                        # Align to symbol precision
                        v = round(v / point) * point
                    else:
                        v = round(v, digits)
                    return v
                except Exception:
                    return None

            # Normalize SL/TP values
            norm_sl = _normalize_price(stop_loss) if stop_loss is not None else (position.sl or 0.0)
            norm_tp = _normalize_price(take_profit) if take_profit is not None else (position.tp or 0.0)

            # Ensure SL/TP values are 0.0 if they should be removed
            if norm_sl is None:
                norm_sl = 0.0
            if norm_tp is None:
                norm_tp = 0.0

            position_type_buy = _safe_int_attr(
                mt5,
                "POSITION_TYPE_BUY",
                _safe_int_attr(mt5, "ORDER_TYPE_BUY", 0),
            )
            try:
                side = "BUY" if int(getattr(position, "type", position_type_buy)) == int(position_type_buy) else "SELL"
            except Exception:
                side = "BUY"
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return {"error": f"Failed to get current price for {position.symbol}"}
            live_protection_error = trading_validation._validate_live_protection_levels(
                symbol_info=symbol_info,
                tick=tick,
                side=side,
                stop_loss=None if float(norm_sl) == 0.0 else float(norm_sl),
                take_profit=None if float(norm_tp) == 0.0 else float(norm_tp),
            )
            if live_protection_error is not None:
                return live_protection_error

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": resolved_ticket,
                "sl": norm_sl,
                "tp": norm_tp,
                "magic": 234000,
                "comment": trading_comments._normalize_trade_comment(comment, default="MCP modify position"),
            }

            result = mt5.order_send(request)
            if result is None:
                # surface the MT5 terminal error for debugging
                try:
                    last_err = mt5.last_error()
                except Exception:
                    last_err = None
                return {"error": "Failed to modify position", "request": request, "last_error": last_err}

            if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
                return {
                    "error": "Failed to modify position",
                    "retcode": result.retcode,
                    "retcode_name": mt5.retcode_name(result.retcode),
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "request": request,
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
            return {"error": str(e)}

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

            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": ticket_id,
                "price": price if price is not None else order.price_open,
                "sl": stop_loss if stop_loss is not None else order.sl,
                "tp": take_profit if take_profit is not None else order.tp,
                "magic": 234000,
                "comment": trading_comments._normalize_trade_comment(comment, default="MCP modify pending order"),
            }

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
                return {"error": "Failed to modify pending order", "request": request, "last_error": last_err}

            if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
                return {
                    "error": "Failed to modify pending order",
                    "retcode": result.retcode,
                    "retcode_name": mt5.retcode_name(result.retcode),
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "request": request,
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
            return {"error": str(e)}

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
                if profit_only and pos.profit <= 0:
                    continue
                if loss_only and pos.profit >= 0:
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
                            "request": request,
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
                            if int(position.type) == int(mt5.ORDER_TYPE_BUY):
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
            for item in results:
                try:
                    if int(item.get("retcode")) == int(mt5.TRADE_RETCODE_DONE):
                        success_count += 1
                except Exception:
                    continue
            return {"cancelled_count": success_count, "attempted_count": len(results), "results": results}

        except Exception as e:
            return {"error": str(e)}

    return _cancel_pending()
