"""Order placement workflows for MetaTrader integration."""

import math
import time
from typing import Optional, Union, List, Dict, Any

from . import trading_comments, trading_time, trading_validation
from .trading_common import _build_trade_preflight, _retcode_name
from .trading_execution import _modify_position
from .trading_gateway import MT5TradingGateway
from .trading_positions import _resolve_open_position
from .trading_time import ExpirationValue
from .trading_validation import MarketOrderTypeInput, OrderTypeInput
from ..utils.mt5 import MT5ConnectionError, ensure_mt5_connection_or_raise, mt5_adapter


def _get_trading_gateway(gateway: Optional[MT5TradingGateway] = None) -> MT5TradingGateway:
    if gateway is not None:
        return gateway
    return MT5TradingGateway(
        adapter=mt5_adapter,
        ensure_connection_impl=ensure_mt5_connection_or_raise,
        build_trade_preflight_impl=_build_trade_preflight,
        retcode_name_impl=_retcode_name,
    )


def _trading_connection_error(gateway: Optional[MT5TradingGateway] = None) -> Optional[Dict[str, Any]]:
    try:
        _get_trading_gateway(gateway).ensure_connection()
    except MT5ConnectionError as exc:
        return {"error": str(exc)}
    return None


def _safe_last_error(mt5: Any) -> Any:
    try:
        if hasattr(mt5, "last_error"):
            return mt5.last_error()
    except Exception:
        return None
    return None


def _invalid_comment_error_text(result: Any, last_error: Any) -> Optional[str]:
    texts: List[str] = []
    try:
        result_comment = getattr(result, "comment", None)
    except Exception:
        result_comment = None
    if isinstance(result_comment, str) and result_comment.strip():
        texts.append(result_comment.strip())
    if isinstance(last_error, tuple):
        if len(last_error) >= 2 and isinstance(last_error[1], str) and last_error[1].strip():
            texts.append(last_error[1].strip())
        elif last_error:
            texts.append(str(last_error))
    elif isinstance(last_error, str) and last_error.strip():
        texts.append(last_error.strip())
    elif last_error not in (None, False):
        texts.append(str(last_error))
    combined = " | ".join(text for text in texts if text)
    lowered = combined.lower()
    if "invalid" in lowered and "comment" in lowered:
        return combined or "Invalid comment field."
    return None


def _send_order_with_comment_fallback(
    mt5: Any,
    request: Dict[str, Any],
) -> tuple[Any, Optional[Dict[str, Any]], Any]:
    result = mt5.order_send(request)
    last_error = _safe_last_error(mt5)
    invalid_comment = _invalid_comment_error_text(result, last_error)
    if invalid_comment is None:
        return result, None, last_error

    fallback_requests: List[tuple[str, Dict[str, Any]]] = []
    minimal_comment = trading_comments._normalize_trade_comment("MCP", default="MCP")
    if request.get("comment") != minimal_comment:
        req_short = dict(request)
        req_short["comment"] = minimal_comment
        fallback_requests.append(("minimal", req_short))
    if "comment" in request:
        req_nocomment = dict(request)
        req_nocomment.pop("comment", None)
        fallback_requests.append(("none", req_nocomment))

    strategies = [strategy for strategy, _req in fallback_requests]
    for strategy, alt_request in fallback_requests:
        alt_result = mt5.order_send(alt_request)
        alt_last_error = _safe_last_error(mt5)
        if alt_result is not None and getattr(alt_result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
            return (
                alt_result,
                {
                    "used": True,
                    "strategy": strategy,
                    "invalid_comment_error": invalid_comment,
                    "request": alt_request,
                },
                alt_last_error,
            )

    return (
        result,
        {
            "used": False,
            "attempted": bool(fallback_requests),
            "strategies": strategies,
            "invalid_comment_error": invalid_comment,
        },
        last_error,
    )


def _place_market_order(
    symbol: str,
    volume: float,
    order_type: MarketOrderTypeInput,
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    comment: Optional[str] = None,
    deviation: int = 20,
    gateway: Optional[MT5TradingGateway] = None,
) -> dict:
    """Internal helper to place a market order."""
    mt5 = _get_trading_gateway(gateway)

    connection_error = _trading_connection_error(mt5)
    if connection_error is not None:
        return connection_error

    def _place_market_order():
        try:
            preflight = mt5.build_trade_preflight()
            if not preflight.get("execution_ready_strict", preflight.get("execution_ready", True)):
                return {
                    "error": "Trading not ready in MT5 terminal/account preflight.",
                    "preflight": preflight,
                }
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"error": f"Symbol {symbol} not found"}

            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    return {"error": f"Failed to select symbol {symbol}"}

            volume_validated, volume_error = trading_validation._validate_volume(volume, symbol_info)
            if volume_error:
                return {"error": volume_error}

            # Normalize and validate requested order type
            t, order_type_error = trading_validation._normalize_order_type_input(order_type)
            if order_type_error:
                return {"error": order_type_error}
            if t not in {"BUY", "SELL"}:
                return {"error": f"Unsupported order_type '{order_type}'. Use BUY or SELL for market orders."}
            side = t

            deviation_validated, deviation_error = trading_validation._validate_deviation(deviation)
            if deviation_error:
                return {"error": deviation_error}

            # Price normalization helper
            point = float(symbol_info.point or 0.0) if hasattr(symbol_info, "point") else 0.0
            digits = int(symbol_info.digits) if hasattr(symbol_info, "digits") else 5

            def _normalize_price(val: Optional[Union[int, float]]) -> Optional[float]:
                try:
                    if val is None:
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

            norm_sl = _normalize_price(stop_loss) if stop_loss not in (None, 0) else None
            norm_tp = _normalize_price(take_profit) if take_profit not in (None, 0) else None

            # Validate against a recent quote, then refresh again right before send.
            validate_tick = mt5.symbol_info_tick(symbol)
            if validate_tick is None:
                return {"error": f"Failed to get current price for {symbol}"}
            validate_price = validate_tick.ask if side == "BUY" else validate_tick.bid

            # SL/TP validation for market orders
            if norm_sl is not None:
                if side == "BUY" and norm_sl >= validate_price:
                    return {"error": f"stop_loss must be below entry for BUY orders. sl={norm_sl}, price={validate_price}"}
                if side == "SELL" and norm_sl <= validate_price:
                    return {"error": f"stop_loss must be above entry for SELL orders. sl={norm_sl}, price={validate_price}"}
            if norm_tp is not None:
                if side == "BUY" and norm_tp <= validate_price:
                    return {"error": f"take_profit must be above entry for BUY orders. tp={norm_tp}, price={validate_price}"}
                if side == "SELL" and norm_tp >= validate_price:
                    return {"error": f"take_profit must be below entry for SELL orders. tp={norm_tp}, price={validate_price}"}
            live_protection_error = trading_validation._validate_live_protection_levels(
                symbol_info=symbol_info,
                tick=validate_tick,
                side=side,
                stop_loss=norm_sl,
                take_profit=norm_tp,
            )
            if live_protection_error is not None:
                return live_protection_error

            # Place market order without TP/SL first (TRADE_ACTION_DEAL doesn't
            # reliably support them)
            send_tick = mt5.symbol_info_tick(symbol)
            if send_tick is None:
                return {"error": f"Failed to get fresh price for {symbol}"}
            price = send_tick.ask if side == "BUY" else send_tick.bid
            if norm_sl is not None:
                if side == "BUY" and norm_sl >= price:
                    return {"error": f"stop_loss must be below entry for BUY orders at send time. sl={norm_sl}, price={price}"}
                if side == "SELL" and norm_sl <= price:
                    return {"error": f"stop_loss must be above entry for SELL orders at send time. sl={norm_sl}, price={price}"}
            if norm_tp is not None:
                if side == "BUY" and norm_tp <= price:
                    return {"error": f"take_profit must be above entry for BUY orders at send time. tp={norm_tp}, price={price}"}
                if side == "SELL" and norm_tp >= price:
                    return {"error": f"take_profit must be below entry for SELL orders at send time. tp={norm_tp}, price={price}"}
            live_protection_error = trading_validation._validate_live_protection_levels(
                symbol_info=symbol_info,
                tick=send_tick,
                side=side,
                stop_loss=norm_sl,
                take_profit=norm_tp,
            )
            if live_protection_error is not None:
                return live_protection_error
            request_comment = trading_comments._normalize_trade_comment(comment, default="MCP order")
            comment_sanitization = trading_comments._comment_sanitization_info(comment, request_comment)
            comment_truncation = trading_comments._comment_truncation_info(comment, request_comment)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume_validated,
                "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": deviation_validated,
                "magic": 234000,
                "comment": request_comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result, comment_fallback, last_error = _send_order_with_comment_fallback(mt5, request)
            if result is None:
                return {
                    "error": "Failed to send order",
                    "last_error": last_error,
                    "request": request,
                    "comment_fallback": comment_fallback,
                }
            if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
                error_message = "Failed to send order"
                invalid_comment = (
                    comment_fallback.get("invalid_comment_error")
                    if isinstance(comment_fallback, dict)
                    else None
                )
                if invalid_comment:
                    error_message = (
                        "Failed to send order: broker rejected the comment field. "
                        "Comments are sanitized to letters/numbers/spaces/_/./-; "
                        "try a simpler comment or omit --comment."
                    )
                return {
                    "error": error_message,
                    "retcode": result.retcode,
                    "retcode_name": mt5.retcode_name(result.retcode),
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "request": request,
                    "last_error": last_error,
                    "comment_fallback": comment_fallback,
                }

            # If TP/SL were specified, modify the position immediately
            order_ticket = trading_validation._safe_int_ticket(getattr(result, "order", None))
            deal_ticket = trading_validation._safe_int_ticket(getattr(result, "deal", None))
            position_ticket_candidates: List[int] = []
            for cand in (order_ticket, deal_ticket):
                if cand is not None and cand not in position_ticket_candidates:
                    position_ticket_candidates.append(cand)
            position_ticket = order_ticket
            position_ticket_resolution: Optional[Dict[str, Any]] = None
            sl_tp_modified = False
            sl_tp_error = None
            sl_tp_requested = bool(norm_sl is not None or norm_tp is not None)
            sl_tp_apply_status = "not_requested"
            sl_applied = None
            tp_applied = None
            sl_tp_broker_adjusted = False
            sl_tp_adjustment: Dict[str, Any] = {}
            sl_tp_attempts = 0
            sl_tp_last_retcode = None
            sl_tp_last_comment = None
            sl_tp_fallback_used = False
            sl_tp_fallback_result: Optional[Dict[str, Any]] = None

            if norm_sl is not None or norm_tp is not None:
                try:
                    # MT5 may report a deal/order ticket that differs from the open
                    # position ticket. Resolve robustly and retry briefly while the
                    # terminal updates its position book.
                    position_obj = None
                    lookup_attempts = 3
                    lookup_wait_seconds = 0.25
                    for attempt_idx in range(lookup_attempts):
                        pos, resolved_ticket, resolve_info = _resolve_open_position(
                            mt5,
                            ticket_candidates=position_ticket_candidates,
                            symbol=symbol,
                            side=side,
                            volume=volume_validated,
                        )
                        if pos is not None and resolved_ticket is not None:
                            position_obj = pos
                            position_ticket = resolved_ticket
                            position_ticket_resolution = {
                                **dict(resolve_info),
                                "attempt": int(attempt_idx + 1),
                            }
                            break
                        if attempt_idx + 1 < lookup_attempts:
                            time.sleep(lookup_wait_seconds)

                    if position_obj is not None and position_ticket is not None:
                        # Use TRADE_ACTION_SLTP to set TP/SL on the position
                        modify_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": symbol,
                            "position": position_ticket,
                            "sl": norm_sl or 0.0,
                            "tp": norm_tp or 0.0,
                            "magic": 234000,
                            "comment": trading_comments._normalize_trade_comment(
                                comment,
                                default=request_comment,
                                suffix=" - set TP/SL",
                            ),
                        }
                        modify_result = None
                        max_modify_attempts = 5
                        for modify_try in range(max_modify_attempts):
                            sl_tp_attempts = int(modify_try + 1)
                            try:
                                modify_result = mt5.order_send(modify_request)
                            except StopIteration:
                                modify_result = None
                                break
                            except Exception as ex:
                                modify_result = None
                                sl_tp_error = f"Error setting TP/SL: {str(ex)}"
                            if modify_result is not None:
                                sl_tp_last_retcode = getattr(modify_result, "retcode", None)
                                sl_tp_last_comment = getattr(modify_result, "comment", None)
                            if modify_result and getattr(modify_result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                                break
                            if modify_try + 1 < max_modify_attempts:
                                time.sleep(0.35)

                        if modify_result and getattr(modify_result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                            sl_tp_modified = True
                            sl_tp_apply_status = "applied"
                            try:
                                positions_after = mt5.positions_get(ticket=position_ticket)
                                if not positions_after:
                                    fallback_pos, _, _ = _resolve_open_position(
                                        mt5,
                                        ticket_candidates=[position_ticket],
                                        symbol=symbol,
                                        side=side,
                                        volume=volume_validated,
                                    )
                                    positions_after = [fallback_pos] if fallback_pos is not None else []
                                if positions_after and len(positions_after) > 0:
                                    pos_after = positions_after[0]
                                    sl_applied = float(getattr(pos_after, "sl", 0.0) or 0.0) or None
                                    tp_applied = float(getattr(pos_after, "tp", 0.0) or 0.0) or None
                            except Exception:
                                pass

                            price_tol = float(getattr(symbol_info, "point", 0.0) or 0.0)
                            if not math.isfinite(price_tol) or price_tol <= 0:
                                price_tol = 1e-9
                            if norm_sl is not None and sl_applied is not None:
                                if abs(float(sl_applied) - float(norm_sl)) > price_tol:
                                    sl_tp_broker_adjusted = True
                                    sl_tp_adjustment["sl"] = {
                                        "requested": float(norm_sl),
                                        "applied": float(sl_applied),
                                    }
                            if norm_tp is not None and tp_applied is not None:
                                if abs(float(tp_applied) - float(norm_tp)) > price_tol:
                                    sl_tp_broker_adjusted = True
                                    sl_tp_adjustment["tp"] = {
                                        "requested": float(norm_tp),
                                        "applied": float(tp_applied),
                                    }
                        else:
                            fallback_out: Dict[str, Any] = {}
                            if position_ticket is not None:
                                sl_tp_fallback_used = True
                                time.sleep(0.35)
                                try:
                                    fallback_out = _modify_position(
                                        ticket=position_ticket,
                                        stop_loss=norm_sl,
                                        take_profit=norm_tp,
                                        comment=comment,
                                        gateway=mt5,
                                    )
                                except Exception as ex:
                                    fallback_out = {"error": f"Fallback modify call failed: {str(ex)}"}
                                sl_tp_fallback_result = fallback_out if isinstance(fallback_out, dict) else {"result": fallback_out}
                                if isinstance(fallback_out, dict) and bool(fallback_out.get("success")):
                                    sl_tp_modified = True
                                    sl_tp_apply_status = "applied"
                                    sl_applied = fallback_out.get("applied_sl")
                                    tp_applied = fallback_out.get("applied_tp")
                                    sl_tp_error = None
                                    # Mark when fallback applied broker-adjusted levels.
                                    if norm_sl is not None and sl_applied is not None:
                                        try:
                                            if abs(float(sl_applied) - float(norm_sl)) > (float(getattr(symbol_info, "point", 0.0) or 1e-9)):
                                                sl_tp_broker_adjusted = True
                                                sl_tp_adjustment["sl"] = {
                                                    "requested": float(norm_sl),
                                                    "applied": float(sl_applied),
                                                }
                                        except Exception:
                                            pass
                                    if norm_tp is not None and tp_applied is not None:
                                        try:
                                            if abs(float(tp_applied) - float(norm_tp)) > (float(getattr(symbol_info, "point", 0.0) or 1e-9)):
                                                sl_tp_broker_adjusted = True
                                                sl_tp_adjustment["tp"] = {
                                                    "requested": float(norm_tp),
                                                    "applied": float(tp_applied),
                                                }
                                        except Exception:
                                            pass
                                else:
                                    sl_tp_error = (
                                        str(fallback_out.get("error"))
                                        if isinstance(fallback_out, dict) and fallback_out.get("error")
                                        else (
                                            "Failed to set TP/SL"
                                            if sl_tp_error is None
                                            else sl_tp_error
                                        )
                                    )
                                    sl_tp_apply_status = "failed"
                            else:
                                sl_tp_error = (
                                    "Failed to set TP/SL"
                                    if sl_tp_error is None
                                    else sl_tp_error
                                )
                                sl_tp_apply_status = "failed"
                    else:
                        checked = ", ".join(str(v) for v in position_ticket_candidates) or "none"
                        sl_tp_error = (
                            "Position not found for TP/SL modification "
                            f"(ticket candidates: {checked})"
                        )
                        sl_tp_apply_status = "failed"
                except Exception as e:
                    sl_tp_error = f"Error setting TP/SL: {str(e)}"
                    sl_tp_apply_status = "failed"

            warnings_out: List[str] = []
            if comment_sanitization:
                warnings_out.append(
                    f"Comment sanitized for broker compatibility: '{comment_sanitization['applied']}'"
                )
            if comment_truncation:
                warnings_out.append(
                    f"Comment truncated to {comment_truncation['max_length']} characters: '{comment_truncation['applied']}'"
                )
            if isinstance(comment_fallback, dict) and comment_fallback.get("used"):
                warnings_out.append(
                    "Broker rejected the comment field; order was retried with a minimal MT5-safe comment."
                )
            if sl_tp_requested and sl_tp_apply_status == "failed":
                fix_hint = (
                    f"trade_modify {position_ticket}"
                    if position_ticket is not None
                    else "trade_modify <position_ticket>"
                )
                warnings_out.append(
                    "CRITICAL: Order filled but TP/SL could not be applied. "
                    f"Use {fix_hint} immediately or close the position."
                )
            if sl_tp_requested and sl_tp_fallback_used and sl_tp_apply_status == "applied":
                warnings_out.append(
                    "TP/SL protection required a post-fill fallback modification. Verify the live position is protected."
                )

            out: Dict[str, Any] = {
                "retcode": result.retcode,
                "retcode_name": mt5.retcode_name(result.retcode),
                "deal": result.deal,
                "order": result.order,
                "volume": result.volume,
                "price": result.price,
                "bid": result.bid,
                "ask": result.ask,
                "comment": result.comment,
                "request_id": result.request_id,
                "position_ticket": position_ticket,
                "position_ticket_candidates": position_ticket_candidates or None,
                "position_ticket_resolution": position_ticket_resolution,
                "sl": norm_sl,
                "tp": norm_tp,
                "sl_tp_requested": sl_tp_requested,
                "sl_tp_apply_status": sl_tp_apply_status,
                "sl_tp_modified": sl_tp_modified,
                "sl_applied": sl_applied,
                "tp_applied": tp_applied,
                "sl_tp_broker_adjusted": sl_tp_broker_adjusted,
                "sl_tp_adjustment": sl_tp_adjustment or None,
                "sl_tp_error": sl_tp_error,
                "sl_tp_attempts": sl_tp_attempts,
                "sl_tp_last_retcode": sl_tp_last_retcode,
                "sl_tp_last_comment": sl_tp_last_comment,
                "sl_tp_fallback_used": sl_tp_fallback_used,
                "sl_tp_fallback_result": sl_tp_fallback_result,
            }
            if comment_sanitization:
                out["comment_sanitization"] = comment_sanitization
            if sl_tp_requested:
                if sl_tp_apply_status == "applied":
                    out["protection_status"] = (
                        "protected_after_fallback"
                        if sl_tp_fallback_used
                        else "protected"
                    )
                elif sl_tp_apply_status == "failed":
                    out["protection_status"] = "unprotected_position"
            if comment_truncation:
                out["comment_truncation"] = comment_truncation
            if comment_fallback:
                out["comment_fallback"] = comment_fallback
            if warnings_out:
                out["warnings"] = warnings_out
            return out

        except Exception as e:
            return {"error": str(e)}

    return _place_market_order()


def _place_pending_order(
    symbol: str,
    volume: float,
    order_type: OrderTypeInput,
    price: Union[int, float],
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    expiration: Optional[ExpirationValue] = None,
    comment: Optional[str] = None,
    deviation: int = 20,
    gateway: Optional[MT5TradingGateway] = None,
) -> dict:
    """Internal helper to place a pending order."""
    mt5 = _get_trading_gateway(gateway)

    connection_error = _trading_connection_error(mt5)
    if connection_error is not None:
        return connection_error

    def _place_pending_order():
        try:
            preflight = mt5.build_trade_preflight()
            if not preflight.get("execution_ready_strict", preflight.get("execution_ready", True)):
                return {
                    "error": "Trading not ready in MT5 terminal/account preflight.",
                    "preflight": preflight,
                }
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"error": f"Symbol {symbol} not found"}

            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    return {"error": f"Failed to select symbol {symbol}"}

            volume_validated, volume_error = trading_validation._validate_volume(volume, symbol_info)
            if volume_error:
                return {"error": volume_error}

            current_price = mt5.symbol_info_tick(symbol)
            if current_price is None:
                return {"error": f"Failed to get current price for {symbol}"}

            deviation_validated, deviation_error = trading_validation._validate_deviation(deviation)
            if deviation_error:
                return {"error": deviation_error}

            # Normalize and validate requested order type
            t, order_type_error = trading_validation._normalize_order_type_input(order_type)
            if order_type_error:
                return {"error": order_type_error}
            explicit_map = {
                "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
                "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
                "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
                "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP,
            }

            # Basic side/price sanity checks for explicit pending types
            bid = float(getattr(current_price, "bid", 0.0) or 0.0)
            ask = float(getattr(current_price, "ask", 0.0) or 0.0)
            point = float(symbol_info.point or 0.0) if hasattr(symbol_info, "point") else 0.0
            digits = int(symbol_info.digits) if hasattr(symbol_info, "digits") else 5

            def _normalize_price(val: Optional[Union[int, float]]) -> Optional[float]:
                try:
                    if val is None:
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

            norm_price = _normalize_price(price)
            if norm_price is None:
                return {"error": "price must be a finite number"}

            order_type_value = None
            if t in explicit_map:
                order_type_value = explicit_map[t]
            elif t == "BUY":
                order_type_value = mt5.ORDER_TYPE_BUY_LIMIT if norm_price < ask else mt5.ORDER_TYPE_BUY_STOP
            elif t == "SELL":
                order_type_value = mt5.ORDER_TYPE_SELL_LIMIT if norm_price > bid else mt5.ORDER_TYPE_SELL_STOP
            else:
                return {
                    "error": (
                        f"Unsupported order_type '{order_type}'. "
                        "Use BUY/SELL or BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP."
                    )
                }
            norm_sl = _normalize_price(stop_loss) if stop_loss not in (None, 0) else None
            norm_tp = _normalize_price(take_profit) if take_profit not in (None, 0) else None

            if order_type_value == mt5.ORDER_TYPE_BUY_LIMIT and not (norm_price < ask):
                return {"error": f"Price must be below ask for BUY_LIMIT. price={norm_price}, ask={ask}"}
            if order_type_value == mt5.ORDER_TYPE_BUY_STOP and not (norm_price > ask):
                return {"error": f"Price must be above ask for BUY_STOP. price={norm_price}, ask={ask}"}
            if order_type_value == mt5.ORDER_TYPE_SELL_LIMIT and not (norm_price > bid):
                return {"error": f"Price must be above bid for SELL_LIMIT. price={norm_price}, bid={bid}"}
            if order_type_value == mt5.ORDER_TYPE_SELL_STOP and not (norm_price < bid):
                return {"error": f"Price must be below bid for SELL_STOP. price={norm_price}, bid={bid}"}

            normalized_expiration, expiration_specified = trading_time._normalize_pending_expiration(expiration)

            # SL/TP sanity relative to entry
            if norm_sl is not None:
                if order_type_value in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP) and norm_sl >= norm_price:
                    return {"error": f"stop_loss must be below entry for BUY orders. sl={norm_sl}, price={norm_price}"}
                if order_type_value in (mt5.ORDER_TYPE_SELL_LIMIT, mt5.ORDER_TYPE_SELL_STOP) and norm_sl <= norm_price:
                    return {"error": f"stop_loss must be above entry for SELL orders. sl={norm_sl}, price={norm_price}"}
            if norm_tp is not None:
                if order_type_value in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP) and norm_tp <= norm_price:
                    return {"error": f"take_profit must be above entry for BUY orders. tp={norm_tp}, price={norm_price}"}
                if order_type_value in (mt5.ORDER_TYPE_SELL_LIMIT, mt5.ORDER_TYPE_SELL_STOP) and norm_tp >= norm_price:
                    return {"error": f"take_profit must be below entry for SELL orders. tp={norm_tp}, price={norm_price}"}

            request_comment = trading_comments._normalize_trade_comment(comment, default="MCP pending order")
            comment_sanitization = trading_comments._comment_sanitization_info(comment, request_comment)
            comment_truncation = trading_comments._comment_truncation_info(comment, request_comment)
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume_validated,
                "type": order_type_value,
                "price": norm_price,
                "sl": norm_sl or 0.0,
                "tp": norm_tp or 0.0,
                "deviation": deviation_validated,
                "magic": 234000,
                "comment": request_comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            if expiration_specified:
                if normalized_expiration is None:
                    request["type_time"] = mt5.ORDER_TIME_GTC
                else:
                    request["type_time"] = mt5.ORDER_TIME_SPECIFIED
                    request["expiration"] = normalized_expiration

            result, comment_fallback, last_error = _send_order_with_comment_fallback(mt5, request)
            if result is None:
                return {
                    "error": "Failed to send pending order",
                    "last_error": last_error,
                    "request": request,
                    "comment_fallback": comment_fallback,
                }

            if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
                error_message = "Failed to send pending order"
                invalid_comment = (
                    comment_fallback.get("invalid_comment_error")
                    if isinstance(comment_fallback, dict)
                    else None
                )
                if invalid_comment:
                    error_message = (
                        "Failed to send pending order: broker rejected the comment field. "
                        "Comments are sanitized to letters/numbers/spaces/_/./-; "
                        "try a simpler comment or omit --comment."
                    )
                return {
                    "error": error_message,
                    "retcode": result.retcode,
                    "retcode_name": mt5.retcode_name(result.retcode),
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "request": request,
                    "last_error": last_error,
                    "comment_fallback": comment_fallback,
                }

            out: Dict[str, Any] = {
                "success": True,
                "retcode": result.retcode,
                "retcode_name": mt5.retcode_name(result.retcode),
                "deal": result.deal,
                "order": result.order,
                "volume": result.volume,
                "price": result.price,
                "bid": result.bid,
                "ask": result.ask,
                "requested_price": float(norm_price),
                "requested_sl": float(norm_sl) if norm_sl is not None else None,
                "requested_tp": float(norm_tp) if norm_tp is not None else None,
                "comment": result.comment,
                "request_id": result.request_id,
            }
            if expiration_specified:
                out["requested_expiration"] = normalized_expiration
            warnings_out: List[str] = []
            if comment_sanitization:
                out["comment_sanitization"] = comment_sanitization
                warnings_out.append(
                    f"Comment sanitized for broker compatibility: '{comment_sanitization['applied']}'"
                )
            if comment_truncation:
                out["comment_truncation"] = comment_truncation
                warnings_out.append(
                    f"Comment truncated to {comment_truncation['max_length']} characters: '{comment_truncation['applied']}'"
                )
            if comment_fallback:
                out["comment_fallback"] = comment_fallback
                if comment_fallback.get("used"):
                    warnings_out.append(
                        "Broker rejected the comment field; pending order was retried with a minimal MT5-safe comment."
                    )
            if warnings_out:
                out["warnings"] = warnings_out
            return out

        except Exception as e:
            return {"error": str(e)}

    return _place_pending_order()
