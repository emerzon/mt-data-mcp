"""Trading functions for MetaTrader integration."""


import math
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Union, List, Dict, Any

from ._mcp_instance import mcp
from . import trading_comments, trading_time, trading_validation
from .trading_account import trade_account_info, trade_history
from .trading_common import _build_trade_preflight, _retcode_name
from .trading_positions import _resolve_open_position, trade_get_open, trade_get_pending
from .trading_risk import trade_risk_analyze
from .trading_time import ExpirationValue
from .trading_validation import MarketOrderTypeInput, OrderTypeInput
from ..utils.mt5 import _auto_connect_wrapper, _mt5_epoch_to_utc, mt5_adapter


def _place_market_order(
    symbol: str,
    volume: float,
    order_type: MarketOrderTypeInput,
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    comment: Optional[str] = None,
    deviation: int = 20,
) -> dict:
    """Internal helper to place a market order."""
    mt5 = mt5_adapter

    @_auto_connect_wrapper
    def _place_market_order():
        try:
            preflight = _build_trade_preflight(mt5)
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
            request_comment = trading_comments._normalize_trade_comment(comment, default="MCP order")
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
            result = mt5.order_send(request)
            if result is None:
                # Surface MetaTrader last_error when available for easier debugging
                try:
                    err = mt5.last_error()
                except Exception:
                    err = None
                return {"error": "Failed to send order", "last_error": err}
            if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
                return {
                    "error": "Failed to send order",
                    "retcode": result.retcode,
                    "retcode_name": _retcode_name(mt5, result.retcode),
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "request": request,
                    "last_error": mt5.last_error() if hasattr(mt5, "last_error") else None,
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
            if comment_truncation:
                warnings_out.append(
                    f"Comment truncated to {comment_truncation['max_length']} characters: '{comment_truncation['applied']}'"
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
                "retcode_name": _retcode_name(mt5, result.retcode),
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
) -> dict:
    """Internal helper to place a pending order."""
    mt5 = mt5_adapter

    @_auto_connect_wrapper
    def _place_pending_order():
        try:
            preflight = _build_trade_preflight(mt5)
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

            result = mt5.order_send(request)
            if result is None:
                # Surface MetaTrader last_error when available for easier debugging
                try:
                    err = mt5.last_error()
                except Exception:
                    err = None
                return {"error": "Failed to send pending order", "last_error": err, "request": request}

            if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
                return {
                    "error": "Failed to send pending order",
                    "retcode": result.retcode,
                    "retcode_name": _retcode_name(mt5, result.retcode),
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "request": request,
                    "last_error": mt5.last_error() if hasattr(mt5, "last_error") else None,
                }

            out: Dict[str, Any] = {
                "success": True,
                "retcode": result.retcode,
                "retcode_name": _retcode_name(mt5, result.retcode),
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
            if comment_truncation:
                out["comment_truncation"] = comment_truncation
                out["warnings"] = [
                    f"Comment truncated to {comment_truncation['max_length']} characters: '{comment_truncation['applied']}'"
                ]
            return out

        except Exception as e:
            return {"error": str(e)}

    return _place_pending_order()


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


def _modify_position(
    ticket: Union[int, str],
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    comment: Optional[str] = None,
) -> dict:
    """Internal helper to modify a position by ticket."""
    mt5 = mt5_adapter

    @_auto_connect_wrapper
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

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
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
                    "retcode_name": _retcode_name(mt5, result.retcode),
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "request": request,
                    "last_error": mt5.last_error() if hasattr(mt5, "last_error") else None,
                }

            return {
                "success": True,
                "retcode": result.retcode,
                "retcode_name": _retcode_name(mt5, result.retcode),
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
) -> dict:
    """Internal helper to modify a pending order by ticket."""
    mt5 = mt5_adapter

    @_auto_connect_wrapper
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
                    "retcode_name": _retcode_name(mt5, result.retcode),
                    "comment": result.comment,
                    "request_id": result.request_id,
                    "request": request,
                    "last_error": mt5.last_error() if hasattr(mt5, "last_error") else None,
                }

            return {
                "success": True,
                "retcode": result.retcode,
                "retcode_name": _retcode_name(mt5, result.retcode),
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


@mcp.tool()
def trade_modify(
    ticket: Union[int, str],
    price: Optional[Union[int, float]] = None,
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    expiration: Optional[ExpirationValue] = None,
    comment: Optional[str] = None,
) -> dict:
    """Modify an open position or pending order by ticket.

    Inference rules:
    - If ``price`` or ``expiration`` is provided, treat the ticket as a pending order.
    - Otherwise, try a position modify first; if not found, fall back to pending order.
    """
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


def _close_positions(
    ticket: Optional[Union[int, str]] = None,
    symbol: Optional[str] = None,
    profit_only: bool = False,
    loss_only: bool = False,
    comment: Optional[str] = None,
    deviation: int = 20,
) -> dict:
    """Internal helper to close open positions."""
    mt5 = mt5_adapter

    @_auto_connect_wrapper
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
                        "volume": position.volume,
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
                                        "retcode_name": _retcode_name(mt5, getattr(alt_res, "retcode", None)),
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
                            "retcode_name": _retcode_name(mt5, retcode_val),
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
                        "retcode_name": _retcode_name(mt5, result.retcode),
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
                        "attempts": attempts,
                    }
                    results.append(res_dict)

            # If only one position was targeted by ticket, return single result
            if ticket is not None and len(results) == 1:
                return results[0]
            success_count = 0
            for item in results:
                try:
                    if int(item.get("retcode")) == int(mt5.TRADE_RETCODE_DONE):
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
) -> dict:
    """Internal helper to cancel pending orders."""
    mt5 = mt5_adapter

    @_auto_connect_wrapper
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
                        "retcode_name": _retcode_name(mt5, result.retcode),
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


@mcp.tool()
def trade_close(
    ticket: Optional[Union[int, str]] = None,
    symbol: Optional[str] = None,
    profit_only: bool = False,
    loss_only: bool = False,
    comment: Optional[str] = None,
    deviation: int = 20,
) -> dict:
    """Close positions or cancel pending orders.

    Behavior:
    - With `profit_only`/`loss_only`: closes positions only.
    - With `ticket`: tries position close first, then pending cancellation for the same ticket.
    - With `symbol`: closes matching positions; if none exist, cancels pending orders for that symbol.
    - With no filters: closes open positions; if none exist, cancels all pending orders.
    """
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


