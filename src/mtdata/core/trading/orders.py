"""Order placement workflows for MetaTrader integration."""

import logging
import math
import time as _stdlib_time
from typing import Any, Dict, List, Optional, TypedDict, Union

from ..config import mt5_config
from . import comments, common, time, validation
from .execution import _modify_position
from .gateway import MT5TradingGateway, create_trading_gateway, trading_connection_error
from .positions import _resolve_open_position
from .time import ExpirationValue
from .validation import MarketOrderTypeInput, OrderTypeInput


class _OrderSymbolContext(TypedDict):
    symbol_info: Any
    volume: float


class _OrderSubmitOutcome(TypedDict):
    result: Any
    comment_fallback: Optional[Dict[str, Any]]
    fill_mode_attempts: List[Dict[str, Any]]
    used_request: Dict[str, Any]


_POSITION_RESOLUTION_WAIT_SCHEDULE_SECONDS = (0.15, 0.3, 0.6, 1.2)
_DEFAULT_ORDER_MAGIC = 234000
logger = logging.getLogger(__name__)


def _configured_order_magic() -> int:
    configured = validation._safe_int_ticket(getattr(mt5_config, "order_magic", None))
    if configured is not None:
        return int(configured)
    return _DEFAULT_ORDER_MAGIC


def _compact_sl_tp_levels(
    *,
    sl: Optional[float],
    tp: Optional[float],
) -> Optional[Dict[str, float]]:
    levels: Dict[str, float] = {}
    if sl is not None:
        levels["sl"] = float(sl)
    if tp is not None:
        levels["tp"] = float(tp)
    return levels or None


def _build_sl_tp_result(
    *,
    requested_sl: Optional[float],
    requested_tp: Optional[float],
    applied_sl: Optional[float],
    applied_tp: Optional[float],
    status: str,
    error: Optional[str],
    broker_adjusted: bool,
    adjustment: Optional[Dict[str, Any]],
    attempts: int,
    last_retcode: Any,
    last_comment: Any,
    comment_fallback: Optional[Dict[str, Any]],
    fallback_used: bool,
    fallback_result: Optional[Dict[str, Any]],
    verification_failed: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": status}
    requested = _compact_sl_tp_levels(sl=requested_sl, tp=requested_tp)
    applied = _compact_sl_tp_levels(sl=applied_sl, tp=applied_tp)
    if requested is not None:
        out["requested"] = requested
    if applied is not None:
        out["applied"] = applied
    if error is not None:
        out["error"] = error
    if broker_adjusted:
        out["broker_adjusted"] = True
    if adjustment:
        out["adjustment"] = adjustment
    if attempts > 0:
        out["attempts"] = int(attempts)
    if last_retcode is not None:
        out["last_retcode"] = last_retcode
    if last_comment is not None:
        out["last_comment"] = last_comment
    if comment_fallback is not None:
        out["comment_fallback"] = comment_fallback
    if fallback_used:
        out["fallback_used"] = True
    if fallback_result is not None:
        out["fallback_result"] = fallback_result
    if verification_failed:
        out["verification_failed"] = True
    return out


class _ProtectionOutcome(TypedDict, total=False):
    position_ticket: Optional[int]
    position_ticket_candidates: List[int]
    position_ticket_resolution: Optional[Dict[str, Any]]
    protection_status: str
    sl_tp_result: Dict[str, Any]
    warnings: List[str]


def _attach_post_fill_protection(
    mt5: Any,
    *,
    symbol: str,
    side: str,
    volume: float,
    position_ticket_candidates: List[int],
    stop_loss: Optional[float],
    take_profit: Optional[float],
    symbol_info: Any,
    comment: Optional[str],
    request_comment: str,
    magic: Optional[int] = None,
) -> _ProtectionOutcome:
    """Resolve the filled position and attach SL/TP protection.

    Returns a structured outcome with position resolution info, the SL/TP
    result dict, an optional protection_status key, and accumulated warnings.
    """
    warnings_out: List[str] = []
    sl_tp_requested = bool(stop_loss is not None or take_profit is not None)

    if not sl_tp_requested:
        return {
            "position_ticket": None,
            "position_ticket_candidates": position_ticket_candidates or None,
            "position_ticket_resolution": None,
            "sl_tp_result": _build_sl_tp_result(
                requested_sl=None,
                requested_tp=None,
                applied_sl=None,
                applied_tp=None,
                status="not_requested",
                error=None,
                broker_adjusted=False,
                adjustment=None,
                attempts=0,
                last_retcode=None,
                last_comment=None,
                comment_fallback=None,
                fallback_used=False,
                fallback_result=None,
            ),
            "warnings": [],
        }

    # --- State variables for the protection flow ---
    position_ticket: Optional[int] = None
    position_ticket_resolution: Optional[Dict[str, Any]] = None
    sl_tp_error: Optional[str] = None
    sl_tp_apply_status = "not_requested"
    sl_applied: Optional[float] = None
    tp_applied: Optional[float] = None
    sl_tp_verification_failed = False
    sl_tp_broker_adjusted = False
    sl_tp_adjustment: Dict[str, Any] = {}
    sl_tp_attempts = 0
    sl_tp_last_retcode = None
    sl_tp_last_comment = None
    sl_tp_comment_fallback: Optional[Dict[str, Any]] = None
    sl_tp_fallback_used = False
    sl_tp_fallback_result: Optional[Dict[str, Any]] = None

    try:
        # --- Phase 1: Resolve the open position ---
        position_obj = None
        lookup_wait_schedule = _POSITION_RESOLUTION_WAIT_SCHEDULE_SECONDS
        lookup_attempts = len(lookup_wait_schedule) + 1
        last_resolve_info: Optional[Dict[str, Any]] = None
        for attempt_idx in range(lookup_attempts):
            pos, resolved_ticket, resolve_info = _resolve_open_position(
                mt5,
                ticket_candidates=position_ticket_candidates,
                symbol=symbol,
                side=side,
                volume=volume,
                magic=magic,
            )
            if isinstance(resolve_info, dict):
                last_resolve_info = dict(resolve_info)
            if pos is not None and resolved_ticket is not None:
                position_obj = pos
                position_ticket = resolved_ticket
                position_ticket_resolution = {
                    **dict(resolve_info),
                    "attempt": int(attempt_idx + 1),
                }
                break
            if attempt_idx + 1 < lookup_attempts:
                _stdlib_time.sleep(float(lookup_wait_schedule[attempt_idx]))
        if position_ticket_resolution is None and last_resolve_info is not None:
            position_ticket_resolution = {
                **last_resolve_info,
                "attempts": int(lookup_attempts),
                "matched": False,
            }

        if position_obj is not None and position_ticket is not None:
            # --- Phase 2: Attach SL/TP via TRADE_ACTION_SLTP ---
            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": position_ticket,
                "sl": 0.0 if stop_loss is None else float(stop_loss),
                "tp": 0.0 if take_profit is None else float(take_profit),
                "comment": comments._normalize_trade_comment(
                    comment,
                    default=request_comment,
                    suffix=" - set TP/SL",
                ),
            }
            modify_magic = validation._safe_int_ticket(getattr(position_obj, "magic", None))
            if modify_magic is not None:
                modify_request["magic"] = modify_magic
            modify_result = None
            max_modify_attempts = 5
            for modify_try in range(max_modify_attempts):
                sl_tp_attempts = int(modify_try + 1)
                try:
                    modify_result, sl_tp_comment_fallback, _sl_tp_last_error = _send_order_with_comment_fallback(
                        mt5,
                        modify_request,
                    )
                except Exception as ex:
                    modify_result = None
                    sl_tp_error = f"Error setting TP/SL: {str(ex)}"
                if modify_result is not None:
                    sl_tp_last_retcode = getattr(modify_result, "retcode", None)
                    sl_tp_last_comment = getattr(modify_result, "comment", None)
                if modify_result and getattr(modify_result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                    break
                if modify_try + 1 < max_modify_attempts:
                    _stdlib_time.sleep(0.35)

            if modify_result and getattr(modify_result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                # --- Phase 3: Verify SL/TP readback ---
                sl_tp_apply_status = "applied"
                try:
                    positions_after = mt5.positions_get(ticket=position_ticket)
                    if not positions_after:
                        fallback_pos, _, _ = _resolve_open_position(
                            mt5,
                            ticket_candidates=[position_ticket],
                            symbol=symbol,
                            side=side,
                            volume=volume,
                            magic=magic,
                        )
                        positions_after = [fallback_pos] if fallback_pos is not None else []
                    if positions_after and len(positions_after) > 0:
                        pos_after = positions_after[0]
                        sl_applied = float(getattr(pos_after, "sl", 0.0) or 0.0) or None
                        tp_applied = float(getattr(pos_after, "tp", 0.0) or 0.0) or None
                except Exception as verify_exc:
                    sl_tp_verification_failed = True
                    logger.warning(
                        "SL/TP verification failed for ticket %s: %s",
                        position_ticket,
                        verify_exc,
                    )

                price_tol = float(getattr(symbol_info, "point", 0.0) or 0.0)
                if not math.isfinite(price_tol) or price_tol <= 0:
                    price_tol = 1e-9
                if stop_loss is not None and sl_applied is not None:
                    if abs(float(sl_applied) - float(stop_loss)) > price_tol:
                        sl_tp_broker_adjusted = True
                        sl_tp_adjustment["sl"] = {
                            "requested": float(stop_loss),
                            "applied": float(sl_applied),
                        }
                if take_profit is not None and tp_applied is not None:
                    if abs(float(tp_applied) - float(take_profit)) > price_tol:
                        sl_tp_broker_adjusted = True
                        sl_tp_adjustment["tp"] = {
                            "requested": float(take_profit),
                            "applied": float(tp_applied),
                        }
            else:
                # --- Phase 4: Fallback via _modify_position ---
                fallback_out: Dict[str, Any] = {}
                if position_ticket is not None:
                    sl_tp_fallback_used = True
                    _stdlib_time.sleep(0.35)
                    try:
                        fallback_out = _modify_position(
                            ticket=position_ticket,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            comment=comment,
                            gateway=mt5,
                        )
                    except Exception as ex:
                        fallback_out = {"error": f"Fallback modify call failed: {str(ex)}"}
                    sl_tp_fallback_result = fallback_out if isinstance(fallback_out, dict) else {"result": fallback_out}
                    if isinstance(fallback_out, dict) and bool(fallback_out.get("success")):
                        sl_tp_apply_status = "applied"
                        sl_applied = fallback_out.get("applied_sl")
                        tp_applied = fallback_out.get("applied_tp")
                        sl_tp_error = None
                        # Mark when fallback applied broker-adjusted levels.
                        if stop_loss is not None and sl_applied is not None:
                            try:
                                if abs(float(sl_applied) - float(stop_loss)) > (float(getattr(symbol_info, "point", 0.0) or 1e-9)):
                                    sl_tp_broker_adjusted = True
                                    sl_tp_adjustment["sl"] = {
                                        "requested": float(stop_loss),
                                        "applied": float(sl_applied),
                                    }
                            except Exception:
                                pass
                        if take_profit is not None and tp_applied is not None:
                            try:
                                if abs(float(tp_applied) - float(take_profit)) > (float(getattr(symbol_info, "point", 0.0) or 1e-9)):
                                    sl_tp_broker_adjusted = True
                                    sl_tp_adjustment["tp"] = {
                                        "requested": float(take_profit),
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

    # --- Build warnings ---
    if isinstance(sl_tp_comment_fallback, dict) and sl_tp_comment_fallback.get("used"):
        warnings_out.append(
            "Broker rejected the comment field on the TP/SL modification; protection was retried without the original comment."
        )
    if sl_tp_apply_status == "failed":
        if position_ticket is not None:
            action_text = f"Run trade_modify {position_ticket} immediately"
        elif position_ticket_candidates:
            candidate_list = ", ".join(str(v) for v in position_ticket_candidates)
            primary_candidate = position_ticket_candidates[0]
            action_text = (
                f"Try trade_modify {primary_candidate} immediately "
                f"(candidate tickets: {candidate_list})"
            )
        else:
            action_text = (
                "Run trade_get_open immediately to find the live position ticket, "
                "then trade_modify it"
            )
        warnings_out.append(
            "CRITICAL: Order filled but TP/SL could not be applied. "
            f"{action_text}, or close the position."
        )
    if sl_tp_fallback_used and sl_tp_apply_status == "applied":
        warnings_out.append(
            "TP/SL protection required a post-fill fallback modification. Verify the live position is protected."
        )
    if sl_tp_verification_failed:
        warnings_out.append(
            "SL/TP verification readback failed after broker acceptance. Verify the live position protection directly."
        )

    # --- Build outcome ---
    outcome: _ProtectionOutcome = {
        "position_ticket": position_ticket,
        "position_ticket_candidates": position_ticket_candidates or None,
        "position_ticket_resolution": position_ticket_resolution,
        "sl_tp_result": _build_sl_tp_result(
            requested_sl=stop_loss,
            requested_tp=take_profit,
            applied_sl=sl_applied,
            applied_tp=tp_applied,
            status=sl_tp_apply_status,
            error=sl_tp_error,
            broker_adjusted=sl_tp_broker_adjusted,
            adjustment=sl_tp_adjustment or None,
            attempts=sl_tp_attempts,
            last_retcode=sl_tp_last_retcode,
            last_comment=sl_tp_last_comment,
            comment_fallback=sl_tp_comment_fallback,
            fallback_used=sl_tp_fallback_used,
            fallback_result=sl_tp_fallback_result,
            verification_failed=sl_tp_verification_failed,
        ),
        "warnings": warnings_out,
    }
    if sl_tp_apply_status == "applied":
        outcome["protection_status"] = (
            "protected_after_fallback"
            if sl_tp_fallback_used
            else "protected"
        )
    elif sl_tp_apply_status == "failed":
        outcome["protection_status"] = "unprotected_position"
    return outcome


def _send_order_with_comment_fallback(
    mt5: Any,
    request: Dict[str, Any],
) -> tuple[Any, Optional[Dict[str, Any]], Any]:
    return comments._send_order_with_comment_fallback(mt5, request)

def _send_order_with_fill_mode_retry(
    mt5: Any,
    request: Dict[str, Any],
    *,
    symbol_info: Any = None,
) -> tuple[Any, Optional[Dict[str, Any]], Any, List[Dict[str, Any]], Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    last_result = None
    last_comment_fallback = None
    last_error = None
    last_request = dict(request)
    for fill_mode in validation._candidate_fill_modes(mt5, symbol_info):
        attempt_request = dict(request)
        attempt_request["type_filling"] = int(fill_mode)
        result, comment_fallback, last_error = _send_order_with_comment_fallback(mt5, attempt_request)
        attempt: Dict[str, Any] = {"type_filling": int(fill_mode)}
        if result is None:
            attempt["error"] = "order_send returned None"
            if last_error is not None:
                attempt["last_error"] = last_error
        else:
            retcode = getattr(result, "retcode", None)
            attempt["retcode"] = retcode
            attempt["retcode_name"] = mt5.retcode_name(retcode)
            attempt["comment"] = getattr(result, "comment", None)
        if comment_fallback:
            attempt["comment_fallback"] = comment_fallback
        attempts.append(attempt)
        last_result = result
        last_comment_fallback = comment_fallback
        last_request = (
            dict(comment_fallback["request"])
            if isinstance(comment_fallback, dict) and comment_fallback.get("used") and isinstance(comment_fallback.get("request"), dict)
            else attempt_request
        )
        try:
            if result is not None and validation._retcode_is_done(mt5, getattr(result, "retcode", -1)):
                return result, comment_fallback, last_error, attempts, last_request
        except Exception:
            pass
    return last_result, last_comment_fallback, last_error, attempts, last_request


def _prepare_order_gateway(
    gateway: Optional[MT5TradingGateway] = None,
) -> tuple[Optional[MT5TradingGateway], Optional[Dict[str, Any]]]:
    mt5 = create_trading_gateway(
        gateway=gateway,
        include_trade_preflight=True,
        include_retcode_name=True,
    )
    connection_error = trading_connection_error(mt5)
    if connection_error is not None:
        return None, connection_error
    return mt5, None


def _prepare_order_symbol_context(
    mt5: Any,
    *,
    symbol: str,
    volume: float,
) -> tuple[Optional[_OrderSymbolContext], Optional[Dict[str, Any]]]:
    preflight = mt5.build_trade_preflight()
    if not preflight.get("execution_ready_strict", preflight.get("execution_ready", True)):
        guidance = common._build_trade_preflight_guidance(preflight)
        return None, {
            "error": "Trading not ready in MT5 terminal/account preflight.",
            "preflight": preflight,
            "hint": guidance[0] if guidance else None,
            "next_steps": guidance,
        }

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return None, {"error": f"Symbol {symbol} not found"}

    if not symbol_info.visible and not mt5.symbol_select(symbol, True):
        return None, {"error": f"Failed to select symbol {symbol}"}

    volume_validated, volume_error = validation._validate_volume(volume, symbol_info)
    if volume_error:
        return None, {"error": volume_error}

    return {
        "symbol_info": symbol_info,
        "volume": volume_validated,
    }, None


def _build_order_comment_payload(
    comment: Optional[str],
    *,
    default: str,
) -> tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    request_comment = comments._normalize_trade_comment(comment, default=default)
    return (
        request_comment,
        comments._comment_sanitization_info(comment, request_comment),
        comments._comment_truncation_info(comment, request_comment),
    )


def _submit_order_request(
    mt5: Any,
    request: Dict[str, Any],
    *,
    base_error: str,
    invalid_comment_error: str,
    symbol_info: Any = None,
) -> tuple[Optional[_OrderSubmitOutcome], Optional[Dict[str, Any]]]:
    result, comment_fallback, last_error, fill_mode_attempts, used_request = _send_order_with_fill_mode_retry(
        mt5,
        request,
        symbol_info=symbol_info,
    )
    if result is None:
        return None, {
            "error": base_error,
            "last_error": last_error,
            "comment_fallback": comment_fallback,
            "fill_mode_attempts": fill_mode_attempts,
        }

    if not validation._retcode_is_done(mt5, getattr(result, "retcode", None)):
        error_message = base_error
        invalid_comment = (
            comment_fallback.get("invalid_comment_error")
            if isinstance(comment_fallback, dict)
            else None
        )
        if invalid_comment:
            error_message = invalid_comment_error
        return None, {
            "error": error_message,
            "retcode": result.retcode,
            "retcode_name": mt5.retcode_name(result.retcode),
            "comment": result.comment,
            "request_id": result.request_id,
            "last_error": last_error,
            "comment_fallback": comment_fallback,
            "fill_mode_attempts": fill_mode_attempts,
        }

    return {
        "result": result,
        "comment_fallback": comment_fallback,
        "fill_mode_attempts": fill_mode_attempts,
        "used_request": used_request,
    }, None


def _attach_comment_response_metadata(
    out: Dict[str, Any],
    warnings_out: List[str],
    *,
    comment_sanitization: Optional[Dict[str, Any]],
    comment_truncation: Optional[Dict[str, Any]],
    comment_fallback: Optional[Dict[str, Any]],
    fallback_warning: str,
) -> None:
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
            warnings_out.append(fallback_warning)


def _place_market_order(  # noqa: C901
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
    mt5, connection_error = _prepare_order_gateway(gateway)
    if connection_error is not None:
        return connection_error

    def _place_market_order():  # noqa: C901
        try:
            order_context, order_context_error = _prepare_order_symbol_context(
                mt5,
                symbol=symbol,
                volume=volume,
            )
            if order_context_error is not None:
                return order_context_error
            symbol_info = order_context["symbol_info"]
            volume_validated = order_context["volume"]

            # Normalize and validate requested order type
            t, order_type_error = validation._normalize_order_type_input(order_type)
            if order_type_error:
                return {"error": order_type_error}
            if t not in {"BUY", "SELL"}:
                return {"error": f"Unsupported order_type '{order_type}'. Use BUY or SELL for market orders."}
            side = t

            deviation_validated, deviation_error = validation._validate_deviation(deviation)
            if deviation_error:
                return {"error": deviation_error}

            price_inputs, price_inputs_error = validation._normalize_trade_price_inputs(
                symbol_info=symbol_info,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            if price_inputs_error is not None:
                return {"error": price_inputs_error}
            norm_sl = price_inputs["stop_loss"]
            norm_tp = price_inputs["take_profit"]

            # Validate against a recent quote, then refresh again right before send.
            validate_tick = mt5.symbol_info_tick(symbol)
            if validate_tick is None:
                return {"error": f"Failed to get current price for {symbol}"}
            validate_reference_price = validate_tick.bid if side == "BUY" else validate_tick.ask

            # SL/TP validation for market orders
            if norm_sl is not None:
                if side == "BUY" and norm_sl >= validate_reference_price:
                    return {
                        "error": (
                            "stop_loss must be below the live bid for BUY orders. "
                            f"sl={norm_sl}, bid={validate_tick.bid}, ask={validate_tick.ask}"
                        )
                    }
                if side == "SELL" and norm_sl <= validate_reference_price:
                    return {
                        "error": (
                            "stop_loss must be above the live ask for SELL orders. "
                            f"sl={norm_sl}, bid={validate_tick.bid}, ask={validate_tick.ask}"
                        )
                    }
            if norm_tp is not None:
                if side == "BUY" and norm_tp <= validate_reference_price:
                    return {
                        "error": (
                            "take_profit must be above the live bid for BUY orders. "
                            f"tp={norm_tp}, bid={validate_tick.bid}, ask={validate_tick.ask}"
                        )
                    }
                if side == "SELL" and norm_tp >= validate_reference_price:
                    return {
                        "error": (
                            "take_profit must be below the live ask for SELL orders. "
                            f"tp={norm_tp}, bid={validate_tick.bid}, ask={validate_tick.ask}"
                        )
                    }
            live_protection_error = validation._validate_live_protection_levels(
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
            send_reference_price = send_tick.bid if side == "BUY" else send_tick.ask
            if norm_sl is not None:
                if side == "BUY" and norm_sl >= send_reference_price:
                    return {
                        "error": (
                            "stop_loss must be below the live bid for BUY orders at send time. "
                            f"sl={norm_sl}, bid={send_tick.bid}, ask={send_tick.ask}"
                        )
                    }
                if side == "SELL" and norm_sl <= send_reference_price:
                    return {
                        "error": (
                            "stop_loss must be above the live ask for SELL orders at send time. "
                            f"sl={norm_sl}, bid={send_tick.bid}, ask={send_tick.ask}"
                        )
                    }
            if norm_tp is not None:
                if side == "BUY" and norm_tp <= send_reference_price:
                    return {
                        "error": (
                            "take_profit must be above the live bid for BUY orders at send time. "
                            f"tp={norm_tp}, bid={send_tick.bid}, ask={send_tick.ask}"
                        )
                    }
                if side == "SELL" and norm_tp >= send_reference_price:
                    return {
                        "error": (
                            "take_profit must be below the live ask for SELL orders at send time. "
                            f"tp={norm_tp}, bid={send_tick.bid}, ask={send_tick.ask}"
                        )
                    }
            live_protection_error = validation._validate_live_protection_levels(
                symbol_info=symbol_info,
                tick=send_tick,
                side=side,
                stop_loss=norm_sl,
                take_profit=norm_tp,
            )
            if live_protection_error is not None:
                return live_protection_error
            request_comment, comment_sanitization, comment_truncation = _build_order_comment_payload(
                comment,
                default="MCP order",
            )
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume_validated,
                "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": deviation_validated,
                "magic": _configured_order_magic(),
                "comment": request_comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": validation._safe_int_attr(mt5, "ORDER_FILLING_IOC", 1),
            }
            send_outcome, send_error = _submit_order_request(
                mt5,
                request,
                base_error="Failed to send order",
                invalid_comment_error=(
                    "Failed to send order: broker rejected the comment field. "
                    "Comments are sanitized to letters/numbers/spaces/_/./-; "
                    "try a simpler comment or omit --comment."
                ),
                symbol_info=symbol_info,
            )
            if send_error is not None:
                return send_error
            result = send_outcome["result"]
            comment_fallback = send_outcome["comment_fallback"]
            fill_mode_attempts = send_outcome["fill_mode_attempts"]
            used_request = send_outcome["used_request"]

            # Resolve position and attach SL/TP protection
            order_ticket = validation._safe_int_ticket(getattr(result, "order", None))
            deal_ticket = validation._safe_int_ticket(getattr(result, "deal", None))
            position_ticket_candidates: List[int] = []
            for cand in (order_ticket, deal_ticket):
                if cand is not None and cand not in position_ticket_candidates:
                    position_ticket_candidates.append(cand)

            protection = _attach_post_fill_protection(
                mt5,
                symbol=symbol,
                side=side,
                volume=volume_validated,
                position_ticket_candidates=position_ticket_candidates,
                stop_loss=norm_sl,
                take_profit=norm_tp,
                symbol_info=symbol_info,
                comment=comment,
                request_comment=request_comment,
                magic=validation._safe_int_ticket(request.get("magic")),
            )

            warnings_out: List[str] = list(protection.get("warnings") or [])
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
                "position_ticket": protection.get("position_ticket"),
                "position_ticket_candidates": protection.get("position_ticket_candidates"),
                "position_ticket_resolution": protection.get("position_ticket_resolution"),
                "type_filling_used": used_request.get("type_filling"),
                "sl_tp_result": protection["sl_tp_result"],
            }
            _attach_comment_response_metadata(
                out,
                warnings_out,
                comment_sanitization=comment_sanitization,
                comment_truncation=comment_truncation,
                comment_fallback=comment_fallback,
                fallback_warning=(
                    "Broker rejected the comment field; order was retried with a minimal MT5-safe comment."
                ),
            )
            if "protection_status" in protection:
                out["protection_status"] = protection["protection_status"]
            if fill_mode_attempts:
                out["fill_mode_attempts"] = fill_mode_attempts
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
    mt5, connection_error = _prepare_order_gateway(gateway)
    if connection_error is not None:
        return connection_error

    def _place_pending_order():
        try:
            order_context, order_context_error = _prepare_order_symbol_context(
                mt5,
                symbol=symbol,
                volume=volume,
            )
            if order_context_error is not None:
                return order_context_error
            symbol_info = order_context["symbol_info"]
            volume_validated = order_context["volume"]

            initial_tick = mt5.symbol_info_tick(symbol)
            if initial_tick is None:
                return {"error": f"Failed to get current price for {symbol}"}

            deviation_validated, deviation_error = validation._validate_deviation(deviation)
            if deviation_error:
                return {"error": deviation_error}

            # Normalize and validate requested order type
            t, order_type_error = validation._normalize_order_type_input(order_type)
            if order_type_error:
                return {"error": order_type_error}
            explicit_map = {
                "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
                "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
                "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
                "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP,
            }

            price_inputs, price_inputs_error = validation._normalize_trade_price_inputs(
                symbol_info=symbol_info,
                price=price,
                require_price=True,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            if price_inputs_error is not None:
                return {"error": price_inputs_error}
            point = float(price_inputs["point"])
            norm_price = float(price_inputs["price"])
            norm_sl = price_inputs["stop_loss"]
            norm_tp = price_inputs["take_profit"]

            normalized_expiration, expiration_specified = time._normalize_pending_expiration(expiration)
            live_tick = mt5.symbol_info_tick(symbol) or initial_tick
            if live_tick is None:
                return {"error": f"Failed to refresh current price for {symbol}"}
            bid = float(getattr(live_tick, "bid", 0.0) or 0.0)
            ask = float(getattr(live_tick, "ask", 0.0) or 0.0)
            price_tol = point * 0.1 if point > 0 else 1e-9

            if t == "BUY" and abs(norm_price - ask) <= price_tol:
                return {
                    "error": (
                        "price is at market for BUY pending order. "
                        f"price={norm_price}, ask={ask}. Use a market order or move price away from ask."
                    )
                }
            if t == "SELL" and abs(norm_price - bid) <= price_tol:
                return {
                    "error": (
                        "price is at market for SELL pending order. "
                        f"price={norm_price}, bid={bid}. Use a market order or move price away from bid."
                    )
                }

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

            pending_level_error = validation._validate_pending_order_levels(
                symbol_info=symbol_info,
                tick=live_tick,
                order_type_value=order_type_value,
                price=float(norm_price),
                stop_loss=None if norm_sl is None else float(norm_sl),
                take_profit=None if norm_tp is None else float(norm_tp),
                mt5=mt5,
            )
            if pending_level_error is not None:
                return pending_level_error

            request_comment, comment_sanitization, comment_truncation = _build_order_comment_payload(
                comment,
                default="MCP pending order",
            )
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume_validated,
                "type": order_type_value,
                "price": norm_price,
                "sl": 0.0 if norm_sl is None else float(norm_sl),
                "tp": 0.0 if norm_tp is None else float(norm_tp),
                "deviation": deviation_validated,
                "magic": _configured_order_magic(),
                "comment": request_comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": validation._safe_int_attr(mt5, "ORDER_FILLING_IOC", 1),
            }

            if expiration_specified:
                if normalized_expiration is None:
                    request["type_time"] = mt5.ORDER_TIME_GTC
                else:
                    request["type_time"] = mt5.ORDER_TIME_SPECIFIED
                    request["expiration"] = normalized_expiration

            send_outcome, send_error = _submit_order_request(
                mt5,
                request,
                base_error="Failed to send pending order",
                invalid_comment_error=(
                    "Failed to send pending order: broker rejected the comment field. "
                    "Comments are sanitized to letters/numbers/spaces/_/./-; "
                    "try a simpler comment or omit --comment."
                ),
                symbol_info=symbol_info,
            )
            if send_error is not None:
                return send_error
            result = send_outcome["result"]
            comment_fallback = send_outcome["comment_fallback"]
            fill_mode_attempts = send_outcome["fill_mode_attempts"]
            used_request = send_outcome["used_request"]

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
                "type_filling_used": used_request.get("type_filling"),
            }
            if expiration_specified:
                out["requested_expiration"] = normalized_expiration
            warnings_out: List[str] = []
            _attach_comment_response_metadata(
                out,
                warnings_out,
                comment_sanitization=comment_sanitization,
                comment_truncation=comment_truncation,
                comment_fallback=comment_fallback,
                fallback_warning=(
                    "Broker rejected the comment field; pending order was retried with a minimal MT5-safe comment."
                ),
            )
            if fill_mode_attempts:
                out["fill_mode_attempts"] = fill_mode_attempts
            if warnings_out:
                out["warnings"] = warnings_out
            return out

        except Exception as e:
            return {"error": str(e)}

    return _place_pending_order()
