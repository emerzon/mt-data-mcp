"""Order placement workflows for MetaTrader integration."""

import logging
import math
import time as _stdlib_time
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

from ...bootstrap.settings import mt5_config, trade_guardrails_config
from . import comments, common, time, validation
from .gateway import MT5TradingGateway, create_trading_gateway, trading_connection_error
from .positions import _resolve_open_position
from .safety import evaluate_trade_guardrails
from .time import ExpirationValue
from .validation import MarketOrderTypeInput, OrderTypeInput


class _OrderSymbolContext(TypedDict):
    symbol_info: Any
    volume: float


class _OrderSubmitOutcome(TypedDict):
    result: Any
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
                    "attempts": int(attempt_idx + 1),
                    "matched": True,
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
            invalid_stops_code = validation._safe_int_attr(mt5, "TRADE_RETCODE_INVALID_STOPS", 10016)
            for modify_try in range(max_modify_attempts):
                sl_tp_attempts = int(modify_try + 1)
                try:
                    modify_result = mt5.order_send(modify_request)
                    _sl_tp_last_error = validation._safe_last_error(mt5)
                except Exception as ex:
                    modify_result = None
                    sl_tp_error = f"Error setting TP/SL: {str(ex)}"
                if modify_result is not None:
                    sl_tp_last_retcode = getattr(modify_result, "retcode", None)
                    sl_tp_last_comment = getattr(modify_result, "comment", None)
                if modify_result and getattr(modify_result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                    break
                if sl_tp_last_retcode == invalid_stops_code:
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
                        sl_applied = validation._safe_float_attr(pos_after, "sl") or None
                        tp_applied = validation._safe_float_attr(pos_after, "tp") or None
                except Exception as verify_exc:
                    sl_tp_verification_failed = True
                    logger.warning(
                        "SL/TP verification failed for ticket %s: %s",
                        position_ticket,
                        verify_exc,
                    )

                price_tol = validation._safe_float_attr(symbol_info, "point")
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
            verification_failed=sl_tp_verification_failed,
        ),
        "warnings": warnings_out,
    }
    if sl_tp_apply_status == "applied":
        outcome["protection_status"] = "protected"
    elif sl_tp_apply_status == "failed":
        outcome["protection_status"] = "unprotected_position"
    return outcome


def _send_order_with_fill_mode_retry(
    mt5: Any,
    request: Dict[str, Any],
    *,
    symbol_info: Any = None,
    price_refresh: Optional[Callable[[Dict[str, Any]], bool]] = None,
    max_price_retries: int = 3,
) -> tuple[Any, Any, List[Dict[str, Any]], Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    last_result = None
    last_error = None
    last_request = dict(request)
    price_changed_codes = {
        validation._safe_int_attr(mt5, "TRADE_RETCODE_PRICE_CHANGED", 10020),
        validation._safe_int_attr(mt5, "TRADE_RETCODE_REQUOTE", 10004),
    }
    terminal_retcode_failures = {
        validation._safe_int_attr(mt5, "TRADE_RETCODE_REJECT", 10006),
        validation._safe_int_attr(mt5, "TRADE_RETCODE_TRADE_DISABLED", 10017),
        validation._safe_int_attr(mt5, "TRADE_RETCODE_MARKET_CLOSED", 10018),
        validation._safe_int_attr(mt5, "TRADE_RETCODE_NO_MONEY", 10019),
        validation._safe_int_attr(mt5, "TRADE_RETCODE_TOO_MANY_REQUESTS", 10024),
        validation._safe_int_attr(mt5, "TRADE_RETCODE_INVALID_STOPS", 10016),
        validation._safe_int_attr(mt5, "TRADE_RETCODE_INVALID", 10013),
    }
    for fill_mode in validation._candidate_fill_modes(mt5, symbol_info):
        attempt_request = dict(request)
        attempt_request["type_filling"] = int(fill_mode)
        price_retry_count = 0
        while True:
            request_to_send = dict(attempt_request)
            result = mt5.order_send(request_to_send)
            last_error = validation._safe_last_error(mt5)
            attempt: Dict[str, Any] = {"type_filling": int(fill_mode)}
            if price_retry_count > 0:
                attempt["price_retry"] = int(price_retry_count)
            if result is None:
                attempt["error"] = "order_send returned None"
                if last_error is not None:
                    attempt["last_error"] = last_error
            else:
                retcode = getattr(result, "retcode", None)
                attempt["retcode"] = retcode
                attempt["retcode_name"] = mt5.retcode_name(retcode)
                attempt["comment"] = getattr(result, "comment", None)
            attempts.append(attempt)
            last_result = result
            last_request = request_to_send
            try:
                if result is not None and validation._retcode_is_done(mt5, getattr(result, "retcode", -1)):
                    return result, last_error, attempts, last_request
            except Exception:
                pass

            retcode = getattr(result, "retcode", None) if result is not None else None
            if retcode in terminal_retcode_failures:
                return result, last_error, attempts, last_request
            should_retry_same_fill = (
                retcode in price_changed_codes
                and price_retry_count < max(0, int(max_price_retries))
            )
            if not should_retry_same_fill:
                break

            refreshed = True
            if price_refresh is not None:
                try:
                    refreshed = bool(price_refresh(attempt_request))
                except Exception:
                    refreshed = False
            if not refreshed:
                break

            price_retry_count += 1
            _stdlib_time.sleep(0.15)
    return last_result, last_error, attempts, last_request


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


def _evaluate_live_trade_guardrails(
    mt5: Any,
    *,
    symbol: str,
    volume: float,
    stop_loss: Optional[float],
    deviation: Optional[int],
    side: str,
    entry_price: float,
    symbol_info: Any,
) -> Optional[Dict[str, Any]]:
    if not trade_guardrails_config.is_enabled():
        return None
    try:
        account_info = mt5.account_info()
    except Exception:
        account_info = None
    try:
        positions = mt5.positions_get()
    except Exception:
        positions = None
    return evaluate_trade_guardrails(
        trade_guardrails_config,
        symbol=symbol,
        volume=volume,
        stop_loss=stop_loss,
        deviation=deviation,
        side=side,
        entry_price=entry_price,
        account_info=account_info,
        existing_positions=list(positions or []),
        symbol_info=symbol_info,
        symbol_info_resolver=mt5.symbol_info,
    )


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
    price_refresh: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> tuple[Optional[_OrderSubmitOutcome], Optional[Dict[str, Any]]]:
    result, last_error, fill_mode_attempts, used_request = _send_order_with_fill_mode_retry(
        mt5,
        request,
        symbol_info=symbol_info,
        price_refresh=price_refresh,
    )
    if result is None:
        return None, {
            "error": base_error,
            "last_error": last_error,
            "fill_mode_attempts": fill_mode_attempts,
        }

    if not validation._retcode_is_done(mt5, getattr(result, "retcode", None)):
        error_message = base_error
        invalid_comment = comments._invalid_comment_error_text(result, last_error)
        if invalid_comment is not None:
            error_message = invalid_comment_error
        return None, {
            "error": error_message,
            "retcode": result.retcode,
            "retcode_name": mt5.retcode_name(result.retcode),
            "comment": result.comment,
            "request_id": result.request_id,
            "last_error": last_error,
            "fill_mode_attempts": fill_mode_attempts,
        }

    return {
        "result": result,
        "fill_mode_attempts": fill_mode_attempts,
        "used_request": used_request,
    }, None


def _attach_comment_response_metadata(
    out: Dict[str, Any],
    warnings_out: List[str],
    *,
    comment_sanitization: Optional[Dict[str, Any]],
    comment_truncation: Optional[Dict[str, Any]],
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


def _round_preview_price(value: Optional[float], *, digits: int) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), max(0, int(digits)))


def _order_type_constant(mt5: Any, order_type: str) -> Optional[int]:
    constant_name = f"ORDER_TYPE_{str(order_type).upper().strip()}"
    return validation._safe_int_attr(mt5, constant_name, None)


def _level_preview_fields(
    prefix: str,
    level: Optional[Union[int, float]],
    *,
    entry_price: float,
    point: float,
) -> Dict[str, Any]:
    try:
        level_value = float(level) if level not in (None, 0) else None
    except (TypeError, ValueError):
        level_value = None
    if level_value is None or not math.isfinite(level_value):
        return {}

    out: Dict[str, Any] = {}
    if point > 0:
        out[f"{prefix}_distance_points"] = round(abs(level_value - entry_price) / point, 2)
    if entry_price:
        out[f"{prefix}_distance_pct"] = round(
            ((level_value - entry_price) / entry_price) * 100.0,
            6,
        )
    return out


def _margin_preview_fields(
    mt5: Any,
    *,
    order_type_value: Optional[int],
    symbol: str,
    volume: float,
    entry_price: float,
) -> Dict[str, Any]:
    if order_type_value is None:
        return {}
    order_calc_margin = getattr(getattr(mt5, "adapter", None), "order_calc_margin", None)
    if not callable(order_calc_margin):
        return {}

    margin = validation.coerce_finite_float(
        order_calc_margin(order_type_value, symbol, volume, entry_price)
    )
    if margin is None:
        return {}

    out: Dict[str, Any] = {"margin_required": round(float(margin), 2)}
    account_info = mt5.account_info()
    margin_free = validation._safe_float_attr(account_info, "margin_free")
    if margin_free is not None and math.isfinite(margin_free):
        out["margin_free"] = round(float(margin_free), 2)
        out["margin_sufficient"] = float(margin_free) >= float(margin)
    return out


def build_trade_place_dry_run_preview(
    *,
    symbol: str,
    volume: float,
    order_type: str,
    pending: bool,
    price: Optional[Union[int, float]],
    stop_loss: Optional[Union[int, float]],
    take_profit: Optional[Union[int, float]],
    gateway: Optional[MT5TradingGateway] = None,
) -> Dict[str, Any]:
    """Build a quote-based order preview without sending or checking an order."""
    mt5 = create_trading_gateway(gateway=gateway)
    connection_error = trading_connection_error(mt5)
    if connection_error is not None:
        return {"preview_error": connection_error.get("error")}

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return {"preview_error": f"Symbol {symbol} not found"}
    if not getattr(symbol_info, "visible", True) and not mt5.symbol_select(symbol, True):
        return {"preview_error": f"Failed to select symbol {symbol}"}

    volume_validated, volume_error = validation._validate_volume(volume, symbol_info)
    if volume_error:
        return {"preview_error": volume_error}

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"preview_error": f"Failed to get current price for {symbol}"}

    bid = validation._safe_float_attr(tick, "bid")
    ask = validation._safe_float_attr(tick, "ask")
    if bid is None or ask is None or not math.isfinite(bid) or not math.isfinite(ask):
        return {"preview_error": f"Failed to get valid bid/ask for {symbol}"}

    digits = validation._safe_int_attr(symbol_info, "digits", 6) or 6
    point = validation._safe_float_attr(symbol_info, "point") or 0.0
    side = "BUY" if str(order_type).upper().startswith("BUY") else "SELL"
    order_type_value = _order_type_constant(mt5, order_type)
    entry_price = float(price) if pending and price not in (None, 0) else (ask if side == "BUY" else bid)

    out: Dict[str, Any] = {
        "bid": _round_preview_price(bid, digits=digits),
        "ask": _round_preview_price(ask, digits=digits),
        "estimated_fill_price": _round_preview_price(entry_price, digits=digits),
    }
    if pending:
        out["entry_price"] = _round_preview_price(entry_price, digits=digits)

    if point > 0:
        spread_price = ask - bid
        out["spread_points"] = round(spread_price / point, 2)
        midpoint = (ask + bid) / 2.0
        if midpoint:
            out["spread_pct"] = round((spread_price / midpoint) * 100.0, 6)
        broker_distance = validation._broker_distance_metadata(symbol_info)
        out["min_distance_points"] = int(broker_distance["min_distance_points"])

    out.update(
        _level_preview_fields(
            "sl",
            stop_loss,
            entry_price=entry_price,
            point=point,
        )
    )
    out.update(
        _level_preview_fields(
            "tp",
            take_profit,
            entry_price=entry_price,
            point=point,
        )
    )

    validation_error: Optional[Dict[str, Any]] = None
    if pending:
        if order_type_value is not None:
            validation_error = validation._validate_pending_order_levels(
                symbol_info=symbol_info,
                tick=tick,
                order_type_value=order_type_value,
                price=float(entry_price),
                stop_loss=float(stop_loss) if stop_loss not in (None, 0) else None,
                take_profit=float(take_profit) if take_profit not in (None, 0) else None,
                mt5=mt5,
            )
    else:
        validation_error = validation._validate_live_protection_levels(
            symbol_info=symbol_info,
            tick=tick,
            side=side,
            stop_loss=float(stop_loss) if stop_loss not in (None, 0) else None,
            take_profit=float(take_profit) if take_profit not in (None, 0) else None,
        )
    if stop_loss not in (None, 0) or take_profit not in (None, 0):
        out["sl_tp_valid"] = validation_error is None
        if validation_error is not None:
            out["sl_tp_error"] = validation_error.get("error")

    out.update(
        _margin_preview_fields(
            mt5,
            order_type_value=order_type_value,
            symbol=symbol,
            volume=float(volume_validated),
            entry_price=float(entry_price),
        )
    )
    return {key: value for key, value in out.items() if value is not None}


def _place_market_order(  # noqa: C901
    symbol: str,
    volume: float,
    order_type: MarketOrderTypeInput,
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
    comment: Optional[str] = None,
    magic: Optional[int] = None,
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
            guardrail_block = _evaluate_live_trade_guardrails(
                mt5,
                symbol=symbol,
                volume=volume_validated,
                stop_loss=norm_sl,
                deviation=deviation_validated,
                side=side,
                entry_price=price,
                symbol_info=symbol_info,
            )
            if guardrail_block is not None:
                return guardrail_block
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
                "magic": magic if magic is not None else _configured_order_magic(),
                "comment": request_comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": validation._safe_int_attr(mt5, "ORDER_FILLING_IOC", 1),
            }
            if norm_sl is not None:
                request["sl"] = float(norm_sl)
            if norm_tp is not None:
                request["tp"] = float(norm_tp)

            def _refresh_market_order_price(attempt_request: Dict[str, Any]) -> bool:
                refreshed_tick = mt5.symbol_info_tick(symbol)
                if refreshed_tick is None:
                    return False
                refreshed_price = refreshed_tick.ask if side == "BUY" else refreshed_tick.bid
                try:
                    refreshed_price = float(refreshed_price)
                except Exception:
                    return False
                if not math.isfinite(refreshed_price) or refreshed_price <= 0.0:
                    return False
                attempt_request["price"] = refreshed_price
                return True

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
                price_refresh=_refresh_market_order_price,
            )
            if send_error is not None:
                return send_error
            result = send_outcome["result"]
            fill_mode_attempts = send_outcome["fill_mode_attempts"]
            used_request = send_outcome["used_request"]

            order_ticket = validation._safe_int_ticket(getattr(result, "order", None))
            deal_ticket = validation._safe_int_ticket(getattr(result, "deal", None))
            position_ticket_candidates: List[int] = []
            for cand in (order_ticket, deal_ticket):
                if cand is not None and cand not in position_ticket_candidates:
                    position_ticket_candidates.append(cand)

            position_ticket = None
            position_ticket_resolution = None
            if position_ticket_candidates:
                position_obj, resolved_ticket, resolve_info = _resolve_open_position(
                    mt5,
                    ticket_candidates=position_ticket_candidates,
                    symbol=symbol,
                    side=side,
                    volume=volume_validated,
                    magic=validation._safe_int_ticket(request.get("magic")),
                )
                position_ticket = resolved_ticket
                position_ticket_resolution = {
                    **dict(resolve_info),
                    "matched": position_obj is not None and resolved_ticket is not None,
                }
            sl_tp_requested = bool(norm_sl is not None or norm_tp is not None)
            sl_tp_result = _build_sl_tp_result(
                requested_sl=norm_sl,
                requested_tp=norm_tp,
                applied_sl=norm_sl if sl_tp_requested else None,
                applied_tp=norm_tp if sl_tp_requested else None,
                status="applied" if sl_tp_requested else "not_requested",
                error=None,
                broker_adjusted=False,
                adjustment=None,
                attempts=1 if sl_tp_requested else 0,
                last_retcode=getattr(result, "retcode", None) if sl_tp_requested else None,
                last_comment=getattr(result, "comment", None) if sl_tp_requested else None,
            )

            warnings_out: List[str] = []
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
                "type_filling_used": used_request.get("type_filling"),
                "sl_tp_result": sl_tp_result,
            }
            _attach_comment_response_metadata(
                out,
                warnings_out,
                comment_sanitization=comment_sanitization,
                comment_truncation=comment_truncation,
            )
            if sl_tp_requested:
                out["protection_status"] = "protected"
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
    magic: Optional[int] = None,
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
            bid = validation._safe_float_attr(live_tick, "bid")
            ask = validation._safe_float_attr(live_tick, "ask")
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
            guardrail_block = _evaluate_live_trade_guardrails(
                mt5,
                symbol=symbol,
                volume=volume_validated,
                stop_loss=None if norm_sl is None else float(norm_sl),
                deviation=deviation_validated,
                side="BUY" if "BUY" in str(t) else "SELL",
                entry_price=float(norm_price),
                symbol_info=symbol_info,
            )
            if guardrail_block is not None:
                return guardrail_block

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
                "magic": magic if magic is not None else _configured_order_magic(),
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
            )
            if fill_mode_attempts:
                out["fill_mode_attempts"] = fill_mode_attempts
            if warnings_out:
                out["warnings"] = warnings_out
            return out

        except Exception as e:
            return {"error": str(e)}

    return _place_pending_order()
