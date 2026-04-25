from __future__ import annotations

import copy
import json
import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ...shared.constants import TIMEFRAME_MAP
from ...shared.result import Err, Ok, Result, to_dict
from ...shared.validators import invalid_timeframe_error
from ...utils.barriers import normalize_trade_direction
from ...utils.mt5 import (
    MT5ConnectionError,
    _ensure_symbol_ready,
    _normalize_times_in_struct,
    _to_utc_history_query_dt,
)
from ..config import trade_guardrails_config
from ..execution_logging import (
    infer_result_success,
    log_operation_finish,
    log_operation_start,
    run_logged_operation,
)
from ..output_contract import resolve_output_contract
from . import validation
from .idempotency import IdempotencyStore
from .requests import (
    TradeCloseRequest,
    TradeGetOpenRequest,
    TradeGetPendingRequest,
    TradeHistoryRequest,
    TradeModifyRequest,
    TradePlaceRequest,
    TradeRiskAnalyzeRequest,
    TradeVarCvarRequest,
)
from .safety import evaluate_trade_guardrails, preview_trade_guardrails
from .sizing import _resolve_risk_tick_value

logger = logging.getLogger(__name__)
_DEFAULT_TRADE_HISTORY_LOOKBACK_DAYS = 7
_TRADE_IDEMPOTENCY_STORE = IdempotencyStore()
_TRADE_HISTORY_RANGE_HINT = (
    "Try narrowing the range with --minutes-back, --days, --start, or --end."
)
_TRADE_PLACE_PREVIEW_KEYS = (
    "success",
    "dry_run",
    "no_action",
    "symbol",
    "order_type",
    "pending",
    "action",
    "volume",
    "message",
    "actionability",
    "preview_scope_summary",
    "require_sl_tp",
    "auto_close_on_sl_tp_fail",
    "requested_price",
    "requested_sl",
    "requested_tp",
    "expiration",
    "expiration_normalized",
)
_TRADE_PLACE_BASIC_KEYS = _TRADE_PLACE_PREVIEW_KEYS + (
    "actionability_reason",
    "validation_not_performed",
    "warnings",
    "guardrails_preview",
)


def _trade_row_to_dict(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "_asdict"):
        return dict(row._asdict())
    try:
        return dict(vars(row))
    except TypeError as exc:
        raise TypeError(f"Unsupported trade row type: {type(row).__name__}") from exc


def _trade_rows_to_dataframe(rows: Any, *, pd_module: Any) -> Any:
    row_dicts = [_trade_row_to_dict(row) for row in list(rows)]
    if not row_dicts:
        return pd_module.DataFrame()
    return pd_module.DataFrame.from_records(
        row_dicts,
        columns=list(row_dicts[0].keys()),
    )


def _resolve_trade_place_preview_detail(request: TradePlaceRequest) -> str:
    contract = resolve_output_contract(
        request,
        detail=request.preview_detail,
        default_detail="basic",
        aliases={
            "compact": "preview",
            "summary": "preview",
        },
    )
    if contract.shape_detail == "full":
        return "full"
    if contract.detail == "preview":
        return "preview"
    return "basic"


def _shape_trade_place_preview(
    payload: Dict[str, Any], *, detail: str
) -> Dict[str, Any]:
    if detail == "full":
        return dict(payload)
    keys = _TRADE_PLACE_BASIC_KEYS if detail == "basic" else _TRADE_PLACE_PREVIEW_KEYS
    return {key: payload[key] for key in keys if key in payload}


def _sl_tp_result_details(result: Dict[str, Any]) -> tuple[bool, str, bool]:
    sl_tp_result = result.get("sl_tp_result")
    if isinstance(sl_tp_result, dict):
        requested = sl_tp_result.get("requested")
        requested_bool = isinstance(requested, dict) and bool(requested)
        status = str(sl_tp_result.get("status") or "").lower()
        fallback_used = bool(sl_tp_result.get("fallback_used"))
        return requested_bool, status, fallback_used
    requested_bool = bool(result.get("sl_tp_requested"))
    status = str(result.get("sl_tp_apply_status") or "").lower()
    fallback_used = bool(result.get("sl_tp_fallback_used"))
    return requested_bool, status, fallback_used


def _guardrail_order_side(order_type: Optional[str]) -> Optional[str]:
    text = str(order_type or "").strip().upper()
    if text.startswith("BUY"):
        return "BUY"
    if text.startswith("SELL"):
        return "SELL"
    return None


def _normalize_idempotency_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip()
    return key or None


def _build_trade_request_signature(request: Any) -> Optional[str]:
    if request is None:
        return None
    try:
        payload = request.model_dump(mode="json")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    payload.pop("idempotency_key", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _idempotency_duplicate_response(
    *,
    key: str,
    original_outcome: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "success": infer_result_success(original_outcome),
        "duplicate": True,
        "idempotency_key": key,
        "message": "Duplicate request suppressed by idempotency key.",
        "original_outcome": original_outcome,
    }


def _begin_trade_idempotency(
    *,
    idempotency_store: Optional[IdempotencyStore],
    key: Optional[str],
    request_signature: Optional[str],
) -> tuple[Optional[Dict[str, Any]], bool]:
    if idempotency_store is None or key is None:
        return None, False
    duplicate = idempotency_store.reserve(
        key,
        request_signature=request_signature,
    )
    if duplicate is None:
        return None, True
    stored_signature = duplicate.get("request_signature")
    if (
        stored_signature is not None
        and request_signature is not None
        and stored_signature != request_signature
    ):
        return {
            "error": (
                "Idempotency key was already used for a different trade request. "
                "Use a new idempotency_key when changing parameters."
            ),
            "idempotency_key": key,
            "idempotency_conflict": True,
        }, False
    original_outcome = duplicate.get("original_outcome")
    if not isinstance(original_outcome, dict):
        return {
            "error": "Stored idempotency outcome is invalid; use a new idempotency_key.",
            "idempotency_key": key,
            "idempotency_conflict": True,
        }, False
    return _idempotency_duplicate_response(
        key=key,
        original_outcome=copy.deepcopy(original_outcome),
    ), False


def _resolve_trade_risk_direction(
    *,
    direction: Any,
    entry: float,
    stop_loss: float,
    take_profit: float | None = None,
) -> tuple[str | None, str | None, str]:
    direction_text = str(direction).strip() if direction is not None else ""
    if direction_text:
        direction_norm, direction_error = normalize_trade_direction(direction_text)
        return direction_norm, direction_error, "explicit"
    if stop_loss < entry:
        return "long", None, "inferred_from_stop_loss"
    if stop_loss > entry:
        return "short", None, "inferred_from_stop_loss"
    if take_profit is not None:
        if take_profit > entry:
            return "long", None, "inferred_from_take_profit"
        if take_profit < entry:
            return "short", None, "inferred_from_take_profit"
    return (
        None,
        "Unable to infer trade direction when stop_loss equals entry "
        "and take_profit is missing or also equals entry. "
        "Provide direction='long' or direction='short'.",
        "unable_to_infer",
    )


def _validate_trade_risk_levels(
    *,
    direction: str,
    entry: float,
    stop_loss: float,
    take_profit: float | None,
) -> str | None:
    if direction == "long":
        if stop_loss > entry:
            return "For long trades, stop_loss must be below entry."
        if take_profit is not None and take_profit <= entry:
            return "For long trades, take_profit must be above entry."
        return None
    if stop_loss < entry:
        return "For short trades, stop_loss must be above entry."
    if take_profit is not None and take_profit >= entry:
        return "For short trades, take_profit must be below entry."
    return None


def _floor_volume_steps(raw_volume: float, volume_step: float) -> int:
    step_ratio = raw_volume / volume_step
    step_count = math.floor(step_ratio)
    if step_count < 0:
        return 0

    remainder_to_next_step = float(step_count + 1) - float(step_ratio)
    if remainder_to_next_step > 0.0:
        # Correct only machine-scale underflow so exact step-sized volumes are kept,
        # while materially smaller values still round down conservatively.
        snap_tolerance = math.ulp(step_ratio) * 8.0
        if remainder_to_next_step <= snap_tolerance:
            return step_count + 1
    return step_count


def _normalize_var_cvar_method(method: Any) -> tuple[Optional[str], Optional[str]]:
    method_text = str(method or "historical").strip().lower()
    if method_text in {"historical", "hist"}:
        return "historical", None
    if method_text in {"gaussian", "normal", "parametric"}:
        return "gaussian", None
    return None, "Invalid method. Valid options: historical, gaussian"


def _normalize_var_cvar_transform(
    transform: Any,
) -> tuple[Optional[str], Optional[str]]:
    transform_text = str(transform or "log_return").strip().lower()
    if transform_text in {"log_return", "log_returns", "log"}:
        return "log_return", None
    if transform_text in {
        "pct",
        "pct_return",
        "pct_returns",
        "percent",
        "percent_return",
        "percent_returns",
        "simple_return",
        "simple_returns",
    }:
        return "pct", None
    return None, "Invalid transform. Valid options: log_return, pct"


def _normalize_var_cvar_confidence(
    confidence: Any,
) -> tuple[Optional[float], Optional[str]]:
    try:
        confidence_value = float(confidence)
    except TypeError, ValueError:
        return None, "confidence must be numeric"
    if not math.isfinite(confidence_value):
        return None, "confidence must be finite"
    if confidence_value > 1.0:
        confidence_value /= 100.0
    if confidence_value <= 0.0 or confidence_value >= 1.0:
        return (
            None,
            "confidence must be between 0 and 1, or between 0 and 100 as a percentage",
        )
    return confidence_value, None


def _historical_var_cvar_tail(
    pnl_values: List[float], confidence: float
) -> tuple[float, float, float]:
    ordered = sorted(float(value) for value in pnl_values)
    if not ordered:
        return 0.0, 0.0, 0.0
    alpha = 1.0 - confidence
    index = max(0, min(len(ordered) - 1, int(math.floor(alpha * (len(ordered) - 1)))))
    threshold = float(ordered[index])
    tail_values = [float(value) for value in ordered[: index + 1]]
    tail_mean = float(sum(tail_values) / len(tail_values)) if tail_values else threshold
    var_value = max(0.0, -threshold)
    cvar_value = max(0.0, -tail_mean)
    return var_value, cvar_value, threshold


def _gaussian_var_cvar_tail(
    pnl_values: List[float], confidence: float
) -> tuple[float, float, float]:
    from scipy.stats import norm

    ordered = [float(value) for value in pnl_values]
    if not ordered:
        return 0.0, 0.0, 0.0
    mean_pnl = float(sum(ordered) / len(ordered))
    if len(ordered) == 1:
        threshold = mean_pnl
        var_value = max(0.0, -threshold)
        return var_value, var_value, threshold
    variance = sum((value - mean_pnl) ** 2 for value in ordered) / float(
        len(ordered) - 1
    )
    std_pnl = math.sqrt(max(0.0, variance))
    if std_pnl <= 0.0:
        threshold = mean_pnl
        var_value = max(0.0, -threshold)
        return var_value, var_value, threshold
    alpha = 1.0 - confidence
    z_score = float(norm.ppf(alpha))
    threshold = mean_pnl + (std_pnl * z_score)
    tail_mean = mean_pnl - (std_pnl * float(norm.pdf(z_score)) / alpha)
    var_value = max(0.0, -threshold)
    cvar_value = max(0.0, -tail_mean)
    return var_value, cvar_value, threshold


def _calculate_var_cvar_from_pnl(
    pnl_values: List[float],
    *,
    confidence: float,
    method: str,
) -> tuple[float, float, float]:
    if method == "historical":
        return _historical_var_cvar_tail(pnl_values, confidence)
    return _gaussian_var_cvar_tail(pnl_values, confidence)


def _extract_var_cvar_return_series(
    *,
    symbol: str,
    rates: Any,
    transform: str,
    pd_module: Any,
    np_module: Any,
) -> tuple[Any, Optional[str]]:
    frame = pd_module.DataFrame(rates)
    if frame.empty:
        return None, f"No candle history returned for {symbol}"
    if "time" not in frame.columns or "close" not in frame.columns:
        return None, f"Candle history for {symbol} is missing time/close columns"
    close = pd_module.to_numeric(frame["close"], errors="coerce")
    timestamps = pd_module.to_datetime(
        frame["time"], unit="s", utc=True, errors="coerce"
    )
    series = pd_module.Series(close.to_numpy(), index=timestamps, name=symbol)
    series = series[~series.index.isna()]
    series = series.replace([np_module.inf, -np_module.inf], np_module.nan).dropna()
    series = series[~series.index.duplicated(keep="last")]
    if len(series) < 2:
        return None, f"Not enough candle history for {symbol}"
    if transform == "log_return":
        returns = np_module.log(series / series.shift(1))
    else:
        returns = series.pct_change()
    returns = returns.replace([np_module.inf, -np_module.inf], np_module.nan).dropna()
    if returns.empty:
        return None, f"No usable returns produced for {symbol}"
    return returns, None


def _format_var_cvar_timestamp(value: Any) -> str:
    try:
        text = value.isoformat()
    except Exception:
        return str(value)
    return text.replace("+00:00", "Z")


def _epoch_series_to_utc_and_text(
    raw_series: Any,
    *,
    pd_module: Any,
    mt5_epoch_to_utc: Any,
    fmt_time: Any,
    require_positive: bool = False,
) -> tuple[Any, Any]:
    numeric = pd_module.to_numeric(raw_series, errors="coerce")
    utc_values: List[float] = []
    text_values: List[Optional[str]] = []
    for raw_value in numeric.tolist():
        if pd_module.isna(raw_value):
            utc_values.append(float("nan"))
            text_values.append(None)
            continue
        epoch_value = float(raw_value)
        if require_positive and epoch_value <= 0.0:
            utc_values.append(float("nan"))
            text_values.append(None)
            continue
        utc_value = float(mt5_epoch_to_utc(epoch_value))
        utc_values.append(utc_value)
        text_values.append(fmt_time(utc_value))
    return (
        pd_module.Series(utc_values, index=numeric.index),
        pd_module.Series(text_values, index=numeric.index),
    )


def run_trade_place(  # noqa: C901
    request: TradePlaceRequest,
    *,
    normalize_order_type_input: Any,
    normalize_pending_expiration: Any,
    prevalidate_trade_place_market_input: Any,
    place_market_order: Any,
    place_pending_order: Any,
    close_positions: Any,
    safe_int_ticket: Any,
    idempotency_store: Optional[IdempotencyStore] = _TRADE_IDEMPOTENCY_STORE,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    missing: List[str] = []
    symbol_norm = str(request.symbol).strip() if request.symbol is not None else ""
    idempotency_key = _normalize_idempotency_key(getattr(request, "idempotency_key", None))
    idempotency_signature = (
        _build_trade_request_signature(request)
        if idempotency_key is not None
        else None
    )
    idempotency_consumed = False
    log_operation_start(
        logger,
        operation="trade_place",
        symbol=symbol_norm or None,
        requested_order_type=request.order_type,
    )

    def _finish(
        result: Dict[str, Any],
        *,
        order_type: Optional[str] = None,
        pending: Optional[bool] = None,
    ) -> Dict[str, Any]:
        nonlocal idempotency_consumed
        if (
            not idempotency_consumed
            and idempotency_store is not None
            and idempotency_key is not None
        ):
            idempotency_store.record(
                idempotency_key,
                copy.deepcopy(result),
                request_signature=idempotency_signature,
            )
            idempotency_consumed = True
        log_operation_finish(
            logger,
            operation="trade_place",
            started_at=started_at,
            success=infer_result_success(result),
            symbol=symbol_norm or None,
            order_type=order_type,
            pending=pending,
        )
        return result

    duplicate_result, idempotency_reserved = _begin_trade_idempotency(
        idempotency_store=idempotency_store,
        key=idempotency_key,
        request_signature=idempotency_signature,
    )
    if duplicate_result is not None:
        idempotency_consumed = True
        return _finish(duplicate_result)

    try:
        def _dry_run_preview(
            *,
            order_type: str,
            pending: bool,
            normalized_expiration: Any,
            expiration_provided: bool,
        ) -> Dict[str, Any]:
            preview_detail = _resolve_trade_place_preview_detail(request)
            validation_scope = "request_routing_only"
            validation_not_performed = [
                "broker_acceptance",
                "live_price_distance_rules",
                "margin_and_funds",
                "fillability",
                "sl_tp_attachment",
            ]
            preview: Dict[str, Any] = {
                "success": True,
                "dry_run": True,
                "no_action": True,
                "symbol": symbol_norm,
                "order_type": order_type,
                "pending": pending,
                "action": "place_pending_order" if pending else "place_market_order",
                "volume": float(request.volume),
                "message": "Dry run only. No order was sent to MT5.",
                "validation_scope": validation_scope,
                "validation_passed": True,
                "trade_gate_passed": False,
                "actionability": "preview_only",
                "actionability_reason": (
                    "Dry run did not execute MT5 or broker-side validation. "
                    "Use this preview for request routing only."
                ),
                "preview_scope_summary": "Routing and local request checks only.",
                "validation_not_performed": list(validation_not_performed),
                "warnings": [
                    "Dry run only. Routing and local safety checks passed; MT5/broker validation was not executed.",
                    (
                        "Not validated in dry run: broker acceptance, live price-distance rules, "
                        "margin/funds, fillability, and SL/TP attachment."
                    ),
                ],
                "require_sl_tp": bool(request.require_sl_tp),
                "auto_close_on_sl_tp_fail": bool(request.auto_close_on_sl_tp_fail),
                "guardrails_preview": preview_trade_guardrails(
                    trade_guardrails_config,
                    symbol=symbol_norm,
                    volume=float(request.volume),
                    stop_loss=request.stop_loss,
                    deviation=request.deviation,
                    side=_guardrail_order_side(order_type),
                ),
            }
            if pending:
                preview["requested_price"] = request.price
            if request.stop_loss not in (None, 0):
                preview["requested_sl"] = request.stop_loss
            if request.take_profit not in (None, 0):
                preview["requested_tp"] = request.take_profit
            if expiration_provided:
                preview["expiration"] = request.expiration
                if normalized_expiration is not None:
                    preview["expiration_normalized"] = normalized_expiration
            return _shape_trade_place_preview(preview, detail=preview_detail)

        if not symbol_norm:
            missing.append("symbol")
        if request.volume is None:
            missing.append("volume")
        if request.order_type is None or (
            isinstance(request.order_type, str) and not request.order_type.strip()
        ):
            missing.append("order_type")
        if missing:
            return _finish(
                {
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
            )

        # Validate volume is positive
        try:
            vol_float = float(request.volume)
        except (TypeError, ValueError):
            return _finish({"error": "volume must be numeric"})
        if not math.isfinite(vol_float):
            return _finish({"error": "volume must be finite"})
        if vol_float <= 0:
            return _finish(
                {
                    "error": "volume must be positive",
                    "volume_received": vol_float,
                    "volume_hint": "volume must be greater than 0",
                }
            )

        order_type_norm, order_type_error = normalize_order_type_input(request.order_type)
        if order_type_error:
            return _finish({"error": order_type_error}, order_type=order_type_norm)
        explicit_pending_types = {"BUY_LIMIT", "BUY_STOP", "SELL_LIMIT", "SELL_STOP"}
        market_side_types = {"BUY", "SELL"}
        supported_order_types = explicit_pending_types.union(market_side_types)
        if order_type_norm not in supported_order_types:
            return _finish(
                {
                    "error": (
                        f"Unsupported order_type '{request.order_type}'. "
                        "Use BUY/SELL or BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP."
                    )
                },
                order_type=order_type_norm,
            )

        price_provided = request.price not in (None, 0)
        try:
            normalized_expiration, expiration_provided = normalize_pending_expiration(
                request.expiration
            )
        except (TypeError, ValueError) as ex:
            return _finish({"error": str(ex)}, order_type=order_type_norm)

        ignore_market_gtc_expiration = (
            order_type_norm in market_side_types
            and not price_provided
            and expiration_provided
            and normalized_expiration is None
        )
        if (
            order_type_norm in market_side_types
            and not price_provided
            and expiration_provided
            and normalized_expiration is not None
        ):
            return _finish(
                {
                    "error": (
                        "expiration only applies to pending orders placed with a price. "
                        "For BUY/SELL market orders, omit expiration or provide price to place a pending order."
                    )
                },
                order_type=order_type_norm,
                pending=False,
            )

        is_pending = (
            order_type_norm in explicit_pending_types
            or price_provided
            or (expiration_provided and not ignore_market_gtc_expiration)
        )
        if bool(request.require_sl_tp) and not is_pending:
            missing_protection: List[str] = []
            if request.stop_loss in (None, 0):
                missing_protection.append("stop_loss")
            if request.take_profit in (None, 0):
                missing_protection.append("take_profit")
            if missing_protection:
                if not bool(request.dry_run):
                    prevalidation_error = prevalidate_trade_place_market_input(
                        symbol_norm,
                        request.volume,
                    )
                    if prevalidation_error is not None:
                        return _finish(
                            prevalidation_error,
                            order_type=order_type_norm,
                            pending=is_pending,
                        )
                return _finish(
                    {
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
                    },
                    order_type=order_type_norm,
                    pending=is_pending,
                )

        if bool(request.dry_run):
            guardrail_preview = preview_trade_guardrails(
                trade_guardrails_config,
                symbol=symbol_norm,
                volume=float(request.volume),
                stop_loss=request.stop_loss,
                deviation=request.deviation,
                side=_guardrail_order_side(order_type_norm),
            )
            if guardrail_preview.get("blocked"):
                violations = list(guardrail_preview.get("violations") or [])
                guardrail_rule = str(guardrail_preview.get("rule") or "").strip()
                error_message = "Trade would be blocked by configured guardrails."
                if violations:
                    prefix = (
                        f"Trade blocked by guardrails ({guardrail_rule})"
                        if guardrail_rule
                        else "Trade blocked by guardrails"
                    )
                    error_message = f"{prefix}: {violations[0]}"
                return _finish(
                    {
                        "error": error_message,
                        "guardrail_blocked": True,
                        "dry_run": True,
                        "no_action": True,
                        "actionability": "blocked_by_guardrails",
                        "guardrails_preview": guardrail_preview,
                        "violations": violations,
                    },
                    order_type=order_type_norm,
                    pending=is_pending,
                )
            if is_pending and request.price is None:
                return _finish(
                    {"error": "price is required for pending orders."},
                    order_type=order_type_norm,
                    pending=is_pending,
                )
            return _finish(
                _dry_run_preview(
                    order_type=order_type_norm,
                    pending=is_pending,
                    normalized_expiration=normalized_expiration,
                    expiration_provided=expiration_provided,
                ),
                order_type=order_type_norm,
                pending=is_pending,
            )

        static_guardrail = evaluate_trade_guardrails(
            trade_guardrails_config,
            symbol=symbol_norm,
            volume=float(request.volume),
            stop_loss=request.stop_loss,
            deviation=request.deviation,
            side=_guardrail_order_side(order_type_norm),
            enforce_account_risk=False,
            enforce_wallet_risk=False,
        )
        if static_guardrail is not None:
            return _finish(static_guardrail, order_type=order_type_norm, pending=is_pending)

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
                sl_tp_requested, sl_tp_status, sl_tp_fallback_used = _sl_tp_result_details(
                    result
                )
                sl_tp_failed = sl_tp_status == "failed"
                if sl_tp_requested and sl_tp_failed:
                    warnings_out: List[str] = list(result.get("warnings") or [])
                    pos_ticket = result.get("position_ticket")
                    candidate_tickets = [
                        ticket
                        for ticket in list(result.get("position_ticket_candidates") or [])
                        if ticket is not None
                    ]
                    if pos_ticket is not None:
                        critical = (
                            "CRITICAL: Order executed without applied TP/SL protection. "
                            f"Run trade_modify {pos_ticket} now, or close the position."
                        )
                    elif candidate_tickets:
                        candidate_list = ", ".join(str(v) for v in candidate_tickets)
                        critical = (
                            "CRITICAL: Order executed without applied TP/SL protection. "
                            f"Try trade_modify {candidate_tickets[0]} now "
                            f"(candidate tickets: {candidate_list}). "
                            "If that fails, run trade_get_open to confirm the live position ticket, "
                            "or close the position."
                        )
                    else:
                        critical = (
                            "CRITICAL: Order executed without applied TP/SL protection. "
                            "Run trade_get_open to find the live position ticket, then trade_modify it now, "
                            "or close the position."
                        )
                    if critical not in warnings_out:
                        warnings_out.append(critical)
                    if warnings_out:
                        result["warnings"] = warnings_out
                    if bool(request.auto_close_on_sl_tp_fail):
                        close_ticket = safe_int_ticket(pos_ticket)
                        if close_ticket is None:
                            for candidate_ticket in candidate_tickets:
                                close_ticket = safe_int_ticket(candidate_ticket)
                                if close_ticket is not None:
                                    break
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
                        if (
                            isinstance(auto_close_result, dict)
                            and "error" not in auto_close_result
                        ):
                            if auto_close_result.get("retcode") is not None:
                                auto_close_ok = True
                            else:
                                try:
                                    auto_close_ok = (
                                        int(auto_close_result.get("closed_count", 0)) > 0
                                    )
                                except Exception:
                                    auto_close_ok = False
                        if auto_close_ok:
                            result["protection_status"] = "auto_closed_after_sl_tp_fail"
                        else:
                            warnings_out = list(result.get("warnings") or [])
                            auto_close_warning = "AUTO-CLOSE FAILED: position remains unprotected; close immediately."
                            if auto_close_warning not in warnings_out:
                                warnings_out.append(auto_close_warning)
                            result["warnings"] = warnings_out

                if (
                    bool(request.require_sl_tp)
                    and sl_tp_requested
                    and sl_tp_failed
                    and "error" not in result
                ):
                    result["error"] = (
                        "Order was executed, but TP/SL protection could not be applied."
                    )
                    result["require_sl_tp"] = bool(request.require_sl_tp)
                    result["protection_status"] = (
                        result.get("protection_status") or "unprotected_position"
                    )
                elif (
                    sl_tp_requested
                    and sl_tp_status == "applied"
                    and sl_tp_fallback_used
                    and "protection_status" not in result
                ):
                    result["protection_status"] = "protected_after_fallback"
            return _finish(result, order_type=order_type_norm, pending=is_pending)
        if request.price is None:
            return _finish(
                {"error": "price is required for pending orders."},
                order_type=order_type_norm,
                pending=is_pending,
            )
        return _finish(
            place_pending_order(
                symbol=symbol_norm,
                volume=float(request.volume),
                order_type=order_type_norm,
                price=request.price,
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
                expiration=request.expiration,
                comment=request.comment,
                deviation=request.deviation,
            ),
            order_type=order_type_norm,
            pending=is_pending,
        )
    finally:
        if (
            idempotency_reserved
            and not idempotency_consumed
            and idempotency_store is not None
            and idempotency_key is not None
        ):
            idempotency_store.release(
                idempotency_key,
                request_signature=idempotency_signature,
            )


def run_trade_modify(
    request: TradeModifyRequest,
    *,
    normalize_pending_expiration: Any,
    modify_pending_order: Any,
    modify_position: Any,
    idempotency_store: Optional[IdempotencyStore] = _TRADE_IDEMPOTENCY_STORE,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    idempotency_key = _normalize_idempotency_key(getattr(request, "idempotency_key", None))
    idempotency_signature = (
        _build_trade_request_signature(request)
        if idempotency_key is not None
        else None
    )
    idempotency_consumed = False
    log_operation_start(
        logger,
        operation="trade_modify",
        ticket=request.ticket,
    )

    def _finish(
        result: Dict[str, Any],
        *,
        pending: Optional[bool] = None,
    ) -> Dict[str, Any]:
        nonlocal idempotency_consumed
        if (
            not idempotency_consumed
            and idempotency_store is not None
            and idempotency_key is not None
        ):
            idempotency_store.record(
                idempotency_key,
                copy.deepcopy(result),
                request_signature=idempotency_signature,
            )
            idempotency_consumed = True
        log_operation_finish(
            logger,
            operation="trade_modify",
            started_at=started_at,
            success=infer_result_success(result),
            ticket=request.ticket,
            pending=pending,
        )
        return result

    duplicate_result, idempotency_reserved = _begin_trade_idempotency(
        idempotency_store=idempotency_store,
        key=idempotency_key,
        request_signature=idempotency_signature,
    )
    if duplicate_result is not None:
        idempotency_consumed = True
        return _finish(duplicate_result)

    try:
        price_val = None if request.price in (None, 0) else request.price
        try:
            _, expiration_specified = normalize_pending_expiration(request.expiration)
        except (TypeError, ValueError) as ex:
            return _finish({"error": str(ex)})

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
                return _finish(
                    {
                        "error_code": "ticket_not_found",
                        "error": (
                            f"Pending order {request.ticket} not found. "
                            "Note: price/expiration only apply to pending orders."
                        ),
                        "ticket": request.ticket,
                        "checked_scopes": ["pending_orders"],
                        "suggestion": "Use trade_get_pending to find active pending-order tickets before retrying trade_modify.",
                    },
                    pending=True,
                )
            return _finish(result, pending=True)

        position_result = modify_position(
            ticket=request.ticket,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            comment=request.comment,
        )
        if position_result.get("success"):
            return _finish(position_result, pending=False)
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
                return _finish(
                    {
                        "error_code": "ticket_not_found",
                        "error": f"Ticket {request.ticket} not found as position or pending order.",
                        "ticket": request.ticket,
                        "checked_scopes": ["positions", "pending_orders"],
                        "suggestion": "Use trade_get_open or trade_get_pending to find active tickets before retrying trade_modify.",
                    },
                    pending=None,
                )
            return _finish(pending_result, pending=True)
        return _finish(position_result, pending=False)
    finally:
        if (
            idempotency_reserved
            and not idempotency_consumed
            and idempotency_store is not None
            and idempotency_key is not None
        ):
            idempotency_store.release(
                idempotency_key,
                request_signature=idempotency_signature,
            )


def run_trade_close(  # noqa: C901
    request: TradeCloseRequest,
    *,
    close_positions: Any,
    cancel_pending: Any,
    lookup_ticket_history: Any = None,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="trade_close",
        ticket=request.ticket,
        close_all=request.close_all,
        symbol=request.symbol,
        volume=request.volume,
        profit_only=request.profit_only,
        loss_only=request.loss_only,
        dry_run=request.dry_run,
    )

    def _finish(
        result: Dict[str, Any],
        *,
        scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        log_operation_finish(
            logger,
            operation="trade_close",
            started_at=started_at,
            success=infer_result_success(result),
            ticket=request.ticket,
            close_all=request.close_all,
            symbol=request.symbol,
            volume=request.volume,
            scope=scope,
            profit_only=request.profit_only,
            loss_only=request.loss_only,
            dry_run=request.dry_run,
        )
        return result

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

    if request.profit_only and request.loss_only:
        return _finish(
            {"error": "profit_only and loss_only cannot both be true."},
            scope="positions",
        )

    if request.volume is not None:
        if request.ticket is None:
            return _finish(
                {
                    "error": (
                        "volume is only supported when closing a specific open position by ticket."
                    )
                },
                scope="positions",
            )
        if request.profit_only or request.loss_only:
            return _finish(
                {
                    "error": (
                        "volume cannot be combined with profit_only or loss_only. "
                        "Use ticket for a specific partial close."
                    )
                },
                scope="positions",
            )

    if request.ticket is not None and request.close_all:
        return _finish(
            {
                "error": (
                    "close_all cannot be combined with ticket. "
                    "Use ticket for a specific position or pending order, "
                    "or omit ticket and pass close_all=true for a bulk close."
                )
            },
            scope="ticket",
        )

    if request.ticket is None and not request.close_all:
        return _finish(
            {
                "error": "Bulk close requires explicit confirmation.",
                "error_code": "CONFIRMATION_REQUIRED",
                "suggestion": "Review matching positions before closing (irreversible action).",
                "alternatives": [
                    "Use ticket=<ticket_number> to close a specific position",
                    "First use trade_get_open to view matching positions",
                    "Then pass close_all=true to proceed with bulk close",
                ],
            },
            scope="bulk_confirmation",
        )

    if request.dry_run:
        scope = (
            "ticket"
            if request.ticket is not None
            else "symbol"
            if request.symbol is not None
            else "positions"
        )
        operation = (
            "partial_close_position"
            if request.volume is not None
            else "close_or_cancel_ticket"
            if request.ticket is not None
            else "close_symbol_positions"
            if request.symbol is not None
            else "close_all_positions"
        )
        preview: Dict[str, Any] = {
            "success": True,
            "dry_run": True,
            "actionability": "preview_only",
            "operation": operation,
            "scope": scope,
            "would_send_order": False,
            "would_cancel_pending_order": False,
            "preview_scope_summary": (
                "Routing and request validation only; no close or cancel request was sent to MT5."
            ),
            "not_estimated": [
                "realized_pnl",
                "slippage",
                "post_close_balance",
                "tax_impact",
            ],
        }
        if request.ticket is not None:
            preview["ticket"] = request.ticket
            preview["ticket_resolution"] = (
                "Would try an open position first, then a pending order if no position matches."
            )
        if request.symbol is not None:
            preview["symbol"] = request.symbol
        if request.volume is not None:
            preview["volume"] = request.volume
            preview["ticket_resolution"] = (
                "Would target only an open position; partial close does not fall back to pending orders."
            )
        if request.close_all:
            preview["close_all"] = True
        if request.profit_only:
            preview["profit_only"] = True
        if request.loss_only:
            preview["loss_only"] = True
        if request.close_priority:
            preview["close_priority"] = request.close_priority
        if request.comment:
            preview["comment"] = request.comment
        if request.deviation != 20:
            preview["deviation"] = request.deviation
        return _finish(preview, scope=scope)

    if request.profit_only or request.loss_only:
        result = close_positions(
            ticket=request.ticket,
            symbol=request.symbol,
            volume=None,
            profit_only=request.profit_only,
            loss_only=request.loss_only,
            close_priority=request.close_priority,
            comment=request.comment,
            deviation=request.deviation,
        )
        if isinstance(result, dict):
            msg = str(result.get("message", "")).strip().lower()
            if (
                msg.startswith("no open positions")
                or msg == "no positions matched criteria"
            ):
                return _finish(_with_no_action(result), scope="positions")
        return _finish(result, scope="positions")

    if request.ticket is not None:
        position_result = close_positions(
            ticket=request.ticket,
            symbol=request.symbol,
            volume=request.volume,
            profit_only=False,
            loss_only=False,
            close_priority=request.close_priority,
            comment=request.comment,
            deviation=request.deviation,
        )
        if (
            request.volume is not None
            and isinstance(position_result, dict)
            and position_result.get("error") == f"Position {request.ticket} not found"
        ):
            return _finish(
                {
                    "error": (
                        f"Position {request.ticket} not found. "
                        "Partial close volume only applies to open positions."
                    ),
                    "checked_scopes": ["positions"],
                },
                scope="positions",
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
                and pending_result.get("error")
                == f"Pending order {request.ticket} not found"
            ):
                history_result = None
                if lookup_ticket_history is not None:
                    try:
                        history_result = lookup_ticket_history(request.ticket)
                    except Exception:
                        history_result = None
                if isinstance(history_result, dict) and history_result:
                    return _finish(history_result, scope="history")
                return _finish(
                    {
                        "error": f"Ticket {request.ticket} not found as position or pending order.",
                        "checked_scopes": ["positions", "pending_orders"],
                    },
                    scope="ticket",
                )
            return _finish(pending_result, scope="pending_orders")
        return _finish(position_result, scope="positions")

    if request.symbol is not None:
        position_result = close_positions(
            symbol=request.symbol,
            volume=None,
            profit_only=False,
            loss_only=False,
            close_priority=request.close_priority,
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
                        return _finish(
                            _with_no_action(
                                message=f"No open positions or pending orders for {request.symbol}"
                            ),
                            scope="symbol",
                        )
                return _finish(pending_result, scope="pending_orders")
        return _finish(position_result, scope="positions")

    position_result = close_positions(
        volume=None,
        profit_only=False,
        loss_only=False,
        close_priority=request.close_priority,
        comment=request.comment,
        deviation=request.deviation,
    )
    if isinstance(position_result, dict):
        msg = str(position_result.get("message", "")).strip().lower()
        if msg == "no open positions":
            pending_result = cancel_pending(comment=request.comment)
            if (
                isinstance(pending_result, dict)
                and str(pending_result.get("message", "")).strip().lower()
                == "no pending orders"
            ):
                return _finish(
                    _with_no_action(message="No open positions or pending orders"),
                    scope="all",
                )
            return _finish(pending_result, scope="pending_orders")
    return _finish(position_result, scope="positions")


def run_trade_history(  # noqa: C901
    request: TradeHistoryRequest,
    *,
    gateway: Any,
    use_client_tz: Any,
    format_time_minimal: Any,
    format_time_minimal_local: Any,
    mt5_epoch_to_utc: Any,
    parse_start_datetime: Any,
    normalize_limit: Any,
    comment_row_metadata: Any,
    normalize_ticket_filter: Any,
    normalize_minutes_back: Any,
    decode_mt5_enum_label: Any,
    mt5_config: Any,
) -> Any:
    import pandas as pd

    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="trade_history",
        symbol=request.symbol,
        history_kind=request.history_kind,
        limit=request.limit,
    )

    def _finish(result: Any) -> Any:
        record_count = None
        if isinstance(result, list):
            record_count = len(result)
        elif isinstance(result, dict):
            items = result.get("items")
            if isinstance(items, list):
                record_count = len(items)
            else:
                count_value = result.get("count")
                if isinstance(count_value, int):
                    record_count = count_value
        log_operation_finish(
            logger,
            operation="trade_history",
            started_at=started_at,
            success=infer_result_success(result),
            symbol=request.symbol,
            history_kind=request.history_kind,
            limit=request.limit,
            record_count=record_count,
        )
        return result

    try:
        gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return _finish({"error": str(exc)})

    def _get_history():  # noqa: C901
        try:
            use_client_tz_value = use_client_tz()
            fmt_time = (
                format_time_minimal_local
                if use_client_tz_value
                else format_time_minimal
            )
            trigger_pattern = re.compile(
                r"\[(sl|tp)\s+([+-]?\d+(?:\.\d+)?)\]", re.IGNORECASE
            )
            default_window_label: Optional[str] = None

            def _normalize_time_col(
                df: "pd.DataFrame", col: str
            ) -> Optional["pd.Series"]:
                if col not in df.columns:
                    return None
                utc, text = _epoch_series_to_utc_and_text(
                    df[col],
                    pd_module=pd,
                    mt5_epoch_to_utc=mt5_epoch_to_utc,
                    fmt_time=fmt_time,
                )
                df[col] = text
                return utc

            if request.start and request.minutes_back not in (None, ""):
                return {"error": "Use either start or minutes_back, not both."}

            position_ticket_value, position_ticket_error = normalize_ticket_filter(
                request.position_ticket,
                name="position_ticket",
            )
            if position_ticket_error:
                return {"error": position_ticket_error}
            deal_ticket_value, deal_ticket_error = normalize_ticket_filter(
                request.deal_ticket,
                name="deal_ticket",
            )
            if deal_ticket_error:
                return {"error": deal_ticket_error}
            order_ticket_value, order_ticket_error = normalize_ticket_filter(
                request.order_ticket,
                name="order_ticket",
            )
            if order_ticket_error:
                return {"error": order_ticket_error}
            side_value, side_error = validation._normalize_trade_side_filter(
                getattr(request, "side", None)
            )
            if side_error:
                return {"error": side_error}
            minutes_back_value, minutes_back_error = normalize_minutes_back(
                request.minutes_back
            )
            if minutes_back_error:
                return {"error": minutes_back_error}

            if request.end:
                to_dt = parse_start_datetime(request.end)
                if not to_dt:
                    return {"error": "Invalid end time."}
            else:
                to_dt = datetime.now(timezone.utc).replace(tzinfo=None)

            if minutes_back_value is not None:
                from_dt = to_dt - timedelta(minutes=minutes_back_value)
            elif request.start:
                from_dt = parse_start_datetime(request.start)
                if not from_dt:
                    return {"error": "Invalid start time."}
            else:
                from_dt = to_dt - timedelta(days=_DEFAULT_TRADE_HISTORY_LOOKBACK_DAYS)
                default_window_label = (
                    f"the last {_DEFAULT_TRADE_HISTORY_LOOKBACK_DAYS} days"
                )

            if from_dt > to_dt:
                return {"error": "start must be before end."}

            # MT5 history_* query bounds must stay in UTC instead of broker-local wall
            # time, otherwise recent windows can shift away from the intended range.
            history_from_dt = _to_utc_history_query_dt(from_dt)
            history_to_dt = _to_utc_history_query_dt(to_dt)

            kind = str(request.history_kind or "deals").strip().lower()
            if kind not in ("deals", "orders"):
                return {"error": "history_kind must be 'deals' or 'orders'."}
            if kind == "orders" and deal_ticket_value is not None:
                return {"error": "deal_ticket is only valid when history_kind='deals'."}

            deal_enum_columns = (
                ("type", "DEAL_TYPE_"),
                ("entry", "DEAL_ENTRY_"),
                ("reason", "DEAL_REASON_"),
            )
            order_enum_columns = (
                ("type", "ORDER_TYPE_"),
                ("state", "ORDER_STATE_"),
                ("type_time", "ORDER_TIME_"),
                ("type_filling", "ORDER_FILLING_"),
                ("reason", "ORDER_REASON_"),
            )

            def _decode_enum_column(df: "pd.DataFrame", col: str, prefix: str) -> None:
                if col not in df.columns:
                    return
                raw = df[col]
                numeric = pd.to_numeric(raw, errors="coerce")
                if numeric.notna().any():
                    df[f"{col}_code"] = numeric.astype("Int64")
                df[col] = raw.apply(
                    lambda v: decode_mt5_enum_label(gateway, v, prefix=prefix) or v
                )

            def _reason_to_exit_trigger(reason: Any) -> Optional[str]:
                txt = str(reason or "").strip().lower()
                if not txt:
                    return None
                if re.search(r"\bsl\b|stop\s*loss", txt):
                    return "SL"
                if re.search(r"\btp\b|take\s*profit", txt):
                    return "TP"
                return None

            def _extract_exit_trigger(
                comment: Any,
                reason: Any,
                entry: Any,
            ) -> tuple[Optional[str], Optional[float], Optional[str]]:
                entry_txt = str(entry or "").strip().lower()
                if entry_txt and "out" not in entry_txt:
                    return None, None, None
                if isinstance(comment, str) and comment:
                    match = trigger_pattern.search(comment)
                    if match:
                        trigger = str(match.group(1)).upper()
                        try:
                            price = float(match.group(2))
                        except Exception:
                            price = None
                        return trigger, price, "comment"
                reason_trigger = _reason_to_exit_trigger(reason)
                if reason_trigger:
                    return reason_trigger, None, "reason"
                return None, None, None

            def _filter_by_ticket_columns(
                df_in: "pd.DataFrame",
                ticket_value: Optional[int],
                *,
                columns: tuple[str, ...],
            ) -> "pd.DataFrame":
                if ticket_value is None:
                    return df_in
                masks: List["pd.Series"] = []
                for col in columns:
                    if col not in df_in.columns:
                        continue
                    masks.append(
                        pd.to_numeric(df_in[col], errors="coerce").eq(ticket_value)
                    )
                if not masks:
                    return df_in.iloc[0:0]
                mask = masks[0]
                for extra in masks[1:]:
                    mask = mask | extra
                return df_in.loc[mask]

            def _is_non_informative_series(series: "pd.Series") -> bool:
                vals = pd.Series(series)
                if vals.dropna().empty:
                    return True
                for value in vals:
                    if value is None:
                        continue
                    if isinstance(value, str):
                        if not value.strip():
                            continue
                        return False
                    try:
                        numeric = float(value)
                        if math.isfinite(numeric) and numeric == 0.0:
                            continue
                        return False
                    except Exception:
                        return False
                return True

            def _history_fetch_error(kind_label: str, exc: Exception) -> Dict[str, str]:
                detail = str(exc).strip()
                if "exception set" in detail.lower():
                    return {
                        "error": f"Failed to fetch {kind_label} history from MT5. {_TRADE_HISTORY_RANGE_HINT}"
                    }
                if detail:
                    return {
                        "error": f"Failed to fetch {kind_label} history from MT5: {detail}"
                    }
                return {"error": f"Failed to fetch {kind_label} history from MT5."}

            def _empty_history_message(kind_label: str) -> Dict[str, str]:
                message = f"No {kind_label} found"
                if side_value and request.symbol:
                    message += f" for {side_value} side on {request.symbol}"
                elif side_value:
                    message += f" for {side_value} side"
                elif request.symbol:
                    message += f" for {request.symbol}"
                if minutes_back_value is not None:
                    message += f" in the last {int(minutes_back_value)} minute(s)"
                elif default_window_label:
                    message += f" in {default_window_label}"
                if kind_label == "deals" and minutes_back_value is None:
                    message += ". For order creation/cancellation events, use --history-kind orders."
                if minutes_back_value is not None and minutes_back_value < 30:
                    message += " Note: MT5 history may take up to a few minutes to reflect very recent events."
                return {"message": message}

            def _filter_by_side(df_in: "pd.DataFrame") -> "pd.DataFrame":
                if side_value is None:
                    return df_in
                if "type" not in df_in.columns:
                    return df_in.iloc[0:0]
                type_text = (
                    df_in["type"]
                    .astype(str)
                    .str.upper()
                    .str.replace(r"[^A-Z0-9]+", "_", regex=True)
                    .str.strip("_")
                )
                mask = type_text.eq(side_value) | type_text.str.startswith(
                    f"{side_value}_"
                )
                return df_in.loc[mask]

            if kind == "deals":
                try:
                    rows = (
                        gateway.history_deals_get(
                            history_from_dt, history_to_dt, symbol=request.symbol
                        )
                        if request.symbol
                        else gateway.history_deals_get(history_from_dt, history_to_dt)
                    )
                except Exception as exc:
                    return _history_fetch_error("deal", exc)
                if rows is None or len(rows) == 0:
                    return _empty_history_message("deals")
                df = _trade_rows_to_dataframe(rows, pd_module=pd)
                if request.symbol and "symbol" in df.columns:
                    df = df.loc[
                        df["symbol"].astype(str).str.upper()
                        == str(request.symbol).upper()
                    ]
                df = _filter_by_ticket_columns(
                    df, deal_ticket_value, columns=("ticket",)
                )
                df = _filter_by_ticket_columns(
                    df, order_ticket_value, columns=("order",)
                )
                df = _filter_by_ticket_columns(
                    df,
                    position_ticket_value,
                    columns=("position_id", "position_by_id"),
                )
                if len(df) == 0:
                    return _empty_history_message("deals")
                sort_src = _normalize_time_col(df, "time")
                for col, prefix in deal_enum_columns:
                    _decode_enum_column(df, col, prefix)
                df = _filter_by_side(df)
                if len(df) == 0:
                    return _empty_history_message("deals")
                if len(df) > 0:
                    triggers = df.apply(
                        lambda row: _extract_exit_trigger(
                            row.get("comment"),
                            row.get("reason"),
                            row.get("entry"),
                        ),
                        axis=1,
                        result_type="expand",
                    )
                    if isinstance(triggers, pd.DataFrame) and triggers.shape[1] == 3:
                        triggers.columns = [
                            "exit_trigger",
                            "exit_trigger_price",
                            "exit_trigger_source",
                        ]
                        for col in triggers.columns:
                            df[col] = triggers[col]
                for noise_col in ("time_msc", "external_id", "fee"):
                    if noise_col in df.columns and _is_non_informative_series(
                        df[noise_col]
                    ):
                        df = df.drop(columns=[noise_col])
            else:
                try:
                    rows = (
                        gateway.history_orders_get(
                            history_from_dt, history_to_dt, symbol=request.symbol
                        )
                        if request.symbol
                        else gateway.history_orders_get(history_from_dt, history_to_dt)
                    )
                except Exception as exc:
                    return _history_fetch_error("order", exc)
                if rows is None or len(rows) == 0:
                    return _empty_history_message("orders")
                df = _trade_rows_to_dataframe(rows, pd_module=pd)
                if request.symbol and "symbol" in df.columns:
                    df = df.loc[
                        df["symbol"].astype(str).str.upper()
                        == str(request.symbol).upper()
                    ]
                df = _filter_by_ticket_columns(
                    df, order_ticket_value, columns=("ticket",)
                )
                df = _filter_by_ticket_columns(
                    df,
                    position_ticket_value,
                    columns=("position_id", "position_by_id"),
                )
                if len(df) == 0:
                    return _empty_history_message("orders")
                sort_src = _normalize_time_col(df, "time_setup")
                if sort_src is None:
                    sort_src = _normalize_time_col(df, "time")
                _normalize_time_col(df, "time_done")
                for col, prefix in order_enum_columns:
                    _decode_enum_column(df, col, prefix)
                df = _filter_by_side(df)
                if len(df) == 0:
                    return _empty_history_message("orders")

            df["__sort_utc"] = (
                sort_src
                if sort_src is not None
                else pd.Series([float("nan")] * len(df))
            )

            limit_value = normalize_limit(request.limit)
            if limit_value and len(df) > limit_value:
                if "__sort_utc" in df.columns:
                    df = df.sort_values("__sort_utc").tail(limit_value)
                else:
                    df = df.tail(limit_value)
            if "__sort_utc" in df.columns:
                df = df.drop(columns=["__sort_utc"])

            df = df.replace([float("inf"), float("-inf")], pd.NA)
            records = (
                df.astype(object).where(df.notna(), None).to_dict(orient="records")
            )
            timezone_label = "UTC"
            if use_client_tz_value:
                try:
                    tz_obj = mt5_config.get_client_tz()
                    timezone_label = str(
                        getattr(tz_obj, "zone", None) or tz_obj or "client_local"
                    )
                except Exception:
                    timezone_label = "client_local"
            for row in records:
                if isinstance(row, dict):
                    row["timestamp_timezone"] = timezone_label
                    row.update(comment_row_metadata(row.get("comment")))
            return records
        except Exception as exc:
            return {"error": str(exc)}

    return _finish(_get_history())


def run_trade_risk_analyze(  # noqa: C901
    request: TradeRiskAnalyzeRequest,
    *,
    gateway: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="trade_risk_analyze",
        symbol=request.symbol,
        desired_risk_pct=request.desired_risk_pct,
    )

    def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
        log_operation_finish(
            logger,
            operation="trade_risk_analyze",
            started_at=started_at,
            success=infer_result_success(result),
            symbol=request.symbol,
            desired_risk_pct=request.desired_risk_pct,
        )
        return result

    try:
        gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return _finish({"error": str(exc)})

    def _analyze_risk():  # noqa: C901
        try:
            account = gateway.account_info()
            if account is None:
                return {"error": "Failed to get account info"}

            equity = validation._safe_float_attr(account, "equity", 0.0)
            currency = getattr(account, "currency", None)
            positions = (
                gateway.positions_get(symbol=request.symbol)
                if request.symbol
                else gateway.positions_get()
            )
            if positions is None:
                positions = []

            position_type_buy = validation._safe_int_attr(
                gateway,
                "POSITION_TYPE_BUY",
                validation._safe_int_attr(gateway, "ORDER_TYPE_BUY", 0),
            )
            position_type_sell = validation._safe_int_attr(
                gateway,
                "POSITION_TYPE_SELL",
                validation._safe_int_attr(gateway, "ORDER_TYPE_SELL", 1),
            )
            position_risks: List[Dict[str, Any]] = []
            risk_calculation_failures: List[Dict[str, Any]] = []
            total_risk_currency = 0.0
            positions_without_sl = 0
            total_notional_exposure = 0.0
            symbol_info_cache: Dict[str, Any] = {}

            for pos in positions:
                try:
                    symbol_key = str(getattr(pos, "symbol", ""))
                    if symbol_key not in symbol_info_cache:
                        symbol_info_cache[symbol_key] = gateway.symbol_info(pos.symbol)
                    sym_info = symbol_info_cache[symbol_key]
                    if sym_info is None:
                        risk_calculation_failures.append(
                            {
                                "ticket": getattr(pos, "ticket", None),
                                "symbol": getattr(pos, "symbol", None),
                                "error": f"Failed to get symbol info for {getattr(pos, 'symbol', None)}",
                                "error_type": "SymbolInfoUnavailable",
                            }
                        )
                        continue

                    entry_price = float(pos.price_open)
                    sl_price = float(pos.sl) if pos.sl and pos.sl > 0 else None
                    tp_price = float(pos.tp) if pos.tp and pos.tp > 0 else None
                    volume = float(pos.volume)

                    contract_size = float(sym_info.trade_contract_size)
                    tick_value = validation._safe_float_attr(
                        sym_info, "trade_tick_value"
                    )
                    tick_value_loss = validation._safe_float_attr(
                        sym_info, "trade_tick_value_loss"
                    )
                    tick_size = validation._safe_float_attr(sym_info, "trade_tick_size")
                    risk_tick_value = _resolve_risk_tick_value(
                        tick_value=tick_value,
                        tick_value_loss=tick_value_loss,
                    )
                    if not math.isfinite(tick_size) or tick_size <= 0:
                        tick_size = 0.0
                    tick_value_valid = (
                        math.isfinite(risk_tick_value) and risk_tick_value > 0
                    )
                    if not math.isfinite(contract_size) or contract_size <= 0:
                        contract_size = 1.0

                    notional_value = abs(volume) * contract_size * entry_price
                    total_notional_exposure += notional_value

                    risk_currency = None
                    risk_pct = None
                    reward_currency = None
                    rr_ratio = None
                    risk_status = "undefined"
                    position_type = validation._safe_int_attr(
                        pos, "type", position_type_sell
                    )
                    is_buy_position = int(position_type) == int(position_type_buy)

                    if sl_price and tick_size > 0 and tick_value_valid:
                        risk_ticks = (
                            (entry_price - sl_price) / tick_size
                            if is_buy_position
                            else (sl_price - entry_price) / tick_size
                        )
                        risk_currency = abs(risk_ticks * risk_tick_value * volume)
                        risk_pct = (
                            (risk_currency / equity) * 100.0 if equity > 0 else 0.0
                        )
                        total_risk_currency += risk_currency
                        risk_status = "defined"

                        if tp_price:
                            reward_ticks = (
                                (tp_price - entry_price) / tick_size
                                if is_buy_position
                                else (entry_price - tp_price) / tick_size
                            )
                            reward_currency = abs(reward_ticks * tick_value * volume)
                            if risk_currency > 0:
                                rr_ratio = reward_currency / risk_currency
                    elif sl_price:
                        risk_status = "undefined"
                        risk_calculation_failures.append(
                            {
                                "ticket": getattr(pos, "ticket", None),
                                "symbol": getattr(pos, "symbol", None),
                                "error": "Stop-loss is set but symbol tick metadata is invalid.",
                                "error_type": "InvalidTickConfiguration",
                            }
                        )
                    else:
                        positions_without_sl += 1
                        risk_status = "unlimited"

                    position_risks.append(
                        {
                            "ticket": pos.ticket,
                            "symbol": pos.symbol,
                            "type": "BUY" if is_buy_position else "SELL",
                            "volume": volume,
                            "entry": entry_price,
                            "sl": sl_price,
                            "tp": tp_price,
                            "risk_currency": round(risk_currency, 2)
                            if risk_currency
                            else None,
                            "risk_pct": round(risk_pct, 2) if risk_pct else None,
                            "risk_status": risk_status,
                            "notional_value": round(notional_value, 2),
                            "reward_currency": round(reward_currency, 2)
                            if reward_currency
                            else None,
                            "rr_ratio": round(rr_ratio, 2) if rr_ratio else None,
                        }
                    )
                except Exception as exc:
                    risk_calculation_failures.append(
                        {
                            "ticket": getattr(pos, "ticket", None),
                            "symbol": getattr(pos, "symbol", None),
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        }
                    )
                    continue

            total_risk_pct = (
                (total_risk_currency / equity) * 100.0 if equity > 0 else 0.0
            )
            notional_exposure_pct = (
                (total_notional_exposure / equity) * 100.0 if equity > 0 else 0.0
            )

            if total_risk_pct > 10:
                quantified_risk_level = "high"
            elif total_risk_pct > 5:
                quantified_risk_level = "moderate"
            else:
                quantified_risk_level = "low"

            if positions_without_sl > 0:
                overall_risk_status = "unlimited"
            elif risk_calculation_failures:
                overall_risk_status = "incomplete"
            else:
                overall_risk_status = "defined"

            result: Dict[str, Any] = {
                "success": True,
                "account": {"equity": round(equity, 2), "currency": currency},
                "portfolio_risk": {
                    "overall_risk_status": overall_risk_status,
                    "quantified_risk_level": quantified_risk_level,
                    "total_risk_currency": round(total_risk_currency, 2),
                    "total_risk_pct": round(total_risk_pct, 2),
                    "positions_count": len(position_risks),
                    "positions_without_sl": positions_without_sl,
                    "positions_with_risk_calculation_failures": len(
                        risk_calculation_failures
                    ),
                    "notional_exposure": round(total_notional_exposure, 2),
                    "notional_exposure_pct": round(notional_exposure_pct, 2),
                },
                "positions": position_risks,
            }
            if risk_calculation_failures:
                result["risk_calculation_failures"] = risk_calculation_failures
            if positions_without_sl > 0:
                result["warning"] = (
                    f"{positions_without_sl} position(s) without stop loss - UNLIMITED RISK!"
                )
            elif risk_calculation_failures:
                result["warning"] = (
                    f"{len(risk_calculation_failures)} position(s) could not be evaluated for risk; "
                    "portfolio risk is incomplete."
                )

            position_sizing_missing = [
                field_name
                for field_name, value in (
                    ("desired_risk_pct", request.desired_risk_pct),
                    ("entry", request.entry),
                    ("stop_loss", request.stop_loss),
                )
                if value is None
            ]
            if position_sizing_missing:
                position_sizing_provided = [
                    field_name
                    for field_name, value in (
                        ("desired_risk_pct", request.desired_risk_pct),
                        ("entry", request.entry),
                        ("stop_loss", request.stop_loss),
                    )
                    if value is not None
                ]
                position_sizing: Dict[str, Any] = {
                    "status": "incomplete",
                    "message": (
                        "Portfolio risk analysis completed. Position sizing is "
                        "available when you provide desired_risk_pct, "
                        "entry, and stop_loss."
                    ),
                    "missing": position_sizing_missing,
                    "required_for_sizing": [
                        "desired_risk_pct",
                        "entry",
                        "stop_loss",
                    ],
                    "note": (
                        "Add desired_risk_pct to specify how much equity to risk "
                        "on the proposed trade."
                    ),
                }
                if position_sizing_provided:
                    position_sizing.update(
                        {
                            "provided": position_sizing_provided,
                            "proposed_trade_context": {
                                key: value
                                for key, value in (
                                    ("desired_risk_pct", request.desired_risk_pct),
                                    ("entry", request.entry),
                                    ("stop_loss", request.stop_loss),
                                    ("take_profit", request.take_profit),
                                    ("direction", request.direction),
                                )
                                if value is not None
                            },
                            "sizing_not_calculated_reason": (
                                "Position sizing requires desired_risk_pct, "
                                "entry, and stop_loss."
                            ),
                        }
                    )
                result["position_sizing"] = position_sizing

            if (
                request.desired_risk_pct is not None
                and request.entry is not None
                and request.stop_loss is not None
            ):
                if not request.symbol:
                    return {"error": "symbol is required for position sizing"}

                sym_info = gateway.symbol_info(request.symbol)
                if sym_info is None:
                    return {"error": f"Symbol {request.symbol} not found"}

                contract_size = float(sym_info.trade_contract_size)
                tick_value = validation._safe_float_attr(sym_info, "trade_tick_value")
                tick_value_loss = validation._safe_float_attr(
                    sym_info, "trade_tick_value_loss"
                )
                tick_size = validation._safe_float_attr(sym_info, "trade_tick_size")
                risk_tick_value = _resolve_risk_tick_value(
                    tick_value=tick_value,
                    tick_value_loss=tick_value_loss,
                )
                if not math.isfinite(tick_size) or tick_size <= 0:
                    tick_size = 0.0
                min_volume = float(sym_info.volume_min)
                max_volume = float(sym_info.volume_max)
                volume_step = float(sym_info.volume_step)
                if not (
                    math.isfinite(risk_tick_value)
                    and risk_tick_value > 0
                    and math.isfinite(tick_size)
                    and tick_size > 0
                ):
                    result["position_sizing_error"] = (
                        "Symbol tick configuration is invalid for risk sizing"
                    )
                    return result
                if not (math.isfinite(volume_step) and volume_step > 0):
                    volume_step = max(min_volume, 0.01)
                if not math.isfinite(contract_size) or contract_size <= 0:
                    contract_size = 1.0

                direction_norm, direction_error, direction_source = (
                    _resolve_trade_risk_direction(
                        direction=request.direction,
                        entry=float(request.entry),
                        stop_loss=float(request.stop_loss),
                        take_profit=float(request.take_profit)
                        if request.take_profit is not None
                        else None,
                    )
                )
                if direction_error or direction_norm is None:
                    result["position_sizing_error"] = (
                        direction_error
                        or "Unable to resolve trade direction for position sizing."
                    )
                    return result
                level_error = _validate_trade_risk_levels(
                    direction=direction_norm,
                    entry=float(request.entry),
                    stop_loss=float(request.stop_loss),
                    take_profit=float(request.take_profit)
                    if request.take_profit is not None
                    else None,
                )
                if level_error:
                    result["position_sizing_error"] = level_error
                    return result

                risk_amount = equity * (request.desired_risk_pct / 100.0)
                if direction_norm == "long":
                    sl_distance_ticks = (
                        request.entry - request.stop_loss
                    ) / tick_size
                else:
                    sl_distance_ticks = (
                        request.stop_loss - request.entry
                    ) / tick_size
                if sl_distance_ticks > 0:
                    raw_volume = risk_amount / (sl_distance_ticks * risk_tick_value)
                    if not math.isfinite(raw_volume) or raw_volume <= 0:
                        result["position_sizing_error"] = "Calculated volume is invalid"
                        return result

                    volume_steps = _floor_volume_steps(raw_volume, volume_step)
                    suggested_volume = volume_steps * volume_step
                    rounding_mode = "rounded_down_to_step"
                    sizing_notes: List[str] = []

                    if suggested_volume < min_volume:
                        suggested_volume = min_volume
                        rounding_mode = "clamped_to_min_volume"
                        sizing_notes.append(
                            "Minimum trade volume forces the size up to the broker minimum."
                        )
                    elif suggested_volume > max_volume:
                        suggested_volume = max_volume
                        rounding_mode = "clamped_to_max_volume"
                        sizing_notes.append(
                            "Maximum trade volume caps the size below the unconstrained target."
                        )
                    elif suggested_volume < raw_volume:
                        sizing_notes.append(
                            "Volume was rounded down to the nearest broker step to avoid exceeding requested risk."
                        )
                    if direction_source == "inferred_from_stop_loss":
                        sizing_notes.append(
                            "Direction was inferred from stop-loss placement."
                        )
                    elif direction_source == "inferred_from_take_profit":
                        sizing_notes.append(
                            "Direction was inferred from take-profit placement because stop-loss matched entry."
                        )

                    step_txt = f"{volume_step:.10f}".rstrip("0")
                    step_decimals = (
                        len(step_txt.split(".")[1]) if "." in step_txt else 0
                    )
                    if step_decimals > 0:
                        suggested_volume = float(
                            f"{suggested_volume:.{step_decimals}f}"
                        )
                    else:
                        suggested_volume = float(round(suggested_volume))

                    actual_risk = sl_distance_ticks * risk_tick_value * suggested_volume
                    actual_risk_pct = (actual_risk / equity) * 100.0
                    risk_pct_diff = actual_risk_pct - float(request.desired_risk_pct)
                    risk_over_target = actual_risk_pct > (
                        float(request.desired_risk_pct) + 1e-9
                    )
                    overshoot_pct = max(
                        0.0, float(actual_risk_pct) - float(request.desired_risk_pct)
                    )
                    overshoot_currency = max(
                        0.0, float(actual_risk) - float(risk_amount)
                    )
                    overshoot_reason = None
                    if risk_over_target:
                        if rounding_mode == "clamped_to_min_volume":
                            overshoot_reason = "min_volume_constraint"
                        elif rounding_mode == "clamped_to_max_volume":
                            overshoot_reason = "max_volume_constraint"
                        elif rounding_mode == "rounded_down_to_step":
                            overshoot_reason = "step_rounding_precision"
                        else:
                            overshoot_reason = "broker_volume_constraints"
                        sizing_notes.append(
                            "Actual risk still exceeds the requested level after broker volume constraints."
                        )

                    rr_ratio = None
                    reward_currency = None
                    if request.take_profit is not None:
                        if direction_norm == "long":
                            tp_distance_ticks = (
                                request.take_profit - request.entry
                            ) / tick_size
                        else:
                            tp_distance_ticks = (
                                request.entry - request.take_profit
                            ) / tick_size
                        reward_currency = (
                            tp_distance_ticks * tick_value * suggested_volume
                        )
                        if actual_risk > 0:
                            rr_ratio = reward_currency / actual_risk

                    result["position_sizing"] = {
                        "symbol": request.symbol,
                        "direction": direction_norm,
                        "direction_source": direction_source,
                        "suggested_volume": suggested_volume,
                        "requested_risk_currency": round(risk_amount, 2),
                        "requested_risk_pct": float(request.desired_risk_pct),
                        "entry": request.entry,
                        "sl": request.stop_loss,
                        "tp": request.take_profit,
                        "risk_currency": round(actual_risk, 2),
                        "risk_pct": round(actual_risk_pct, 2),
                        "risk_pct_diff": round(risk_pct_diff, 2),
                        "risk_over_target": risk_over_target,
                        "risk_compliance": (
                            "exceeds_requested_risk"
                            if risk_over_target
                            else "within_requested_risk"
                        ),
                        "risk_overshoot_pct": round(overshoot_pct, 2),
                        "risk_overshoot_currency": round(overshoot_currency, 2),
                        "risk_over_target_reason": overshoot_reason,
                        "raw_volume": round(raw_volume, 8),
                        "volume_step": volume_step,
                        "volume_min": min_volume,
                        "volume_max": max_volume,
                        "volume_rounding": rounding_mode,
                        "reward_currency": round(reward_currency, 2)
                        if reward_currency
                        else None,
                        "rr_ratio": round(rr_ratio, 2) if rr_ratio else None,
                        "sizing_notes": sizing_notes,
                    }
                    if risk_over_target:
                        result["position_sizing_warning"] = (
                            f"Requested risk {float(request.desired_risk_pct):.2f}% but actual risk is "
                            f"{float(actual_risk_pct):.2f}% (+{overshoot_pct:.2f}%) after broker volume constraints."
                        )
                        result["risk_alert"] = {
                            "severity": "warning",
                            "code": "risk_overshoot_after_volume_constraints",
                            "reason": overshoot_reason,
                            "requested_risk_pct": float(request.desired_risk_pct),
                            "actual_risk_pct": round(actual_risk_pct, 2),
                            "overshoot_pct": round(overshoot_pct, 2),
                            "requested_risk_currency": round(risk_amount, 2),
                            "actual_risk_currency": round(actual_risk, 2),
                            "overshoot_currency": round(overshoot_currency, 2),
                        }
                else:
                    result["position_sizing_error"] = (
                        "SL distance must be greater than 0"
                    )

            return result
        except Exception as exc:
            return {"error": str(exc)}

    return _finish(_analyze_risk())


def run_trade_var_cvar_calculate(  # noqa: C901
    request: TradeVarCvarRequest,
    *,
    gateway: Any,
) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="trade_var_cvar_calculate",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        confidence=request.confidence,
    )

    def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
        log_operation_finish(
            logger,
            operation="trade_var_cvar_calculate",
            started_at=started_at,
            success=infer_result_success(result),
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            confidence=request.confidence,
        )
        return result

    try:
        gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return _finish({"error": str(exc)})

    timeframe_value = str(request.timeframe or "").strip().upper()
    if timeframe_value not in TIMEFRAME_MAP:
        return _finish(
            {"error": invalid_timeframe_error(timeframe_value, TIMEFRAME_MAP)}
        )

    method_value, method_error = _normalize_var_cvar_method(request.method)
    if method_error or method_value is None:
        return _finish({"error": method_error})

    transform_value, transform_error = _normalize_var_cvar_transform(request.transform)
    if transform_error or transform_value is None:
        return _finish({"error": transform_error})

    confidence_value, confidence_error = _normalize_var_cvar_confidence(
        request.confidence
    )
    if confidence_error or confidence_value is None:
        return _finish({"error": confidence_error})

    try:
        lookback = int(request.lookback)
    except TypeError, ValueError:
        return _finish({"error": "lookback must be an integer"})
    if lookback < 2:
        return _finish({"error": "lookback must be at least 2"})

    try:
        min_observations = int(request.min_observations)
    except TypeError, ValueError:
        return _finish({"error": "min_observations must be an integer"})
    if min_observations < 2:
        return _finish({"error": "min_observations must be at least 2"})

    try:
        account = gateway.account_info()
    except Exception:
        account = None
    equity = None
    currency = None
    if account is not None:
        equity_value = validation._safe_float_attr(account, "equity", 0.0)
        if equity_value > 0.0:
            equity = float(equity_value)
        currency_text = str(getattr(account, "currency", "") or "").strip()
        if currency_text:
            currency = currency_text

    try:
        positions = (
            gateway.positions_get(symbol=request.symbol)
            if request.symbol
            else gateway.positions_get()
        )
    except Exception as exc:
        return _finish({"error": str(exc)})
    if positions is None:
        positions = []
    if not positions:
        message = (
            f"No open positions found for symbol {request.symbol}"
            if request.symbol
            else "No open positions found for VaR/CVaR calculation."
        )
        summary: Dict[str, Any] = {
            "method": method_value,
            "confidence": round(float(confidence_value), 6),
            "transform": transform_value,
            "timeframe": timeframe_value,
            "lookback": int(lookback),
            "min_observations": int(min_observations),
            "observations": 0,
            "positions": 0,
            "symbols": 0,
            "gross_notional": 0.0,
            "net_exposure": 0.0,
            "var": 0.0,
            "cvar": 0.0,
        }
        if equity is not None and equity > 0.0:
            summary["equity"] = round(float(equity), 2)
            summary["var_pct_of_equity"] = 0.0
            summary["cvar_pct_of_equity"] = 0.0
        if currency:
            summary["currency"] = currency
        result: Dict[str, Any] = {
            "success": True,
            "message": message,
            "empty": True,
            "reason": message,
        }
        if request.detail == "full":
            result.update(
                {
                    "summary": summary,
                    "symbol_exposures": [],
                    "positions": [],
                    "worst_observations": [],
                }
            )
        else:
            result["summary"] = {
                key: summary[key]
                for key in ("positions", "symbols", "equity", "currency")
                if key in summary
            }
        return _finish(result)

    position_type_buy = validation._safe_int_attr(
        gateway,
        "POSITION_TYPE_BUY",
        validation._safe_int_attr(gateway, "ORDER_TYPE_BUY", 0),
    )
    position_type_sell = validation._safe_int_attr(
        gateway,
        "POSITION_TYPE_SELL",
        validation._safe_int_attr(gateway, "ORDER_TYPE_SELL", 1),
    )
    mt5_timeframe = TIMEFRAME_MAP[timeframe_value]
    symbol_info_cache: Dict[str, Any] = {}
    history_failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    position_exposures: List[Dict[str, Any]] = []
    symbol_exposures: Dict[str, Dict[str, Any]] = {}

    for position in positions:
        symbol = str(getattr(position, "symbol", "") or "").strip()
        if not symbol:
            warnings.append(
                {
                    "ticket": getattr(position, "ticket", None),
                    "warning": "Position has no symbol",
                }
            )
            continue
        if symbol not in symbol_info_cache:
            symbol_info_cache[symbol] = gateway.symbol_info(symbol)
        symbol_info = symbol_info_cache[symbol]
        if symbol_info is None:
            warnings.append(
                {
                    "ticket": getattr(position, "ticket", None),
                    "symbol": symbol,
                    "warning": "Symbol info unavailable",
                }
            )
            continue

        volume = validation._safe_float_attr(position, "volume", 0.0)
        if not math.isfinite(volume) or volume <= 0.0:
            warnings.append(
                {
                    "ticket": getattr(position, "ticket", None),
                    "symbol": symbol,
                    "warning": "Position volume is invalid",
                }
            )
            continue

        contract_size = validation._safe_float_attr(
            symbol_info, "trade_contract_size", 1.0
        )
        if not math.isfinite(contract_size) or contract_size <= 0.0:
            contract_size = 1.0
        mark_price = validation._safe_float_attr(position, "price_current", 0.0)
        if mark_price <= 0.0:
            mark_price = validation._safe_float_attr(position, "price_open", 0.0)
        if not math.isfinite(mark_price) or mark_price <= 0.0:
            warnings.append(
                {
                    "ticket": getattr(position, "ticket", None),
                    "symbol": symbol,
                    "warning": "Position mark price is invalid",
                }
            )
            continue

        position_type = validation._safe_int_attr(position, "type", position_type_sell)
        side = "BUY" if int(position_type) == int(position_type_buy) else "SELL"
        side_sign = 1.0 if side == "BUY" else -1.0
        signed_notional = side_sign * volume * contract_size * mark_price

        position_exposures.append(
            {
                "ticket": getattr(position, "ticket", None),
                "symbol": symbol,
                "side": side,
                "volume": float(volume),
                "mark_price": round(float(mark_price), 6),
                "contract_size": round(float(contract_size), 6),
                "signed_notional": round(float(signed_notional), 2),
                "unrealized_profit": round(
                    validation._safe_float_attr(position, "profit", 0.0), 2
                ),
            }
        )

        exposure = symbol_exposures.setdefault(
            symbol,
            {
                "symbol": symbol,
                "signed_notional": 0.0,
                "gross_notional": 0.0,
                "positions": 0,
            },
        )
        exposure["signed_notional"] += float(signed_notional)
        exposure["gross_notional"] += abs(float(signed_notional))
        exposure["positions"] += 1

    if not position_exposures:
        return _finish(
            {"error": "No usable open positions available for VaR/CVaR calculation."}
        )

    return_series: Dict[str, Any] = {}
    for symbol in list(symbol_exposures.keys()):
        try:
            rates = gateway.copy_rates_from_pos(symbol, mt5_timeframe, 0, lookback)
            if rates is not None:
                rates = _normalize_times_in_struct(rates)
        except Exception as exc:
            history_failures.append({"symbol": symbol, "error": str(exc)})
            continue
        returns, history_error = _extract_var_cvar_return_series(
            symbol=symbol,
            rates=rates,
            transform=transform_value,
            pd_module=pd,
            np_module=np,
        )
        if history_error:
            history_failures.append({"symbol": symbol, "error": history_error})
            continue
        return_series[symbol] = returns

    if not return_series:
        result: Dict[str, Any] = {
            "error": "Unable to build return series for any open-position symbols.",
        }
        if history_failures:
            result["history_failures"] = history_failures
        if warnings:
            result["warnings"] = warnings
        return _finish(result)

    valid_symbols = set(return_series)
    position_exposures = [
        item for item in position_exposures if item["symbol"] in valid_symbols
    ]
    symbol_exposure_frame = {
        symbol: data
        for symbol, data in symbol_exposures.items()
        if symbol in valid_symbols
    }
    if not position_exposures or not symbol_exposure_frame:
        return _finish(
            {"error": "No open positions remained after filtering unavailable history."}
        )

    aligned_returns = pd.concat(
        [return_series[symbol].rename(symbol) for symbol in symbol_exposure_frame],
        axis=1,
        join="inner",
    ).dropna(how="any")
    if len(aligned_returns) < min_observations:
        result = {
            "error": (
                f"Not enough aligned return observations for VaR/CVaR calculation "
                f"({len(aligned_returns)} available, {min_observations} required)."
            ),
            "available_observations": int(len(aligned_returns)),
        }
        if history_failures:
            result["history_failures"] = history_failures
        if warnings:
            result["warnings"] = warnings
        return _finish(result)

    exposure_vector = pd.Series(
        {
            symbol: float(data["signed_notional"])
            for symbol, data in symbol_exposure_frame.items()
        }
    )
    portfolio_pnl = (
        aligned_returns[exposure_vector.index].mul(exposure_vector, axis=1).sum(axis=1)
    )
    pnl_values = [
        float(value) for value in portfolio_pnl.tolist() if math.isfinite(float(value))
    ]
    if len(pnl_values) < min_observations:
        return _finish(
            {
                "error": (
                    f"Not enough finite portfolio PnL observations for VaR/CVaR calculation "
                    f"({len(pnl_values)} available, {min_observations} required)."
                )
            }
        )

    try:
        var_value, cvar_value, threshold = _calculate_var_cvar_from_pnl(
            pnl_values,
            confidence=confidence_value,
            method=method_value,
        )
    except Exception as exc:
        return _finish({"error": str(exc)})

    total_abs_notional = float(
        sum(abs(float(item["signed_notional"])) for item in position_exposures)
    )
    net_exposure = float(
        sum(float(item["signed_notional"]) for item in position_exposures)
    )
    mean_pnl = float(sum(pnl_values) / len(pnl_values))
    if len(pnl_values) > 1:
        variance = sum((value - mean_pnl) ** 2 for value in pnl_values) / float(
            len(pnl_values) - 1
        )
        volatility_pnl = math.sqrt(max(0.0, variance))
    else:
        volatility_pnl = 0.0

    symbol_rows: List[Dict[str, Any]] = []
    for symbol, data in symbol_exposure_frame.items():
        gross_notional = float(data["gross_notional"])
        symbol_rows.append(
            {
                "symbol": symbol,
                "positions": int(data["positions"]),
                "signed_notional": round(float(data["signed_notional"]), 2),
                "gross_notional": round(gross_notional, 2),
                "gross_weight": round((gross_notional / total_abs_notional), 6)
                if total_abs_notional > 0.0
                else 0.0,
            }
        )
    symbol_rows.sort(
        key=lambda item: (-abs(float(item["signed_notional"])), item["symbol"])
    )

    worst_bars = portfolio_pnl.nsmallest(min(5, len(portfolio_pnl)))
    worst_observations = [
        {
            "time": _format_var_cvar_timestamp(timestamp),
            "simulated_pnl": round(float(value), 2),
        }
        for timestamp, value in worst_bars.items()
    ]

    summary: Dict[str, Any] = {
        "method": method_value,
        "confidence": round(float(confidence_value), 6),
        "transform": transform_value,
        "timeframe": timeframe_value,
        "lookback": int(lookback),
        "min_observations": int(min_observations),
        "observations": int(len(pnl_values)),
        "positions": int(len(position_exposures)),
        "symbols": int(len(symbol_rows)),
        "gross_notional": round(total_abs_notional, 2),
        "net_exposure": round(net_exposure, 2),
        "var": round(float(var_value), 2),
        "cvar": round(float(cvar_value), 2),
        "tail_threshold": round(float(threshold), 2),
        "mean_pnl": round(mean_pnl, 2),
        "volatility_pnl": round(float(volatility_pnl), 2),
        "worst_observed_pnl": round(min(pnl_values), 2),
        "best_observed_pnl": round(max(pnl_values), 2),
    }
    if equity is not None and equity > 0.0:
        summary["equity"] = round(float(equity), 2)
        summary["var_pct_of_equity"] = round((float(var_value) / equity) * 100.0, 4)
        summary["cvar_pct_of_equity"] = round((float(cvar_value) / equity) * 100.0, 4)
    if currency:
        summary["currency"] = currency

    result = {
        "success": True,
        "summary": summary,
        "symbol_exposures": symbol_rows,
        "positions": position_exposures,
        "worst_observations": worst_observations,
    }
    if history_failures:
        result["history_failures"] = history_failures
    if warnings:
        result["warnings"] = warnings
    return _finish(result)


def run_trade_get_open(
    request: TradeGetOpenRequest,
    *,
    gateway: Any,
    use_client_tz: Any,
    format_time_minimal: Any,
    format_time_minimal_local: Any,
    mt5_epoch_to_utc: Any,
    normalize_limit: Any,
    comment_row_metadata: Any,
) -> List[Dict[str, Any]]:
    import pandas as pd

    result = run_logged_operation(
        logger,
        operation="trade_get_open",
        symbol=request.symbol,
        ticket=request.ticket,
        limit=request.limit,
        func=lambda: _run_trade_get_open_impl(
            request=request,
            gateway=gateway,
            use_client_tz=use_client_tz,
            format_time_minimal=format_time_minimal,
            format_time_minimal_local=format_time_minimal_local,
            mt5_epoch_to_utc=mt5_epoch_to_utc,
            normalize_limit=normalize_limit,
            comment_row_metadata=comment_row_metadata,
            pd_module=pd,
        ),
    )
    if isinstance(result, Ok):
        return result.value
    if isinstance(result, Err):
        return [to_dict(result)]
    return result


def run_trade_get_pending(
    request: TradeGetPendingRequest,
    *,
    gateway: Any,
    use_client_tz: Any,
    format_time_minimal: Any,
    format_time_minimal_local: Any,
    mt5_epoch_to_utc: Any,
    normalize_limit: Any,
    comment_row_metadata: Any,
) -> List[Dict[str, Any]]:
    import pandas as pd

    result = run_logged_operation(
        logger,
        operation="trade_get_pending",
        symbol=request.symbol,
        ticket=request.ticket,
        limit=request.limit,
        func=lambda: _run_trade_get_pending_impl(
            request=request,
            gateway=gateway,
            use_client_tz=use_client_tz,
            format_time_minimal=format_time_minimal,
            format_time_minimal_local=format_time_minimal_local,
            mt5_epoch_to_utc=mt5_epoch_to_utc,
            normalize_limit=normalize_limit,
            comment_row_metadata=comment_row_metadata,
            pd_module=pd,
        ),
    )
    if isinstance(result, Ok):
        return result.value
    if isinstance(result, Err):
        return [to_dict(result)]
    return result


def _mt5_int_const(gateway: Any, name: str, fallback: int) -> int:
    return validation._safe_int_attr(gateway, name, fallback)


def _pick_trade_series(df: Any, pd_module: Any, *names: str):
    out = None
    for name in names:
        if name in df.columns:
            out = df[name] if out is None else out.where(out.notna(), df[name])
    if out is None:
        return pd_module.Series([None] * len(df), index=df.index)
    return out


def _fetch_trade_query_rows(
    request: Any,
    *,
    fetch_rows: Any,
    no_ticket_message: Any,
    no_symbol_message: Any,
    no_rows_message: str,
) -> tuple[Optional[Any], Optional[List[Dict[str, Any]]]]:
    if request.ticket is not None:
        ticket_int = int(request.ticket)
        rows = fetch_rows(ticket=ticket_int)
        if rows is None or len(rows) == 0:
            return None, [{"message": no_ticket_message(request.ticket)}]
    elif request.symbol is not None:
        # Validate symbol before querying
        symbol_error = _ensure_symbol_ready(request.symbol)
        if symbol_error:
            return None, [{"error": symbol_error}]
        rows = fetch_rows(symbol=request.symbol)
        if rows is None or len(rows) == 0:
            return None, [{"message": no_symbol_message(request.symbol)}]
    else:
        rows = fetch_rows()
        if rows is None or len(rows) == 0:
            return None, [{"message": no_rows_message}]
    return rows, None


def _build_trade_time_columns(
    df: Any,
    *,
    time_source_fields: tuple[str, ...],
    pd_module: Any,
    mt5_epoch_to_utc: Any,
    fmt_time: Any,
) -> tuple[Any, Any]:
    time_src = None
    for field in time_source_fields:
        if field in df.columns:
            time_src = df[field]
            break
    if time_src is None:
        time_utc = pd_module.Series([float("nan")] * len(df), index=df.index)
        time_txt = pd_module.Series([None] * len(df), index=df.index)
    else:
        time_utc, time_txt = _epoch_series_to_utc_and_text(
            time_src,
            pd_module=pd_module,
            mt5_epoch_to_utc=mt5_epoch_to_utc,
            fmt_time=fmt_time,
        )
    return time_utc, time_txt


def _append_trade_comment_metadata(
    out_df: Any,
    *,
    comment_series: Any,
    comment_row_metadata: Any,
) -> None:
    comment_lengths: List[Any] = []
    comment_limits: List[Any] = []
    comment_truncation: List[Any] = []
    for comment_value in comment_series.tolist():
        metadata = comment_row_metadata(comment_value)
        if not isinstance(metadata, dict):
            metadata = {}
        comment_lengths.append(metadata.get("comment_visible_length"))
        comment_limits.append(metadata.get("comment_max_length"))
        comment_truncation.append(metadata.get("comment_may_be_truncated"))
    out_df["comment_visible_length"] = comment_lengths
    out_df["comment_max_length"] = comment_limits
    out_df["comment_may_be_truncated"] = comment_truncation


def _apply_trade_query_limit(
    out_df: Any,
    *,
    time_utc: Any,
    limit: Any,
    normalize_limit: Any,
) -> Any:
    limit_value = normalize_limit(limit)
    if not limit_value or len(out_df) <= limit_value:
        return out_df
    sorted_index = (
        time_utc.sort_values(
            kind="stable",
            na_position="first",
        )
        .tail(limit_value)
        .index
    )
    return out_df.loc[sorted_index].copy()


def _build_trade_get_open_output(
    *,
    df: Any,
    gateway: Any,
    request: Any,
    time_txt: Any,
    pd_module: Any,
    **_kwargs: Any,
) -> Any:
    open_df = df.drop(
        columns=[
            col
            for col in ("time_msc", "time_update", "time_update_msc")
            if col in df.columns
        ]
    ).copy()
    if "type" in open_df.columns:
        mapped = open_df["type"].map(
            {
                _mt5_int_const(gateway, "POSITION_TYPE_BUY", 0): "BUY",
                _mt5_int_const(gateway, "POSITION_TYPE_SELL", 1): "SELL",
            }
        )
        open_df["type"] = mapped.fillna(open_df["type"].astype(str))
    return pd_module.DataFrame(
        {
            "symbol": _pick_trade_series(open_df, pd_module, "symbol"),
            "ticket": _pick_trade_series(open_df, pd_module, "ticket"),
            "time": time_txt,
            "type": _pick_trade_series(open_df, pd_module, "type"),
            "volume": _pick_trade_series(open_df, pd_module, "volume"),
            "price_open": _pick_trade_series(open_df, pd_module, "price_open"),
            "sl": _pick_trade_series(open_df, pd_module, "sl"),
            "tp": _pick_trade_series(open_df, pd_module, "tp"),
            "price_current": _pick_trade_series(open_df, pd_module, "price_current"),
            "swap": pd_module.to_numeric(
                _pick_trade_series(open_df, pd_module, "swap"),
                errors="coerce",
            ).fillna(0.0),
            "profit": pd_module.to_numeric(
                _pick_trade_series(open_df, pd_module, "profit"),
                errors="coerce",
            ).fillna(0.0),
            "comment": _pick_trade_series(open_df, pd_module, "comment"),
            "magic": _pick_trade_series(open_df, pd_module, "magic"),
        }
    )


def _build_trade_get_pending_output(
    *,
    df: Any,
    gateway: Any,
    request: Any,
    time_txt: Any,
    pd_module: Any,
    fmt_time: Any,
    mt5_epoch_to_utc: Any,
    **_kwargs: Any,
) -> Any:
    pending_df = df.copy()
    if "time_expiration" in pending_df.columns:
        exp_raw = pd_module.to_numeric(pending_df["time_expiration"], errors="coerce")
        _, exp_text = _epoch_series_to_utc_and_text(
            exp_raw,
            pd_module=pd_module,
            mt5_epoch_to_utc=mt5_epoch_to_utc,
            fmt_time=fmt_time,
            require_positive=True,
        )
        expiration = pd_module.Series(
            [
                None
                if pd_module.isna(raw_value)
                else "GTC"
                if float(raw_value) <= 0.0
                else text_value
                for raw_value, text_value in zip(exp_raw.tolist(), exp_text.tolist())
            ],
            index=exp_raw.index,
        )
    else:
        expiration = pd_module.Series([None] * len(pending_df), index=pending_df.index)

    pending_df = pending_df.drop(
        columns=[
            col
            for col in (
                "time_setup",
                "time_setup_msc",
                "time_done",
                "time_done_msc",
                "time_expiration",
                "time_msc",
            )
            if col in pending_df.columns
        ]
    ).copy()
    if "type" in pending_df.columns:
        mapped = pending_df["type"].map(
            {
                _mt5_int_const(gateway, "ORDER_TYPE_BUY", 0): "BUY",
                _mt5_int_const(gateway, "ORDER_TYPE_SELL", 1): "SELL",
                _mt5_int_const(gateway, "ORDER_TYPE_BUY_LIMIT", 2): "BUY_LIMIT",
                _mt5_int_const(gateway, "ORDER_TYPE_SELL_LIMIT", 3): "SELL_LIMIT",
                _mt5_int_const(gateway, "ORDER_TYPE_BUY_STOP", 4): "BUY_STOP",
                _mt5_int_const(gateway, "ORDER_TYPE_SELL_STOP", 5): "SELL_STOP",
                _mt5_int_const(
                    gateway, "ORDER_TYPE_BUY_STOP_LIMIT", 6
                ): "BUY_STOP_LIMIT",
                _mt5_int_const(
                    gateway, "ORDER_TYPE_SELL_STOP_LIMIT", 7
                ): "SELL_STOP_LIMIT",
            }
        )
        pending_df["type"] = mapped.fillna(pending_df["type"].astype(str))
    return pd_module.DataFrame(
        {
            "symbol": _pick_trade_series(pending_df, pd_module, "symbol"),
            "ticket": _pick_trade_series(pending_df, pd_module, "ticket"),
            "time": time_txt,
            "expiration": expiration,
            "type": _pick_trade_series(pending_df, pd_module, "type"),
            "volume": _pick_trade_series(
                pending_df,
                pd_module,
                "volume",
                "volume_current",
                "volume_initial",
            ),
            "price_open": _pick_trade_series(pending_df, pd_module, "price_open"),
            "sl": _pick_trade_series(pending_df, pd_module, "sl"),
            "tp": _pick_trade_series(pending_df, pd_module, "tp"),
            "price_current": _pick_trade_series(pending_df, pd_module, "price_current"),
            "comment": _pick_trade_series(pending_df, pd_module, "comment"),
            "magic": _pick_trade_series(pending_df, pd_module, "magic"),
        }
    )


def _run_trade_query_impl(
    *,
    request: Any,
    gateway: Any,
    use_client_tz: Any,
    format_time_minimal: Any,
    format_time_minimal_local: Any,
    mt5_epoch_to_utc: Any,
    normalize_limit: Any,
    comment_row_metadata: Any,
    pd_module: Any,
    fetch_rows: Any,
    no_ticket_message: Any,
    no_symbol_message: Any,
    no_rows_message: str,
    time_source_fields: tuple[str, ...],
    build_output: Any,
) -> Result[List[Dict[str, Any]]]:
    try:
        gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return Err(str(exc), code="MT5_CONNECTION")

    try:
        fmt_time = format_time_minimal_local if use_client_tz() else format_time_minimal
        rows, empty_response = _fetch_trade_query_rows(
            request,
            fetch_rows=fetch_rows,
            no_ticket_message=no_ticket_message,
            no_symbol_message=no_symbol_message,
            no_rows_message=no_rows_message,
        )
        if empty_response is not None:
            return Ok(empty_response)

        df = _trade_rows_to_dataframe(rows, pd_module=pd_module)
        time_utc, time_txt = _build_trade_time_columns(
            df,
            time_source_fields=time_source_fields,
            pd_module=pd_module,
            mt5_epoch_to_utc=mt5_epoch_to_utc,
            fmt_time=fmt_time,
        )
        comment_series = _pick_trade_series(df, pd_module, "comment")
        out_df = build_output(
            df=df,
            gateway=gateway,
            request=request,
            time_txt=time_txt,
            pd_module=pd_module,
            fmt_time=fmt_time,
            mt5_epoch_to_utc=mt5_epoch_to_utc,
        )
        _append_trade_comment_metadata(
            out_df,
            comment_series=comment_series,
            comment_row_metadata=comment_row_metadata,
        )
        out_df = _apply_trade_query_limit(
            out_df,
            time_utc=time_utc,
            limit=request.limit,
            normalize_limit=normalize_limit,
        )
        return Ok(out_df.to_dict(orient="records"))
    except Exception as exc:
        return Err(str(exc))


def _run_trade_get_open_impl(
    *,
    request: TradeGetOpenRequest,
    gateway: Any,
    use_client_tz: Any,
    format_time_minimal: Any,
    format_time_minimal_local: Any,
    mt5_epoch_to_utc: Any,
    normalize_limit: Any,
    comment_row_metadata: Any,
    pd_module: Any,
) -> Result[List[Dict[str, Any]]]:
    return _run_trade_query_impl(
        request=request,
        gateway=gateway,
        use_client_tz=use_client_tz,
        format_time_minimal=format_time_minimal,
        format_time_minimal_local=format_time_minimal_local,
        mt5_epoch_to_utc=mt5_epoch_to_utc,
        normalize_limit=normalize_limit,
        comment_row_metadata=comment_row_metadata,
        pd_module=pd_module,
        fetch_rows=gateway.positions_get,
        no_ticket_message=lambda ticket: f"No position found with ID {ticket}",
        no_symbol_message=lambda symbol: f"No open positions for {symbol}",
        no_rows_message="No open positions",
        time_source_fields=("time_update", "time"),
        build_output=_build_trade_get_open_output,
    )


def _run_trade_get_pending_impl(
    *,
    request: TradeGetPendingRequest,
    gateway: Any,
    use_client_tz: Any,
    format_time_minimal: Any,
    format_time_minimal_local: Any,
    mt5_epoch_to_utc: Any,
    normalize_limit: Any,
    comment_row_metadata: Any,
    pd_module: Any,
) -> Result[List[Dict[str, Any]]]:
    return _run_trade_query_impl(
        request=request,
        gateway=gateway,
        use_client_tz=use_client_tz,
        format_time_minimal=format_time_minimal,
        format_time_minimal_local=format_time_minimal_local,
        mt5_epoch_to_utc=mt5_epoch_to_utc,
        normalize_limit=normalize_limit,
        comment_row_metadata=comment_row_metadata,
        pd_module=pd_module,
        fetch_rows=gateway.orders_get,
        no_ticket_message=lambda ticket: f"No pending order found with ID {ticket}",
        no_symbol_message=lambda symbol: f"No pending orders for {symbol}",
        no_rows_message="No pending orders",
        time_source_fields=("time_setup", "time"),
        build_output=_build_trade_get_pending_output,
    )
