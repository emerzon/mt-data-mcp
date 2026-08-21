"""Trade modification use case."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from mtdata.core.error_envelope import normalize_error_payload
from mtdata.core.execution_logging import (
    infer_result_success,
    log_operation_finish,
    log_operation_start,
)
from mtdata.core.trading import comments
from mtdata.core.trading.requests import TradeModifyRequest
from mtdata.core.trading.use_cases.common import (
    _TRADE_IDEMPOTENCY_STORE,
    TradeIdempotencyStore,
    _annotate_idempotency_scope,
    _attach_live_guardrail_status,
    _attach_trade_correlation,
    _begin_trade_idempotency,
    _build_trade_request_signature,
    _invalid_pending_expiration_payload,
    _log_trade_correlation,
    _normalize_idempotency_key,
    _record_or_release_idempotency,
    logger,
)


def run_trade_modify(
    request: TradeModifyRequest,
    *,
    normalize_pending_expiration: Any,
    modify_pending_order: Any,
    modify_position: Any,
    idempotency_store: Optional[TradeIdempotencyStore] = _TRADE_IDEMPOTENCY_STORE,
    correlation_id: Optional[str] = None,
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
        correlation_id=correlation_id,
        ticket=request.ticket,
        dry_run=request.dry_run,
    )

    def _finish(
        result: Dict[str, Any],
        *,
        pending: Optional[bool] = None,
    ) -> Dict[str, Any]:
        nonlocal idempotency_consumed
        if (
            request.dry_run
            and result.get("success") is True
            and not str(result.get("error") or "").strip()
        ):
            result.setdefault("preview_ok", True)
            result.setdefault("would_send_order", False)
            if pending is not None:
                result = comments._attach_comment_preview_metadata(
                    result,
                    request.comment,
                    default=(
                        "mtdata modify pending order"
                        if pending
                        else "mtdata modify position"
                    ),
                )
        if correlation_id and str(result.get("error") or "").strip():
            result = normalize_error_payload(
                result,
                default_code="trade_modify_error",
                request_id=correlation_id,
                operation="trade_modify",
            )
        result = _attach_live_guardrail_status(result, dry_run=request.dry_run)
        result = _annotate_idempotency_scope(result, idempotency_key, idempotency_store)
        result = _attach_trade_correlation(result, correlation_id=correlation_id)
        if not idempotency_consumed:
            if _record_or_release_idempotency(
                idempotency_store,
                idempotency_key,
                result,
                request_signature=idempotency_signature,
            ):
                idempotency_consumed = True
        _log_trade_correlation(operation="trade_modify", result=result)
        log_operation_finish(
            logger,
            operation="trade_modify",
            started_at=started_at,
            success=infer_result_success(result),
            correlation_id=correlation_id,
            ticket=request.ticket,
            pending=pending,
            dry_run=request.dry_run,
        )
        return result

    if "comment" in request.model_fields_set and request.comment is not None:
        return _finish(
            {
                "success": False,
                "error_code": "unsupported_field",
                "error": (
                    "trade_modify cannot change broker comments on positions or "
                    "pending orders."
                ),
                "remediation": (
                    "Set the comment when placing or closing the order. MT5 does "
                    "not support retagging an existing ticket via trade_modify."
                ),
                "ticket": request.ticket,
                "unsupported_fields": ["comment"],
            }
        )
    mutable_fields = {
        "price",
        "stop_limit_price",
        "stop_loss",
        "take_profit",
        "expiration",
    }
    if not (request.model_fields_set & mutable_fields):
        return _finish(
            {
                "success": False,
                "error_code": "no_modification_fields",
                "error": (
                    "trade_modify requires at least one field to change: price, "
                    "stop_limit_price, stop_loss, take_profit, or expiration."
                ),
                "remediation": (
                    "Provide at least one modification field. Price and expiration "
                    "apply only to pending orders."
                ),
                "ticket": request.ticket,
            }
        )

    duplicate_result, idempotency_reserved = _begin_trade_idempotency(
        idempotency_store=idempotency_store,
        key=idempotency_key,
        request_signature=idempotency_signature,
    )
    if duplicate_result is not None:
        idempotency_consumed = True
        return _finish(duplicate_result)

    try:
        price_val = request.price
        try:
            _, expiration_specified = normalize_pending_expiration(request.expiration)
        except (TypeError, ValueError) as ex:
            return _finish(
                _invalid_pending_expiration_payload(
                    ex,
                    dry_run=bool(request.dry_run),
                )
            )

        if (
            price_val is not None
            or request.stop_limit_price is not None
            or expiration_specified
        ):
            pending_kwargs = {
                "ticket": request.ticket,
                "price": price_val,
                "stop_limit_price": request.stop_limit_price,
                "stop_loss": request.stop_loss,
                "take_profit": request.take_profit,
                "expiration": request.expiration,
                "comment": request.comment,
            }
            if request.dry_run:
                pending_kwargs["dry_run"] = True
            result = modify_pending_order(
                **pending_kwargs,
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

        position_kwargs = {
            "ticket": request.ticket,
            "stop_loss": request.stop_loss,
            "take_profit": request.take_profit,
            "comment": request.comment,
        }
        if request.dry_run:
            position_kwargs["dry_run"] = True
        position_result = modify_position(
            **position_kwargs,
        )
        if position_result.get("success"):
            return _finish(position_result, pending=False)
        if position_result.get("error") == f"Position {request.ticket} not found":
            pending_kwargs = {
                "ticket": request.ticket,
                "price": None,
                "stop_limit_price": request.stop_limit_price,
                "stop_loss": request.stop_loss,
                "take_profit": request.take_profit,
                "expiration": None,
                "comment": request.comment,
            }
            if request.dry_run:
                pending_kwargs["dry_run"] = True
            pending_result = modify_pending_order(
                **pending_kwargs,
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
