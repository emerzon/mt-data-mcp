"""Trading position resolution and read-only views."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from ...utils.mt5 import _mt5_epoch_to_utc
from ...utils.utils import (
    _format_time_minimal,
    _format_time_minimal_local,
    _normalize_limit,
    _use_client_tz,
)
from .._mcp_instance import mcp
from ..execution_logging import run_logged_operation
from ..output_contract import resolve_output_contract
from . import comments, validation
from .gateway import create_trading_gateway
from .requests import TradeGetOpenRequest, TradeGetPendingRequest
from .use_cases import run_trade_get_open, run_trade_get_pending

logger = logging.getLogger(__name__)


def _position_sort_key(position: Any) -> float:
    """Prefer the most recently updated position when multiple candidates exist."""
    for field in ("time_update_msc", "time_msc", "time_update", "time"):
        try:
            value = float(getattr(position, field, 0.0) or 0.0)
            if math.isfinite(value):
                return value
        except Exception:
            continue
    return 0.0


def _order_sort_key(order: Any) -> float:
    """Prefer the most recently updated pending order when multiple candidates exist."""
    for field in ("time_done_msc", "time_setup_msc", "time_done", "time_setup", "time"):
        try:
            value = float(getattr(order, field, 0.0) or 0.0)
            if math.isfinite(value):
                return value
        except Exception:
            continue
    return 0.0


def _position_side_matches(position: Any, side: Optional[str], mt5: Any) -> bool:
    if side not in {"BUY", "SELL"}:
        return True
    return validation._resolve_position_side(position, mt5) == side


def _position_matches_required_filters(
    position: Any,
    *,
    symbol: Optional[str],
    side: Optional[str],
    mt5: Any,
) -> bool:
    if symbol is not None:
        position_symbol = str(getattr(position, "symbol", "")).upper()
        if position_symbol != str(symbol).upper():
            return False
    if side in {"BUY", "SELL"}:
        raw_type = getattr(position, "type", None)
        if isinstance(raw_type, (int, float, str)) and not isinstance(raw_type, bool):
            resolved_side = validation._resolve_position_side(position, mt5)
            if resolved_side is not None and resolved_side != side:
                return False
    return True


def _position_ticket_fields(position: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for field in ("ticket", "identifier", "position_id", "position", "order", "deal"):
        ticket = validation._safe_int_ticket(getattr(position, field, None))
        if ticket is not None:
            out[field] = ticket
    return out


def _resolved_position_ticket(
    position: Any, *, fallback: Optional[int] = None
) -> Optional[int]:
    fields = _position_ticket_fields(position)
    for field in ("ticket", "identifier", "position_id", "position", "order", "deal"):
        ticket = fields.get(field)
        if ticket is not None:
            return ticket
    return validation._safe_int_ticket(fallback)


def _pending_order_ticket_fields(order: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for field in ("ticket", "identifier", "position_id", "position", "order", "deal"):
        ticket = validation._safe_int_ticket(getattr(order, field, None))
        if ticket is not None:
            out[field] = ticket
    return out


def _resolved_pending_order_ticket(
    order: Any, *, fallback: Optional[int] = None
) -> Optional[int]:
    fields = _pending_order_ticket_fields(order)
    for field in ("ticket", "identifier", "position_id", "position", "order", "deal"):
        ticket = fields.get(field)
        if ticket is not None:
            return ticket
    return validation._safe_int_ticket(fallback)


def _trade_read_scope(request: Any) -> str:
    if getattr(request, "ticket", None) is not None:
        return "ticket"
    for field in ("position_ticket", "deal_ticket", "order_ticket"):
        if getattr(request, field, None) is not None:
            return "ticket"
    if getattr(request, "symbol", None) is not None:
        return "symbol"
    return "all"


def _preserve_trade_error_metadata(out: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if key in {"error", "items", "success"}:
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if key in out:
            continue
        out[key] = value


def _include_trade_read_request_metadata(request: Any) -> bool:
    contract = resolve_output_contract(request, default_detail="full")
    return contract.shape_detail == "full"


def _normalize_trade_read_output(
    rows: Any,
    *,
    request: Any,
    kind: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "success": True,
        "kind": kind,
        "scope": _trade_read_scope(request),
        "count": 0,
        "items": [],
    }
    if _include_trade_read_request_metadata(request):
        symbol = getattr(request, "symbol", None)
        ticket = getattr(request, "ticket", None)
        limit = getattr(request, "limit", None)
        if symbol is not None:
            out["symbol"] = symbol
        if ticket is not None:
            out["ticket"] = ticket
        if limit is not None:
            out["limit"] = limit

    if isinstance(rows, dict):
        error_text = str(rows.get("error", "")).strip()
        if error_text:
            out["success"] = False
            out["error"] = error_text
            _preserve_trade_error_metadata(out, rows)
            return out

        items = rows.get("items")
        if isinstance(items, list):
            out["items"] = items
            out["count"] = len(items)
            if rows.get("no_action"):
                out["no_action"] = True
            message_text = str(rows.get("message", "")).strip()
            if message_text:
                out["message"] = message_text
            return out

        message_text = str(rows.get("message", "")).strip()
        if message_text:
            out["message"] = message_text
            out["no_action"] = True
            return out

    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        first = rows[0]
        error_text = str(first.get("error", "")).strip()
        if error_text:
            out["success"] = False
            out["error"] = error_text
            _preserve_trade_error_metadata(out, first)
            return out
        message_text = str(first.get("message", "")).strip()
        if message_text:
            out["message"] = message_text
            out["no_action"] = True
            return out

    if not isinstance(rows, list):
        out["success"] = False
        out["error"] = f"Unexpected {kind} payload type: {type(rows).__name__}"
        return out

    out["items"] = rows
    out["count"] = len(rows)
    return out


def normalize_trade_history_output(
    rows: Any,
    *,
    request: Any,
) -> Dict[str, Any]:
    """Normalize trade history into the standard trade read envelope."""
    out = _normalize_trade_read_output(rows, request=request, kind="trade_history")
    if _include_trade_read_request_metadata(request):
        history_kind = getattr(request, "history_kind", None)
        if history_kind is not None:
            out["history_kind"] = history_kind
        for field in (
            "start",
            "end",
            "side",
            "minutes_back",
            "position_ticket",
            "deal_ticket",
            "order_ticket",
        ):
            value = getattr(request, field, None)
            if value is not None:
                if field == "side":
                    normalized_side, _ = validation._normalize_trade_side_filter(value)
                    out[field] = normalized_side or value
                else:
                    out[field] = value
    elif getattr(request, "history_kind", None) is not None:
        out["history_kind"] = getattr(request, "history_kind")
    return out


def _select_position_candidate(
    rows: List[Any],
    *,
    symbol: Optional[str],
    side: Optional[str],
    volume: Optional[float],
    magic: Optional[int] = None,
    ticket_candidates: Optional[List[int]] = None,
    mt5: Any,
) -> Optional[Any]:
    if not rows:
        return None
    volume_tol = 1e-9
    if volume is not None and symbol:
        try:
            symbol_info = mt5.symbol_info(str(symbol))
        except Exception:
            symbol_info = None
        try:
            volume_step = float(getattr(symbol_info, "volume_step", float("nan")))
        except Exception:
            volume_step = float("nan")
        if math.isfinite(volume_step) and volume_step > 0.0:
            volume_tol = max(volume_tol, volume_step / 2.0)
    candidates = list(rows)
    # Prefer positions matching known tickets when multiple are available
    if ticket_candidates and len(candidates) > 1:
        ticket_filtered = [
            pos
            for pos in candidates
            if any(
                v in ticket_candidates for v in _position_ticket_fields(pos).values()
            )
        ]
        if ticket_filtered:
            candidates = ticket_filtered
    required_filtered = [
        pos
        for pos in candidates
        if _position_matches_required_filters(
            pos,
            symbol=symbol,
            side=side,
            mt5=mt5,
        )
    ]
    if symbol is not None or side in {"BUY", "SELL"}:
        candidates = required_filtered
    if magic is not None:
        magic_filtered = [
            pos
            for pos in candidates
            if validation._safe_int_ticket(getattr(pos, "magic", None)) == magic
        ]
        if magic_filtered:
            candidates = magic_filtered
    if volume is not None:
        volume_filtered: List[Any] = []
        for pos in candidates:
            try:
                if math.isclose(
                    float(getattr(pos, "volume", float("nan"))),
                    float(volume),
                    abs_tol=volume_tol,
                ):
                    volume_filtered.append(pos)
            except Exception:
                continue
        if volume_filtered:
            candidates = volume_filtered
    candidates.sort(key=_position_sort_key, reverse=True)
    return candidates[0] if candidates else None


def _select_pending_order_candidate(
    rows: List[Any],
    *,
    symbol: Optional[str],
) -> Optional[Any]:
    if not rows:
        return None
    candidates = list(rows)
    if symbol:
        symbol_upper = str(symbol).upper()
        symbol_filtered = [
            order
            for order in candidates
            if str(getattr(order, "symbol", "")).upper() == symbol_upper
        ]
        if symbol_filtered:
            candidates = symbol_filtered
    candidates.sort(key=_order_sort_key, reverse=True)
    return candidates[0] if candidates else None


def _resolve_open_position(
    mt5: Any,
    *,
    ticket_candidates: Optional[List[int]] = None,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    volume: Optional[float] = None,
    magic: Optional[int] = None,
    require_exact_ticket_match: bool = False,
    allow_alternate_ticket_match: bool = False,
) -> Tuple[Optional[Any], Optional[int], Dict[str, Any]]:
    """Resolve an open position robustly across ticket/identifier mismatches."""
    candidate_ids: List[int] = []
    for raw in list(ticket_candidates or []):
        ticket = validation._safe_int_ticket(raw)
        if ticket is not None and ticket not in candidate_ids:
            candidate_ids.append(ticket)

    for candidate in candidate_ids:
        try:
            rows = mt5.positions_get(ticket=int(candidate))
        except Exception:
            rows = None
        rows_list = list(rows) if rows else []
        picked = _select_position_candidate(
            rows_list,
            symbol=symbol,
            side=side,
            volume=volume,
            magic=magic,
            mt5=mt5,
        )
        if picked is not None:
            direct_ticket = validation._safe_int_ticket(getattr(picked, "ticket", None))
            if require_exact_ticket_match and direct_ticket != candidate:
                continue
            resolved = (
                direct_ticket
                if require_exact_ticket_match
                else _resolved_position_ticket(picked, fallback=candidate)
            )
            diag: Dict[str, Any] = {
                "method": "positions_get(ticket)",
                "candidate": candidate,
            }
            if magic is not None:
                diag["magic_filter"] = magic
            if require_exact_ticket_match:
                diag["exact_ticket_required"] = True
            return picked, resolved, diag

    try:
        rows_fallback = (
            mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        )
    except Exception:
        rows_fallback = None
    rows_list = list(rows_fallback) if rows_fallback else []
    if not rows_list:
        return (
            None,
            None,
            {
                "method": "positions_get",
                "candidate_ids": candidate_ids,
                "matched": False,
            },
        )

    exact_matches: List[Tuple[Any, str, int]] = []
    if candidate_ids:
        for pos in rows_list:
            for field, value in _position_ticket_fields(pos).items():
                if (
                    require_exact_ticket_match
                    and not allow_alternate_ticket_match
                    and field != "ticket"
                ):
                    continue
                if value in candidate_ids:
                    exact_matches.append((pos, field, value))
        exact_matches = [
            (pos, field, value)
            for pos, field, value in exact_matches
            if _position_matches_required_filters(
                pos,
                symbol=symbol,
                side=side,
                mt5=mt5,
            )
        ]
        if exact_matches:
            exact_matches.sort(
                key=lambda item: _position_sort_key(item[0]), reverse=True
            )
            pos, field, matched_value = exact_matches[0]
            resolved = _resolved_position_ticket(pos, fallback=matched_value)
            return (
                pos,
                resolved,
                {
                    "method": "positions_get(fallback_exact)",
                    "matched_field": field,
                    "matched_value": matched_value,
                    "exact_ticket_required": require_exact_ticket_match,
                },
            )

    if candidate_ids and require_exact_ticket_match:
        return (
            None,
            None,
            {
                "method": "positions_get(fallback_heuristic)",
                "candidate_ids": candidate_ids,
                "matched": False,
                "exact_ticket_required": True,
            },
        )

    picked = _select_position_candidate(
        rows_list,
        symbol=symbol,
        side=side,
        volume=volume,
        magic=magic,
        ticket_candidates=candidate_ids or None,
        mt5=mt5,
    )
    if picked is None:
        return (
            None,
            None,
            {
                "method": "positions_get(fallback_heuristic)",
                "candidate_ids": candidate_ids,
                "matched": False,
            },
        )
    resolved = _resolved_position_ticket(picked)
    diag = {"method": "positions_get(fallback_heuristic)"}
    if magic is not None:
        diag["magic_filter"] = magic
    if len(rows_list) > 1:
        diag["candidates_count"] = len(rows_list)
    return picked, resolved, diag


def _resolve_pending_order(
    mt5: Any,
    *,
    ticket_candidates: Optional[List[int]] = None,
    symbol: Optional[str] = None,
    require_exact_ticket_match: bool = False,
) -> Tuple[Optional[Any], Optional[int], Dict[str, Any]]:
    """Resolve a pending order robustly across ticket/identifier mismatches."""
    candidate_ids: List[int] = []
    for raw in list(ticket_candidates or []):
        ticket = validation._safe_int_ticket(raw)
        if ticket is not None and ticket not in candidate_ids:
            candidate_ids.append(ticket)

    for candidate in candidate_ids:
        try:
            rows = mt5.orders_get(ticket=int(candidate))
        except Exception:
            rows = None
        rows_list = list(rows) if rows else []
        picked = _select_pending_order_candidate(rows_list, symbol=symbol)
        if picked is not None:
            direct_ticket = validation._safe_int_ticket(getattr(picked, "ticket", None))
            if require_exact_ticket_match and direct_ticket != candidate:
                continue
            resolved = (
                direct_ticket
                if require_exact_ticket_match
                else _resolved_pending_order_ticket(picked, fallback=candidate)
            )
            return (
                picked,
                resolved,
                {
                    "method": "orders_get(ticket)",
                    "candidate": candidate,
                    "exact_ticket_required": require_exact_ticket_match,
                },
            )

    try:
        rows_fallback = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
    except Exception:
        rows_fallback = None
    rows_list = list(rows_fallback) if rows_fallback else []
    if not rows_list:
        return (
            None,
            None,
            {"method": "orders_get", "candidate_ids": candidate_ids, "matched": False},
        )

    exact_matches: List[Tuple[Any, str, int]] = []
    if candidate_ids:
        for order in rows_list:
            for field, value in _pending_order_ticket_fields(order).items():
                if require_exact_ticket_match and field != "ticket":
                    continue
                if value in candidate_ids:
                    exact_matches.append((order, field, value))
        if exact_matches:
            exact_matches.sort(key=lambda item: _order_sort_key(item[0]), reverse=True)
            order, field, matched_value = exact_matches[0]
            resolved = _resolved_pending_order_ticket(order, fallback=matched_value)
            return (
                order,
                resolved,
                {
                    "method": "orders_get(fallback_exact)",
                    "matched_field": field,
                    "matched_value": matched_value,
                    "exact_ticket_required": require_exact_ticket_match,
                },
            )

    if candidate_ids and require_exact_ticket_match:
        return (
            None,
            None,
            {
                "method": "orders_get(fallback_heuristic)",
                "candidate_ids": candidate_ids,
                "matched": False,
                "exact_ticket_required": True,
            },
        )

    picked = _select_pending_order_candidate(rows_list, symbol=symbol)
    if picked is None:
        return (
            None,
            None,
            {
                "method": "orders_get(fallback_heuristic)",
                "candidate_ids": candidate_ids,
                "matched": False,
            },
        )
    resolved = _resolved_pending_order_ticket(picked)
    return picked, resolved, {"method": "orders_get(fallback_heuristic)"}


@mcp.tool()
def trade_get_open(
    request: TradeGetOpenRequest,
) -> Dict[str, Any]:
    """Get open positions. Use detail="compact" to omit echoed request metadata."""
    return run_logged_operation(
        logger,
        operation="trade_get_open",
        symbol=request.symbol,
        limit=request.limit,
        func=lambda: _normalize_trade_read_output(
            run_trade_get_open(
                request,
                gateway=create_trading_gateway(),
                use_client_tz=_use_client_tz,
                format_time_minimal=_format_time_minimal,
                format_time_minimal_local=_format_time_minimal_local,
                mt5_epoch_to_utc=_mt5_epoch_to_utc,
                normalize_limit=_normalize_limit,
                comment_row_metadata=comments._comment_row_metadata,
            ),
            request=request,
            kind="open_positions",
        ),
    )


@mcp.tool()
def trade_get_pending(
    request: TradeGetPendingRequest,
) -> Dict[str, Any]:
    """Get pending orders. Use detail="compact" to omit echoed request metadata."""
    return run_logged_operation(
        logger,
        operation="trade_get_pending",
        symbol=request.symbol,
        limit=request.limit,
        func=lambda: _normalize_trade_read_output(
            run_trade_get_pending(
                request,
                gateway=create_trading_gateway(),
                use_client_tz=_use_client_tz,
                format_time_minimal=_format_time_minimal,
                format_time_minimal_local=_format_time_minimal_local,
                mt5_epoch_to_utc=_mt5_epoch_to_utc,
                normalize_limit=_normalize_limit,
                comment_row_metadata=comments._comment_row_metadata,
            ),
            request=request,
            kind="pending_orders",
        ),
    )
