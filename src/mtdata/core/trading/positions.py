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
    expected_buy = getattr(mt5, "POSITION_TYPE_BUY", getattr(mt5, "ORDER_TYPE_BUY", None))
    expected_sell = getattr(mt5, "POSITION_TYPE_SELL", getattr(mt5, "ORDER_TYPE_SELL", None))
    expected = expected_buy if side == "BUY" else expected_sell
    if expected is None:
        return True
    try:
        return int(getattr(position, "type", -99999)) == int(expected)
    except Exception:
        return False


def _position_ticket_fields(position: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for field in ("ticket", "identifier", "position_id", "position", "order", "deal"):
        ticket = validation._safe_int_ticket(getattr(position, field, None))
        if ticket is not None:
            out[field] = ticket
    return out


def _resolved_position_ticket(position: Any, *, fallback: Optional[int] = None) -> Optional[int]:
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


def _resolved_pending_order_ticket(order: Any, *, fallback: Optional[int] = None) -> Optional[int]:
    fields = _pending_order_ticket_fields(order)
    for field in ("ticket", "identifier", "position_id", "position", "order", "deal"):
        ticket = fields.get(field)
        if ticket is not None:
            return ticket
    return validation._safe_int_ticket(fallback)


def _trade_read_scope(request: Any) -> str:
    if getattr(request, "ticket", None) is not None:
        return "ticket"
    if getattr(request, "symbol", None) is not None:
        return "symbol"
    return "all"


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
    symbol = getattr(request, "symbol", None)
    ticket = getattr(request, "ticket", None)
    limit = getattr(request, "limit", None)
    if symbol is not None:
        out["symbol"] = symbol
    if ticket is not None:
        out["ticket"] = ticket
    if limit is not None:
        out["limit"] = limit

    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        first = rows[0]
        error_text = str(first.get("error", "")).strip()
        if error_text:
            out["success"] = False
            out["error"] = error_text
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


def _select_position_candidate(
    rows: List[Any],
    *,
    symbol: Optional[str],
    side: Optional[str],
    volume: Optional[float],
    magic: Optional[int] = None,
    mt5: Any,
) -> Optional[Any]:
    if not rows:
        return None
    candidates = list(rows)
    if symbol:
        symbol_upper = str(symbol).upper()
        symbol_filtered = [pos for pos in candidates if str(getattr(pos, "symbol", "")).upper() == symbol_upper]
        if symbol_filtered:
            candidates = symbol_filtered
    side_filtered = [pos for pos in candidates if _position_side_matches(pos, side, mt5)]
    if side_filtered:
        candidates = side_filtered
    if magic is not None:
        magic_filtered = [
            pos for pos in candidates
            if validation._safe_int_ticket(getattr(pos, "magic", None)) == magic
        ]
        if magic_filtered:
            candidates = magic_filtered
    if volume is not None:
        volume_filtered: List[Any] = []
        for pos in candidates:
            try:
                if abs(float(getattr(pos, "volume", float("nan"))) - float(volume)) <= 1e-9:
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
        symbol_filtered = [order for order in candidates if str(getattr(order, "symbol", "")).upper() == symbol_upper]
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
            resolved = _resolved_position_ticket(picked, fallback=candidate)
            diag: Dict[str, Any] = {"method": "positions_get(ticket)", "candidate": candidate}
            if magic is not None:
                diag["magic_filter"] = magic
            return picked, resolved, diag

    try:
        rows_fallback = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    except Exception:
        rows_fallback = None
    rows_list = list(rows_fallback) if rows_fallback else []
    if not rows_list:
        return None, None, {"method": "positions_get", "candidate_ids": candidate_ids, "matched": False}

    exact_matches: List[Tuple[Any, str, int]] = []
    if candidate_ids:
        for pos in rows_list:
            for field, value in _position_ticket_fields(pos).items():
                if value in candidate_ids:
                    exact_matches.append((pos, field, value))
        if exact_matches:
            exact_matches.sort(key=lambda item: _position_sort_key(item[0]), reverse=True)
            pos, field, matched_value = exact_matches[0]
            resolved = _resolved_position_ticket(pos, fallback=matched_value)
            return pos, resolved, {
                "method": "positions_get(fallback_exact)",
                "matched_field": field,
                "matched_value": matched_value,
            }

    picked = _select_position_candidate(
        rows_list,
        symbol=symbol,
        side=side,
        volume=volume,
        magic=magic,
        mt5=mt5,
    )
    if picked is None:
        return None, None, {"method": "positions_get(fallback_heuristic)", "candidate_ids": candidate_ids, "matched": False}
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
            resolved = _resolved_pending_order_ticket(picked, fallback=candidate)
            return picked, resolved, {"method": "orders_get(ticket)", "candidate": candidate}

    try:
        rows_fallback = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
    except Exception:
        rows_fallback = None
    rows_list = list(rows_fallback) if rows_fallback else []
    if not rows_list:
        return None, None, {"method": "orders_get", "candidate_ids": candidate_ids, "matched": False}

    exact_matches: List[Tuple[Any, str, int]] = []
    if candidate_ids:
        for order in rows_list:
            for field, value in _pending_order_ticket_fields(order).items():
                if value in candidate_ids:
                    exact_matches.append((order, field, value))
        if exact_matches:
            exact_matches.sort(key=lambda item: _order_sort_key(item[0]), reverse=True)
            order, field, matched_value = exact_matches[0]
            resolved = _resolved_pending_order_ticket(order, fallback=matched_value)
            return order, resolved, {
                "method": "orders_get(fallback_exact)",
                "matched_field": field,
                "matched_value": matched_value,
            }

    picked = _select_pending_order_candidate(rows_list, symbol=symbol)
    if picked is None:
        return None, None, {"method": "orders_get(fallback_heuristic)", "candidate_ids": candidate_ids, "matched": False}
    resolved = _resolved_pending_order_ticket(picked)
    return picked, resolved, {"method": "orders_get(fallback_heuristic)"}


@mcp.tool()
def trade_get_open(
    request: TradeGetOpenRequest,
) -> Dict[str, Any]:
    """Get open positions."""
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
    """Get pending orders (open orders)."""
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
