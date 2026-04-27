"""Trading position resolution and read-only views."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

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


def _utc_epoch_identity(value: Any) -> float:
    return float(value)


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


def _mark_trade_read_empty(out: Dict[str, Any], message: Optional[str] = None) -> None:
    out["empty"] = True
    out["no_action"] = True


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
            message_text = str(rows.get("message", "")).strip()
            if message_text:
                out["message"] = message_text
            if len(items) == 0:
                _mark_trade_read_empty(out, message_text or None)
            return _compact_trade_read_output(out, request=request)

        message_text = str(rows.get("message", "")).strip()
        if message_text:
            out["message"] = message_text
            _mark_trade_read_empty(out, message_text)
            return _compact_trade_read_output(out, request=request)

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
            _mark_trade_read_empty(out, message_text)
            return _compact_trade_read_output(out, request=request)

    if not isinstance(rows, list):
        out["success"] = False
        out["error"] = f"Unexpected {kind} payload type: {type(rows).__name__}"
        return out

    out["items"] = rows
    out["count"] = len(rows)
    if len(rows) == 0:
        _mark_trade_read_empty(out)
    return _compact_trade_read_output(out, request=request)


def _compact_trade_read_output(out: Dict[str, Any], *, request: Any) -> Dict[str, Any]:
    if (
        out.get("kind") == "trade_history"
        or _include_trade_read_request_metadata(request)
        or not out.get("success", False)
    ):
        return out
    if int(out.get("count") or 0) == 0:
        return {"success": True, "count": 0}
    compact = dict(out)
    for key in ("kind", "scope", "empty", "no_action"):
        compact.pop(key, None)
    return compact


def _first_present(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _compact_non_empty_mapping(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in row.items()
        if value is not None and not (isinstance(value, str) and not value.strip())
    }


_TRADE_MONEY_FIELDS = {"profit", "commission", "swap", "fee"}


def _round_trade_money_value(value: Any) -> Any:
    try:
        numeric = float(value)
    except Exception:
        return value
    if not math.isfinite(numeric):
        return value
    return float(round(numeric, 2))


def _round_trade_money_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        key_text = str(key)
        if key_text in _TRADE_MONEY_FIELDS:
            out[key] = _round_trade_money_value(value)
        elif isinstance(value, dict):
            out[key] = _round_trade_money_fields(value)
        else:
            out[key] = value
    return out


def _normalize_trade_history_row(
    row: Dict[str, Any],
    *,
    history_kind: Optional[str],
) -> Dict[str, Any]:
    row = _round_trade_money_fields(row)
    item_kind = "order" if history_kind == "orders" else "deal"
    if item_kind == "order":
        timestamp = _first_present(row, "time_done", "time_setup", "time")
        created_at = _first_present(row, "time_setup", "time")
        state = _first_present(row, "state_label", "state")
        price = _first_present(row, "price_current", "price_open", "price")
        native_key = "order_details"
    else:
        timestamp = _first_present(row, "time", "time_msc")
        created_at = timestamp
        state = _first_present(row, "entry_label", "entry", "type_label", "type")
        price = _first_present(row, "price")
        native_key = "deal_details"

    normalized: Dict[str, Any] = {
        "kind": item_kind,
        "ticket": _first_present(row, "ticket", "order", "deal"),
        "symbol": row.get("symbol"),
        "timestamp": timestamp,
        "created_at": created_at,
        "volume": _first_present(row, "volume", "volume_initial", "volume_current"),
        "price": price,
        "state": state,
        native_key: _compact_non_empty_mapping(row),
    }
    return {
        key: value
        for key, value in normalized.items()
        if value is not None and value != {}
    }


def _normalize_trade_history_items(
    items: List[Any],
    *,
    history_kind: Optional[str],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(
                _normalize_trade_history_row(item, history_kind=history_kind)
            )
    return normalized


def _trade_history_request_echo(request: Any, *, history_kind: Any) -> Dict[str, Any]:
    echo: Dict[str, Any] = {}
    if history_kind is not None:
        echo["history_kind"] = history_kind
    column_style = getattr(request, "column_style", None)
    if column_style is not None:
        echo["column_style"] = column_style
    for field in (
        "start",
        "end",
        "side",
        "minutes_back",
        "position_ticket",
        "deal_ticket",
        "order_ticket",
        "symbol",
        "limit",
    ):
        value = getattr(request, field, None)
        if value is None:
            continue
        if field == "side":
            normalized_side, _ = validation._normalize_trade_side_filter(value)
            echo[field] = normalized_side or value
        else:
            echo[field] = value
    return echo


def _trade_history_humanized_key(key: str) -> str:
    overrides = {
        "sl": "SL",
        "tp": "TP",
        "time": "Time",
        "time_setup": "Setup Time",
        "time_done": "Done Time",
        "time_msc": "Time Msc",
        "ticket": "Ticket",
        "order": "Order",
        "deal": "Deal",
        "position_id": "Position ID",
        "position_by_id": "Position By ID",
        "symbol": "Symbol",
        "type": "Type",
        "type_code": "Type Code",
        "entry": "Entry",
        "entry_code": "Entry Code",
        "reason": "Reason",
        "reason_code": "Reason Code",
        "state": "State",
        "state_code": "State Code",
        "volume": "Volume",
        "volume_initial": "Initial Volume",
        "volume_current": "Current Volume",
        "price": "Price",
        "price_open": "Open Price",
        "price_current": "Current Price",
        "profit": "Profit",
        "commission": "Commission",
        "swap": "Swap",
        "fee": "Fee",
        "comment": "Comments",
        "magic": "Magic",
        "timestamp_timezone": "Timestamp Timezone",
        "exit_trigger": "Exit Trigger",
        "exit_trigger_price": "Exit Trigger Price",
        "exit_trigger_source": "Exit Trigger Source",
    }
    return overrides.get(key, key.replace("_", " ").title())


def _style_trade_history_items(items: List[Any], *, column_style: Any) -> List[Any]:
    style = str(column_style or "snake_case").strip().lower()
    if style != "humanized":
        return items
    styled: List[Any] = []
    for item in items:
        if not isinstance(item, dict):
            styled.append(item)
            continue
        styled.append(
            {_trade_history_humanized_key(str(key)): value for key, value in item.items()}
        )
    return styled


def normalize_trade_history_output(
    rows: Any,
    *,
    request: Any,
) -> Dict[str, Any]:
    """Normalize trade history into the standard trade read envelope."""
    out = _normalize_trade_read_output(rows, request=request, kind="trade_history")
    history_kind = getattr(request, "history_kind", None)
    include_request_metadata = _include_trade_read_request_metadata(request)
    if out.get("success") is True and isinstance(out.get("items"), list):
        raw_items = list(out["items"])
        if include_request_metadata:
            out["items"] = _normalize_trade_history_items(
                raw_items,
                history_kind=history_kind,
            )
            out["item_schema"] = "normalized_trade_history.v1"
        else:
            out["items"] = _style_trade_history_items(
                [
                    _round_trade_money_fields(item) if isinstance(item, dict) else item
                    for item in raw_items
                ],
                column_style=getattr(request, "column_style", "snake_case"),
            )
    if include_request_metadata:
        for key in ("symbol", "ticket", "limit"):
            out.pop(key, None)
        request_echo = _trade_history_request_echo(request, history_kind=history_kind)
        if request_echo:
            out["request_echo"] = request_echo
    elif history_kind is not None:
        out["history_kind"] = history_kind
    if out.get("success") is True:
        timezone_label = "UTC"
        items = out.get("items")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("timestamp_timezone"):
                    timezone_label = str(item["timestamp_timezone"])
                    break
        out.setdefault("timezone", timezone_label)
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
                mt5_epoch_to_utc=_utc_epoch_identity,
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
                mt5_epoch_to_utc=_utc_epoch_identity,
                normalize_limit=_normalize_limit,
                comment_row_metadata=comments._comment_row_metadata,
            ),
            request=request,
            kind="pending_orders",
        ),
    )
