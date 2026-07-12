"""Trading position resolution and read-only views."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from ...utils.time import (
    _format_datetime_second_explicit,
    _format_time_minimal,
)
from ...utils.utils import (
    _normalize_limit,
    _parse_start_datetime,
)
from .._mcp_instance import mcp
from ..execution_logging import run_logged_operation
from ..output_contract import resolve_output_contract
from . import comments, validation
from .gateway import create_trading_gateway
from .requests import TradeGetOpenRequest, TradeGetPendingRequest
from .use_cases import (
    _DEFAULT_TRADE_HISTORY_LOOKBACK_DAYS,
    run_trade_get_open,
    run_trade_get_pending,
)

logger = logging.getLogger(__name__)

_TRADE_VOLUME_UNITS = {
    "volume": "lots",
    "volume_initial": "lots",
    "volume_current": "lots",
    "requested_volume": "lots",
    "remaining_volume": "lots",
    "Volume": "lots",
    "Initial Volume": "lots",
    "Current Volume": "lots",
}


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


_MT5_TICKET_FIELDS = (
    "ticket",
    "identifier",
    "position_id",
    "position",
    "order",
    "deal",
)


def _ticket_fields(obj: Any) -> Dict[str, int]:
    """Return valid values from the standard MT5 ticket fields."""
    out: Dict[str, int] = {}
    for field in _MT5_TICKET_FIELDS:
        ticket = validation._safe_int_ticket(getattr(obj, field, None))
        if ticket is not None:
            out[field] = ticket
    return out


def _resolved_ticket(obj: Any, *, fallback: Optional[int] = None) -> Optional[int]:
    fields = _ticket_fields(obj)
    for field in _MT5_TICKET_FIELDS:
        ticket = fields.get(field)
        if ticket is not None:
            return ticket
    return validation._safe_int_ticket(fallback)


def _trade_read_scope(request: Any) -> str:
    has_temporal_filter = any(
        getattr(request, field, None) is not None
        for field in ("start", "end", "minutes_back")
    )
    if getattr(request, "ticket", None) is not None:
        return "ticket"
    for field in ("position_ticket", "deal_ticket", "order_ticket"):
        if getattr(request, field, None) is not None:
            return "ticket"
    if getattr(request, "symbol", None) is not None:
        if has_temporal_filter:
            return "symbol_date_range"
        return "symbol"
    if (
        getattr(request, "start", None) is not None
        or getattr(request, "end", None) is not None
    ):
        return "date_range"
    if getattr(request, "minutes_back", None) is not None:
        return "lookback"
    for field in ("side", "magic", "order_type"):
        if getattr(request, field, None) is not None:
            return "filtered"
    if bool(getattr(request, "profit_only", False)) or bool(
        getattr(request, "loss_only", False)
    ):
        return "filtered"
    return "all"


def _trade_history_filters_applied(request: Any) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    for field in (
        "start",
        "end",
        "minutes_back",
        "symbol",
        "side",
        "position_ticket",
        "deal_ticket",
        "order_ticket",
    ):
        value = getattr(request, field, None)
        if value is not None:
            filters[field] = value
    return filters


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


def _trade_read_timezone_label(items: Any) -> Optional[str]:
    if not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, dict) and item.get("timezone"):
            return str(item["timezone"])
    return None


def _strip_active_trade_row_metadata(items: List[Any], *, kind: str) -> List[Any]:
    if kind not in {"open_positions", "pending_orders"}:
        return items
    out: List[Any] = []
    for item in items:
        if isinstance(item, dict):
            row = dict(item)
            row.pop("timezone", None)
            out.append(row)
        else:
            out.append(item)
    return out


def _attach_trade_volume_units(out: Dict[str, Any]) -> None:
    items = out.get("items")
    if not isinstance(items, list):
        return
    seen_fields = {
        str(key)
        for item in items
        if isinstance(item, dict)
        for key, value in item.items()
        if value is not None
    }
    units = {
        key: unit
        for key, unit in _TRADE_VOLUME_UNITS.items()
        if key in seen_fields
    }
    if units:
        out["units"] = units


def _gateway_account_currency(gateway: Any) -> Optional[str]:
    account_info = getattr(gateway, "account_info", None)
    if not callable(account_info):
        return None
    try:
        account = account_info()
    except Exception:
        return None
    currency = str(getattr(account, "currency", "") or "").strip()
    return currency or None


def _normalize_trade_read_output(
    rows: Any,
    *,
    request: Any,
    kind: str,
    account_currency: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "success": True,
        "kind": kind,
        "scope": _trade_read_scope(request),
        "count": 0,
        "items": [],
    }
    if kind in ("open_positions", "pending_orders"):
        out["as_of"] = (
            datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        )
    if kind == "trade_history":
        filters_applied = _trade_history_filters_applied(request)
        if filters_applied:
            out["filters_applied"] = filters_applied
    if account_currency:
        out["currency"] = account_currency
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
            normalized_items = [
                _round_trade_money_fields(item) if isinstance(item, dict) else item
                for item in items
            ]
            timezone_label = _trade_read_timezone_label(normalized_items)
            out["items"] = _strip_active_trade_row_metadata(
                normalized_items,
                kind=kind,
            )
            out["count"] = len(items)
            if timezone_label:
                out["timezone"] = timezone_label
            _attach_trade_volume_units(out)
            message_text = str(rows.get("message", "")).strip()
            if message_text:
                out["message"] = message_text
            for key in (
                "total_count",
                "offset",
                "limit",
                "has_more",
                "truncated",
                "more_available",
            ):
                if key in rows:
                    out[key] = rows.get(key)
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

    normalized_items = [
        _round_trade_money_fields(row) if isinstance(row, dict) else row for row in rows
    ]
    timezone_label = _trade_read_timezone_label(normalized_items)
    out["items"] = _strip_active_trade_row_metadata(
        normalized_items,
        kind=kind,
    )
    out["count"] = len(rows)
    if timezone_label:
        out["timezone"] = timezone_label
    _attach_trade_volume_units(out)
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
        kind = str(out.get("kind") or "")
        default_message = {
            "open_positions": "No open positions matched the request.",
            "pending_orders": "No pending orders matched the request.",
        }.get(kind, "No rows matched the request.")
        compact = {
            "success": True,
            "kind": out.get("kind"),
            "count": 0,
            "items": [],
            "empty": True,
        }
        if out.get("as_of"):
            compact["as_of"] = out.get("as_of")
        compact["message"] = out.get("message") or default_message
        compact["hint"] = (
            "Normal when flat; relax symbol/ticket filters or check trade_account_info."
            if kind == "open_positions"
            else "Normal when no working orders; relax symbol/ticket filters or check trade_account_info."
        )
        return compact
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
_TRADE_PRICE_FIELDS = {
    "price",
    "entry_price",
    "trigger_price",
    "price_open",
    "price_current",
    "price_stoplimit",
    "sl",
    "tp",
    "exit_trigger_price",
}
_TRADE_MILLISECOND_TIME_FIELDS = {
    "time_msc",
    "time_setup_msc",
    "time_done_msc",
    "time_update_msc",
}
_TRADE_HISTORY_DIAGNOSTIC_FIELDS = {
    "comment_visible_length",
    "comment_max_length",
    "comment_may_be_truncated",
    "type_code",
    "entry_code",
    "state_code",
    "type_time_code",
    "type_filling_code",
    "external_id",
}
_TRADE_HISTORY_ROW_METADATA_FIELDS = {"timezone"}
_TRADE_HISTORY_DEAL_TOP_LEVEL_FIELDS = (
    "ticket",
    "deal_ticket",
    "order",
    "order_ticket",
    "time",
    "time_msc",
    "type",
    "type_label",
    "entry",
    "entry_label",
    "magic",
    "position_id",
    "position_by_id",
    "position_ticket",
    "reason",
    "reason_label",
    "reason_code",
    "volume",
    "price",
    "commission",
    "swap",
    "profit",
    "fee",
    "symbol",
    "comment",
    "exit_trigger",
    "exit_trigger_price",
    "exit_trigger_source",
)
_TRADE_HISTORY_ORDER_TOP_LEVEL_FIELDS = (
    "ticket",
    "order_ticket",
    "time_setup",
    "time_done",
    "time_setup_msc",
    "time_done_msc",
    "type",
    "state",
    "state_label",
    "reason",
    "reason_label",
    "reason_code",
    "magic",
    "position_id",
    "position_by_id",
    "position_ticket",
    "volume_initial",
    "volume_current",
    "price_open",
    "price_current",
    "price_stoplimit",
    "sl",
    "tp",
    "symbol",
    "comment",
)
_TRADE_HISTORY_COMPACT_DEAL_FIELDS = (
    "fill_time",
    "ticket",
    "deal_ticket",
    "order_ticket",
    "position_ticket",
    "symbol",
    "type",
    "action",
    "deal_effect",
    "position_side",
    "volume",
    "price",
    "profit",
    "commission",
    "swap",
    "fee",
    "comment",
    "comment_truncated",
    "exit_trigger",
    "exit_trigger_price",
)
_TRADE_HISTORY_COMPACT_ORDER_FIELDS = (
    "placed_time",
    "done_time",
    "ticket",
    "order_ticket",
    "position_ticket",
    "symbol",
    "type",
    "state",
    "volume_initial",
    "volume_current",
    "price_open",
    "price_current",
    "sl",
    "tp",
    "comment",
)


def _round_trade_money_value(value: Any) -> Any:
    try:
        numeric = float(value)
    except Exception:
        return value
    if not math.isfinite(numeric):
        return value
    return float(round(numeric, 2))


def _round_trade_price_value(value: Any) -> Any:
    try:
        numeric = float(value)
    except Exception:
        return value
    if not math.isfinite(numeric):
        return value
    return float(f"{numeric:.12g}")


def _normalize_trade_millisecond_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    try:
        numeric = float(value)
    except Exception:
        return value
    if not math.isfinite(numeric):
        return value
    rounded = round(numeric)
    if abs(numeric - rounded) <= 1e-6:
        return int(rounded)
    return value


def _round_trade_money_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        key_text = str(key)
        if key_text in _TRADE_MONEY_FIELDS:
            out[key] = _round_trade_money_value(value)
        elif key_text in {"sl", "tp"}:
            rounded = _round_trade_price_value(value)
            try:
                is_unset = math.isclose(float(rounded), 0.0, abs_tol=1e-12)
                out[key] = None if is_unset else rounded
            except (TypeError, ValueError):
                out[key] = rounded
        elif key_text in _TRADE_PRICE_FIELDS:
            out[key] = _round_trade_price_value(value)
        elif key_text in _TRADE_MILLISECOND_TIME_FIELDS:
            out[key] = _normalize_trade_millisecond_value(value)
        elif isinstance(value, dict):
            out[key] = _round_trade_money_fields(value)
        else:
            out[key] = value
    return out


def _trade_history_action(row: Dict[str, Any], *, history_kind: Optional[str]) -> Any:
    if history_kind == "orders":
        return None
    raw_entry = _first_present(row, "entry_label", "entry")
    entry_text = str(raw_entry or "").strip().lower().replace("_", " ")
    if not entry_text:
        return None
    if "inout" in entry_text or "in out" in entry_text:
        return "reverse"
    if "out by" in entry_text:
        return "close_by"
    if "out" in entry_text:
        return "close"
    if "in" in entry_text:
        return "open"
    return None


def _trade_history_position_side(
    row: Dict[str, Any],
    *,
    action: Optional[str],
    history_kind: Optional[str],
) -> Optional[str]:
    if history_kind == "orders":
        return None
    raw_type = _first_present(row, "type_label", "type")
    type_text = str(raw_type or "").strip().lower().replace("_", " ")
    if not type_text:
        return None
    if "buy" in type_text:
        return "short" if action in {"close", "close_by"} else "long"
    if "sell" in type_text:
        return "long" if action in {"close", "close_by"} else "short"
    return None


def _compact_trade_history_row(
    row: Dict[str, Any],
    *,
    history_kind: Optional[str],
) -> Dict[str, Any]:
    compact = _round_trade_money_fields(row)
    if history_kind == "orders":
        if "time_setup" in compact:
            compact["placed_time"] = compact["time_setup"]
        if "time_done" in compact:
            compact["done_time"] = compact["time_done"]
        fields = _TRADE_HISTORY_COMPACT_ORDER_FIELDS
    else:
        if "time" in compact:
            compact["fill_time"] = compact["time"]
        action = _trade_history_action(compact, history_kind=history_kind)
        if action is not None:
            compact["action"] = action
            compact["deal_effect"] = action
        position_side = _trade_history_position_side(
            compact,
            action=action,
            history_kind=history_kind,
        )
        if position_side is not None:
            compact["position_side"] = position_side
        if compact.get("comment_may_be_truncated") is True:
            compact["comment_truncated"] = True
        fields = _TRADE_HISTORY_COMPACT_DEAL_FIELDS
    return {
        key: compact[key]
        for key in fields
        if key in compact
        and compact[key] is not None
        and not (isinstance(compact[key], str) and not compact[key].strip())
    }


def _public_trade_history_details(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in _compact_non_empty_mapping(row).items()
        if str(key) not in _TRADE_HISTORY_DIAGNOSTIC_FIELDS
        and str(key) not in _TRADE_HISTORY_ROW_METADATA_FIELDS
    }


def _normalize_trade_history_row(
    row: Dict[str, Any],
    *,
    history_kind: Optional[str],
) -> Dict[str, Any]:
    row = _round_trade_money_fields(row)
    item_kind = "order" if history_kind == "orders" else "deal"
    if item_kind == "order":
        price = _first_present(row, "price_current", "price_open", "price")
        native_key = "order_details"
        top_level_fields = _TRADE_HISTORY_ORDER_TOP_LEVEL_FIELDS
    else:
        price = _first_present(row, "price")
        native_key = "deal_details"
        top_level_fields = _TRADE_HISTORY_DEAL_TOP_LEVEL_FIELDS

    normalized: Dict[str, Any] = {
        "ticket": _first_present(row, "ticket", "order", "deal"),
        "symbol": row.get("symbol"),
        "volume": _first_present(row, "volume", "volume_initial", "volume_current"),
        "price": price,
    }
    action = _trade_history_action(row, history_kind=history_kind)
    if action is not None:
        normalized["action"] = action
        normalized["deal_effect"] = action
    position_side = _trade_history_position_side(
        row,
        action=action,
        history_kind=history_kind,
    )
    if position_side is not None:
        normalized["position_side"] = position_side
    public_details = _public_trade_history_details(row)
    for key in top_level_fields:
        if key in public_details:
            normalized[key] = public_details[key]
    remaining_details = {
        key: value
        for key, value in public_details.items()
        if key not in top_level_fields
    }
    if remaining_details:
        normalized[native_key] = remaining_details
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
        "offset",
        "page",
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
        "fill_time": "Fill Time",
        "placed_time": "Placed Time",
        "done_time": "Done Time",
        "time_setup": "Setup Time",
        "time_done": "Done Time",
        "time_msc": "Time Msc",
        "ticket": "Ticket",
        "deal_ticket": "Deal Ticket",
        "order": "Order",
        "order_ticket": "Order Ticket",
        "deal": "Deal",
        "position_id": "Position ID",
        "position_by_id": "Position By ID",
        "symbol": "Symbol",
        "type": "Type",
        "type_code": "Type Code",
        "position_side": "Position Side",
        "entry": "Entry",
        "entry_code": "Entry Code",
        "deal_effect": "Deal Effect",
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


def _trade_history_period_context(request: Any) -> Dict[str, Any]:
    def _format_period_dt(value: Any) -> Optional[str]:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return _format_datetime_second_explicit(value)

    end_value = getattr(request, "end", None)
    to_dt = _parse_start_datetime(end_value) if end_value else None
    if to_dt is None:
        to_dt = datetime.now(timezone.utc).replace(tzinfo=None)

    minutes_back_value, minutes_back_error = validation._normalize_minutes_back(
        getattr(request, "minutes_back", None)
    )
    if minutes_back_error:
        minutes_back_value = None

    start_value = getattr(request, "start", None)
    if minutes_back_value is not None:
        from_dt = to_dt - timedelta(minutes=minutes_back_value)
    elif start_value:
        from_dt = _parse_start_datetime(start_value)
    else:
        minutes_back_value = int(_DEFAULT_TRADE_HISTORY_LOOKBACK_DAYS * 24 * 60)
        from_dt = to_dt - timedelta(minutes=minutes_back_value)

    out: Dict[str, Any] = {
        "period_start": _format_period_dt(from_dt),
        "period_end": _format_period_dt(to_dt),
        "period_timezone": "UTC",
    }
    if minutes_back_value is not None:
        out["minutes_back_effective"] = int(minutes_back_value)
        if getattr(request, "minutes_back", None) is not None:
            out["period_source"] = "minutes_back"
            out["minutes_back_requested"] = int(minutes_back_value)
        else:
            out["period_source"] = "default_lookback"
            out["defaults_applied"] = {"lookback_minutes": int(minutes_back_value)}
            out["note"] = (
                f"Period limited to default {int(minutes_back_value)}-minute "
                f"({_DEFAULT_TRADE_HISTORY_LOOKBACK_DAYS}-day) lookback. "
                "Set minutes_back or start/end to change."
            )
    elif start_value or end_value:
        out["period_source"] = "explicit_range"
    return out


def _insert_trade_history_period_context(
    out: Dict[str, Any],
    period_context: Dict[str, Any],
) -> Dict[str, Any]:
    if not period_context:
        return out
    ordered: Dict[str, Any] = {}
    inserted = False
    for key, value in out.items():
        ordered[key] = value
        if key == "count":
            for period_key, period_value in period_context.items():
                ordered.setdefault(period_key, period_value)
            inserted = True
    if not inserted:
        for period_key, period_value in period_context.items():
            ordered.setdefault(period_key, period_value)
    return ordered


def normalize_trade_history_output(
    rows: Any,
    *,
    request: Any,
) -> Dict[str, Any]:
    """Normalize trade history into the standard trade read envelope."""
    out = _normalize_trade_read_output(rows, request=request, kind="trade_history")
    history_kind = getattr(request, "history_kind", None)
    include_request_metadata = _include_trade_read_request_metadata(request)
    if out.get("success") is True:
        detail_value = str(getattr(request, "detail", "compact") or "compact").lower()
        if detail_value != "compact":
            period_context = _trade_history_period_context(request)
            out = _insert_trade_history_period_context(out, period_context)
    timezone_label = "UTC"
    if out.get("success") is True and isinstance(out.get("items"), list):
        out.setdefault("row_key", "items")
        raw_items = list(out["items"])
        for item in raw_items:
            if isinstance(item, dict) and item.get("timezone"):
                timezone_label = str(item["timezone"])
                break
        if include_request_metadata:
            out["items"] = _normalize_trade_history_items(
                raw_items,
                history_kind=history_kind,
            )
            out["item_schema"] = "normalized_trade_history.v2"
        else:
            out["items"] = _style_trade_history_items(
                [
                    _compact_trade_history_row(item, history_kind=history_kind)
                    if isinstance(item, dict)
                    else item
                    for item in raw_items
                ],
                column_style=getattr(request, "column_style", "snake_case"),
            )
    if include_request_metadata:
        for key in ("symbol", "ticket"):
            out.pop(key, None)
        if "total_count" not in out:
            out.pop("limit", None)
        request_echo = _trade_history_request_echo(request, history_kind=history_kind)
        if request_echo:
            out["request_echo"] = request_echo
    elif history_kind is not None:
        out["history_kind"] = history_kind
    if out.get("success") is True:
        out.setdefault("timezone", timezone_label)
        _attach_trade_volume_units(out)
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
                v in ticket_candidates for v in _ticket_fields(pos).values()
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
                else _resolved_ticket(picked, fallback=candidate)
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
            for field, value in _ticket_fields(pos).items():
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
            resolved = _resolved_ticket(pos, fallback=matched_value)
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
    resolved = _resolved_ticket(picked)
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
                else _resolved_ticket(picked, fallback=candidate)
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
            for field, value in _ticket_fields(order).items():
                if require_exact_ticket_match and field != "ticket":
                    continue
                if value in candidate_ids:
                    exact_matches.append((order, field, value))
        if exact_matches:
            exact_matches.sort(key=lambda item: _order_sort_key(item[0]), reverse=True)
            order, field, matched_value = exact_matches[0]
            resolved = _resolved_ticket(order, fallback=matched_value)
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
    resolved = _resolved_ticket(picked)
    return picked, resolved, {"method": "orders_get(fallback_heuristic)"}


@mcp.tool()
def trade_get_open(
    request: TradeGetOpenRequest,
) -> Dict[str, Any]:
    """Get open positions. Compact output omits echoed request metadata by default.

    Each row's `ticket` is the position ticket; it equals `position_ticket` in
    `trade_history`, so join the two tools on
    `trade_get_open.ticket == trade_history.position_ticket`.
    """
    def _run() -> Dict[str, Any]:
        gateway = create_trading_gateway()
        return _normalize_trade_read_output(
            run_trade_get_open(
                request,
                gateway=gateway,
                use_client_tz=lambda: False,
                format_time_minimal=_format_time_minimal,
                format_time_minimal_local=_format_time_minimal,
                mt5_epoch_to_utc=_utc_epoch_identity,
                normalize_limit=_normalize_limit,
                comment_row_metadata=comments._comment_row_metadata,
            ),
            request=request,
            kind="open_positions",
            account_currency=_gateway_account_currency(gateway),
        )

    return run_logged_operation(
        logger,
        operation="trade_get_open",
        symbol=request.symbol,
        limit=request.limit,
        func=_run,
    )


@mcp.tool()
def trade_get_pending(
    request: TradeGetPendingRequest,
) -> Dict[str, Any]:
    """Get pending orders. Compact output omits echoed request metadata by default."""
    def _run() -> Dict[str, Any]:
        gateway = create_trading_gateway()
        return _normalize_trade_read_output(
            run_trade_get_pending(
                request,
                gateway=gateway,
                use_client_tz=lambda: False,
                format_time_minimal=_format_time_minimal,
                format_time_minimal_local=_format_time_minimal,
                mt5_epoch_to_utc=_utc_epoch_identity,
                normalize_limit=_normalize_limit,
                comment_row_metadata=comments._comment_row_metadata,
            ),
            request=request,
            kind="pending_orders",
            account_currency=_gateway_account_currency(gateway),
        )

    return run_logged_operation(
        logger,
        operation="trade_get_pending",
        symbol=request.symbol,
        limit=request.limit,
        func=_run,
    )
