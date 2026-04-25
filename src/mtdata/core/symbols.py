
import logging
import math
import time
from typing import Any, Dict, List, Literal, Optional

from ..shared.schema import CompactFullDetailLiteral, TimeframeLiteral
from ..shared.validators import invalid_timeframe_error
from ..utils.mt5 import (
    MT5ConnectionError,
    _mt5_copy_rates_from_pos,
    _symbol_ready_guard,
    ensure_mt5_connection_or_raise,
    mt5,
)
from ..utils.mt5_enums import decode_mt5_bitmask_labels, decode_mt5_enum_label
from ..utils.symbol import _extract_group_path as _extract_group_path_util
from ..utils.utils import _format_time_minimal, _normalize_limit, _table_from_rows
from ._mcp_instance import mcp
from .constants import DEFAULT_ROW_LIMIT, GROUP_SEARCH_THRESHOLD, TIMEFRAME_MAP
from .execution_logging import run_logged_operation
from .mt5_gateway import get_mt5_gateway
from .output_contract import (
    attach_collection_contract,
    resolve_output_contract,
    resolve_output_detail,
)

logger = logging.getLogger(__name__)
_MARKET_SCAN_STALE_BAR_SECONDS = 7 * 24 * 60 * 60


def _case_insensitive_sort_key(value: Any) -> tuple[str, str]:
    text = str(value or "").strip()
    return text.casefold(), text


def _normalize_symbol_search_term(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_SYMBOL_DESCRIBE_PRICE_FIELDS = frozenset(
    {
        "bidlow",
        "bidhigh",
        "asklow",
        "askhigh",
        "session_open",
        "session_close",
    }
)

_SYMBOL_DESCRIBE_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "path",
    "bank",
    "currency_base",
    "currency_profit",
    "currency_margin",
    "digits",
    "point",
    "bidlow",
    "bidhigh",
    "asklow",
    "askhigh",
    "price_change",
    "session_open",
    "session_close",
    "trade_mode",
    "trade_exemode",
    "trade_calc_mode",
    "order_mode",
    "expiration_mode",
    "filling_mode",
    "trade_contract_size",
    "trade_tick_size",
    "trade_tick_value",
    "trade_tick_value_profit",
    "trade_tick_value_loss",
    "trade_stops_level",
    "trade_freeze_level",
    "volume_min",
    "volume_max",
    "volume_step",
    "volume_limit",
    "swap_mode",
    "swap_long",
    "swap_short",
    "swap_rollover3days",
    "spread_float",
    "ticks_bookdepth",
    "time",
    "select",
)

_SYMBOL_DESCRIBE_COMPACT_DIRECT_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "currency_base",
    "currency_profit",
    "digits",
    "point",
    "bidlow",
    "bidhigh",
    "asklow",
    "askhigh",
    "trade_contract_size",
    "trade_tick_size",
    "trade_tick_value",
    "volume_min",
    "volume_max",
    "volume_step",
    "spread_float",
    "time",
)
_SYMBOL_DESCRIBE_COMPACT_ENUM_FIELDS: tuple[str, ...] = (
    "trade_mode",
    "order_mode",
)


def _copy_symbol_describe_field(
    out: Dict[str, Any],
    source: Dict[str, Any],
    field: str,
) -> bool:
    if field not in source:
        return False
    value = source.get(field)
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    if isinstance(value, list) and not value:
        return False
    out[field] = value
    return True


def _compact_symbol_describe_payload(symbol_data: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for field in _SYMBOL_DESCRIBE_COMPACT_DIRECT_FIELDS:
        _copy_symbol_describe_field(compact, symbol_data, field)

    for field in _SYMBOL_DESCRIBE_COMPACT_ENUM_FIELDS:
        if _copy_symbol_describe_field(compact, symbol_data, f"{field}_label"):
            continue
        if _copy_symbol_describe_field(compact, symbol_data, f"{field}_labels"):
            continue
        _copy_symbol_describe_field(compact, symbol_data, field)

    if "time_epoch" in symbol_data:
        compact["time_epoch"] = symbol_data["time_epoch"]
    return compact


@mcp.tool()
def symbols_list(  # noqa: C901
    search_term: Optional[str] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
    list_mode: Literal["symbols", "groups"] = "symbols",  # type: ignore
) -> Dict[str, Any]:
    """List symbols or symbol groups."""
    normalized_search_term = _normalize_symbol_search_term(search_term)

    def _run() -> Dict[str, Any]:
        try:
            mt5_gateway = get_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            mode = str(list_mode or "symbols").strip().lower()
            if mode not in ("symbols", "groups"):
                return {"error": "list_mode must be 'symbols' or 'groups'."}
            if mode == "groups":
                return _list_symbol_groups(
                    search_term=normalized_search_term,
                    limit=limit,
                    mt5_gateway=mt5_gateway,
                )

            matched_symbols = []

            if normalized_search_term:
                search_upper = normalized_search_term.upper()

                all_symbols = mt5_gateway.symbols_get()
                if all_symbols is None:
                    return {"error": f"Failed to get symbols: {mt5_gateway.last_error()}"}

                groups = {}
                for symbol in all_symbols:
                    group_path = _extract_group_path_util(symbol)
                    if group_path not in groups:
                        groups[group_path] = []
                    groups[group_path].append(symbol)

                matching_groups = []
                group_search_threshold = GROUP_SEARCH_THRESHOLD

                for group_name in groups.keys():
                    if search_upper in group_name.upper():
                        matching_groups.append(group_name)

                if matching_groups and len(matching_groups) <= group_search_threshold:
                    for group_name in matching_groups:
                        matched_symbols.extend(groups[group_name])
                else:
                    symbol_name_matches = []
                    for symbol in all_symbols:
                        if search_upper in symbol.name.upper():
                            symbol_name_matches.append(symbol)

                    if symbol_name_matches:
                        matched_symbols = symbol_name_matches
                    elif matching_groups:
                        for group_name in matching_groups:
                            matched_symbols.extend(groups[group_name])
                    else:
                        description_matches = []
                        for symbol in all_symbols:
                            if hasattr(symbol, "description") and symbol.description:
                                if search_upper in symbol.description.upper():
                                    description_matches.append(symbol)
                                    continue

                            group_path = getattr(symbol, "path", "")
                            if search_upper in group_path.upper():
                                description_matches.append(symbol)

                        matched_symbols = description_matches
            else:
                matched_symbols = list(mt5_gateway.symbols_get() or [])

            matched_symbols = sorted(
                matched_symbols,
                key=lambda symbol: _case_insensitive_sort_key(getattr(symbol, "name", "")),
            )
            only_visible = not bool(normalized_search_term)
            symbol_list = []
            for symbol in matched_symbols:
                if only_visible and not symbol.visible:
                    continue
                symbol_list.append({
                    "name": symbol.name,
                    "group": _extract_group_path_util(symbol),
                    "description": symbol.description,
                })

            limit_value = _normalize_limit(limit)
            if limit_value:
                symbol_list = symbol_list[:limit_value]
            rows = [[s["name"], s["group"], s["description"]] for s in symbol_list]
            result = _table_from_rows(["name", "group", "description"], rows)
            return attach_collection_contract(
                result,
                collection_kind="table",
                rows=result.get("data"),
                include_contract_meta=False,
            )
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error getting symbols: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="symbols_list",
        search_term=normalized_search_term,
        limit=limit,
        list_mode=list_mode,
        func=_run,
    )

def _list_symbol_groups(
    search_term: Optional[str] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
    mt5_gateway: Any = None,
) -> Dict[str, Any]:
    """List group paths as a tabular result with a single column: group."""
    try:
        gateway = mt5_gateway or get_mt5_gateway(
            adapter=mt5,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )
        # Get all symbols first
        all_symbols = gateway.symbols_get()
        if all_symbols is None:
            return {"error": f"Failed to get symbols: {gateway.last_error()}"}
        
        # Collect unique groups and counts
        groups = {}
        for symbol in all_symbols:
            group_path = _extract_group_path_util(symbol)
            if group_path not in groups:
                groups[group_path] = {"count": 0}
            groups[group_path]["count"] += 1
        
        # Filter by search term if provided
        filtered_items = list(groups.items())
        if search_term:
            q = search_term.strip().lower()
            filtered_items = [(k, v) for (k, v) in filtered_items if q in (k or '').lower()]

        # Sort groups by count (most symbols first)
        filtered_items.sort(
            key=lambda item: (
                -item[1]["count"],
                *_case_insensitive_sort_key(item[0]),
            )
        )

        # Apply limit
        limit_value = _normalize_limit(limit)
        if limit_value:
            filtered_items = filtered_items[:limit_value]

        # Build tabular result with only group names
        group_names = [name for name, _ in filtered_items]
        rows = [[g] for g in group_names]
        result = _table_from_rows(["group"], rows)
        return attach_collection_contract(
            result,
            collection_kind="table",
            rows=result.get("data"),
            include_contract_meta=False,
        )
    except Exception as e:
        return {"error": f"Error getting symbol groups: {str(e)}"}

@mcp.tool()
def symbols_describe(
    symbol: str,
    detail: CompactFullDetailLiteral = "full",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Return symbol information as JSON for `symbol`.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., "EURUSD")
    detail : str, optional (default="full")
        Output verbosity level:
        - "compact": Essential fields only (name, bid/ask, volume limits, contract size, tick size/value)
        - "full": Complete metadata including all trading modes, swap details, and session times
    verbose : bool, optional (default=False)
        Include time_epoch field for timestamps
    
    Returns:
    --------
    dict
        Symbol information with requested detail level
    """
    def _run() -> Dict[str, Any]:
        try:
            contract = resolve_output_contract(
                detail=detail,
                verbose=verbose,
                default_detail="full",
            )
            mt5_gateway = get_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            symbol_info = mt5_gateway.symbol_info(symbol)
            if symbol_info is None:
                return {"error": f"Symbol {symbol} not found"}

            enum_specs = {
                "trade_mode": {"prefixes": ("SYMBOL_TRADE_MODE_",), "bitmask": False},
                "trade_exemode": {"prefixes": ("SYMBOL_TRADE_EXECUTION_",), "bitmask": False},
                "trade_calc_mode": {"prefixes": ("SYMBOL_CALC_MODE_",), "bitmask": False},
                "swap_mode": {"prefixes": ("SYMBOL_SWAP_MODE_",), "bitmask": False},
                "filling_mode": {"prefixes": ("ORDER_FILLING_", "SYMBOL_FILLING_"), "bitmask": True},
                "expiration_mode": {"prefixes": ("SYMBOL_EXPIRATION_",), "bitmask": True},
                "order_mode": {"prefixes": ("SYMBOL_ORDER_",), "bitmask": True},
            }

            symbol_data = {}
            available_attrs = set(dir(symbol_info))
            for attr in _SYMBOL_DESCRIBE_FIELDS:
                if attr not in available_attrs:
                    continue
                try:
                    value = getattr(symbol_info, attr)
                except Exception:
                    continue
                if callable(value):
                    continue
                if value is None:
                    continue
                if isinstance(value, str) and value == "":
                    continue
                if attr == "time":
                    try:
                        from ..utils.mt5 import _mt5_epoch_to_utc

                        epoch = float(value)
                        utc_epoch = _mt5_epoch_to_utc(epoch)
                        if verbose:
                            symbol_data["time_epoch"] = utc_epoch
                        symbol_data["time"] = _format_time_minimal(utc_epoch)
                    except Exception:
                        if verbose:
                            symbol_data["time_epoch"] = value
                        symbol_data["time"] = str(value)
                else:
                    if attr in _SYMBOL_DESCRIBE_PRICE_FIELDS:
                        digits = max(0, int(getattr(symbol_info, "digits", 0) or 0))
                        value = _market_scan_round(_market_scan_float(value), digits=digits)
                    symbol_data[attr] = value

                spec = enum_specs.get(attr)
                if not spec:
                    continue
                prefixes = spec.get("prefixes", ())
                is_bitmask = bool(spec.get("bitmask"))
                if is_bitmask:
                    labels = []
                    for prefix in prefixes:
                        labels = decode_mt5_bitmask_labels(mt5_gateway, value, prefix=prefix)
                        if labels:
                            break
                    if labels:
                        symbol_data[f"{attr}_labels"] = labels
                        symbol_data[f"{attr}_label"] = ", ".join(labels)
                else:
                    label = None
                    for prefix in prefixes:
                        label = decode_mt5_enum_label(mt5_gateway, value, prefix=prefix)
                        if label:
                            break
                    if label:
                        symbol_data[f"{attr}_label"] = label

            if contract.shape_detail == "compact":
                symbol_data = _compact_symbol_describe_payload(symbol_data)

            return {
                "success": True,
                "symbol": symbol_data,
            }
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error getting symbol info: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="symbols_describe",
        symbol=symbol,
        detail=detail,
        func=_run,
    )


def _market_scan_is_tradable(symbol: Any) -> bool:
    disabled_trade_mode = getattr(mt5, "SYMBOL_TRADE_MODE_DISABLED", None)
    if disabled_trade_mode is None:
        return True
    return getattr(symbol, "trade_mode", None) != disabled_trade_mode


def _market_scan_base_row(symbol: Any) -> Dict[str, Any]:
    return {
        "symbol": getattr(symbol, "name", None),
        "group": _extract_group_path_util(symbol),
        "description": getattr(symbol, "description", None),
    }


def _market_scan_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _market_scan_bar_int(value: Any) -> Optional[int]:
    try:
        out = int(value)
    except Exception:
        return None
    return out


def _market_scan_round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), max(0, int(digits)))


def _market_scan_freshness_fields(bar_time: Optional[float]) -> Dict[str, Any]:
    if bar_time is None:
        return {}
    try:
        age_seconds = max(0.0, float(time.time()) - float(bar_time))
    except Exception:
        return {}
    fields: Dict[str, Any] = {
        "bar_age_hours": _market_scan_round(age_seconds / 3600.0, digits=3),
        "data_stale": age_seconds > _MARKET_SCAN_STALE_BAR_SECONDS,
    }
    if fields["data_stale"]:
        fields["stale_warning"] = (
            "Completed bar data may be stale; latest bar is "
            f"{fields['bar_age_hours']} hours old."
        )
    return fields


def _build_market_scan_spread_row(
    symbol: Any,
    mt5_gateway: Any,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    tick = mt5_gateway.symbol_info_tick(symbol.name)
    if tick is None:
        return None, f"Failed to get tick data: {mt5_gateway.last_error()}"

    bid = _market_scan_float(getattr(tick, "bid", None))
    ask = _market_scan_float(getattr(tick, "ask", None))
    if bid is None or ask is None:
        return None, "Bid/ask quote is unavailable."
    if ask < bid:
        return None, "Bid/ask quote is invalid."

    point = _market_scan_float(getattr(symbol, "point", 0.0)) or 0.0
    tick_size = _market_scan_float(getattr(symbol, "trade_tick_size", 0.0)) or 0.0
    tick_value = _market_scan_float(getattr(symbol, "trade_tick_value", 0.0)) or 0.0
    digits = max(0, int(getattr(symbol, "digits", 0) or 0))

    spread_abs = float(ask - bid)
    mid = (ask + bid) / 2.0
    spread_points = (spread_abs / point) if point > 0 else None
    spread_pct = ((spread_abs / mid) * 100.0) if mid > 0 else None
    spread_usd = None
    pricing_basis = "quote_only"
    if tick_size > 0 and tick_value > 0:
        spread_usd = (spread_abs / tick_size) * tick_value
        pricing_basis = "per_1_lot_estimate"

    row = _market_scan_base_row(symbol)
    row.update(
        {
            "bid": _market_scan_round(bid, digits=digits),
            "ask": _market_scan_round(ask, digits=digits),
            "spread": _market_scan_round(spread_abs, digits=digits),
            "spread_points": _market_scan_round(spread_points, digits=4),
            "spread_pct": _market_scan_round(spread_pct, digits=6),
            "spread_usd": _market_scan_round(spread_usd, digits=6),
            "pricing_basis": pricing_basis,
        }
    )
    return row, None


def _build_market_scan_bar_row(
    symbol: Any,
    timeframe: str,
    mt5_timeframe: Any,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    rates = _mt5_copy_rates_from_pos(symbol.name, mt5_timeframe, 1, 1)
    if rates is None or len(rates) < 1:
        return None, f"No completed {timeframe} bar data returned."

    latest_bar = rates[-1]
    open_price = _market_scan_float(latest_bar["open"])
    close_price = _market_scan_float(latest_bar["close"])
    if open_price is None or close_price is None:
        return None, "Completed bar is missing open/close prices."
    if open_price == 0:
        return None, "Completed bar open price is zero."

    digits = max(0, int(getattr(symbol, "digits", 0) or 0))
    bar_time = _market_scan_float(latest_bar["time"])
    tick_volume = _market_scan_bar_int(latest_bar["tick_volume"])
    real_volume = _market_scan_bar_int(latest_bar["real_volume"])
    row = _market_scan_base_row(symbol)
    row.update(
        {
            "timeframe": timeframe,
            "bar_time": _format_time_minimal(bar_time) if bar_time is not None else None,
            **_market_scan_freshness_fields(bar_time),
            "open": _market_scan_round(open_price, digits=digits),
            "close": _market_scan_round(close_price, digits=digits),
            "tick_volume": tick_volume,
            "real_volume": real_volume,
            "price_change_pct": _market_scan_round(
                ((close_price - open_price) / open_price) * 100.0,
                digits=6,
            ),
        }
    )
    return row, None


def _market_scan_table(
    headers: List[str],
    rows: List[Dict[str, Any]],
    *,
    include_contract_meta: bool = True,
) -> Dict[str, Any]:
    ordered_rows = [[row.get(header) for header in headers] for row in rows]
    result = _table_from_rows(headers, ordered_rows)
    return attach_collection_contract(
        result,
        collection_kind="table",
        rows=result.get("data"),
        include_contract_meta=include_contract_meta,
    )


def _strip_nested_market_scan_meta(table: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in table.items() if k not in ("success", "count")}


def _market_scan_contract_table(
    headers: List[str],
    rows: List[Dict[str, Any]],
    *,
    include_columns: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "rows": [dict(row) for row in rows],
        "row_count": int(len(rows)),
    }
    if include_columns:
        out["columns"] = [str(header) for header in headers]
    return out


def _market_scan_contract_meta(
    *,
    request: Dict[str, Any],
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "tool": "market_scan",
        "request": {
            key: value for key, value in request.items() if value is not None
        },
        "runtime": {},
    }
    if stats:
        out["stats"] = {
            key: value for key, value in stats.items() if value is not None
        }
    return out


def _market_scan_error(
    message: str,
    *,
    code: str,
    request: Dict[str, Any],
    stats: Optional[Dict[str, Any]] = None,
    details: Any = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "success": False,
        "error": str(message),
        "error_code": str(code),
        "meta": _market_scan_contract_meta(request=request, stats=stats),
    }
    if details not in (None, [], {}):
        out["details"] = details
    if warnings:
        out["warnings"] = warnings
    return out


def _top_markets_headers(metric: str, *, detail_mode: str) -> List[str]:
    full_headers = {
        "spread": [
            "symbol",
            "group",
            "description",
            "bid",
            "ask",
            "spread",
            "spread_points",
            "spread_pct",
            "spread_usd",
            "pricing_basis",
        ],
        "volume": [
            "symbol",
            "group",
            "description",
            "timeframe",
            "bar_time",
            "bar_age_hours",
            "data_stale",
            "stale_warning",
            "tick_volume",
            "real_volume",
            "open",
            "close",
            "price_change_pct",
        ],
        "price_change": [
            "symbol",
            "group",
            "description",
            "timeframe",
            "bar_time",
            "bar_age_hours",
            "data_stale",
            "stale_warning",
            "open",
            "close",
            "price_change_pct",
            "tick_volume",
            "real_volume",
        ],
    }
    compact_headers = {
        "spread": ["symbol", "group", "spread_pct", "spread_points"],
        "volume": [
            "symbol",
            "group",
            "timeframe",
            "bar_age_hours",
            "data_stale",
            "tick_volume",
            "price_change_pct",
        ],
        "price_change": [
            "symbol",
            "group",
            "timeframe",
            "bar_age_hours",
            "data_stale",
            "price_change_pct",
            "tick_volume",
        ],
    }
    header_map = compact_headers if detail_mode == "compact" else full_headers
    return list(header_map[metric])


def _parse_market_scan_symbols(symbols: Optional[str]) -> List[str]:
    text = str(symbols or "").replace(";", ",").replace("\n", ",")
    parsed: List[str] = []
    seen: set[str] = set()
    for chunk in text.split(","):
        name = chunk.strip()
        if not name:
            continue
        upper = name.upper()
        if upper in seen:
            continue
        seen.add(upper)
        parsed.append(name)
    return parsed


def _resolve_market_scan_group_path(
    all_symbols: List[Any],
    group: str,
) -> tuple[Optional[str], Optional[str]]:
    requested = str(group or "").strip()
    if not requested:
        return None, "group must not be empty."

    groups: Dict[str, str] = {}
    for symbol in all_symbols:
        group_path = str(_extract_group_path_util(symbol) or "").strip()
        if not group_path:
            continue
        groups.setdefault(group_path.lower(), group_path)

    requested_lower = requested.lower()
    exact = groups.get(requested_lower)
    if exact is not None:
        return exact, None

    partial_matches = sorted(
        (
            value for value in groups.values()
            if requested_lower in value.lower()
        ),
        key=_case_insensitive_sort_key,
    )
    if len(partial_matches) == 1:
        return partial_matches[0], None
    if partial_matches:
        return None, (
            f"Ambiguous group '{requested}'. Matching groups: "
            + ", ".join(partial_matches[:10])
        )
    return None, f"No symbol group matched '{requested}'."


def _select_market_scan_symbols(
    all_symbols: List[Any],
    *,
    symbols: Optional[str] = None,
    group: Optional[str],
    universe: str,
) -> tuple[List[Any], Dict[str, Any], Optional[str]]:
    requested_names = _parse_market_scan_symbols(symbols)
    selection_meta: Dict[str, Any] = {}
    if requested_names:
        selection_meta["symbols_input"] = list(requested_names)
    if requested_names and group:
        return [], selection_meta, "Provide either symbols or group, not both."

    tradable_symbols = [symbol for symbol in all_symbols if _market_scan_is_tradable(symbol)]

    if requested_names:
        by_upper: Dict[str, Any] = {}
        for tradable_symbol in tradable_symbols:
            name = str(getattr(tradable_symbol, "name", "") or "").strip()
            if not name:
                continue
            by_upper.setdefault(name.upper(), tradable_symbol)
        selected: List[Any] = []
        missing: List[str] = []
        for name in requested_names:
            symbol_obj = by_upper.get(name.upper())
            if symbol_obj is None:
                missing.append(name)
                continue
            selected.append(symbol_obj)
        if not selected:
            return [], selection_meta, "None of the requested symbols matched the MT5 symbol list."
        return selected, {
            **selection_meta,
            "scope": "symbols",
            "requested_symbols": requested_names,
            "missing_symbols": missing,
        }, None

    if group:
        resolved_group, group_error = _resolve_market_scan_group_path(tradable_symbols, group)
        if group_error or resolved_group is None:
            return [], {}, group_error
        selected = sorted(
            [
                symbol for symbol in tradable_symbols
                if str(_extract_group_path_util(symbol) or "").strip() == resolved_group
                and (universe == "all" or bool(getattr(symbol, "visible", False)))
            ],
            key=lambda symbol: _case_insensitive_sort_key(getattr(symbol, "name", "")),
        )
        return selected, {
            **selection_meta,
            "scope": "group",
            "group": resolved_group,
        }, None

    selected = sorted(
        [
            symbol for symbol in tradable_symbols
            if universe == "all" or bool(getattr(symbol, "visible", False))
        ],
        key=lambda symbol: _case_insensitive_sort_key(getattr(symbol, "name", "")),
    )
    return selected, {**selection_meta, "scope": "universe"}, None


def _market_scan_compute_rsi(closes: List[float], length: int) -> Optional[float]:
    if length <= 0 or len(closes) < (length + 1):
        return None

    gains: List[float] = []
    losses: List[float] = []
    for prev_close, close in zip(closes[:-1], closes[1:]):
        delta = float(close - prev_close)
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains[:length]) / float(length)
    avg_loss = sum(losses[:length]) / float(length)
    for gain, loss in zip(gains[length:], losses[length:]):
        avg_gain = ((avg_gain * float(length - 1)) + float(gain)) / float(length)
        avg_loss = ((avg_loss * float(length - 1)) + float(loss)) / float(length)

    if avg_loss <= 0.0:
        if avg_gain <= 0.0:
            return 50.0
        return 100.0

    relative_strength = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _build_market_scan_signal_row(
    symbol: Any,
    *,
    timeframe: str,
    mt5_timeframe: Any,
    lookback: int,
    rsi_length: int,
    sma_period: int,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    rates = _mt5_copy_rates_from_pos(symbol.name, mt5_timeframe, 1, lookback)
    if rates is None or len(rates) < 1:
        return None, f"No completed {timeframe} bar data returned."

    latest_bar = rates[-1]
    open_price = _market_scan_float(latest_bar["open"])
    close_price = _market_scan_float(latest_bar["close"])
    if open_price is None or close_price is None:
        return None, "Completed bar is missing open/close prices."
    if open_price == 0:
        return None, "Completed bar open price is zero."

    close_values: List[float] = []
    for bar in rates:
        close_value = _market_scan_float(bar["close"])
        if close_value is not None:
            close_values.append(close_value)

    digits = max(0, int(getattr(symbol, "digits", 0) or 0))
    bar_time = _market_scan_float(latest_bar["time"])
    tick_volume = _market_scan_bar_int(latest_bar["tick_volume"])
    real_volume = _market_scan_bar_int(latest_bar["real_volume"])
    sma_value = None
    if len(close_values) >= max(1, int(sma_period)):
        sma_window = close_values[-int(sma_period):]
        sma_value = float(sum(sma_window) / len(sma_window))
    rsi_value = _market_scan_compute_rsi(close_values, int(rsi_length))
    sma_distance_pct = None
    if sma_value is not None and sma_value != 0:
        sma_distance_pct = ((close_price - sma_value) / sma_value) * 100.0

    row = _market_scan_base_row(symbol)
    row.update(
        {
            "timeframe": timeframe,
            "bar_time": _format_time_minimal(bar_time) if bar_time is not None else None,
            **_market_scan_freshness_fields(bar_time),
            "open": _market_scan_round(open_price, digits=digits),
            "close": _market_scan_round(close_price, digits=digits),
            "tick_volume": tick_volume,
            "real_volume": real_volume,
            "price_change_pct": _market_scan_round(
                ((close_price - open_price) / open_price) * 100.0,
                digits=6,
            ),
            "rsi": _market_scan_round(rsi_value, digits=4),
            "sma_value": _market_scan_round(sma_value, digits=digits),
            "sma_distance_pct": _market_scan_round(sma_distance_pct, digits=6),
        }
    )
    return row, None


def _market_scan_missing_required_metric(
    row: Dict[str, Any],
    *,
    rank_by: str,
    rsi_above: Optional[float],
    rsi_below: Optional[float],
    price_vs_sma: Optional[str],
    max_spread_pct: Optional[float],
    min_tick_volume: Optional[int],
    min_price_change_pct: Optional[float],
    max_price_change_pct: Optional[float],
    rsi_length: int,
    sma_period: int,
) -> Optional[str]:
    requirements: List[tuple[str, str]] = []
    if rank_by in {"abs_price_change_pct", "price_change_pct"}:
        requirements.append(("price_change_pct", "price-change data is unavailable."))
    elif rank_by == "tick_volume":
        requirements.append(("tick_volume", "Tick-volume data is unavailable."))
    elif rank_by == "rsi":
        requirements.append(("rsi", f"Not enough history to compute RSI({int(rsi_length)})."))
    elif rank_by == "spread_pct":
        requirements.append(("spread_pct", "Spread data is unavailable."))

    if min_price_change_pct is not None or max_price_change_pct is not None:
        requirements.append(("price_change_pct", "price-change data is unavailable."))
    if max_spread_pct is not None:
        requirements.append(("spread_pct", "Spread data is unavailable."))
    if min_tick_volume is not None:
        requirements.append(("tick_volume", "Tick-volume data is unavailable."))
    if rsi_above is not None or rsi_below is not None:
        requirements.append(("rsi", f"Not enough history to compute RSI({int(rsi_length)})."))
    if price_vs_sma is not None:
        requirements.append(("sma_value", f"Not enough history to compute SMA({int(sma_period)})."))

    for key, message in requirements:
        if row.get(key) is None:
            return message
    return None


def _market_scan_row_matches_filters(
    row: Dict[str, Any],
    *,
    min_price_change_pct: Optional[float],
    max_price_change_pct: Optional[float],
    max_spread_pct: Optional[float],
    min_tick_volume: Optional[int],
    rsi_below: Optional[float],
    rsi_above: Optional[float],
    price_vs_sma: Optional[str],
) -> bool:
    price_change_pct = _market_scan_float(row.get("price_change_pct"))
    spread_pct = _market_scan_float(row.get("spread_pct"))
    tick_volume = _market_scan_bar_int(row.get("tick_volume"))
    rsi_value = _market_scan_float(row.get("rsi"))
    close_price = _market_scan_float(row.get("close"))
    sma_value = _market_scan_float(row.get("sma_value"))

    if min_price_change_pct is not None and (price_change_pct is None or price_change_pct < float(min_price_change_pct)):
        return False
    if max_price_change_pct is not None and (price_change_pct is None or price_change_pct > float(max_price_change_pct)):
        return False
    if max_spread_pct is not None and (spread_pct is None or spread_pct > float(max_spread_pct)):
        return False
    if min_tick_volume is not None and (tick_volume is None or tick_volume < int(min_tick_volume)):
        return False
    if rsi_below is not None and (rsi_value is None or rsi_value > float(rsi_below)):
        return False
    if rsi_above is not None and (rsi_value is None or rsi_value < float(rsi_above)):
        return False
    if price_vs_sma == "above" and (close_price is None or sma_value is None or close_price <= sma_value):
        return False
    if price_vs_sma == "below" and (close_price is None or sma_value is None or close_price >= sma_value):
        return False
    return True


def _market_scan_sort_rows(
    rows: List[Dict[str, Any]],
    *,
    rank_by: str,
    rsi_above: Optional[float],
    rsi_below: Optional[float],
) -> None:
    if rank_by == "spread_pct":
        rows.sort(
            key=lambda row: (
                row.get("spread_pct") is None,
                row.get("spread_pct") if row.get("spread_pct") is not None else float("inf"),
                row.get("symbol") or "",
            )
        )
        return

    if rank_by == "rsi" and rsi_below is not None and rsi_above is None:
        rows.sort(
            key=lambda row: (
                row.get("rsi") is None,
                row.get("rsi") if row.get("rsi") is not None else float("inf"),
                row.get("symbol") or "",
            )
        )
        return

    if rank_by == "abs_price_change_pct":
        rows.sort(
            key=lambda row: (
                row.get("price_change_pct") is None,
                -abs(float(row.get("price_change_pct") or 0.0)),
                row.get("symbol") or "",
            )
        )
        return

    rows.sort(
        key=lambda row: (
            row.get(rank_by) is None,
            -(float(row.get(rank_by) or 0.0)),
            row.get("symbol") or "",
        )
    )


_MARKET_SCAN_RANK_BY_ALIASES = {
    "abs_price_change": "abs_price_change_pct",
    "price_change": "price_change_pct",
    "volume": "tick_volume",
    "spread": "spread_pct",
}


_MARKET_SCAN_RANK_BY_CHOICES = (
    "abs_price_change_pct",
    "abs_price_change",
    "price_change_pct",
    "price_change",
    "tick_volume",
    "volume",
    "rsi",
    "spread_pct",
    "spread",
)


def _normalize_market_scan_rank_by(value: Any) -> tuple[str, Optional[str]]:
    raw_value = str(value or "abs_price_change_pct").strip().lower()
    return _MARKET_SCAN_RANK_BY_ALIASES.get(raw_value, raw_value), raw_value


@mcp.tool()
def symbols_top_markets(  # noqa: C901
    rank_by: Literal["all", "spread", "volume", "price_change"] = "all",  # type: ignore
    limit: Optional[int] = 10,
    universe: Literal["visible", "all"] = "visible",  # type: ignore
    timeframe: TimeframeLiteral = "H1",
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    """Scan MT5 symbols and rank the top markets by spread, recent volume, or recent price change.

    Defaults to visible tradable symbols for responsiveness. Set `universe="all"` to
    include hidden tradable symbols too; that mode is slower because MT5 may need to
    activate quotes for instruments that are not already visible. Volume and
    price-change rankings use the most recent completed bar on `timeframe`.
    Uses compact leaderboard rows by default. Set `detail="full"` for the
    expanded row shape and collection metadata.
    """

    detail_mode = resolve_output_detail(detail=detail, default="compact")

    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            rank_by_value = str(rank_by or "all").strip().lower()
            if rank_by_value not in {"all", "spread", "volume", "price_change"}:
                return {
                    "error": "rank_by must be one of: all, spread, volume, price_change."
                }

            universe_value = str(universe or "visible").strip().lower()
            if universe_value not in {"visible", "all"}:
                return {"error": "universe must be 'visible' or 'all'."}

            timeframe_value = str(timeframe or "H1").strip().upper()
            needs_bar_data = rank_by_value in {"all", "volume", "price_change"}
            if needs_bar_data and timeframe_value not in TIMEFRAME_MAP:
                return {"error": invalid_timeframe_error(timeframe_value, TIMEFRAME_MAP)}
            mt5_timeframe = TIMEFRAME_MAP.get(timeframe_value)

            mt5_gateway = get_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()

            raw_symbols = mt5_gateway.symbols_get()
            if raw_symbols is None:
                return {"error": f"Failed to get symbols: {mt5_gateway.last_error()}"}
            all_symbols = list(raw_symbols)

            selected_symbols = sorted(
                [
                    symbol for symbol in all_symbols
                    if _market_scan_is_tradable(symbol)
                    and (universe_value == "all" or bool(getattr(symbol, "visible", False)))
                ],
                key=lambda symbol: _case_insensitive_sort_key(getattr(symbol, "name", "")),
            )

            limit_value = _normalize_limit(limit) or 10
            started_at = time.perf_counter()

            spread_rows: List[Dict[str, Any]] = []
            volume_rows: List[Dict[str, Any]] = []
            price_change_rows: List[Dict[str, Any]] = []
            metric_issues: Dict[str, List[Dict[str, str]]] = {
                "spread": [],
                "volume": [],
                "price_change": [],
            }
            metric_skips: Dict[str, int] = {
                "spread": 0,
                "volume": 0,
                "price_change": 0,
            }

            def _record_issue(metric_name: str, symbol_name: str, reason: str) -> None:
                metric_skips[metric_name] += 1
                if len(metric_issues[metric_name]) < 10:
                    metric_issues[metric_name].append(
                        {"symbol": symbol_name, "reason": reason}
                    )

            def _collect_for_symbol(symbol: Any) -> None:
                symbol_name = str(getattr(symbol, "name", "") or "")

                if rank_by_value in {"all", "spread"}:
                    spread_row, spread_error = _build_market_scan_spread_row(symbol, mt5_gateway)
                    if spread_error:
                        _record_issue("spread", symbol_name, spread_error)
                    elif spread_row is not None:
                        spread_rows.append(spread_row)

                if needs_bar_data and mt5_timeframe is not None:
                    bar_row, bar_error = _build_market_scan_bar_row(
                        symbol,
                        timeframe=timeframe_value,
                        mt5_timeframe=mt5_timeframe,
                    )
                    if bar_error:
                        if rank_by_value in {"all", "volume"}:
                            _record_issue("volume", symbol_name, bar_error)
                        if rank_by_value in {"all", "price_change"}:
                            _record_issue("price_change", symbol_name, bar_error)
                    elif bar_row is not None:
                        if rank_by_value in {"all", "volume"}:
                            volume_rows.append(dict(bar_row))
                        if rank_by_value in {"all", "price_change"}:
                            price_change_rows.append(dict(bar_row))

            for symbol in selected_symbols:
                symbol_name = str(getattr(symbol, "name", "") or "")
                is_hidden = not bool(getattr(symbol, "visible", False))
                if universe_value == "all" and is_hidden:
                    with _symbol_ready_guard(symbol_name, info_before=symbol) as (err, _):
                        if err:
                            if rank_by_value in {"all", "spread"}:
                                _record_issue("spread", symbol_name, err)
                            if rank_by_value in {"all", "volume"}:
                                _record_issue("volume", symbol_name, err)
                            if rank_by_value in {"all", "price_change"}:
                                _record_issue("price_change", symbol_name, err)
                            continue
                        _collect_for_symbol(symbol)
                    continue
                _collect_for_symbol(symbol)

            spread_rows.sort(
                key=lambda row: (
                    row.get("spread_pct") is None,
                    row.get("spread_pct") if row.get("spread_pct") is not None else float("inf"),
                    row.get("symbol") or "",
                )
            )
            volume_rows.sort(
                key=lambda row: (
                    row.get("tick_volume") is None,
                    -(row.get("tick_volume") or 0),
                    row.get("symbol") or "",
                )
            )
            price_change_rows.sort(
                key=lambda row: (
                    row.get("price_change_pct") is None,
                    -(row.get("price_change_pct") or 0.0),
                    row.get("symbol") or "",
                )
            )

            evaluated_counts = {
                "spread": len(spread_rows),
                "volume": len(volume_rows),
                "price_change": len(price_change_rows),
            }

            spread_rows = spread_rows[:limit_value]
            volume_rows = volume_rows[:limit_value]
            price_change_rows = price_change_rows[:limit_value]

            scan_meta = {
                "success": True,
                "rank_by": rank_by_value,
                "limit": limit_value,
                "universe": universe_value,
                "detail": detail_mode,
                "timeframe": timeframe_value if needs_bar_data else None,
                "timeframe_requested": timeframe_value,
                "timeframe_used": timeframe_value if needs_bar_data else None,
                "scanned_symbols": len(selected_symbols),
                "query_latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            }

            if rank_by_value == "spread":
                out = _market_scan_table(
                    _top_markets_headers("spread", detail_mode=detail_mode),
                    spread_rows,
                    include_contract_meta=detail_mode == "full",
                )
                out.update(scan_meta)
                out["evaluated_symbols"] = evaluated_counts["spread"]
                out["skipped_symbols"] = metric_skips["spread"]
                out["skipped_examples"] = metric_issues["spread"]
                out["ranking"] = "lowest_spread"
                return out

            if rank_by_value == "volume":
                out = _market_scan_table(
                    _top_markets_headers("volume", detail_mode=detail_mode),
                    volume_rows,
                    include_contract_meta=detail_mode == "full",
                )
                out.update(scan_meta)
                out["evaluated_symbols"] = evaluated_counts["volume"]
                out["skipped_symbols"] = metric_skips["volume"]
                out["skipped_examples"] = metric_issues["volume"]
                out["ranking"] = "highest_volume"
                return out

            if rank_by_value == "price_change":
                out = _market_scan_table(
                    _top_markets_headers("price_change", detail_mode=detail_mode),
                    price_change_rows,
                    include_contract_meta=detail_mode == "full",
                )
                out.update(scan_meta)
                out["evaluated_symbols"] = evaluated_counts["price_change"]
                out["skipped_symbols"] = metric_skips["price_change"]
                out["skipped_examples"] = metric_issues["price_change"]
                out["ranking"] = "highest_price_change"
                return attach_collection_contract(
                    out,
                    collection_kind="table",
                    rows=out.get("data"),
                    include_contract_meta=detail_mode == "full",
                )

            results = {
                "lowest_spread": _strip_nested_market_scan_meta(
                    _market_scan_table(
                        _top_markets_headers("spread", detail_mode=detail_mode),
                        spread_rows,
                        include_contract_meta=detail_mode == "full",
                    )
                ),
                "highest_volume": _strip_nested_market_scan_meta(
                    _market_scan_table(
                        _top_markets_headers("volume", detail_mode=detail_mode),
                        volume_rows,
                        include_contract_meta=detail_mode == "full",
                    )
                ),
                "highest_price_change": _strip_nested_market_scan_meta(
                    _market_scan_table(
                        _top_markets_headers("price_change", detail_mode=detail_mode),
                        price_change_rows,
                        include_contract_meta=detail_mode == "full",
                    )
                ),
            }
            return attach_collection_contract(
                {
                    **scan_meta,
                    "results": results,
                    "scan_stats": {
                        "spread": {
                            "evaluated_symbols": evaluated_counts["spread"],
                            "skipped_symbols": metric_skips["spread"],
                            "skipped_examples": metric_issues["spread"],
                        },
                        "volume": {
                            "evaluated_symbols": evaluated_counts["volume"],
                            "skipped_symbols": metric_skips["volume"],
                            "skipped_examples": metric_issues["volume"],
                        },
                        "price_change": {
                            "evaluated_symbols": evaluated_counts["price_change"],
                            "skipped_symbols": metric_skips["price_change"],
                            "skipped_examples": metric_issues["price_change"],
                        },
                    },
                },
                collection_kind="groups",
                groups=results,
                include_contract_meta=detail_mode == "full",
            )
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error scanning top markets: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="symbols_top_markets",
        rank_by=rank_by,
        limit=limit,
        universe=universe,
        timeframe=timeframe,
        detail=detail_mode,
        func=_run,
    )


@mcp.tool()
def market_scan(  # noqa: C901
    symbols: Optional[str] = None,
    symbol: Optional[str] = None,
    group: Optional[str] = None,
    limit: Optional[int] = 20,
    universe: Literal["visible", "all"] = "visible",  # type: ignore
    timeframe: TimeframeLiteral = "H1",
    detail: CompactFullDetailLiteral = "compact",
    lookback: int = 100,
    rsi_length: int = 14,
    sma_period: int = 20,
    min_price_change_pct: Optional[float] = None,
    max_price_change_pct: Optional[float] = None,
    max_spread_pct: Optional[float] = None,
    min_tick_volume: Optional[int] = None,
    rsi_below: Optional[float] = None,
    rsi_above: Optional[float] = None,
    price_vs_sma: Optional[Literal["above", "below"]] = None,  # type: ignore
    rank_by: Literal["abs_price_change_pct", "abs_price_change", "price_change_pct", "price_change", "tick_volume", "volume", "rsi", "spread_pct", "spread"] = "abs_price_change_pct",  # type: ignore
) -> Dict[str, Any]:
    """Scan MT5 symbols with explicit price, spread, volume, RSI, and SMA filters.

    Pass `symbol` for one instrument or `symbols` for a comma-separated list.
    `data.table.rows` is the canonical table payload. Compact detail is the
    default; use `detail="full"` when you also want the explicit `columns`
    ordering hint for compatibility.
    """

    detail_mode = resolve_output_detail(detail=detail, default="compact")

    def _run() -> Dict[str, Any]:  # noqa: C901
        request: Dict[str, Any] = {
            "symbol": symbol,
            "symbols": symbols,
            "group": group,
            "limit": limit,
            "universe": universe,
            "timeframe": timeframe,
            "detail": detail_mode,
            "lookback": lookback,
            "rank_by": rank_by,
            "filters": {
                key: value
                for key, value in {
                    "min_price_change_pct": min_price_change_pct,
                    "max_price_change_pct": max_price_change_pct,
                    "max_spread_pct": max_spread_pct,
                    "min_tick_volume": min_tick_volume,
                    "rsi_below": rsi_below,
                    "rsi_above": rsi_above,
                    "price_vs_sma": price_vs_sma,
                    "rsi_length": rsi_length,
                    "sma_period": sma_period,
                }.items()
                if value is not None
            },
        }
        try:
            universe_value = str(universe or "visible").strip().lower()
            request["universe"] = universe_value
            if universe_value not in {"visible", "all"}:
                return _market_scan_error(
                    "universe must be 'visible' or 'all'.",
                    code="invalid_input",
                    request=request,
                )

            symbol_value = str(symbol or "").strip()
            symbols_value = str(symbols or "").strip()
            if symbol_value and symbols_value:
                return _market_scan_error(
                    "Provide either symbol or symbols, not both.",
                    code="invalid_input",
                    request=request,
                )
            symbols_filter = symbols_value or symbol_value or None
            if symbol_value:
                request["symbols"] = symbol_value
                request["symbol_alias_used"] = True

            timeframe_value = str(timeframe or "H1").strip().upper()
            request["timeframe"] = timeframe_value
            if timeframe_value not in TIMEFRAME_MAP:
                return _market_scan_error(
                    invalid_timeframe_error(timeframe_value, TIMEFRAME_MAP),
                    code="invalid_timeframe",
                    request=request,
                )
            mt5_timeframe = TIMEFRAME_MAP[timeframe_value]

            rank_by_value, rank_by_input = _normalize_market_scan_rank_by(rank_by)
            request["rank_by"] = rank_by_value
            if rank_by_input != rank_by_value:
                request["rank_by_input"] = rank_by_input
            if rank_by_value not in {"abs_price_change_pct", "price_change_pct", "tick_volume", "rsi", "spread_pct"}:
                return _market_scan_error(
                    (
                        "rank_by must be one of: "
                        f"{', '.join(_MARKET_SCAN_RANK_BY_CHOICES)}."
                    ),
                    code="invalid_input",
                    request=request,
                )

            price_vs_sma_value = None
            if price_vs_sma is not None:
                price_vs_sma_value = str(price_vs_sma).strip().lower()
                request["filters"] = {
                    **dict(request.get("filters", {})),
                    "price_vs_sma": price_vs_sma_value,
                }
                if price_vs_sma_value not in {"above", "below"}:
                    return _market_scan_error(
                        "price_vs_sma must be 'above' or 'below'.",
                        code="invalid_input",
                        request=request,
                    )

            try:
                lookback_value = int(lookback)
                rsi_length_value = int(rsi_length)
                sma_period_value = int(sma_period)
            except Exception:
                return _market_scan_error(
                    "lookback, rsi_length, and sma_period must be integers.",
                    code="invalid_input",
                    request=request,
                )
            request["lookback"] = lookback_value
            request["filters"] = {
                **dict(request.get("filters", {})),
                "rsi_length": rsi_length_value,
                "sma_period": sma_period_value,
            }
            if lookback_value < 2:
                return _market_scan_error(
                    "lookback must be at least 2.",
                    code="invalid_input",
                    request=request,
                )
            if rsi_length_value < 1:
                return _market_scan_error(
                    "rsi_length must be at least 1.",
                    code="invalid_input",
                    request=request,
                )
            if sma_period_value < 1:
                return _market_scan_error(
                    "sma_period must be at least 1.",
                    code="invalid_input",
                    request=request,
                )

            required_lookback = 2
            if rank_by_value == "rsi" or rsi_above is not None or rsi_below is not None:
                required_lookback = max(required_lookback, rsi_length_value + 1)
            if price_vs_sma_value is not None:
                required_lookback = max(required_lookback, sma_period_value)
            if lookback_value < required_lookback:
                return _market_scan_error(
                    (
                        f"lookback={lookback_value} is too small for the requested filters; "
                        f"need at least {required_lookback} bars."
                    ),
                    code="invalid_input",
                    request=request,
                )

            mt5_gateway = get_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()

            raw_symbols = mt5_gateway.symbols_get()
            if raw_symbols is None:
                return _market_scan_error(
                    f"Failed to get symbols: {mt5_gateway.last_error()}",
                    code="data_fetch_failed",
                    request=request,
                )
            all_symbols = list(raw_symbols)

            selected_symbols, selection_meta, selection_error = _select_market_scan_symbols(
                all_symbols,
                symbols=symbols_filter,
                group=group,
                universe=universe_value,
            )
            if selection_error:
                error_code = "invalid_input"
                if group and selection_meta.get("symbols_input") is None:
                    error_code = "symbol_group_error"
                return _market_scan_error(
                    selection_error,
                    code=error_code,
                    request=request,
                )

            limit_value = _normalize_limit(limit) or 20
            request["limit"] = limit_value
            if selection_meta.get("symbols_input") is not None:
                request["symbols_input"] = selection_meta.get("symbols_input")
            request["scope"] = selection_meta.get("scope")
            if selection_meta.get("group") is not None:
                request["group"] = selection_meta.get("group")
            if selection_meta.get("requested_symbols") is not None:
                request["requested_symbols"] = selection_meta.get("requested_symbols")
            if selection_meta.get("missing_symbols") is not None:
                request["missing_symbols"] = selection_meta.get("missing_symbols")
            started_at = time.perf_counter()
            matched_rows: List[Dict[str, Any]] = []
            skipped_examples: List[Dict[str, str]] = []
            skipped_symbols = 0
            evaluated_symbols = 0

            def _record_issue(symbol_name: str, reason: str) -> None:
                nonlocal skipped_symbols
                skipped_symbols += 1
                if len(skipped_examples) < 10:
                    skipped_examples.append({"symbol": symbol_name, "reason": reason})

            for missing_symbol in selection_meta.get("missing_symbols", []):
                if len(skipped_examples) < 10:
                    skipped_examples.append({"symbol": missing_symbol, "reason": "Requested symbol not found."})

            def _evaluate_symbol(symbol_obj: Any) -> None:
                nonlocal evaluated_symbols
                symbol_name = str(getattr(symbol_obj, "name", "") or "")

                spread_row, spread_error = _build_market_scan_spread_row(symbol_obj, mt5_gateway)
                if spread_error or spread_row is None:
                    _record_issue(symbol_name, spread_error or "Spread data is unavailable.")
                    return

                signal_row, signal_error = _build_market_scan_signal_row(
                    symbol_obj,
                    timeframe=timeframe_value,
                    mt5_timeframe=mt5_timeframe,
                    lookback=lookback_value,
                    rsi_length=rsi_length_value,
                    sma_period=sma_period_value,
                )
                if signal_error or signal_row is None:
                    _record_issue(symbol_name, signal_error or "Bar data is unavailable.")
                    return

                row = dict(spread_row)
                row.update(signal_row)
                metric_error = _market_scan_missing_required_metric(
                    row,
                    rank_by=rank_by_value,
                    rsi_above=rsi_above,
                    rsi_below=rsi_below,
                    price_vs_sma=price_vs_sma_value,
                    max_spread_pct=max_spread_pct,
                    min_tick_volume=min_tick_volume,
                    min_price_change_pct=min_price_change_pct,
                    max_price_change_pct=max_price_change_pct,
                    rsi_length=rsi_length_value,
                    sma_period=sma_period_value,
                )
                if metric_error:
                    _record_issue(symbol_name, metric_error)
                    return

                evaluated_symbols += 1
                if not _market_scan_row_matches_filters(
                    row,
                    min_price_change_pct=min_price_change_pct,
                    max_price_change_pct=max_price_change_pct,
                    max_spread_pct=max_spread_pct,
                    min_tick_volume=min_tick_volume,
                    rsi_below=rsi_below,
                    rsi_above=rsi_above,
                    price_vs_sma=price_vs_sma_value,
                ):
                    return
                matched_rows.append(row)

            for symbol_obj in selected_symbols:
                symbol_name = str(getattr(symbol_obj, "name", "") or "")
                is_hidden = not bool(getattr(symbol_obj, "visible", False))
                if is_hidden:
                    with _symbol_ready_guard(symbol_name, info_before=symbol_obj) as (err, _):
                        if err:
                            _record_issue(symbol_name, err)
                            continue
                        _evaluate_symbol(symbol_obj)
                    continue
                _evaluate_symbol(symbol_obj)

            _market_scan_sort_rows(
                matched_rows,
                rank_by=rank_by_value,
                rsi_above=rsi_above,
                rsi_below=rsi_below,
            )
            total_matches = len(matched_rows)
            limited_rows = matched_rows[:limit_value]

            headers = [
                "symbol",
                "group",
                "description",
                "timeframe",
                "bar_time",
                "bar_age_hours",
                "data_stale",
                "stale_warning",
                "close",
                "price_change_pct",
                "tick_volume",
                "spread_pct",
                "rsi",
                "sma_value",
                "sma_distance_pct",
            ]
            request["filters"] = {
                key: value
                for key, value in {
                    "min_price_change_pct": min_price_change_pct,
                    "max_price_change_pct": max_price_change_pct,
                    "max_spread_pct": max_spread_pct,
                    "min_tick_volume": min_tick_volume,
                    "rsi_below": rsi_below,
                    "rsi_above": rsi_above,
                    "price_vs_sma": price_vs_sma_value,
                    "rsi_length": rsi_length_value
                    if (rsi_above is not None or rsi_below is not None or rank_by_value == "rsi")
                    else None,
                    "sma_period": sma_period_value
                    if price_vs_sma_value is not None
                    else None,
                }.items()
                if value is not None
            }
            stats = {
                "scanned_symbols": len(selected_symbols),
                "evaluated_symbols": evaluated_symbols,
                "matched_symbols": total_matches,
                "filtered_out_symbols": max(0, evaluated_symbols - total_matches),
                "skipped_symbols": skipped_symbols,
                "skipped_examples": skipped_examples,
                "query_latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            }
            out: Dict[str, Any] = {
                "success": True,
                "data": {
                    "table": _market_scan_contract_table(
                        headers,
                        limited_rows,
                        include_columns=detail_mode == "full",
                    ),
                },
                "summary": {
                    "counts": {
                        "scanned_symbols": int(stats["scanned_symbols"]),
                        "evaluated_symbols": int(stats["evaluated_symbols"]),
                        "matched_symbols": int(stats["matched_symbols"]),
                        "filtered_out_symbols": int(stats["filtered_out_symbols"]),
                        "skipped_symbols": int(stats["skipped_symbols"]),
                    }
                },
                "meta": _market_scan_contract_meta(request=request, stats=stats),
            }
            if total_matches == 0:
                out["summary"]["empty"] = True
                out["message"] = "No symbols matched the requested market scan filters."
            return attach_collection_contract(
                out,
                collection_kind="table",
                rows=limited_rows,
                include_contract_meta=detail_mode == "full",
            )
        except MT5ConnectionError as exc:
            return _market_scan_error(
                str(exc),
                code="mt5_connection_error",
                request=request,
            )
        except Exception as exc:
            return _market_scan_error(
                f"Error running market scan: {str(exc)}",
                code="market_scan_failed",
                request=request,
            )

    return run_logged_operation(
        logger,
        operation="market_scan",
        symbols=symbols,
        group=group,
        limit=limit,
        universe=universe,
        timeframe=timeframe,
        func=_run,
    )
