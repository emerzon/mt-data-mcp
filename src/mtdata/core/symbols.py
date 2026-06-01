
import logging
import math
import re
import time
from typing import Any, Dict, List, Literal, Optional

from ..shared.constants import (
    DEFAULT_ROW_LIMIT,
    GROUP_SEARCH_THRESHOLD,
    TIMEFRAME_MAP,
    TIMEFRAME_SECONDS,
)
from ..shared.schema import CompactFullDetailLiteral, TimeframeLiteral
from ..shared.validators import invalid_timeframe_error
from ..utils.freshness import format_age_seconds, format_freshness_label
from ..utils.mt5 import (
    MT5ConnectionError,
    _mt5_copy_rates_from_pos,
    _symbol_ready_guard,
    ensure_mt5_connection_or_raise,
    mt5,
)
from ..utils.mt5_enums import decode_mt5_bitmask_labels, decode_mt5_enum_label
from ..utils.symbol import (
    _extract_group_path as _extract_group_path_util,
)
from ..utils.symbol import (
    _normalize_group_path_query,
)
from ..utils.utils import _format_time_minimal, _normalize_limit, _table_from_rows
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway
from .output_contract import (
    attach_collection_contract,
    normalize_output_detail,
    normalize_output_verbosity_detail,
    resolve_output_contract,
)
from .quote_freshness import QUOTE_STALE_SECONDS, quote_closed_session_context

logger = logging.getLogger(__name__)
_MARKET_SCAN_STALE_BAR_SECONDS = 7 * 24 * 60 * 60
_MARKET_SCAN_STALE_QUOTE_SECONDS = QUOTE_STALE_SECONDS
_FOREX_CURRENCY_CODES = {
    "AUD",
    "CAD",
    "CHF",
    "EUR",
    "GBP",
    "JPY",
    "NZD",
    "USD",
}


def _case_insensitive_sort_key(value: Any) -> tuple[str, str]:
    text = str(value or "").strip()
    return text.casefold(), text


def _normalize_symbol_search_term(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _nonempty_symbol_string(value: Any) -> Optional[str]:
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
    "price_change_pct",
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
    "currency_base_inferred",
    "currency_base_warning",
    "currency_profit",
    "time",
    "freshness",
    "market_status",
    "market_status_reason",
    "note",
    "warning",
    "price_change_pct",
    "price_change_pct_unit",
    "digits",
    "point",
    "trade_contract_size",
    "trade_tick_size",
    "trade_tick_value",
    "volume_min",
    "volume_max",
    "volume_step",
    "spread_float",
    "trade_mode_label",
    "order_mode_labels",
)

_SYMBOL_DESCRIBE_SUMMARY_DIRECT_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "currency_base",
    "currency_base_inferred",
    "currency_base_warning",
    "currency_profit",
    "time",
    "freshness",
    "market_status",
    "market_status_reason",
    "note",
    "warning",
    "price_change_pct",
    "price_change_pct_unit",
    "trade_mode_label",
    "order_mode_labels",
)

_COMMON_CRYPTO_BASES = (
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "LTC",
    "BCH",
    "ADA",
    "DOT",
    "DOGE",
    "BNB",
    "AVAX",
    "LINK",
    "XLM",
    "TRX",
    "UNI",
    "USDC",
    "USDT",
)

_SYMBOL_SEARCH_MODES = frozenset(
    {"auto", "name", "description", "group", "exact", "all"}
)


def _symbols_from_groups(
    groups: Dict[str, List[Any]],
    group_names: List[str],
) -> List[Any]:
    matched: List[Any] = []
    for group_name in group_names:
        matched.extend(groups[group_name])
    return matched


def _match_symbols_for_search(
    all_symbols: List[Any],
    search_term: str,
    search_mode: str,
) -> List[Any]:
    search_upper = search_term.upper()
    groups: Dict[str, List[Any]] = {}
    symbol_name_matches: List[Any] = []
    description_matches: List[Any] = []
    all_field_matches: List[Any] = []

    for symbol in all_symbols:
        group_path = _extract_group_path_util(symbol)
        groups.setdefault(group_path, []).append(symbol)

        symbol_name = str(getattr(symbol, "name", "") or "")
        description = str(getattr(symbol, "description", "") or "")
        name_hit = search_upper in symbol_name.upper()
        description_hit = search_upper in description.upper()
        group_hit = search_upper in str(group_path or "").upper()

        if search_mode == "exact":
            if symbol_name.upper() == search_upper:
                symbol_name_matches.append(symbol)
        elif name_hit:
            symbol_name_matches.append(symbol)
        if description_hit:
            description_matches.append(symbol)
        if name_hit or description_hit or group_hit:
            all_field_matches.append(symbol)

    matching_groups = [
        group_name
        for group_name in groups.keys()
        if search_upper in group_name.upper()
    ]

    if search_mode in {"exact", "name"}:
        return symbol_name_matches
    if search_mode == "description":
        return description_matches
    if search_mode == "group":
        return _symbols_from_groups(groups, matching_groups)
    if search_mode == "all":
        return all_field_matches

    if matching_groups and len(matching_groups) <= GROUP_SEARCH_THRESHOLD:
        return _symbols_from_groups(groups, matching_groups)
    if symbol_name_matches:
        return symbol_name_matches
    if matching_groups:
        return _symbols_from_groups(groups, matching_groups)
    return all_field_matches


_COMMON_QUOTE_CURRENCIES = (
    "USD",
    "USDT",
    "USDC",
    "EUR",
    "GBP",
    "JPY",
    "CHF",
    "AUD",
    "CAD",
    "NZD",
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


def _infer_symbol_base_from_name(symbol_name: Any, quote_currency: Any) -> Optional[str]:
    name = re.sub(r"[^A-Z0-9]", "", str(symbol_name or "").upper())
    quote = str(quote_currency or "").strip().upper()
    if not name or quote not in _COMMON_QUOTE_CURRENCIES:
        return None
    if not name.endswith(quote):
        return None
    base = name[: -len(quote)]
    for crypto_base in _COMMON_CRYPTO_BASES:
        if base == crypto_base or base.endswith(crypto_base):
            return crypto_base
    return None


def _add_symbol_currency_diagnostics(symbol_data: Dict[str, Any]) -> None:
    currency_base = str(symbol_data.get("currency_base") or "").strip().upper()
    currency_profit = str(symbol_data.get("currency_profit") or "").strip().upper()
    if not currency_base or not currency_profit or currency_base != currency_profit:
        return
    inferred_base = _infer_symbol_base_from_name(symbol_data.get("name"), currency_profit)
    if not inferred_base or inferred_base == currency_base:
        return
    symbol_data["currency_base_inferred"] = inferred_base
    symbol_data["currency_base_warning"] = (
        "MT5 reports identical currency_base and currency_profit; verify broker metadata."
    )


def _compact_symbol_describe_payload(symbol_data: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for field in _SYMBOL_DESCRIBE_COMPACT_DIRECT_FIELDS:
        _copy_symbol_describe_field(compact, symbol_data, field)

    _apply_symbol_currency_diagnostics(compact)

    if "time_epoch" in symbol_data:
        compact["time_epoch"] = symbol_data["time_epoch"]
    return compact


def _summary_symbol_describe_payload(symbol_data: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for field in _SYMBOL_DESCRIBE_SUMMARY_DIRECT_FIELDS:
        _copy_symbol_describe_field(summary, symbol_data, field)

    _apply_symbol_currency_diagnostics(summary)
    return summary


def _apply_symbol_currency_diagnostics(payload: Dict[str, Any]) -> None:
    inferred_base = payload.get("currency_base_inferred")
    reported_base = payload.get("currency_base")
    if inferred_base and payload.get("currency_base_warning"):
        payload["currency_base_reported"] = reported_base
        payload["currency_base_source"] = "reported_by_mt5"
        payload["currency_base_inference_source"] = "inferred_from_symbol_name"


def _symbol_session_type(
    *,
    name: Any,
    group: Any = None,
    description: Any = None,
) -> Optional[str]:
    text = " ".join(
        str(value or "").upper()
        for value in (name, group, description)
        if value not in (None, "")
    )
    if any(token in text for token in ("-24", "24HR", "24/5", "24H")):
        return "extended_24h"
    if "." in str(name or "") and any(token in text for token in ("STOCK", "CFD")):
        return "regular"
    return None


def _symbol_suggestion_from_info(symbol_info: Any) -> Dict[str, Any]:
    group = _extract_group_path_util(symbol_info)
    description = getattr(symbol_info, "description", None)
    suggestion: Dict[str, Any] = {
        "symbol": getattr(symbol_info, "name", None),
        "group": group,
    }
    if description not in (None, ""):
        suggestion["description"] = description
    session_type = _symbol_session_type(
        name=getattr(symbol_info, "name", None),
        group=group,
        description=description,
    )
    if session_type is not None:
        suggestion["session_type"] = session_type
    return {key: value for key, value in suggestion.items() if value not in (None, "")}


def _symbol_list_optional_attr(symbol_info: Any, attr: str) -> Any:
    try:
        if attr not in dir(symbol_info):
            return None
        value = getattr(symbol_info, attr)
    except Exception:
        return None
    if callable(value) or value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _find_symbol_suggestions(
    mt5_gateway: Any,
    query: str,
    *,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    text = str(query or "").strip()
    if not text:
        return []
    query_upper = text.upper()
    try:
        symbols = list(mt5_gateway.symbols_get() or [])
    except Exception:
        return []
    matches = []
    for symbol_info in symbols:
        name = str(getattr(symbol_info, "name", "") or "")
        description = str(getattr(symbol_info, "description", "") or "")
        group = str(_extract_group_path_util(symbol_info) or "")
        searchable = f"{name} {description} {group}".upper()
        if query_upper in searchable:
            matches.append(symbol_info)
    matches.sort(
        key=lambda info: (
            not str(getattr(info, "name", "") or "").upper().startswith(query_upper),
            *_case_insensitive_sort_key(getattr(info, "name", "")),
        )
    )
    return [_symbol_suggestion_from_info(symbol) for symbol in matches[: max(1, int(limit))]]


@mcp.tool()
def symbols_list(  # noqa: C901
    search_term: Optional[str] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
    offset: int = 0,
    list_mode: Literal["symbols", "groups"] = "symbols",  # type: ignore
    search_mode: Literal[  # type: ignore
        "auto",
        "name",
        "description",
        "group",
        "exact",
        "all",
    ] = "auto",
    detail: CompactFullDetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """List symbols or symbol groups."""
    normalized_search_term = _normalize_symbol_search_term(search_term)
    detail_mode = normalize_output_detail(detail, default="compact")
    search_mode_value = str(search_mode or "auto").strip().lower()

    def _run() -> Dict[str, Any]:
        try:
            mt5_gateway = create_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            mode = str(list_mode or "symbols").strip().lower()
            if mode not in ("symbols", "groups"):
                return {"error": "list_mode must be 'symbols' or 'groups'."}
            if search_mode_value not in _SYMBOL_SEARCH_MODES:
                return {
                    "error": (
                        "search_mode must be one of auto, name, description, "
                        "group, exact, or all."
                    )
                }
            if mode == "groups":
                return _list_symbol_groups(
                    search_term=normalized_search_term,
                    limit=limit,
                    offset=offset,
                    mt5_gateway=mt5_gateway,
                    detail=detail_mode,
                )

            matched_symbols = []

            if normalized_search_term:
                all_symbols = mt5_gateway.symbols_get()
                if all_symbols is None:
                    return {"error": f"Failed to get symbols: {mt5_gateway.last_error()}"}
                matched_symbols = _match_symbols_for_search(
                    list(all_symbols),
                    normalized_search_term,
                    search_mode_value,
                )
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
                row = {
                    "symbol": symbol.name,
                    "group": _extract_group_path_util(symbol),
                    "description": symbol.description,
                    "visible": bool(getattr(symbol, "visible", False)),
                    "session_type": _symbol_session_type(
                        name=symbol.name,
                        group=_extract_group_path_util(symbol),
                        description=symbol.description,
                    ),
                }
                for attr in (
                    "currency_base",
                    "currency_profit",
                    "digits",
                    "spread_float",
                ):
                    value = _symbol_list_optional_attr(symbol, attr)
                    if value is not None:
                        row[attr] = value
                symbol_list.append(row)

            limit_value = _normalize_limit(limit)
            try:
                offset_value = int(offset or 0)
            except Exception:
                return {"error": "offset must be a non-negative integer."}
            if offset_value < 0:
                return {"error": "offset must be >= 0."}
            total_count = len(symbol_list)
            if offset_value:
                symbol_list = symbol_list[offset_value:]
            if limit_value:
                symbol_list = symbol_list[:limit_value]
            has_more = offset_value + len(symbol_list) < total_count
            if detail_mode == "summary":
                out = {
                    "success": True,
                    "list_mode": "symbols",
                    "count": len(symbol_list),
                    "search_term": normalized_search_term,
                    "search_mode": search_mode_value,
                    "limit": limit_value,
                }
                if offset_value or has_more:
                    out["total_count"] = total_count
                    out["offset"] = offset_value
                    out["has_more"] = has_more
                return out
            if detail_mode == "compact":
                headers = ["symbol", "group", "description"]
                for optional_header in (
                    "currency_base",
                    "currency_profit",
                    "digits",
                    "spread_float",
                ):
                    if any(s.get(optional_header) is not None for s in symbol_list):
                        headers.append(optional_header)
                if any(s.get("session_type") for s in symbol_list):
                    headers.append("session_type")
                rows = [[s.get(header) for header in headers] for s in symbol_list]
            elif detail_mode == "full":
                headers = ["symbol", "group", "description", "visible"]
                rows = [
                    [s["symbol"], s["group"], s["description"], s["visible"]]
                    for s in symbol_list
                ]
            else:
                headers = ["symbol", "group", "description"]
                rows = [[s["symbol"], s["group"], s["description"]] for s in symbol_list]
            result = _table_from_rows(headers, rows)
            if offset_value or has_more:
                result["total_count"] = total_count
                result["offset"] = offset_value
                result["limit"] = limit_value
                result["has_more"] = has_more
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
        offset=offset,
        list_mode=list_mode,
        search_mode=search_mode_value,
        detail=detail_mode,
        func=_run,
    )

def _list_symbol_groups(
    search_term: Optional[str] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
    offset: int = 0,
    mt5_gateway: Any = None,
    detail: CompactFullDetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """List group paths as a tabular result with a single column: group."""
    try:
        gateway = mt5_gateway or create_mt5_gateway(
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
        try:
            offset_value = int(offset or 0)
        except Exception:
            return {"error": "offset must be a non-negative integer."}
        if offset_value < 0:
            return {"error": "offset must be >= 0."}
        total_count = len(filtered_items)
        if offset_value:
            filtered_items = filtered_items[offset_value:]
        if limit_value:
            filtered_items = filtered_items[:limit_value]
        has_more = offset_value + len(filtered_items) < total_count

        detail_mode = normalize_output_detail(detail, default="compact")
        if detail_mode == "summary":
            out = {
                "success": True,
                "list_mode": "groups",
                "count": len(filtered_items),
                "search_term": search_term,
                "limit": limit_value,
            }
            if offset_value or has_more:
                out["total_count"] = total_count
                out["offset"] = offset_value
                out["has_more"] = has_more
            return out
        if detail_mode in {"standard", "full"}:
            rows = [[name, meta["count"]] for name, meta in filtered_items]
            result = _table_from_rows(["group", "count"], rows)
        else:
            group_names = [name for name, _ in filtered_items]
            rows = [[g] for g in group_names]
            result = _table_from_rows(["group"], rows)
        if offset_value or has_more:
            result["total_count"] = total_count
            result["offset"] = offset_value
            result["limit"] = limit_value
            result["has_more"] = has_more
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
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    """Return symbol information as JSON for `symbol`.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., "EURUSD")
    detail : str, optional (default="compact")
        Output verbosity level:
        - "summary": Symbol identity, currencies, quote freshness, and session/trade labels
        - "compact": Essential fields only (identifier, volume limits, contract size, tick size/value)
        - "standard": Same concise field set as compact for this single-symbol metadata tool
        - "full": Complete metadata including all trading modes, swap details, and session times
    Returns:
    --------
    dict
        Symbol identifier plus requested detail fields
    """
    def _run() -> Dict[str, Any]:
        try:
            contract = resolve_output_contract(
                detail=detail,
                default_detail="compact",
            )
            mt5_gateway = create_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            symbol_info = mt5_gateway.symbol_info(symbol)
            if symbol_info is None:
                suggestions = _find_symbol_suggestions(mt5_gateway, symbol)
                details: Dict[str, Any] = {
                    "symbol": symbol,
                    "search_hint": f"Use symbols_list(search_term='{symbol}') to browse matching broker symbols.",
                }
                if suggestions:
                    details["did_you_mean"] = suggestions
                return build_error_payload(
                    f"Symbol '{symbol}' not found in MT5 terminal.",
                    code="symbol_not_found",
                    operation="symbols_describe",
                    details=details,
                )

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
                        if contract.shape_detail == "full":
                            symbol_data["time_epoch"] = utc_epoch
                        symbol_data["time"] = _format_time_minimal(utc_epoch)
                        symbol_data.update(
                            {
                                key: value
                                for key, value in _quote_staleness_fields(
                                    utc_epoch,
                                    symbol=(
                                        _nonempty_symbol_string(
                                            getattr(symbol_info, "name", None)
                                        )
                                        or symbol
                                    ),
                                ).items()
                                if key
                                in {
                                    "data_age_seconds",
                                    "data_age",
                                    "data_stale",
                                    "freshness",
                                    "stale_after_seconds",
                                    "market_status",
                                    "market_status_reason",
                                    "market_status_source",
                                    "note",
                                    "warning",
                                }
                            }
                        )
                    except Exception:
                        if contract.shape_detail == "full":
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

            price_change_value = _market_scan_float(symbol_data.get("price_change"))
            if price_change_value is not None:
                symbol_data["price_change_pct"] = _market_scan_round(
                    price_change_value,
                    digits=6,
                )
                symbol_data["price_change_pct_unit"] = "percentage_points (1.0 = 1%)"
            else:
                session_open = _market_scan_float(symbol_data.get("session_open"))
                session_close = _market_scan_float(symbol_data.get("session_close"))
                if (
                    session_open is not None
                    and session_close is not None
                    and abs(session_open) > 1e-12
                ):
                    symbol_data["price_change_pct"] = _market_scan_round(
                        ((session_close - session_open) / abs(session_open)) * 100.0,
                        digits=6,
                    )
                    symbol_data["price_change_pct_unit"] = "percentage_points (1.0 = 1%)"

            _add_symbol_currency_diagnostics(symbol_data)
            if contract.detail == "summary":
                symbol_data = _summary_symbol_describe_payload(symbol_data)
            elif contract.shape_detail == "compact":
                symbol_data = _compact_symbol_describe_payload(symbol_data)

            symbol_name = _nonempty_symbol_string(symbol_data.pop("name", None))
            return {
                "success": True,
                "symbol": symbol_name or _nonempty_symbol_string(symbol) or symbol,
                "timezone": "UTC",
                "details": symbol_data,
            }
        except MT5ConnectionError as exc:
            return build_error_payload(
                str(exc),
                code="mt5_connection_error",
                operation="symbols_describe",
            )
        except Exception as exc:
            return build_error_payload(
                f"Error getting symbol info: {str(exc)}",
                code="symbols_describe_failed",
                operation="symbols_describe",
            )

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


def _market_scan_stale_bar_seconds(timeframe: Optional[str]) -> int:
    seconds = TIMEFRAME_SECONDS.get(str(timeframe or "").strip().upper())
    if seconds:
        return max(1, int(seconds) * 2)
    return int(_MARKET_SCAN_STALE_BAR_SECONDS)


def _market_scan_freshness_fields(
    bar_time: Optional[float],
    *,
    timeframe: Optional[str] = None,
    symbol: Any = None,
) -> Dict[str, Any]:
    if bar_time is None:
        return {}
    try:
        now_epoch = float(time.time())
        age_seconds = max(0.0, now_epoch - float(bar_time))
    except Exception:
        return {}
    stale_after_seconds = _market_scan_stale_bar_seconds(timeframe)
    data_stale = age_seconds > stale_after_seconds
    closed_session = quote_closed_session_context(symbol, now_epoch=now_epoch)
    if data_stale and closed_session:
        data_stale = False
    fields: Dict[str, Any] = {
        "data_freshness_seconds": _market_scan_round(age_seconds, digits=3),
        "stale_after_seconds": stale_after_seconds,
        "bar_age_hours": _market_scan_round(age_seconds / 3600.0, digits=3),
        "data_stale": data_stale,
        "freshness": format_freshness_label(
            data_stale=data_stale,
            age_seconds=age_seconds,
            item="bar",
        ),
    }
    if closed_session:
        fields.update(closed_session)
        fields["freshness"] = format_freshness_label(
            data_stale=False,
            market_status=fields.get("market_status"),
            market_status_reason=fields.get("market_status_reason"),
            age_seconds=age_seconds,
            item="bar",
        )
    if fields["data_stale"]:
        fields["stale_warning"] = (
            "Completed bar data may be stale; latest bar is "
            f"{fields['bar_age_hours']} hours old."
        )
    return fields


def _quote_staleness_fields(
    tick_time: Optional[float],
    *,
    symbol: Any = None,
) -> Dict[str, Any]:
    if tick_time is None:
        return {}
    try:
        now_epoch = float(time.time())
        age_seconds = max(0.0, now_epoch - float(tick_time))
    except Exception:
        return {}
    fields: Dict[str, Any] = {
        "data_age_seconds": _market_scan_round(age_seconds, digits=3),
        "data_age": format_age_seconds(age_seconds),
        "stale_after_seconds": int(_MARKET_SCAN_STALE_QUOTE_SECONDS),
    }
    closed_session = quote_closed_session_context(symbol, now_epoch=now_epoch)
    if closed_session:
        fields["data_stale"] = False
        fields.update(closed_session)
        fields["freshness"] = format_freshness_label(
            data_stale=False,
            market_status=fields.get("market_status"),
            market_status_reason=fields.get("market_status_reason"),
            age_seconds=age_seconds,
            item="tick",
        )
        return fields
    fields["data_stale"] = age_seconds > float(_MARKET_SCAN_STALE_QUOTE_SECONDS)
    fields["freshness"] = format_freshness_label(
        data_stale=fields["data_stale"],
        age_seconds=age_seconds,
        item="tick",
    )
    if fields["data_stale"]:
        fields["warning"] = (
            "Live quote timestamp is older than "
            f"{int(_MARKET_SCAN_STALE_QUOTE_SECONDS)} seconds."
        )
    return fields


def _market_scan_quote_freshness_fields(
    tick_time: Optional[float],
    *,
    symbol: Any = None,
) -> Dict[str, Any]:
    if tick_time is None:
        return {}
    return {
        "tick_time": _format_time_minimal(tick_time),
        **_quote_staleness_fields(tick_time, symbol=symbol),
    }


def _market_scan_points_per_pip(symbol: Any, *, point: float, digits: int) -> Optional[float]:
    path = str(getattr(symbol, "path", "") or "").casefold()
    name_letters = re.sub(r"[^A-Z]", "", str(getattr(symbol, "name", "") or "").upper())
    pair_prefix = name_letters[:6]
    is_currency_pair = (
        len(pair_prefix) == 6
        and pair_prefix[:3] in _FOREX_CURRENCY_CODES
        and pair_prefix[3:] in _FOREX_CURRENCY_CODES
    )
    if not is_currency_pair and "forex" not in path and "\\fx" not in path and "/fx" not in path:
        return None

    if digits in {3, 5}:
        return 10.0
    if digits in {2, 4}:
        return 1.0
    if point in {0.00001, 0.001}:
        return 10.0
    if point in {0.0001, 0.01}:
        return 1.0
    return None


def _build_market_scan_spread_row(
    symbol: Any,
    mt5_gateway: Any,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    tick = mt5_gateway.symbol_info_tick(symbol.name)
    if tick is None:
        return None, f"Failed to get tick data: {mt5_gateway.last_error()}"

    bid = _market_scan_float(getattr(tick, "bid", None))
    ask = _market_scan_float(getattr(tick, "ask", None))
    tick_time = _market_scan_float(getattr(tick, "time", None))
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
    points_per_pip = _market_scan_points_per_pip(symbol, point=point, digits=digits)
    spread_pips = (
        (spread_points / points_per_pip)
        if spread_points is not None and points_per_pip is not None and points_per_pip > 0
        else None
    )
    spread_pct = ((spread_abs / mid) * 100.0) if mid > 0 else None
    spread_cost_per_lot = None
    spread_cost_currency = str(
        getattr(symbol, "currency_profit", None)
        or getattr(symbol, "currency_margin", None)
        or ""
    ).strip() or None
    pricing_basis = "quote_only"
    if tick_size > 0 and tick_value > 0:
        spread_cost_per_lot = (spread_abs / tick_size) * tick_value
        pricing_basis = "per_1_lot_estimate"

    row = _market_scan_base_row(symbol)
    row.update(
        {
            "bid": _market_scan_round(bid, digits=digits),
            "ask": _market_scan_round(ask, digits=digits),
            **_market_scan_quote_freshness_fields(tick_time, symbol=symbol.name),
            "spread": _market_scan_round(spread_abs, digits=digits),
            "spread_points": _market_scan_round(spread_points, digits=4),
            "spread_pips": _market_scan_round(spread_pips, digits=4),
            "spread_pct": _market_scan_round(spread_pct, digits=6),
            "spread_cost_per_lot": _market_scan_round(spread_cost_per_lot, digits=6),
            "pricing_basis": pricing_basis,
        }
    )
    if spread_cost_per_lot is not None and spread_cost_currency:
        row["spread_cost_currency"] = spread_cost_currency
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
            "data_source": f"{timeframe}_bars",
            "time": _format_time_minimal(bar_time) if bar_time is not None else None,
            **_market_scan_freshness_fields(bar_time, timeframe=timeframe, symbol=symbol.name),
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
        columns = [str(header) for header in headers]
        seen = set(columns)
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in row.keys():
                column = str(key)
                if column not in seen:
                    columns.append(column)
                    seen.add(column)
        out["columns"] = columns
    return out


def _project_market_scan_rows(
    headers: List[str],
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    projected: List[Dict[str, Any]] = []
    for row in rows:
        out = {header: row.get(header) for header in headers}
        projected.append(out)
    return projected


_MARKET_SCAN_UNITS = {
    "close": "price",
    "price_change_pct": "percentage_points (1.0 = 1%)",
    "tick_volume": "broker_tick_count",
    "real_volume": "traded_volume",
    "spread_points": "broker_points",
    "spread_pips": "pips",
    "spread_pct": "percentage_points (1.0 = 1%)",
    "spread_cost_per_lot": "currency_per_lot_estimate",
    "rsi": "0_100",
    "sma_distance_pct": "percentage_points (1.0 = 1%)",
    "data_age_seconds": "seconds",
    "data_freshness_seconds": "seconds",
    "stale_after_seconds": "seconds",
    "bar_age_hours": "hours",
}


def _market_scan_units_for_rows(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    seen_fields = {
        str(key)
        for row in rows
        if isinstance(row, dict)
        for key, value in row.items()
        if value is not None
    }
    return {
        key: unit
        for key, unit in _MARKET_SCAN_UNITS.items()
        if key in seen_fields
    }


def _attach_top_markets_units(
    out: Dict[str, Any],
    *row_groups: List[Dict[str, Any]],
) -> None:
    rows = [
        row
        for group in row_groups
        for row in group
        if isinstance(row, dict)
    ]
    units = _market_scan_units_for_rows(rows)
    if units:
        out["units"] = units


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
    out = build_error_payload(
        message,
        code=str(code),
        operation="market_scan",
    )
    out["meta"] = _market_scan_contract_meta(request=request, stats=stats)
    if details not in (None, [], {}):
        out["details"] = details
    if warnings:
        out["warnings"] = warnings
    return out


def _market_scan_freshness_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    stale_count = sum(1 for row in rows if bool(row.get("data_stale")))
    row_count = len(rows)
    if stale_count == row_count:
        freshness = "stale"
    elif stale_count:
        freshness = f"mixed, {stale_count}/{row_count} stale"
    else:
        freshness = "fresh"

    out: Dict[str, Any] = {
        "freshness": freshness,
        "stale_rows": int(stale_count),
    }
    if stale_count:
        out["stale_symbols"] = [
            str(row.get("symbol"))
            for row in rows
            if bool(row.get("data_stale")) and str(row.get("symbol") or "").strip()
        ]
    row_times = [
        str(row.get("time") or "").strip()
        for row in rows
        if str(row.get("time") or "").strip()
    ]
    if row_times:
        out["data_as_of"] = max(row_times)

    now_epoch = time.time()
    closed_count = sum(
        1
        for row in rows
        if quote_closed_session_context(row.get("symbol"), now_epoch=now_epoch)
    )
    if closed_count == row_count:
        out["session_status"] = "closed_weekend"
    elif closed_count:
        out["session_status"] = f"mixed, {closed_count}/{row_count} closed_weekend"
    return out


_TOP_MARKETS_COMPACT_BASE_HEADERS = [
    "symbol",
    "group",
    "timeframe",
    "data_source",
    "time",
    "data_stale",
    "freshness",
]

_TOP_MARKETS_COMPACT_SPREAD_HEADERS = [
    "bid",
    "ask",
    "spread_pct",
    "spread_points",
    "spread_pips",
]

_TOP_MARKETS_COMPACT_BAR_HEADERS = [
    "close",
    "tick_volume",
    "price_change_pct",
]

_TOP_MARKETS_COMPACT_HEADERS = [
    *_TOP_MARKETS_COMPACT_BASE_HEADERS,
    "close",
    "bid",
    "ask",
    "spread_pct",
    "spread_points",
    "tick_volume",
    "price_change_pct",
]

_TOP_MARKETS_FULL_BASE_HEADERS = [
    "symbol",
    "group",
    "description",
    "timeframe",
    "data_source",
    "time",
    "data_age_seconds",
    "data_freshness_seconds",
    "stale_after_seconds",
    "bar_age_hours",
    "data_stale",
    "freshness",
    "warning",
    "stale_warning",
]

_TOP_MARKETS_FULL_SPREAD_HEADERS = [
    "bid",
    "ask",
    "spread",
    "spread_points",
    "spread_pct",
    "spread_cost_per_lot",
    "spread_cost_currency",
    "pricing_basis",
]

_TOP_MARKETS_FULL_BAR_HEADERS = [
    "open",
    "close",
    "tick_volume",
    "real_volume",
    "price_change_pct",
]

_TOP_MARKETS_FULL_HEADERS = [
    *_TOP_MARKETS_FULL_BASE_HEADERS,
    *_TOP_MARKETS_FULL_SPREAD_HEADERS,
    *_TOP_MARKETS_FULL_BAR_HEADERS,
]


def _top_markets_headers(metric: str, *, detail_mode: str) -> List[str]:
    if metric == "spread":
        if detail_mode == "compact":
            return [
                *_TOP_MARKETS_COMPACT_BASE_HEADERS,
                *_TOP_MARKETS_COMPACT_SPREAD_HEADERS,
            ]
        return [
            *_TOP_MARKETS_FULL_BASE_HEADERS,
            *_TOP_MARKETS_FULL_SPREAD_HEADERS,
        ]
    if detail_mode == "compact":
        return [
            *_TOP_MARKETS_COMPACT_BASE_HEADERS,
            *_TOP_MARKETS_COMPACT_BAR_HEADERS,
        ]
    return [
        *_TOP_MARKETS_FULL_BASE_HEADERS,
        *_TOP_MARKETS_FULL_BAR_HEADERS,
    ]


def _top_markets_all_headers(*, detail_mode: str) -> List[str]:
    compact_headers = [
        "rank_category",
        "rank",
        *_TOP_MARKETS_COMPACT_HEADERS,
    ]
    if detail_mode == "compact":
        return compact_headers
    return [
        "rank_category",
        "rank",
        *_TOP_MARKETS_FULL_HEADERS,
    ]


def _ranked_top_market_rows(
    ranking: str,
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "rank_category": ranking,
            "rank": rank,
            **row,
        }
        for rank, row in enumerate(rows, start=1)
    ]


def _compact_top_market_leaderboard_rows(
    metric: str,
    rows: List[Dict[str, Any]],
    *,
    detail_mode: str,
) -> List[Dict[str, Any]]:
    headers = _top_markets_headers(metric, detail_mode=detail_mode)
    return [
        {
            "rank": rank,
            **{
                header: row.get(header)
                for header in headers
            },
        }
        for rank, row in enumerate(rows, start=1)
    ]


def _top_market_data_source(metric: str, timeframe: str) -> str:
    return "live_tick" if metric == "spread" else f"{timeframe}_bars"


def _top_market_data_time_key(metric: str) -> str:
    return "tick_time" if metric == "spread" else "time"


def _top_market_rows_with_data_context(
    metric: str,
    rows: List[Dict[str, Any]],
    *,
    timeframe: str,
) -> List[Dict[str, Any]]:
    data_source = _top_market_data_source(metric, timeframe)
    data_time_key = _top_market_data_time_key(metric)
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        mapped = dict(row)
        mapped["data_source"] = data_source
        mapped["time"] = row.get(data_time_key)
        normalized.append(mapped)
    return normalized


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
) -> tuple[List[str], Optional[str]]:
    requested = str(group or "").strip()
    if not requested:
        return [], "group must not be empty."

    groups: Dict[str, str] = {}
    for symbol in all_symbols:
        group_path = str(_extract_group_path_util(symbol) or "").strip()
        if not group_path:
            continue
        groups.setdefault(_normalize_group_path_query(group_path).lower(), group_path)

    requested_lower = _normalize_group_path_query(requested).lower()
    exact = groups.get(requested_lower)
    if exact is not None:
        return [exact], None

    partial_matches = sorted(
        (
            value for value in groups.values()
            if requested_lower in _normalize_group_path_query(value).lower()
        ),
        key=_case_insensitive_sort_key,
    )
    if len(partial_matches) == 1:
        return [partial_matches[0]], None
    if partial_matches:
        return partial_matches, None
    available = sorted(groups.values(), key=_case_insensitive_sort_key)
    if available:
        preview = ", ".join(available[:5])
        suffix = ", ..." if len(available) > 5 else ""
        return [], f"No symbol group matched '{requested}'. Available groups: {preview}{suffix}"
    return [], f"No symbol group matched '{requested}'."


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
        resolved_groups, group_error = _resolve_market_scan_group_path(tradable_symbols, group)
        if group_error or not resolved_groups:
            return [], {}, group_error
        resolved_group_set = {
            _normalize_group_path_query(str(group_path).strip()).lower()
            for group_path in resolved_groups
        }
        selected = sorted(
            [
                symbol for symbol in tradable_symbols
                if (
                    _normalize_group_path_query(
                        str(_extract_group_path_util(symbol) or "").strip()
                    ).lower()
                    in resolved_group_set
                )
                and (universe == "all" or bool(getattr(symbol, "visible", False)))
            ],
            key=lambda symbol: _case_insensitive_sort_key(getattr(symbol, "name", "")),
        )
        return selected, {
            **selection_meta,
            "scope": "group",
            "group": resolved_groups[0] if len(resolved_groups) == 1 else str(group).strip(),
            "groups": resolved_groups,
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
    include_rsi: bool,
    include_sma: bool,
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
    if include_sma and len(close_values) >= max(1, int(sma_period)):
        sma_window = close_values[-int(sma_period):]
        sma_value = float(sum(sma_window) / len(sma_window))
    rsi_value = _market_scan_compute_rsi(close_values, int(rsi_length)) if include_rsi else None
    sma_distance_pct = None
    if sma_value is not None and sma_value != 0:
        sma_distance_pct = ((close_price - sma_value) / sma_value) * 100.0

    row = _market_scan_base_row(symbol)
    row.update(
        {
            "timeframe": timeframe,
            "data_source": f"{timeframe}_bars",
            "time": _format_time_minimal(bar_time) if bar_time is not None else None,
            **_market_scan_freshness_fields(bar_time, timeframe=timeframe, symbol=symbol.name),
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
    if include_rsi:
        row["rsi"] = _market_scan_round(rsi_value, digits=4)
    if include_sma:
        row["sma_value"] = _market_scan_round(sma_value, digits=digits)
        row["sma_distance_pct"] = _market_scan_round(sma_distance_pct, digits=6)
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
    rank_order: str,
    rsi_above: Optional[float],
    rsi_below: Optional[float],
) -> None:
    order = str(rank_order or "auto").strip().lower()
    if order == "auto":
        if rank_by == "spread_pct":
            order = "asc"
        elif rank_by == "rsi" and rsi_below is not None and rsi_above is None:
            order = "asc"
        else:
            order = "desc"

    if rank_by == "abs_price_change_pct":
        rows.sort(
            key=lambda row: (
                row.get("price_change_pct") is None,
                (
                    abs(float(row.get("price_change_pct") or 0.0))
                    if order == "asc"
                    else -abs(float(row.get("price_change_pct") or 0.0))
                ),
                row.get("symbol") or "",
            )
        )
        return

    missing_value = float("inf") if order == "asc" else 0.0

    rows.sort(
        key=lambda row: (
            row.get(rank_by) is None,
            (
                float(row.get(rank_by) if row.get(rank_by) is not None else missing_value)
                if order == "asc"
                else -(float(row.get(rank_by) or 0.0))
            ),
            row.get("symbol") or "",
        )
    )


_MARKET_SCAN_RANK_BY_ALIASES = {
    "abs_price_change": "abs_price_change_pct",
    "price_change": "price_change_pct",
    "volume": "tick_volume",
    "spread": "spread_pct",
}

_SYMBOLS_TOP_MARKETS_RANK_BY_ALIASES = {
    "abs_price_change": "abs_price_change_pct",
    "price_change": "price_change_pct",
    "volume": "tick_volume",
    "spread": "spread_pct",
}

_SYMBOLS_TOP_MARKETS_INTERNAL_RANK_BY = {
    "abs_price_change_pct": "abs_price_change",
    "price_change_pct": "price_change",
    "tick_volume": "volume",
    "spread_pct": "spread",
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


def _normalize_market_scan_rank_order(value: Any) -> tuple[str, Optional[str]]:
    raw_value = str(value or "auto").strip().lower()
    aliases = {"ascending": "asc", "descending": "desc"}
    return aliases.get(raw_value, raw_value), raw_value


def _market_scan_ranking_label(
    rank_by: str,
    *,
    rsi_above: Optional[float] = None,
    rsi_below: Optional[float] = None,
) -> str:
    if rank_by == "abs_price_change_pct":
        return "largest_abs_price_change_pct"
    if rank_by == "price_change_pct":
        return "highest_price_change_pct"
    if rank_by == "tick_volume":
        return "highest_tick_volume"
    if rank_by == "spread_pct":
        return "lowest_spread_pct"
    if rank_by == "rsi" and rsi_below is not None and rsi_above is None:
        return "lowest_rsi"
    if rank_by == "rsi":
        return "highest_rsi"
    return str(rank_by)


@mcp.tool()
def symbols_top_markets(  # noqa: C901
    rank_by: Literal[
        "all",
        "spread",
        "volume",
        "price_change",
        "spread_pct",
        "tick_volume",
        "price_change_pct",
        "abs_price_change",
        "abs_price_change_pct",
    ] = "abs_price_change_pct",  # type: ignore
    limit: Optional[int] = 10,
    universe: Literal["visible", "all"] = "visible",  # type: ignore
    timeframe: TimeframeLiteral = "H1",
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    """Quick MT5 market overview ranked by spread, volume, or price change.

    Defaults to visible tradable symbols for responsiveness. Set `universe="all"` to
    include hidden tradable symbols too; that mode is slower because MT5 may need to
    activate quotes for instruments that are not already visible. Defaults to a
    single absolute-price-change leaderboard; set `rank_by="price_change"` for
    gainers only or `rank_by="all"` for spread, volume, and signed price-change
    leaderboards. Volume and price-change rankings use the most recent completed
    bar on `timeframe`. Uses compact leaderboard rows by default. Set
    `detail="full"` for the expanded row shape and collection metadata. Use
    `market_scan` instead when you need symbol/group inputs, RSI/SMA filters, or
    a single flat scanner table.
    """

    detail_mode = normalize_output_verbosity_detail(detail, default="compact")

    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            raw_rank_by_value = str(rank_by or "abs_price_change_pct").strip().lower()
            rank_by_value = _SYMBOLS_TOP_MARKETS_RANK_BY_ALIASES.get(
                raw_rank_by_value,
                raw_rank_by_value,
            )
            rank_kind = _SYMBOLS_TOP_MARKETS_INTERNAL_RANK_BY.get(
                rank_by_value,
                rank_by_value,
            )
            if rank_kind not in {
                "all",
                "spread",
                "volume",
                "price_change",
                "abs_price_change",
            }:
                return {
                    "error": (
                        "rank_by must be one of: all, spread/spread_pct, "
                        "volume/tick_volume, price_change/price_change_pct, "
                        "abs_price_change/abs_price_change_pct."
                    )
                }

            universe_value = str(universe or "visible").strip().lower()
            if universe_value not in {"visible", "all"}:
                return {"error": "universe must be 'visible' or 'all'."}

            timeframe_value = str(timeframe or "H1").strip().upper()
            needs_bar_data = rank_kind in {
                "all",
                "volume",
                "price_change",
                "abs_price_change",
            }
            if needs_bar_data and timeframe_value not in TIMEFRAME_MAP:
                return {"error": invalid_timeframe_error(timeframe_value, TIMEFRAME_MAP)}
            mt5_timeframe = TIMEFRAME_MAP.get(timeframe_value)

            mt5_gateway = create_mt5_gateway(
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

                if rank_kind in {"all", "spread"}:
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
                        if rank_kind in {"all", "volume"}:
                            _record_issue("volume", symbol_name, bar_error)
                        if rank_kind in {
                            "all",
                            "price_change",
                            "abs_price_change",
                        }:
                            _record_issue("price_change", symbol_name, bar_error)
                    elif bar_row is not None:
                        if rank_kind in {"all", "volume"}:
                            volume_rows.append(dict(bar_row))
                        if rank_kind in {
                            "all",
                            "price_change",
                            "abs_price_change",
                        }:
                            price_change_rows.append(dict(bar_row))

            for symbol in selected_symbols:
                symbol_name = str(getattr(symbol, "name", "") or "")
                is_hidden = not bool(getattr(symbol, "visible", False))
                if universe_value == "all" and is_hidden:
                    with _symbol_ready_guard(symbol_name, info_before=symbol) as (err, _):
                        if err:
                            if rank_kind in {"all", "spread"}:
                                _record_issue("spread", symbol_name, err)
                            if rank_kind in {"all", "volume"}:
                                _record_issue("volume", symbol_name, err)
                            if rank_kind in {
                                "all",
                                "price_change",
                                "abs_price_change",
                            }:
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
                    (
                        -abs(float(row.get("price_change_pct") or 0.0))
                        if rank_kind == "abs_price_change"
                        else -(row.get("price_change_pct") or 0.0)
                    ),
                    row.get("symbol") or "",
                )
            )

            evaluated_counts = {
                "spread": len(spread_rows),
                "volume": len(volume_rows),
                "price_change": len(price_change_rows),
            }

            spread_rows = _top_market_rows_with_data_context(
                "spread",
                spread_rows[:limit_value],
                timeframe=timeframe_value,
            )
            volume_rows = _top_market_rows_with_data_context(
                "volume",
                volume_rows[:limit_value],
                timeframe=timeframe_value,
            )
            price_change_rows = _top_market_rows_with_data_context(
                "price_change",
                price_change_rows[:limit_value],
                timeframe=timeframe_value,
            )

            def _scope_fields(
                metric_name: str,
                rows: List[Dict[str, Any]],
            ) -> Dict[str, Any]:
                available_count = int(evaluated_counts[metric_name])
                returned_count = int(len(rows))
                fields: Dict[str, Any] = {
                    "requested_limit": int(limit_value),
                    "returned_count": returned_count,
                    "universe_size": int(len(selected_symbols)),
                    "available_count": available_count,
                }
                if returned_count < int(limit_value):
                    fields["note"] = (
                        f"Requested {int(limit_value)} rows but only "
                        f"{available_count} symbols had usable {metric_name} data "
                        f"in the {universe_value} universe."
                    )
                return fields

            scan_meta = {"success": True}
            if detail_mode == "full":
                scan_meta.update(
                    {
                        "rank_by": rank_by_value,
                        "rank_by_input": raw_rank_by_value
                        if raw_rank_by_value != rank_by_value
                        else None,
                        "limit": limit_value,
                        "universe": universe_value,
                        "detail": detail_mode,
                        "timeframe": timeframe_value if needs_bar_data else None,
                        "timeframe_requested": timeframe_value,
                        "timeframe_used": timeframe_value if needs_bar_data else None,
                        "scanned_symbols": len(selected_symbols),
                        "query_latency_ms": round(
                            (time.perf_counter() - started_at) * 1000.0,
                            3,
                        ),
                    }
                )

            if rank_kind == "spread":
                out = _market_scan_table(
                    _top_markets_headers("spread", detail_mode=detail_mode),
                    spread_rows,
                    include_contract_meta=detail_mode == "full",
                )
                out.update(scan_meta)
                out["ranking"] = "lowest_spread"
                out.update(_scope_fields("spread", spread_rows))
                _attach_top_markets_units(out, spread_rows)
                if detail_mode == "full":
                    out["evaluated_symbols"] = evaluated_counts["spread"]
                    out["skipped_symbols"] = metric_skips["spread"]
                    out["skipped_examples"] = metric_issues["spread"]
                return out

            if rank_kind == "volume":
                out = _market_scan_table(
                    _top_markets_headers("volume", detail_mode=detail_mode),
                    volume_rows,
                    include_contract_meta=detail_mode == "full",
                )
                out.update(scan_meta)
                out["ranking"] = "highest_volume"
                out.update(_scope_fields("volume", volume_rows))
                _attach_top_markets_units(out, volume_rows)
                if detail_mode == "full":
                    out["evaluated_symbols"] = evaluated_counts["volume"]
                    out["skipped_symbols"] = metric_skips["volume"]
                    out["skipped_examples"] = metric_issues["volume"]
                return out

            if rank_kind in {"price_change", "abs_price_change"}:
                out = _market_scan_table(
                    _top_markets_headers("price_change", detail_mode=detail_mode),
                    price_change_rows,
                    include_contract_meta=detail_mode == "full",
                )
                out.update(scan_meta)
                out["ranking"] = (
                    "largest_abs_price_change_pct"
                    if rank_kind == "abs_price_change"
                    else "highest_price_change_pct"
                )
                out.update(_scope_fields("price_change", price_change_rows))
                _attach_top_markets_units(out, price_change_rows)
                if detail_mode == "full":
                    out["evaluated_symbols"] = evaluated_counts["price_change"]
                    out["skipped_symbols"] = metric_skips["price_change"]
                    out["skipped_examples"] = metric_issues["price_change"]
                return attach_collection_contract(
                    out,
                    collection_kind="table",
                    rows=out.get("data"),
                    include_contract_meta=detail_mode == "full",
                )

            if detail_mode == "compact":
                returned_counts = {
                    "lowest_spread": len(spread_rows),
                    "highest_volume": len(volume_rows),
                    "highest_price_change_pct": len(price_change_rows),
                }
                available_counts = {
                    "lowest_spread": evaluated_counts["spread"],
                    "highest_volume": evaluated_counts["volume"],
                    "highest_price_change_pct": evaluated_counts["price_change"],
                }
                notes = [
                    fields["note"]
                    for metric_name, rows in (
                        ("spread", spread_rows),
                        ("volume", volume_rows),
                        ("price_change", price_change_rows),
                    )
                    for fields in (_scope_fields(metric_name, rows),)
                    if fields.get("note")
                ]
                out = {
                    "success": True,
                    "ranking": "all",
                    "requested_limit": int(limit_value),
                    "universe_size": int(len(selected_symbols)),
                    "returned_counts": returned_counts,
                    "available_counts": available_counts,
                    "lowest_spread": _compact_top_market_leaderboard_rows(
                        "spread",
                        spread_rows,
                        detail_mode=detail_mode,
                    ),
                    "highest_volume": _compact_top_market_leaderboard_rows(
                        "volume",
                        volume_rows,
                        detail_mode=detail_mode,
                    ),
                    "highest_price_change_pct": _compact_top_market_leaderboard_rows(
                        "price_change",
                        price_change_rows,
                        detail_mode=detail_mode,
                    ),
                    **({"notes": notes} if notes else {}),
                }
                _attach_top_markets_units(out, spread_rows, volume_rows, price_change_rows)
                return out

            all_rows = [
                *_ranked_top_market_rows("lowest_spread", spread_rows),
                *_ranked_top_market_rows("highest_volume", volume_rows),
                *_ranked_top_market_rows("highest_price_change_pct", price_change_rows),
            ]
            out = _market_scan_table(
                _top_markets_all_headers(detail_mode=detail_mode),
                all_rows,
                include_contract_meta=detail_mode == "full",
            )
            out.update(scan_meta)
            out["ranking"] = "all"
            out["rank_categories"] = [
                "lowest_spread",
                "highest_volume",
                "highest_price_change_pct",
            ]
            out["requested_limit"] = int(limit_value)
            out["universe_size"] = int(len(selected_symbols))
            out["returned_counts"] = {
                "lowest_spread": len(spread_rows),
                "highest_volume": len(volume_rows),
                "highest_price_change_pct": len(price_change_rows),
            }
            out["available_counts"] = {
                "lowest_spread": evaluated_counts["spread"],
                "highest_volume": evaluated_counts["volume"],
                "highest_price_change_pct": evaluated_counts["price_change"],
            }
            if detail_mode == "full":
                out["scan_stats"] = {
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
                }
            _attach_top_markets_units(out, all_rows)
            return out
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


_MARKET_SCAN_PRESETS: Dict[str, Dict[str, Any]] = {
    "oversold": {"rsi_below": 30.0, "min_tick_volume": 1000, "rank_by": "rsi"},
    "overbought": {"rsi_above": 70.0, "min_tick_volume": 1000, "rank_by": "rsi"},
    "high_volume": {"min_price_change_pct": 1.0, "rank_by": "tick_volume"},
    "tight_spread": {"max_spread_pct": 0.01, "min_tick_volume": 500, "rank_by": "spread_pct"},
    "gap_up": {"min_price_change_pct": 2.0, "rank_by": "price_change_pct"},
    "gap_down": {"max_price_change_pct": -2.0, "rank_by": "price_change_pct"},
}


@mcp.tool()
def market_scan(  # noqa: C901
    symbols: Optional[str] = None,
    group: Optional[str] = None,
    preset: Optional[str] = None,
    limit: Optional[int] = 50,
    offset: int = 0,
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
    rank_order: Literal["auto", "asc", "desc", "ascending", "descending"] = "auto",  # type: ignore
) -> Dict[str, Any]:
    """Filtered MT5 market scanner with one flat table and technical filters.

    Pass `symbols` for one instrument or a comma-separated list. `data` is
    the canonical flat row payload. Compact detail is the default; use
    `detail="full"` when you also want the explicit `columns` ordering hint
    for compatibility. Broad scans use the visible universe; `universe="all"`
    must be combined with `symbols` or `group` to avoid unbounded hidden-symbol
    activation. Use `symbols_top_markets` for a quick all-market overview with
    separate spread, volume, and mover leaderboards.
    """

    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    preset_value = str(preset or "").strip().lower().replace("-", "_")
    preset_error = None
    preset_config = _MARKET_SCAN_PRESETS.get(preset_value) if preset_value else None
    if preset_value and preset_config is None:
        preset_error = (
            "preset must be one of: "
            + ", ".join(sorted(_MARKET_SCAN_PRESETS))
            + "."
        )
    elif preset_config:
        if min_price_change_pct is None and "min_price_change_pct" in preset_config:
            min_price_change_pct = preset_config["min_price_change_pct"]
        if max_price_change_pct is None and "max_price_change_pct" in preset_config:
            max_price_change_pct = preset_config["max_price_change_pct"]
        if max_spread_pct is None and "max_spread_pct" in preset_config:
            max_spread_pct = preset_config["max_spread_pct"]
        if min_tick_volume is None and "min_tick_volume" in preset_config:
            min_tick_volume = preset_config["min_tick_volume"]
        if rsi_below is None and "rsi_below" in preset_config:
            rsi_below = preset_config["rsi_below"]
        if rsi_above is None and "rsi_above" in preset_config:
            rsi_above = preset_config["rsi_above"]
        if rank_by in {None, "abs_price_change_pct"} and "rank_by" in preset_config:
            rank_by = preset_config["rank_by"]

    def _run() -> Dict[str, Any]:  # noqa: C901
        request: Dict[str, Any] = {
            "symbols": symbols,
            "group": group,
            "preset": preset_value or None,
            "limit": limit,
            "offset": offset,
            "universe": universe,
            "timeframe": timeframe,
            "detail": detail_mode,
            "lookback": lookback,
            "rank_by": rank_by,
            "rank_order": rank_order,
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
            if preset_error:
                return _market_scan_error(
                    preset_error,
                    code="invalid_input",
                    request=request,
                )

            universe_value = str(universe or "visible").strip().lower()
            request["universe"] = universe_value
            if universe_value not in {"visible", "all"}:
                return _market_scan_error(
                    "universe must be 'visible' or 'all'.",
                    code="invalid_input",
                    request=request,
                )

            symbols_value = str(symbols or "").strip()
            symbols_filter = symbols_value or None
            if universe_value == "all" and not symbols_filter and not group:
                return _market_scan_error(
                    (
                        "market_scan universe='all' requires symbols or group "
                        "to bound the scan. Use universe='visible' for broad scans "
                        "or symbols_top_markets for quick all-market leaderboards."
                    ),
                    code="invalid_input",
                    request=request,
                )

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
            rank_order_value, rank_order_input = _normalize_market_scan_rank_order(rank_order)
            request["rank_order"] = rank_order_value
            if rank_order_input != rank_order_value:
                request["rank_order_input"] = rank_order_input
            if rank_order_value not in {"auto", "asc", "desc"}:
                return _market_scan_error(
                    "rank_order must be one of: auto, asc, desc, ascending, descending.",
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

            include_rsi = (
                detail_mode != "compact"
                or rank_by_value == "rsi"
                or rsi_above is not None
                or rsi_below is not None
            )
            include_sma = detail_mode != "compact" or price_vs_sma_value is not None
            signal_lookback = lookback_value if (include_rsi or include_sma) else 1

            mt5_gateway = create_mt5_gateway(
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

            limit_value = _normalize_limit(limit) or 50
            request["limit"] = limit_value
            try:
                offset_value = int(offset or 0)
            except Exception:
                return _market_scan_error(
                    "offset must be a non-negative integer.",
                    code="invalid_input",
                    request=request,
                )
            if offset_value < 0:
                return _market_scan_error(
                    "offset must be >= 0.",
                    code="invalid_input",
                    request=request,
                )
            request["offset"] = offset_value
            if selection_meta.get("symbols_input") is not None:
                request["symbols_input"] = selection_meta.get("symbols_input")
            request["scope"] = selection_meta.get("scope")
            if selection_meta.get("group") is not None:
                request["group"] = selection_meta.get("group")
            if selection_meta.get("groups") is not None:
                request["groups"] = selection_meta.get("groups")
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
                    lookback=signal_lookback,
                    rsi_length=rsi_length_value,
                    sma_period=sma_period_value,
                    include_rsi=include_rsi,
                    include_sma=include_sma,
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
                rank_order=rank_order_value,
                rsi_above=rsi_above,
                rsi_below=rsi_below,
            )
            total_matches = len(matched_rows)
            limited_rows = matched_rows[offset_value : offset_value + limit_value]

            full_headers = [
                "symbol",
                "group",
                "description",
                "timeframe",
                "time",
                "data_freshness_seconds",
                "stale_after_seconds",
                "bar_age_hours",
                "data_stale",
                "stale_warning",
                "close",
                "price_change_pct",
                "tick_volume",
                "spread_pct",
                "spread_cost_per_lot",
                "spread_cost_currency",
                "rsi",
                "sma_value",
                "sma_distance_pct",
            ]
            compact_headers = [
                "symbol",
                "group",
                "timeframe",
                "data_source",
                "time",
                "data_stale",
                "freshness",
                "close",
                "price_change_pct",
                "tick_volume",
                "spread_pct",
                "spread_points",
                "spread_pips",
            ]
            if include_rsi:
                compact_headers.append("rsi")
            if include_sma:
                compact_headers.append("sma_distance_pct")
            headers = compact_headers if detail_mode == "compact" else full_headers
            output_rows = (
                _project_market_scan_rows(headers, limited_rows)
                if detail_mode == "compact"
                else limited_rows
            )
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
            table_payload = _market_scan_contract_table(
                headers,
                output_rows,
                include_columns=detail_mode == "full",
            )
            freshness_summary = _market_scan_freshness_summary(limited_rows)
            out: Dict[str, Any] = {
                "success": True,
                "status": "ok" if total_matches > 0 else "no_matches",
                "message": (
                    f"{int(total_matches)} symbol(s) matched the requested market scan filters."
                    if total_matches > 0
                    else "No symbols matched the requested market scan filters."
                ),
                "data": table_payload["rows"],
                "count": table_payload["row_count"],
                "rank_by": rank_by_value,
                "rank_order": rank_order_value,
                "ranking": _market_scan_ranking_label(
                    rank_by_value,
                    rsi_above=rsi_above,
                    rsi_below=rsi_below,
                ),
                "requested_limit": int(limit_value),
                "offset": int(offset_value),
                "returned_count": int(table_payload["row_count"]),
                "total_count": int(total_matches),
                "has_more": bool(offset_value + table_payload["row_count"] < total_matches),
                "universe_size": int(len(selected_symbols)),
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
            out.update(freshness_summary)
            units = _market_scan_units_for_rows(table_payload["rows"])
            if units:
                out["units"] = units
            if "columns" in table_payload:
                out["columns"] = table_payload["columns"]
            if len(selected_symbols) < int(limit_value):
                out["note"] = (
                    f"Requested {int(limit_value)} rows but only "
                    f"{len(selected_symbols)} symbols were available in the "
                    f"{universe_value} universe."
                )
            if total_matches == 0:
                out["summary"]["empty"] = True
            return attach_collection_contract(
                out,
                collection_kind="table",
                rows=output_rows,
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
        offset=offset,
        universe=universe,
        timeframe=timeframe,
        func=_run,
    )

