
from typing import Any, Dict, Optional, Literal
import logging

from ..utils.utils import _table_from_rows, _normalize_limit
from ..utils.utils import _format_time_minimal
from ..utils.symbol import _extract_group_path as _extract_group_path_util
from ..utils.mt5_enums import decode_mt5_enum_label, decode_mt5_bitmask_labels
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .mt5_gateway import get_mt5_gateway
from .constants import GROUP_SEARCH_THRESHOLD, DEFAULT_ROW_LIMIT
from ..utils.mt5 import MT5ConnectionError, ensure_mt5_connection_or_raise, mt5

logger = logging.getLogger(__name__)


@mcp.tool()
def symbols_list(
    search_term: Optional[str] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
    list_mode: Literal["symbols", "groups"] = "symbols",  # type: ignore
) -> Dict[str, Any]:
    """List symbols or symbol groups."""
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
                return _list_symbol_groups(search_term=search_term, limit=limit, mt5_gateway=mt5_gateway)

            matched_symbols = []

            if search_term:
                search_upper = search_term.upper()

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

            only_visible = not bool(search_term)
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
            return _table_from_rows(["name", "group", "description"], rows)
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error getting symbols: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="symbols_list",
        search_term=search_term,
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
        filtered_items.sort(key=lambda x: x[1]["count"], reverse=True)

        # Apply limit
        limit_value = _normalize_limit(limit)
        if limit_value:
            filtered_items = filtered_items[:limit_value]

        # Build tabular result with only group names
        group_names = [name for name, _ in filtered_items]
        rows = [[g] for g in group_names]
        return _table_from_rows(["group"], rows)
    except Exception as e:
        return {"error": f"Error getting symbol groups: {str(e)}"}

@mcp.tool()
def symbols_describe(symbol: str) -> Dict[str, Any]:
    """Return symbol information as JSON for `symbol`.
       Parameters: symbol
       Includes information such as Symbol Description, Swap Values, Tick Size/Value, etc.
    """
    def _run() -> Dict[str, Any]:
        try:
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
            excluded = {"spread", "ask", "bid", "visible", "custom", "n_fields", "n_sequence_fields"}
            for attr in dir(symbol_info):
                if attr.startswith("_"):
                    continue
                if attr in excluded:
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
                if isinstance(value, (int, float)) and value == 0:
                    continue
                if attr == "time":
                    try:
                        epoch = float(value)
                        symbol_data["time_epoch"] = epoch
                        symbol_data["time"] = _format_time_minimal(epoch)
                    except Exception:
                        symbol_data[attr] = value
                else:
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
        func=_run,
    )
