
from typing import Any, Dict, Optional, Literal
import logging
import time

from ..utils.utils import _table_from_rows, _normalize_limit
from ..utils.utils import _format_time_minimal
from ..utils.symbol import _extract_group_path as _extract_group_path_util
from ..utils.mt5_enums import decode_mt5_enum_label, decode_mt5_bitmask_labels
from ._mcp_instance import mcp
from .execution_logging import infer_result_success, log_operation_finish, log_operation_start
from .mt5_gateway import create_mt5_gateway
from .constants import GROUP_SEARCH_THRESHOLD, DEFAULT_ROW_LIMIT
from ..utils.mt5 import MT5ConnectionError, ensure_mt5_connection_or_raise, mt5

logger = logging.getLogger(__name__)


def _get_mt5_gateway():
    return create_mt5_gateway(adapter=mt5, ensure_connection_impl=ensure_mt5_connection_or_raise)



@mcp.tool()
def symbols_list(
    search_term: Optional[str] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
    list_mode: Literal["symbols", "groups"] = "symbols",  # type: ignore
) -> Dict[str, Any]:
    """List symbols or symbol groups."""
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="symbols_list",
        search_term=search_term,
        limit=limit,
        list_mode=list_mode,
    )

    def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
        log_operation_finish(
            logger,
            operation="symbols_list",
            started_at=started_at,
            success=infer_result_success(result),
            search_term=search_term,
            limit=limit,
            list_mode=list_mode,
        )
        return result

    try:
        mt5_gateway = _get_mt5_gateway()
        mt5_gateway.ensure_connection()
        mode = str(list_mode or "symbols").strip().lower()
        if mode not in ("symbols", "groups"):
            return _finish({"error": "list_mode must be 'symbols' or 'groups'."})
        if mode == "groups":
            return _finish(
                _list_symbol_groups(search_term=search_term, limit=limit, mt5_gateway=mt5_gateway)
            )

        matched_symbols = []
        
        if search_term:
            search_upper = search_term.upper()
            
            # Strategy 1: Search for matching group names first
            all_symbols = mt5_gateway.symbols_get()
            if all_symbols is None:
                return _finish({"error": f"Failed to get symbols: {mt5_gateway.last_error()}"})
            
            # Get all unique groups
            groups = {}
            for symbol in all_symbols:
                group_path = _extract_group_path_util(symbol)
                if group_path not in groups:
                    groups[group_path] = []
                groups[group_path].append(symbol)
            
            # Strategy 1: Try group search first, but only if it looks like a group name
            # (avoid matching individual symbol groups for currency searches)
            matching_groups = []
            group_search_threshold = GROUP_SEARCH_THRESHOLD  # centralized threshold
            
            for group_name in groups.keys():
                if search_upper in group_name.upper():
                    matching_groups.append(group_name)
            
            # If we find many groups with the search term, it's probably a symbol search (like EUR, USD)
            # If we find few groups, it's probably a real group search (like Majors, Forex)
            if matching_groups and len(matching_groups) <= group_search_threshold:
                # Use symbols from matching groups
                for group_name in matching_groups:
                    matched_symbols.extend(groups[group_name])
            else:
                # Strategy 2: Partial match in symbol names (primary strategy for currencies)
                symbol_name_matches = []
                for symbol in all_symbols:
                    if search_upper in symbol.name.upper():
                        symbol_name_matches.append(symbol)
                
                if symbol_name_matches:
                    matched_symbols = symbol_name_matches
                elif matching_groups:  # Fall back to group matches if we had many
                    for group_name in matching_groups:
                        matched_symbols.extend(groups[group_name])
                else:
                    # Strategy 3: Partial match in descriptions
                    description_matches = []
                    for symbol in all_symbols:
                        # Check symbol description
                        if hasattr(symbol, 'description') and symbol.description:
                            if search_upper in symbol.description.upper():
                                description_matches.append(symbol)
                                continue
                        
                        # Check group path as description
                        group_path = getattr(symbol, 'path', '')
                        if search_upper in group_path.upper():
                            description_matches.append(symbol)
                    
                    if description_matches:
                        matched_symbols = description_matches
                    else:
                        matched_symbols = []
        else:
            # No search term - return all symbols
            matched_symbols = list(mt5_gateway.symbols_get() or [])
        
        # Build symbol list with visibility rule
        only_visible = False if search_term else True
        symbol_list = []
        for symbol in matched_symbols:
            if only_visible and not symbol.visible:
                continue
            symbol_list.append({
                "name": symbol.name,
                "group": _extract_group_path_util(symbol),
                "description": symbol.description,
            })
        
        # Apply limit
        limit_value = _normalize_limit(limit)
        if limit_value:
            symbol_list = symbol_list[:limit_value]
        # Build tabular result
        rows = [[s["name"], s["group"], s["description"]] for s in symbol_list]
        return _finish(_table_from_rows(["name", "group", "description"], rows))
    except MT5ConnectionError as exc:
        return _finish({"error": str(exc)})
    except Exception as e:
        return _finish({"error": f"Error getting symbols: {str(e)}"})

def _list_symbol_groups(
    search_term: Optional[str] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
    mt5_gateway: Any = None,
) -> Dict[str, Any]:
    """List group paths as a tabular result with a single column: group."""
    try:
        gateway = mt5_gateway or _get_mt5_gateway()
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
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="symbols_describe",
        symbol=symbol,
    )

    def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
        log_operation_finish(
            logger,
            operation="symbols_describe",
            started_at=started_at,
            success=infer_result_success(result),
            symbol=symbol,
        )
        return result

    try:
        mt5_gateway = _get_mt5_gateway()
        mt5_gateway.ensure_connection()
        symbol_info = mt5_gateway.symbol_info(symbol)
        if symbol_info is None:
            return _finish({"error": f"Symbol {symbol} not found"})

        enum_specs = {
            "trade_mode": {"prefixes": ("SYMBOL_TRADE_MODE_",), "bitmask": False},
            "trade_exemode": {"prefixes": ("SYMBOL_TRADE_EXECUTION_",), "bitmask": False},
            "trade_calc_mode": {"prefixes": ("SYMBOL_CALC_MODE_",), "bitmask": False},
            "swap_mode": {"prefixes": ("SYMBOL_SWAP_MODE_",), "bitmask": False},
            "filling_mode": {"prefixes": ("ORDER_FILLING_", "SYMBOL_FILLING_"), "bitmask": True},
            "expiration_mode": {"prefixes": ("SYMBOL_EXPIRATION_",), "bitmask": True},
            "order_mode": {"prefixes": ("SYMBOL_ORDER_",), "bitmask": True},
        }
        
        # Build symbol info dynamically: include all available attributes
        # except excluded ones; skip empty/default values when possible.
        symbol_data = {}
        excluded = {"spread", "ask", "bid", "visible", "custom", "n_fields", "n_sequence_fields"}
        for attr in dir(symbol_info):
            if attr.startswith('_'):
                continue
            if attr in excluded:
                continue
            try:
                value = getattr(symbol_info, attr)
            except Exception:
                continue
            # Skip callables and descriptors
            if callable(value):
                continue
            # Skip empty/defaults for readability
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
        
        return _finish({
            "success": True,
            "symbol": symbol_data
        })
    except MT5ConnectionError as exc:
        return _finish({"error": str(exc)})
    except Exception as e:
        return _finish({"error": f"Error getting symbol info: {str(e)}"})
