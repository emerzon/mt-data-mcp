
from typing import Any, Dict, Optional, Literal

from ..utils.utils import _table_from_rows, _normalize_limit
from ..utils.symbol import _extract_group_path as _extract_group_path_util
from .server import mcp, _auto_connect_wrapper
from .constants import GROUP_SEARCH_THRESHOLD, DEFAULT_ROW_LIMIT
import MetaTrader5 as mt5



@mcp.tool()
@_auto_connect_wrapper
def symbols_list(
    search_term: Optional[str] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
    list_mode: Literal["symbols", "groups"] = "symbols",  # type: ignore
) -> Dict[str, Any]:
    """List symbols or symbol groups."""
    try:
        mode = str(list_mode or "symbols").strip().lower()
        if mode not in ("symbols", "groups"):
            return {"error": "list_mode must be 'symbols' or 'groups'."}
        if mode == "groups":
            return _list_symbol_groups(search_term=search_term, limit=limit)

        matched_symbols = []
        
        if search_term:
            search_upper = search_term.upper()
            
            # Strategy 1: Search for matching group names first
            all_symbols = mt5.symbols_get()
            if all_symbols is None:
                return {"error": f"Failed to get symbols: {mt5.last_error()}"}
            
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
            matched_symbols = list(mt5.symbols_get() or [])
        
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
        return _table_from_rows(["name", "group", "description"], rows)
    except Exception as e:
        return {"error": f"Error getting symbols: {str(e)}"}

def _list_symbol_groups(
    search_term: Optional[str] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
) -> Dict[str, Any]:
    """List group paths as a tabular result with a single column: group."""
    try:
        # Get all symbols first
        all_symbols = mt5.symbols_get()
        if all_symbols is None:
            return {"error": f"Failed to get symbols: {mt5.last_error()}"}
        
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
@_auto_connect_wrapper
def symbols_describe(symbol: str) -> Dict[str, Any]:
    """Return symbol information as JSON for `symbol`.
       Parameters: symbol
       Includes information such as Symbol Description, Swap Values, Tick Size/Value, etc.
    """
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {"error": f"Symbol {symbol} not found"}
        
        # Build symbol info dynamically: include all available attributes
        # except excluded ones; skip empty/default values when possible.
        symbol_data = {}
        excluded = {"spread", "ask", "bid", "visible", "custom"}
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
            symbol_data[attr] = value
        
        return {
            "success": True,
            "symbol": symbol_data
        }
    except Exception as e:
        return {"error": f"Error getting symbol info: {str(e)}"}
