#!/usr/bin/env python3
import logging
import atexit
import functools
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List
import io
import csv
import time
import dateparser
import MetaTrader5 as mt5
from mcp.server.fastmcp import FastMCP
from config import mt5_config

# Constants (centralize defaults instead of hardcoding inline)
SERVICE_NAME = "MetaTrader5 Market Data Server"
GROUP_SEARCH_THRESHOLD = 5   # threshold for treating a search as group vs symbol search
TICKS_LOOKBACK_DAYS = 1      # lookback days for ticks when no start_datetime provided
DATA_READY_TIMEOUT = 3.0     # seconds to wait for feed to become ready after selection
DATA_POLL_INTERVAL = 0.2     # seconds between readiness polls
FETCH_RETRY_ATTEMPTS = 3     # attempts to fetch data if none returned
FETCH_RETRY_DELAY = 0.3      # delay between fetch retries
SANITY_BARS_TOLERANCE = 3    # acceptable lag in bars when checking freshness

# Shared timeframe mapping (per MetaTrader5 docs)
TIMEFRAME_MAP = {
    # Minutes
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    # Hours
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6,
    "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    # Days / Weeks / Months
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

# Approximate seconds per bar for timeframe window calculations
TIMEFRAME_SECONDS = {
    "M1": 60,
    "M2": 120,
    "M3": 180,
    "M4": 240,
    "M5": 300,
    "M6": 360,
    "M10": 600,
    "M12": 720,
    "M15": 900,
    "M20": 1200,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H3": 10800,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D1": 86400,
    "W1": 604800,
    # For months, use a rough average of 30 days
    "MN1": 2592000,
}

mcp = FastMCP(SERVICE_NAME)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MT5Connection:
    def __init__(self):
        self.connected = False
        # Delay connection until a tool is actually invoked
        
    def _ensure_connection(self) -> bool:
        """Ensure MT5 connection is active, connect if needed"""
        if self.is_connected():
            return True
            
        try:
            if mt5_config.has_credentials():
                login = mt5_config.get_login()
                password = mt5_config.get_password()
                server = mt5_config.get_server()
                if not mt5.initialize(login=login, password=password, server=server):
                    logger.warning(f"Failed to initialize MT5 with credentials: {mt5.last_error()}")
                    if not mt5.initialize():
                        logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
                        return False
                else:
                    logger.info(f"Connected to MT5 with account {login}")
            else:
                if not mt5.initialize():
                    logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
                    return False
                else:
                    logger.info("Connected to MT5 using terminal's current login")
            
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MetaTrader5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MetaTrader5")
    
    def is_connected(self) -> bool:
        """Check if connected to MetaTrader5"""
        if not self.connected:
            return False
        terminal_info = mt5.terminal_info()
        return terminal_info is not None and terminal_info.connected

mt5_connection = MT5Connection()

def _auto_connect_wrapper(func):
    """Decorator to ensure MT5 connection before tool execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not mt5_connection._ensure_connection():
            return {"error": "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."}
        return func(*args, **kwargs)
    return wrapper

atexit.register(mt5_connection.disconnect)


# Flexible datetime parsing helper using dateparser
def _parse_start_datetime(value: str) -> Optional[datetime]:
    """Parse a flexible date/time string into a UTC-naive datetime.

    Accepts formats like:
    - 2025-08-29
    - 2025-08-29 14:30
    - yesterday 14:00
    - 2 days ago
    - 2025/08/29 14:30 UTC
    """
    if not value:
        return None
    dt = dateparser.parse(
        value,
        settings={
            'RETURN_AS_TIMEZONE_AWARE': True,
            'TIMEZONE': 'UTC',
            'TO_TIMEZONE': 'UTC',
            'PREFER_DAY_OF_MONTH': 'first',
        },
    )
    if not dt:
        return None
    # Convert to UTC-naive for MT5 APIs
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# Helpers
def _ensure_symbol_ready(symbol: str) -> Optional[str]:
    """Ensure a symbol is selected and tick info is available. Returns error string or None."""
    info_before = mt5.symbol_info(symbol)
    was_visible = bool(info_before.visible) if info_before is not None else None
    if not mt5.symbol_select(symbol, True):
        return f"Failed to select symbol {symbol}: {mt5.last_error()}"
    # If we just made it visible, wait for fresh tick data to arrive (poll up to timeout)
    if was_visible is False:
        deadline = time.time() + DATA_READY_TIMEOUT
        while time.time() < deadline:
            tick = mt5.symbol_info_tick(symbol)
            if tick and (getattr(tick, 'time', 0) or getattr(tick, 'bid', 0) or getattr(tick, 'ask', 0)):
                break
            time.sleep(DATA_POLL_INTERVAL)
    # Final check
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return f"Failed to refresh {symbol} data: {mt5.last_error()}"
    return None
def _csv_from_rows(headers: List[str], rows: List[List[Any]]) -> Dict[str, str]:
    """Build CSV payload with proper escaping (returns header and data strings)."""
    data_buf = io.StringIO()
    writer = csv.writer(data_buf, lineterminator="\n")
    for row in rows:
        writer.writerow(row)
    return {
        "csv_header": ",".join(headers),
        "csv_data": data_buf.getvalue().rstrip("\n"),
    }

def _utc_iso(epoch_seconds: float) -> str:
    """UTC ISO8601 without TZ suffix to match existing examples."""
    return datetime.utcfromtimestamp(epoch_seconds).isoformat()

def _extract_group_path(sym) -> str:
    """Extract pure group path from a symbol, stripping the symbol name if present.

    MT5 sometimes reports `symbol.path` including the symbol at the tail. This trims the
    last component when it equals the symbol name (case-insensitive).
    """
    raw = getattr(sym, 'path', '') or ''
    name = getattr(sym, 'name', '') or ''
    if not raw:
        return 'Unknown'
    parts = raw.split('\\')
    if parts and name and parts[-1].lower() == name.lower():
        parts = parts[:-1]
    group = '\\'.join(parts).strip('\\')
    return group or 'Unknown'

# Removed grouping helper; get_symbols is simplified to CSV list only

@mcp.tool()
@_auto_connect_wrapper
def get_symbols(
    search_term: Optional[str] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get trading symbols (CSV output: name,description).

    Search strategy:
    - Group name match first, then symbol name, then description.
    - When browsing (no search term), only visible symbols are listed.
    - When searching, all matches are included regardless of visibility.

    Args:
        search_term: Optional search term.
        limit: Optional maximum number of symbols to return.
    """
    try:
        search_strategy = "none"
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
                group_path = _extract_group_path(symbol)
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
                search_strategy = "group_match"
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
                    search_strategy = "symbol_name_match"
                    matched_symbols = symbol_name_matches
                elif matching_groups:  # Fall back to group matches if we had many
                    search_strategy = "group_match"
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
                        search_strategy = "description_match"
                        matched_symbols = description_matches
                    else:
                        search_strategy = "no_match"
                        matched_symbols = []
        else:
            # No search term - return all symbols
            search_strategy = "all"
            matched_symbols = list(mt5.symbols_get() or [])
        
        # Build symbol list with visibility rule
        only_visible = False if search_term else True
        symbol_list = []
        for symbol in matched_symbols:
            if only_visible and not symbol.visible:
                continue
            symbol_list.append({
                "name": symbol.name,
                "group": _extract_group_path(symbol),
                "description": symbol.description,
            })
        
        # Apply limit
        if limit and limit > 0:
            symbol_list = symbol_list[:limit]
        # Convert to CSV format using proper escaping
        rows = [[s["name"], s["group"], s["description"]] for s in symbol_list]
        return _csv_from_rows(["name", "group", "description"], rows)
    except Exception as e:
        return {"error": f"Error getting symbols: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def get_symbol_groups(search_term: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Get available symbol groups from MetaTrader5 (CSV output: group).

    Args:
        search_term: Optional substring to match against group path (case-insensitive).
        limit: Optional maximum number of groups to return (after sorting by count, desc).

    Returns:
        CSV with header "group" and one row per group path.
    """
    try:
        # Get all symbols first
        all_symbols = mt5.symbols_get()
        if all_symbols is None:
            return {"error": f"Failed to get symbols: {mt5.last_error()}"}
        
        # Collect unique groups and counts
        groups = {}
        for symbol in all_symbols:
            group_path = _extract_group_path(symbol)
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
        if limit and limit > 0:
            filtered_items = filtered_items[:limit]

        # Build CSV with only group names
        group_names = [name for name, _ in filtered_items]
        rows = [[g] for g in group_names]
        return _csv_from_rows(["group"], rows)
    except Exception as e:
        return {"error": f"Error getting symbol groups: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def get_symbol_info(symbol: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific symbol
    
    Args:
        symbol: Symbol name (e.g., "EURUSD", "GBPUSD")
    
    Returns:
        Dictionary with symbol information
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

@mcp.tool()
@_auto_connect_wrapper
def get_rates(
    symbol: str,
    timeframe: str = "H1",
    candles: int = 100,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get historical rates for a symbol in CSV format
    
    Args:
        symbol: Symbol name (e.g., "EURUSD", "GBPUSD")
        timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
        candles: Number of candles to retrieve (default 100). Ignored if both start and end are provided.
        start_datetime: Flexible start datetime (e.g., "2025-06-01", "yesterday 14:00").
        end_datetime: Flexible end datetime (optional). If provided with start, fetches the range.
    
    Returns:
        Dictionary with CSV-formatted historical rates data
    """
    try:
        # Validate timeframe using the shared map
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_timeframe = TIMEFRAME_MAP[timeframe]
        
        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = mt5.symbol_info(symbol)
        _was_visible = bool(_info_before.visible) if _info_before is not None else None
        err = _ensure_symbol_ready(symbol)
        if err:
            return {"error": err}
        
        try:
            if start_datetime and end_datetime:
                from_date = _parse_start_datetime(start_datetime)
                to_date = _parse_start_datetime(end_datetime)
                if not from_date or not to_date:
                    return {"error": "Invalid date format. Try '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00'."}
                if from_date > to_date:
                    return {"error": "start_datetime must be before end_datetime"}
                rates = None
                for _ in range(FETCH_RETRY_ATTEMPTS):
                    rates = mt5.copy_rates_range(symbol, mt5_timeframe, from_date, to_date)
                    if rates is not None and len(rates) > 0:
                        # Sanity: last bar should be close to end
                        last_t = rates[-1]["time"]
                        seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe, 60)
                        if last_t >= (to_date.timestamp() - seconds_per_bar * SANITY_BARS_TOLERANCE):
                            break
                    time.sleep(FETCH_RETRY_DELAY)
            elif start_datetime:
                from_date = _parse_start_datetime(start_datetime)
                if not from_date:
                    return {"error": "Invalid date format. Try '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00'."}
                # Fetch forward from the provided start by using a to_date window
                seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe)
                if not seconds_per_bar:
                    return {"error": f"Unable to determine timeframe seconds for {timeframe}"}
                to_date = from_date + timedelta(seconds=seconds_per_bar * (candles + 2))
                rates = None
                for _ in range(FETCH_RETRY_ATTEMPTS):
                    rates = mt5.copy_rates_range(symbol, mt5_timeframe, from_date, to_date)
                    if rates is not None and len(rates) > 0:
                        # Sanity: last bar should be close to computed to_date
                        last_t = rates[-1]["time"]
                        if last_t >= (to_date.timestamp() - seconds_per_bar * SANITY_BARS_TOLERANCE):
                            break
                    time.sleep(FETCH_RETRY_DELAY)
            elif end_datetime:
                to_date = _parse_start_datetime(end_datetime)
                if not to_date:
                    return {"error": "Invalid date format. Try '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00'."}
                # Get the last 'count' bars ending at end_datetime
                rates = None
                seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe, 60)
                for _ in range(FETCH_RETRY_ATTEMPTS):
                    rates = mt5.copy_rates_from(symbol, mt5_timeframe, to_date, candles)
                    if rates is not None and len(rates) > 0:
                        # Sanity: last bar near end
                        last_t = rates[-1]["time"]
                        if last_t >= (to_date.timestamp() - seconds_per_bar * SANITY_BARS_TOLERANCE):
                            break
                    time.sleep(FETCH_RETRY_DELAY)
            else:
                # Use copy_rates_from with current time (now) to force fresh data retrieval
                utc_now = datetime.utcnow()
                rates = None
                seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe, 60)
                for _ in range(FETCH_RETRY_ATTEMPTS):
                    rates = mt5.copy_rates_from(symbol, mt5_timeframe, utc_now, candles)
                    if rates is not None and len(rates) > 0:
                        last_t = rates[-1]["time"]
                        if last_t >= (utc_now.timestamp() - seconds_per_bar * SANITY_BARS_TOLERANCE):
                            break
                    time.sleep(FETCH_RETRY_DELAY)
        finally:
            # Restore original visibility if we changed it
            if _was_visible is False:
                try:
                    mt5.symbol_select(symbol, False)
                except Exception:
                    pass
        
        if rates is None:
            return {"error": f"Failed to get rates for {symbol}: {mt5.last_error()}"}

        # Generate CSV-like format with dynamic column filtering
        if len(rates) == 0:
            return {"error": "No data available"}
        
        # Check which optional columns have meaningful data (at least one non-zero/different value)
        tick_volumes = [int(rate["tick_volume"]) for rate in rates]
        spreads = [int(rate["spread"]) for rate in rates]
        real_volumes = [int(rate["real_volume"]) for rate in rates]
        
        has_tick_volume = len(set(tick_volumes)) > 1 or any(v != 0 for v in tick_volumes)
        has_spread = len(set(spreads)) > 1 or any(v != 0 for v in spreads)
        has_real_volume = len(set(real_volumes)) > 1 or any(v != 0 for v in real_volumes)
        
        # Build header dynamically
        headers = ["time", "open", "high", "low", "close"]
        if has_tick_volume:
            headers.append("tick_volume")
        if has_spread:
            headers.append("spread")
        if has_real_volume:
            headers.append("real_volume")
        
        csv_header = ",".join(headers)
        csv_rows = []
        
        # Build data rows with matching columns
        rows = []
        for rate in rates:
            time_str = _utc_iso(rate["time"])  # use UTC-based timestamp
            values = [str(time_str), str(rate['open']), str(rate['high']), str(rate['low']), str(rate['close'])]
            if has_tick_volume:
                values.append(str(rate['tick_volume']))
            if has_spread:
                values.append(str(rate['spread']))
            if has_real_volume:
                values.append(str(rate['real_volume']))
            rows.append(values)

        # Build CSV via writer for escaping
        payload = _csv_from_rows(headers, rows)
        payload.update({
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": len(rates),
        })
        return payload
    except Exception as e:
        return {"error": f"Error getting rates: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def get_ticks(symbol: str, count: int = 100, start_datetime: Optional[str] = None) -> Dict[str, Any]:
    """
    Get tick data for a symbol in CSV format
    
    Args:
        symbol: Symbol name (e.g., "EURUSD", "GBPUSD")
        count: Number of ticks to retrieve (default 100)
        start_datetime: Start datetime in YYYY-MM-DD HH:MM:SS format (optional, defaults to now)
    
    Returns:
        Dictionary with CSV-formatted tick data
    """
    try:
        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = mt5.symbol_info(symbol)
        _was_visible = bool(_info_before.visible) if _info_before is not None else None
        err = _ensure_symbol_ready(symbol)
        if err:
            return {"error": err}
        
        try:
            if start_datetime:
                from_date = _parse_start_datetime(start_datetime)
                if not from_date:
                    return {"error": "Invalid date format. Try examples like '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00', '2 days ago'."}
                ticks = None
                for _ in range(FETCH_RETRY_ATTEMPTS):
                    ticks = mt5.copy_ticks_from(symbol, from_date, count, mt5.COPY_TICKS_ALL)
                    if ticks is not None and len(ticks) > 0:
                        break
                    time.sleep(FETCH_RETRY_DELAY)
            else:
                # Get recent ticks from current time (now)
                to_date = datetime.utcnow()
                from_date = to_date - timedelta(days=TICKS_LOOKBACK_DAYS)  # look back a configurable window
                ticks = None
                for _ in range(FETCH_RETRY_ATTEMPTS):
                    ticks = mt5.copy_ticks_range(symbol, from_date, to_date, mt5.COPY_TICKS_ALL)
                    if ticks is not None and len(ticks) > 0:
                        break
                    time.sleep(FETCH_RETRY_DELAY)
                if ticks is not None and len(ticks) > count:
                    ticks = ticks[-count:]  # Get the last 'count' ticks
        finally:
            # Restore original visibility if we changed it
            if _was_visible is False:
                try:
                    mt5.symbol_select(symbol, False)
                except Exception:
                    pass
        
        if ticks is None:
            return {"error": f"Failed to get ticks for {symbol}: {mt5.last_error()}"}
        
        # Generate CSV-like format with dynamic column filtering
        if len(ticks) == 0:
            return {"error": "No tick data available"}
        
        # Check which optional columns have meaningful data
        lasts = [float(tick["last"]) for tick in ticks]
        volumes = [float(tick["volume"]) for tick in ticks]
        flags = [int(tick["flags"]) for tick in ticks]
        
        has_last = len(set(lasts)) > 1 or any(v != 0 for v in lasts)
        has_volume = len(set(volumes)) > 1 or any(v != 0 for v in volumes)
        has_flags = len(set(flags)) > 1 or any(v != 0 for v in flags)
        
        # Build header dynamically (time, bid, ask are always included)
        headers = ["time", "bid", "ask"]
        if has_last:
            headers.append("last")
        if has_volume:
            headers.append("volume")
        if has_flags:
            headers.append("flags")
        
        # Build data rows with matching columns and escape properly
        rows = []
        for tick in ticks:
            time_str = _utc_iso(tick["time"])  # use UTC-based timestamp
            values = [time_str, str(tick['bid']), str(tick['ask'])]
            if has_last:
                values.append(str(tick['last']))
            if has_volume:
                values.append(str(tick['volume']))
            if has_flags:
                values.append(str(tick['flags']))
            rows.append(values)

        payload = _csv_from_rows(headers, rows)
        payload.update({
            "success": True,
            "symbol": symbol,
            "count": len(ticks),
        })
        return payload
    except Exception as e:
        return {"error": f"Error getting ticks: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def get_market_depth(symbol: str) -> Dict[str, Any]:
    """
    Get market depth (DOM - Depth of Market) for a symbol
    
    Args:
        symbol: Symbol name (e.g., "EURUSD", "GBPUSD")
    
    Returns:
        Dictionary with market depth data or current bid/ask if DOM unavailable
    """
    try:
        # Ensure symbol is selected
        if not mt5.symbol_select(symbol, True):
            return {"error": f"Failed to select symbol {symbol}: {mt5.last_error()}"}
        
        # Try to get market depth first
        depth = mt5.market_book_get(symbol)
        
        if depth is not None and len(depth) > 0:
            # Process DOM levels
            buy_orders = []
            sell_orders = []
            
            for level in depth:
                order_data = {
                    "price": float(level["price"]),
                    "volume": float(level["volume"]),
                    "volume_real": float(level["volume_real"])
                }
                
                if int(level["type"]) == 0:  # Buy order
                    buy_orders.append(order_data)
                else:  # Sell order
                    sell_orders.append(order_data)
            
            return {
                "success": True,
                "symbol": symbol,
                "type": "full_depth",
                "data": {
                    "buy_orders": buy_orders,
                    "sell_orders": sell_orders
                }
            }
        else:
            # DOM not available, fall back to symbol tick info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"error": f"Symbol {symbol} not found"}
            
            # Get current tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"error": f"Failed to get tick data for {symbol}"}
            
            return {
                "success": True,
                "symbol": symbol,
                "type": "tick_data",
                "data": {
                    "bid": float(tick.bid) if tick.bid else None,
                    "ask": float(tick.ask) if tick.ask else None,
                    "last": float(tick.last) if tick.last else None,
                    "volume": int(tick.volume) if tick.volume else None,
                    "time": int(tick.time) if tick.time else None,
                    "spread": symbol_info.spread,
                    "note": "Full market depth not available, showing current bid/ask"
                }
            }
    except Exception as e:
        return {"error": f"Error getting market depth: {str(e)}"}

def main():
    """Main entry point for the MCP server"""
    mcp.run()

if __name__ == "__main__":
    main()
