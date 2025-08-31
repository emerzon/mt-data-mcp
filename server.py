#!/usr/bin/env python3
import logging
import atexit
import functools
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List, Tuple, Literal
from typing_extensions import TypedDict
import io
import csv
import time
import re
import inspect
import pydoc
import pandas as pd
import warnings
import pandas_ta as pta
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
TI_NAN_RETRY_ATTEMPTS = 1    # extra attempts if TI columns contain NaNs
TI_NAN_WARMUP_FACTOR = 2     # multiply warmup by this on retry
TI_NAN_WARMUP_MIN_ADD = 50   # at least add this many bars on retry
PRECISION_REL_TOL = 1e-6     # relative tolerance for rounding optimization
PRECISION_ABS_TOL = 1e-12    # absolute tolerance for rounding optimization
PRECISION_MAX_DECIMALS = 10  # upper bound on decimal places

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

# Build a Literal type for timeframe so MCP input schema exposes valid enum values
_TIMEFRAME_CHOICES = tuple(sorted(TIMEFRAME_MAP.keys()))
TimeframeLiteral = Literal[_TIMEFRAME_CHOICES]  # type: ignore

# Build a Literal for single OHLCV letters; the parameter will be a list of these
OhlcvCharLiteral = Literal['O', 'H', 'L', 'C', 'V']  # type: ignore

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

def _format_time_minimal(epoch_seconds: float) -> str:
    """Format UTC time compactly by stripping trailing zero components.

    - If seconds == 0 and minutes == 0 and hours == 0: YYYY-MM-DD
    - Else if seconds == 0 and minutes == 0: YYYY-MM-DDTHH
    - Else if seconds == 0: YYYY-MM-DDTHH:MM
    - Else: YYYY-MM-DDTHH:MM:SS
    """
    dt = datetime.utcfromtimestamp(epoch_seconds)
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        return dt.strftime("%Y-%m-%d")
    if dt.minute == 0 and dt.second == 0:
        return dt.strftime("%Y-%m-%dT%H")
    if dt.second == 0:
        return dt.strftime("%Y-%m-%dT%H:%M")
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def _time_format_from_epochs(epochs: List[float]) -> str:
    """Choose a single consistent time format for a series of epoch timestamps.

    - If any timestamp has non-zero seconds -> include seconds
    - Else if any has non-zero minutes -> include minutes
    - Else if any has non-zero hours -> include hours
    - Else -> date only
    """
    any_sec = False
    any_min = False
    any_hour = False
    for e in epochs:
        dt = datetime.utcfromtimestamp(e)
        if dt.second != 0:
            any_sec = True
            break
        if dt.minute != 0:
            any_min = True
        if dt.hour != 0:
            any_hour = True
    if any_sec:
        return "%Y-%m-%dT%H:%M:%S"
    if any_min:
        return "%Y-%m-%dT%H:%M"
    if any_hour:
        return "%Y-%m-%dT%H"
    return "%Y-%m-%d"

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


# ---- Numeric formatting helpers ----
def _optimal_decimals(values: List[float], rel_tol: float = PRECISION_REL_TOL, abs_tol: float = PRECISION_ABS_TOL,
                      max_decimals: int = PRECISION_MAX_DECIMALS) -> int:
    """Find minimal decimals d such that rounding error is within tolerance for the whole column.

    Uses max absolute error <= max(abs_tol, rel_tol * scale), where scale is max(|v|) or 1 if small.
    """
    if not values:
        return 0
    # Filter NaNs/None
    nums = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not nums:
        return 0
    scale = max(1.0, max(abs(v) for v in nums))
    tol = max(abs_tol, rel_tol * scale)
    # Try from fewest decimals up
    for d in range(0, max_decimals + 1):
        ok = True
        factor = 10.0 ** d
        for v in nums:
            rv = round(v * factor) / factor
            if abs(rv - v) > tol:
                ok = False
                break
        if ok:
            return d
    return max_decimals


def _format_float(v: float, d: int) -> str:
    s = f"{v:.{d}f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s


def _format_numeric_rows_from_df(df: pd.DataFrame, headers: List[str]) -> List[List[str]]:
    """Return rows of strings with optimized decimal places per float column in headers."""
    out_rows: List[List[str]] = []
    # Determine decimals per float column
    decimals_by_col: Dict[str, int] = {}
    for col in headers:
        if col not in df.columns or col == 'time':
            continue
        if pd.api.types.is_float_dtype(df[col]):
            decimals_by_col[col] = _optimal_decimals(df[col].tolist())
    # Build rows
    for _, row in df[headers].iterrows():
        out_row: List[str] = []
        for col in headers:
            val = row[col]
            if col == 'time':
                out_row.append(str(val))
            elif isinstance(val, (float,)) and col in decimals_by_col:
                out_row.append(_format_float(float(val), decimals_by_col[col]))
            else:
                out_row.append(str(val))
        out_rows.append(out_row)
    return out_rows


# ---- Technical Indicators (dynamic discovery and application) ----
def _list_ta_indicators() -> List[Dict[str, Any]]:
    """Dynamically list TA indicators available via pandas_ta.

    Returns a list of dicts with: name, params (name,type,default), description.
    """
    # Create a minimal DataFrame to get the .ta accessor
    tmp = pd.DataFrame({
        'open': [1], 'high': [1], 'low': [1], 'close': [1], 'volume': [1]
    })
    ind_list: List[Dict[str, Any]] = []
    seen = set()
    for attr in dir(tmp.ta):
        if attr.startswith('_'):
            continue
        func = getattr(tmp.ta, attr, None)
        if not callable(func):
            continue
        name = attr.lower()
        if name in seen:
            continue
        seen.add(name)
        # Prefer original pandas_ta function (for better docs) if available
        lib_func = getattr(pta, name, None)
        target_for_sig = lib_func if callable(lib_func) else func
        try:
            sig = inspect.signature(target_for_sig)
        except (TypeError, ValueError):
            continue
        # Collect parameters excluding self and implicit OHLCV columns
        params = []
        for p in sig.parameters.values():
            if p.name in {"self", "open", "high", "low", "close", "volume"}:
                continue
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            entry = {"name": p.name}
            if p.default is not inspect._empty and p.default is not None:
                entry["default"] = p.default
            params.append(entry)
        # Full help text: prefer library function docs if present
        try:
            if callable(lib_func):
                raw = pydoc.render_doc(lib_func)
                desc = _clean_help_text(raw, func_name=name, func=lib_func)
            else:
                raw = pydoc.render_doc(func)
                desc = _clean_help_text(raw, func_name=name, func=func)
        except Exception:
            # Fallback to raw docstring
            desc = inspect.getdoc(lib_func or func) or ''

        # Try to infer missing defaults from the doc text
        try:
            doc_text = inspect.getdoc(lib_func or func) or raw if 'raw' in locals() else ''
            _infer_defaults_from_doc(name, doc_text, params)
        except Exception:
            pass
        # Derive category from module path (e.g., pandas_ta.momentum.rsi -> momentum)
        category = ''
        try:
            mod = (lib_func or func).__module__
            parts = mod.split('.')
            if len(parts) >= 2 and parts[0] == 'pandas_ta':
                # usually pandas_ta.<category>.<func>
                category = parts[1]
        except Exception:
            category = ''

        ind_list.append({
            "name": name,
            "params": params,
            "description": desc,
            "category": category,
        })
    # Sort by name
    ind_list.sort(key=lambda x: x["name"])
    return ind_list


def _infer_defaults_from_doc(func_name: str, doc_text: str, params: List[Dict[str, Any]]):
    """Infer parameter defaults from doc_text when signature uses None but docs specify defaults.

    Attempts two strategies:
    1) Parse a signature-like line: func_name(param=123, other=4.5)
    2) Parse prose patterns: 'param ... Default: 20' or 'param=20' in descriptions
    """
    if not doc_text:
        return
    text = doc_text
    # Remove overstrikes just in case
    text = re.sub(r'.\x08', '', text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Strategy 1: look for a signature line containing func_name(
    sig_line = None
    for ln in lines:
        if ln.startswith(func_name + '(') or re.match(rf"^\s*{re.escape(func_name)}\s*\(.*\)", ln):
            sig_line = ln
            break
    if sig_line:
        inside = sig_line[sig_line.find('(') + 1 : sig_line.rfind(')')] if '(' in sig_line and ')' in sig_line else ''
        for part in re.split(r'[\s,]+', inside):
            if '=' in part:
                k, v = part.split('=', 1)
                k = k.strip()
                v = v.strip().strip(',)')
                num = _try_number(v)
                if num is not None:
                    for p in params:
                        if p.get('name') == k and 'default' not in p:
                            p['default'] = num
    # Strategy 2: prose patterns like 'length ... Default: 20'
    for p in params:
        if 'default' in p:
            continue
        k = p.get('name')
        if not k:
            continue
        m = re.search(rf"{re.escape(k)}[^\n]*?(?:Default|default)\s*:?[\s]*([0-9]+(?:\.[0-9]+)?)", text)
        if m:
            p['default'] = _try_number(m.group(1))


def _try_number(s: str):
    try:
        if '.' in s:
            return float(s)
        return int(s)
    except Exception:
        return None


def _clean_help_text(text: str, func_name: Optional[str] = None, func: Optional[Any] = None) -> str:
    """Return full pydoc help text (cleaned), starting at the function signature.

    - Always uses the rendered pydoc text provided as `text`
    - Removes overstrike sequences
    - Drops everything before the first signature line
    - Cleans trailing "method of ... instance" blurb on the signature line and direct next line
    """
    if not isinstance(text, str):
        return ''
    cleaned = re.sub(r'.\x08', '', text)
    lines = [ln.rstrip() for ln in cleaned.splitlines()]
    # Find signature line
    sig_re = re.compile(rf"^\s*{re.escape(func_name)}\s*\(.*\)") if func_name else re.compile(r"^\s*\w+\s*\(.*\)")
    start = 0
    for i, ln in enumerate(lines):
        if sig_re.match(ln):
            start = i
            break
    kept = lines[start:]
    if kept:
        kept[0] = re.sub(r"\s+method of.*", "", kept[0], flags=re.IGNORECASE)
        if len(kept) > 1 and re.search(r"method of", kept[1], re.IGNORECASE):
            kept.pop(1)
    return "\n".join(kept).strip()


def _parse_ti_specs(spec: str) -> List[Tuple[str, List[Any], Dict[str, Any]]]:
    """Parse TI spec string into a list of (name, args, kwargs).

    Supported formats:
        "sma(14), ema(length=50), macd(12,26,9)"
    Returns empty list on parse error.
    """
    results: List[Tuple[str, List[Any], Dict[str, Any]]] = []
    
    # Split by commas but respect parentheses
    parts = []
    current_part = ""
    paren_depth = 0
    
    for char in spec:
        if char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1
        elif char == ',' and paren_depth == 0:
            if current_part.strip():
                parts.append(current_part.strip())
            current_part = ""
            continue
        current_part += char
    
    if current_part.strip():
        parts.append(current_part.strip())
    
    for part in parts:
        # Tolerate surrounding quotes and whitespace
        part = part.strip().strip("\"'")
        if not part:
            continue
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)(?:\((.*)\))?$", part)
        if not m:
            continue
        raw_name = m.group(1)
        inner = (m.group(2) or '').strip().strip("\"'")
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        if inner:
            for token in re.split(r"[\s,;]+", inner):
                if not token:
                    continue
                if '=' in token:
                    k, v = token.split('=', 1)
                    kwargs[k.strip()] = _coerce_scalar(v.strip())
                else:
                    args.append(_coerce_scalar(token))
        # Flex: detect trailing numeric length in the indicator name (e.g., RSI_48 or EMA21)
        base_name = raw_name
        mlen = re.match(r"^([A-Za-z]+)[_-]?(\d+)$", raw_name)
        if mlen and not inner:
            base_name = mlen.group(1)
            try:
                length_val = int(mlen.group(2))
                args = [length_val] + args
            except Exception:
                pass
        name = base_name.lower()
        results.append((name, args, kwargs))
    return results


def _coerce_scalar(val: str) -> Any:
    """Best-effort type coercion for numeric strings."""
    # Try int
    try:
        return int(val)
    except Exception:
        pass
    # Try float
    try:
        return float(val)
    except Exception:
        pass
    # Strip quotes if present
    if ((val.startswith('"') and val.endswith('"')) or
        (val.startswith("'") and val.endswith("'"))):
        return val[1:-1]
    return val


def _apply_ta_indicators(df: pd.DataFrame, ti_spec: str) -> List[str]:
    """Apply indicators specified by ti_spec to df in-place, return list of added column names.

    Uses pandas_ta via the DataFrame.ta accessor and introspection to pass args.
    """
    added_cols: List[str] = []
    specs = _parse_ti_specs(ti_spec)
    if not specs:
        return added_cols
    
    # Always set up DatetimeIndex when we have epoch timestamps available
    # This ensures compatibility with all datetime-dependent indicators
    original_index = None
    if '__epoch' in df.columns:
        original_index = df.index
        try:
            # Create DatetimeIndex from epoch timestamps
            dt_index = pd.to_datetime(df['__epoch'], unit='s')
            df.index = dt_index
        except Exception:
            # If datetime index setup fails, continue with original index
            pass
    
    before = set(df.columns)
    for name, args, kwargs in specs:
        # Resolve method on accessor
        method = getattr(df.ta, name, None)
        if not callable(method):
            continue
        try:
            # Bind positional args to parameter names where possible
            sig = inspect.signature(method)
            ba = sig.bind_partial(*args, **kwargs)
            ba.apply_defaults()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                method(*ba.args, **ba.kwargs, append=True)
        except Exception:
            # If binding failed, try best-effort call with append
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    method(*args, **kwargs, append=True)
            except Exception:
                continue
        # capture newly added columns
        new_cols = [c for c in df.columns if c not in before]
        added_cols.extend(new_cols)
        before = set(df.columns)
    
    # Restore original index if we changed it
    if original_index is not None:
        df.index = original_index
    
    return added_cols


def _estimate_warmup_bars(ti_spec: Optional[str]) -> int:
    """Estimate extra candles needed so indicators have values at the first target row.

    Heuristics based on common parameters; falls back to 50 if unknown.
    """
    if not ti_spec:
        return 0
    max_warmup = 0
    specs = _parse_ti_specs(ti_spec)
    for name, args, kwargs in specs:
        lname = name.lower()
        # Extract common parameter names
        def geti(key, default):
            if key in kwargs:
                try:
                    return int(kwargs[key])
                except Exception:
                    return default
            if args:
                try:
                    return int(args[0])
                except Exception:
                    return default
            return default
        warm = 0
        if lname in ("sma", "ema", "rsi"):
            warm = geti("length", 14)
        elif lname == "macd":
            fast = kwargs.get("fast", args[0] if len(args) > 0 else 12)
            slow = kwargs.get("slow", args[1] if len(args) > 1 else 26)
            try:
                warm = int(max(int(fast), int(slow)))
            except Exception:
                warm = 26
        elif lname == "stoch":
            k = kwargs.get("k", args[0] if len(args) > 0 else 14)
            d = kwargs.get("d", args[1] if len(args) > 1 else 3)
            s = kwargs.get("smooth", args[2] if len(args) > 2 else 3)
            try:
                warm = int(k) + int(d) + int(s)
            except Exception:
                warm = 20
        elif lname in ("bbands", "bb"):
            length = kwargs.get("length", args[0] if len(args) > 0 else 20)
            try:
                warm = int(length)
            except Exception:
                warm = 20
        else:
            # Unknown indicator: conservative default
            warm = 50
        if warm > max_warmup:
            max_warmup = warm
    # Scale up to ensure stabilization of EMA-based indicators
    scaled = max(int(max_warmup * 3), 50) if max_warmup > 0 else 0
    return scaled


# Build category Literal before tool registration so MCP captures it in the schema
try:
    _CATEGORY_CHOICES = sorted({it.get('category') for it in _list_ta_indicators() if it.get('category')})
except Exception:
    _CATEGORY_CHOICES = []

if _CATEGORY_CHOICES:
    # Create a Literal type alias dynamically
    CategoryLiteral = Literal[tuple(_CATEGORY_CHOICES)]  # type: ignore
else:
    CategoryLiteral = str  # fallback

# Build indicator name Literal so details endpoint has enum name choices
try:
    _INDICATOR_NAME_CHOICES = sorted({it.get('name') for it in _list_ta_indicators() if it.get('name')})
except Exception:
    _INDICATOR_NAME_CHOICES = []

if _INDICATOR_NAME_CHOICES:
    IndicatorNameLiteral = Literal[tuple(_INDICATOR_NAME_CHOICES)]  # type: ignore
else:
    IndicatorNameLiteral = str  # fallback

class IndicatorSpec(TypedDict, total=False):
    """Structured TI spec: name with optional numeric params."""
    name: IndicatorNameLiteral  # type: ignore
    params: List[float]

@mcp.tool()
def get_indicators(search_term: Optional[str] = None, category: Optional[CategoryLiteral] = None) -> Dict[str, Any]:  # type: ignore
    """List indicators as CSV with columns: name,category. Optional filters: search_term, category."""
    try:
        items = _list_ta_indicators()
        if search_term:
            q = search_term.strip().lower()
            filtered = []
            for it in items:
                name = it.get('name', '').lower()
                desc = (it.get('description') or '').lower()
                cat = (it.get('category') or '').lower()
                if q in name or q in desc or q in cat:
                    filtered.append(it)
            items = filtered
        if category:
            cat_q = category.strip().lower()
            items = [it for it in items if (it.get('category') or '').lower() == cat_q]
        items.sort(key=lambda x: (x.get('category') or '', x.get('name') or ''))
        rows = [[it.get('name',''), it.get('category','')] for it in items]
        return _csv_from_rows(["name", "category"], rows)
    except Exception as e:
        return {"error": f"Error listing indicators: {e}"}


# Note: category annotation is set at definition time above to be captured in the MCP schema

@mcp.tool()
def get_indicator_details(name: IndicatorNameLiteral) -> Dict[str, Any]:  # type: ignore
    """Return detailed indicator information (name, category, params, description)."""
    try:
        items = _list_ta_indicators()
        target = next((it for it in items if it.get('name','').lower() == str(name).lower()), None)
        if not target:
            return {"error": f"Indicator '{name}' not found"}
        return {"success": True, "indicator": target}
    except Exception as e:
        return {"error": f"Error getting indicator details: {e}"}

# Removed grouping helper; get_symbols is simplified to CSV list only

@mcp.tool()
@_auto_connect_wrapper
def get_symbols(
    search_term: Optional[str] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """List symbols as CSV with columns: name,group,description.

    - If `search_term` is provided, matches group name, then symbol name, then description.
    - If omitted, returns only visible symbols. When searching, includes non‑visible matches.
    - `limit` caps the number of returned rows.
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
    """List group paths as CSV with a single column: group.

    - Filters by `search_term` (substring, case‑insensitive) when provided.
    - Sorted by group size (desc); `limit` caps the number of groups.
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
    """Return symbol information as JSON for `symbol`.
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

@mcp.tool()
@_auto_connect_wrapper
def get_rates(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    candles: int = 10,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    ohlcv: Optional[List[OhlcvCharLiteral]] = ('C',),
    ti: Optional[List[IndicatorSpec]] = None,
) -> Dict[str, Any]:
    """Return historical candles as CSV.
       Can include OHLCV data, optionally along with technical indicators.
       Returns the last candles by default, unless a date range is specified.
         Parameters:
         - symbol: The symbol to retrieve data for (e.g., "EURUSD").
         - timeframe: The timeframe to use (e.g., "H1", "M30").
         - candles: The number of candles to retrieve (default is 10).
         - start_datetime: Optional start date for the data (e.g., "2025-08-29").
         - end_datetime: Optional end date for the data (e.g., "2025-08-30").
         - ohlcv: Optional list of OHLCV fields to include (e.g., ["O", "H", "L", "C", "V"]).
         - ti: Optional technical indicators to include (e.g., "rsi(20),macd(12,26,9),ema(26)")
       The full list of supported technical indicators can be retrieved from `get_indicators`.
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
            # Normalize TI spec from structured list to string for internal processing
            ti_spec = None
            if ti is not None:
                if isinstance(ti, (list, tuple)):
                    parts = []
                    for item in ti:
                        if isinstance(item, dict) and 'name' in item:
                            nm = str(item['name'])
                            params = item.get('params') or []
                            if isinstance(params, (list, tuple)) and len(params) > 0:
                                args_str = ",".join(str(_coerce_scalar(str(p))) for p in params)
                                parts.append(f"{nm}({args_str})")
                            else:
                                parts.append(nm)
                        else:
                            parts.append(str(item))
                    ti_spec = ",".join(parts)
                else:
                    ti_spec = str(ti)
            # Determine warmup bars if technical indicators requested
            warmup_bars = _estimate_warmup_bars(ti_spec)

            if start_datetime and end_datetime:
                from_date = _parse_start_datetime(start_datetime)
                to_date = _parse_start_datetime(end_datetime)
                if not from_date or not to_date:
                    return {"error": "Invalid date format. Try '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00'."}
                if from_date > to_date:
                    return {"error": "start_datetime must be before end_datetime"}
                # Expand range backward by warmup bars for TI calculation
                seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe, 60)
                from_date_internal = from_date - timedelta(seconds=seconds_per_bar * warmup_bars)
                rates = None
                for _ in range(FETCH_RETRY_ATTEMPTS):
                    rates = mt5.copy_rates_range(symbol, mt5_timeframe, from_date_internal, to_date)
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
                # Expand backward for warmup
                from_date_internal = from_date - timedelta(seconds=seconds_per_bar * warmup_bars)
                rates = None
                for _ in range(FETCH_RETRY_ATTEMPTS):
                    rates = mt5.copy_rates_range(symbol, mt5_timeframe, from_date_internal, to_date)
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
                    rates = mt5.copy_rates_from(symbol, mt5_timeframe, to_date, candles + warmup_bars)
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
                    rates = mt5.copy_rates_from(symbol, mt5_timeframe, utc_now, candles + warmup_bars)
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
        
        # Determine requested columns (O,H,L,C,V) if provided
        requested: Optional[set] = None
        if ohlcv:
            letters: List[str] = []
            if isinstance(ohlcv, str):
                letters = list(ohlcv)
            elif isinstance(ohlcv, (list, tuple, set)):
                letters = list(ohlcv)
            requested = {c.upper() for c in letters if c and c.upper() in {"O", "H", "L", "C", "V"}}
            if not requested:
                requested = None
        
        # Build header dynamically
        headers = ["time"]
        if requested is not None:
            # Include only requested subset
            if "O" in requested:
                headers.append("open")
            if "H" in requested:
                headers.append("high")
            if "L" in requested:
                headers.append("low")
            if "C" in requested:
                headers.append("close")
            if "V" in requested:
                headers.append("tick_volume")
        else:
            # Default: OHLC always; include extras if meaningful
            headers.extend(["open", "high", "low", "close"])
            if has_tick_volume:
                headers.append("tick_volume")
            if has_spread:
                headers.append("spread")
            if has_real_volume:
                headers.append("real_volume")
        
        csv_header = ",".join(headers)
        csv_rows = []
        
        # Construct DataFrame to support indicators and consistent CSV building
        df = pd.DataFrame(rates)
        # Keep epoch for filtering and convert readable time; ensure 'volume' exists for TA
        df['__epoch'] = df['time']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df["time"] = df["time"].apply(_format_time_minimal)
        if 'volume' not in df.columns and 'tick_volume' in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df['volume'] = df['tick_volume']

        # Apply technical indicators if requested (dynamic)
        ti_cols: List[str] = []
        if ti_spec:
            ti_cols = _apply_ta_indicators(df, ti_spec)
            headers.extend([c for c in ti_cols if c not in headers])

        # Build final header list when not using OHLCV subset
        if requested is None:
            # headers already includes OHLC and optional extras
            pass

        # Filter out warmup region to return the intended target window only
        if start_datetime and end_datetime:
            # Keep within original [from_date, to_date]
            target_from = _parse_start_datetime(start_datetime).timestamp()
            target_to = _parse_start_datetime(end_datetime).timestamp()
            df = df.loc[(df['__epoch'] >= target_from) & (df['__epoch'] <= target_to)].copy()
        elif start_datetime:
            target_from = _parse_start_datetime(start_datetime).timestamp()
            df = df.loc[df['__epoch'] >= target_from].copy()
            if len(df) > candles:
                df = df.iloc[:candles].copy()
        elif end_datetime:
            if len(df) > candles:
                df = df.iloc[-candles:].copy()
        else:
            if len(df) > candles:
                df = df.iloc[-candles:].copy()

        # If TI requested, check for NaNs and retry once with increased warmup
        if ti_spec and ti_cols:
            try:
                if df[ti_cols].isna().any().any():
                    # Increase warmup and refetch once
                    warmup_bars_retry = max(int(warmup_bars * TI_NAN_WARMUP_FACTOR), warmup_bars + TI_NAN_WARMUP_MIN_ADD)
                    seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe, 60)
                    # Refetch rates with larger warmup
                    if start_datetime and end_datetime:
                        target_from_dt = _parse_start_datetime(start_datetime)
                        target_to_dt = _parse_start_datetime(end_datetime)
                        from_date_internal = target_from_dt - timedelta(seconds=seconds_per_bar * warmup_bars_retry)
                        rates = mt5.copy_rates_range(symbol, mt5_timeframe, from_date_internal, target_to_dt)
                    elif start_datetime:
                        target_from_dt = _parse_start_datetime(start_datetime)
                        to_date_dt = target_from_dt + timedelta(seconds=seconds_per_bar * (candles + 2))
                        from_date_internal = target_from_dt - timedelta(seconds=seconds_per_bar * warmup_bars_retry)
                        rates = mt5.copy_rates_range(symbol, mt5_timeframe, from_date_internal, to_date_dt)
                    elif end_datetime:
                        target_to_dt = _parse_start_datetime(end_datetime)
                        rates = mt5.copy_rates_from(symbol, mt5_timeframe, target_to_dt, candles + warmup_bars_retry)
                    else:
                        utc_now = datetime.utcnow()
                        rates = mt5.copy_rates_from(symbol, mt5_timeframe, utc_now, candles + warmup_bars_retry)
                    # Rebuild df and indicators with the larger window
                    if rates is not None and len(rates) > 0:
                        df = pd.DataFrame(rates)
                        df['__epoch'] = df['time']
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            df['time'] = df['time'].apply(_format_time_minimal)
                        if 'volume' not in df.columns and 'tick_volume' in df.columns:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                df['volume'] = df['tick_volume']
                        # Re-apply indicators and re-extend headers
                        ti_cols = _apply_ta_indicators(df, ti_spec)
                        headers.extend([c for c in ti_cols if c not in headers])
                        # Re-trim to target window
                        if start_datetime and end_datetime:
                            target_from = _parse_start_datetime(start_datetime).timestamp()
                            target_to = _parse_start_datetime(end_datetime).timestamp()
                            df = df[(df['__epoch'] >= target_from) & (df['__epoch'] <= target_to)]
                        elif start_datetime:
                            target_from = _parse_start_datetime(start_datetime).timestamp()
                            df = df[df['__epoch'] >= target_from]
                            if len(df) > candles:
                                df = df.iloc[:candles]
                        elif end_datetime:
                            if len(df) > candles:
                                df = df.iloc[-candles:]
                        else:
                            if len(df) > candles:
                                df = df.iloc[-candles:]
            except Exception:
                pass

        # Ensure headers are unique and exist in df
        headers = [h for h in headers if h in df.columns or h == 'time']

        # Reformat time consistently across rows
        if 'time' in headers and len(df) > 0:
            fmt = _time_format_from_epochs(df['__epoch'].tolist())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df['time'] = df['__epoch'].apply(lambda t: datetime.utcfromtimestamp(t).strftime(fmt))

        # Assemble rows from DataFrame for selected headers with optimized precision
        rows = _format_numeric_rows_from_df(df, headers)

        # Build CSV via writer for escaping
        payload = _csv_from_rows(headers, rows)
        payload.update({
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": len(df),
        })
        return payload
    except Exception as e:
        return {"error": f"Error getting rates: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def get_ticks(symbol: str, count: int = 100, start_datetime: Optional[str] = None) -> Dict[str, Any]:
    """Return latest ticks as CSV with columns: time,bid,ask and optional last,volume,flags.
    - `count` limits the number of rows; `start_datetime` starts from a flexible date/time.
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
        # Choose a consistent time format for all rows
        fmt = _time_format_from_epochs([float(t["time"]) for t in ticks])
        rows = []
        for tick in ticks:
            time_str = datetime.utcfromtimestamp(tick["time"]).strftime(fmt)
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
    """Return DOM if available; otherwise current bid/ask snapshot for `symbol`."""
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

