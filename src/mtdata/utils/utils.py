from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple, Set
from numbers import Number

import pandas as pd
import dateparser

from .constants import (
    PRECISION_ABS_TOL,
    PRECISION_MAX_DECIMALS,
    PRECISION_MAX_LOSS_PCT,
    PRECISION_REL_TOL,
    TIME_DISPLAY_FORMAT,
)
from .formatting import (
    format_number,
    optimal_decimals as _optimal_decimals_shared,
    format_float as _format_float_shared,
)


def _coerce_scalar(s: str):
    """Try to coerce a scalar string to int or float; otherwise return original string."""
    try:
        if s is None:
            return s
        st = str(s).strip()
        if st == "":
            return st
        if st.isdigit() or (st.startswith('-') and st[1:].isdigit()):
            return int(st)
        v = float(st)
        return v
    except Exception:
        return s


def _normalize_ohlcv_arg(ohlcv: Optional[str]) -> Optional[Set[str]]:
    """Normalize user-provided OHLCV selection into a set of letters.

    Accepts forms like: 'close', 'price', 'ohlc', 'ohlcv', 'all', 'cl', 'OHLCV',
    or names 'open,high,low,close,volume'. Returns None when not specified.
    """
    if ohlcv is None:
        return None
    text = str(ohlcv).strip()
    if text == "":
        return None
    t = text.lower()
    if t in ("all", "ohlcv"):
        return {"O", "H", "L", "C", "V"}
    if t in ("ohlc",):
        return {"O", "H", "L", "C"}
    if t in ("price", "close"):
        return {"C"}
    # Compact letters like 'cl', 'oh', etc.
    if all(ch in "ohlcv" for ch in t):
        return {ch.upper() for ch in t}
    # Comma separated names
    parts = [p.strip().lower() for p in t.replace(";", ",").split(",") if p.strip() != ""]
    if not parts:
        return None
    mapping = {
        "o": "O", "open": "O",
        "h": "H", "high": "H",
        "l": "L", "low": "L",
        "c": "C", "close": "C", "price": "C",
        "v": "V", "vol": "V", "volume": "V", "tick_volume": "V",
    }
    out: Set[str] = set()
    for p in parts:
        key = mapping.get(p)
        if key:
            out.add(key)
    return out or None


def _normalize_limit(limit: Optional[Any]) -> Optional[int]:
    try:
        if limit is None:
            return None
        if isinstance(limit, str):
            limit = limit.strip()
            if not limit:
                return None
        value = int(float(limit))
        return value if value > 0 else None
    except Exception:
        return None


def _table_from_rows(headers: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    """Build a normalized tabular payload for results.

    Returns a dict with at least:
    - data: list[dict] rows (keys follow the provided headers order)
    - success: True
    - count: number of data rows
    """
    cols = [str(h) for h in (headers or [])]
    items: List[Dict[str, Any]] = []
    for row in rows or []:
        item: Dict[str, Any] = {}
        for idx, col in enumerate(cols):
            item[col] = row[idx] if idx < len(row) else None
        items.append(item)
    return {
        "data": items,
        "success": True,
        "count": len(items),
    }

def _format_time_minimal(epoch_seconds: float) -> str:
    """Format epoch seconds into a normalized UTC datetime string.

    Normalized format everywhere: YYYY-MM-DD HH:MM
    """
    dt = datetime.utcfromtimestamp(epoch_seconds)
    return dt.strftime(TIME_DISPLAY_FORMAT)

def _format_time_minimal_local(epoch_seconds: float) -> str:
    """Format epoch seconds into a normalized local/client datetime string.

    Normalized format everywhere: YYYY-MM-DD HH:MM (local/client tz)
    Falls back to UTC if tz resolution fails.
    """
    from ..core.config import mt5_config
    try:
        tz = mt5_config.get_client_tz()
        if tz is not None:
            dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).astimezone(tz)
        else:
            dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).astimezone()
        return dt.strftime(TIME_DISPLAY_FORMAT)
    except Exception:
        return _format_time_minimal(epoch_seconds)

def _use_client_tz(_: object = None) -> bool:
    """Return True when a client timezone is configured."""
    from ..core.config import mt5_config
    try:
        return mt5_config.get_client_tz() is not None
    except Exception:
        return False

def _resolve_client_tz(_: object = None):
    """Return the configured client timezone, if any."""
    from ..core.config import mt5_config
    try:
        return mt5_config.get_client_tz()
    except Exception:
        return None

def _time_format_from_epochs(epochs: List[float]) -> str:
    """Return the normalized display format regardless of epoch contents."""
    return TIME_DISPLAY_FORMAT

def _maybe_strip_year(fmt: str, epochs: List[float]) -> str:
    """No-op when normalization is requested; keep full year for consistency."""
    return fmt

def _style_time_format(fmt: str) -> str:
    """No special styling; keep normalized spacing."""
    try:
        if 'T' in fmt:
            return fmt.replace('T', ' ')
    except Exception:
        pass
    return fmt

def _optimal_decimals(
    values: List[float],
    rel_tol: float = PRECISION_REL_TOL,
    abs_tol: float = PRECISION_ABS_TOL,
    max_decimals: int = PRECISION_MAX_DECIMALS,
    max_loss_pct: float = PRECISION_MAX_LOSS_PCT,
) -> int:
    return _optimal_decimals_shared(
        values,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        max_decimals=max_decimals,
        max_loss_pct=max_loss_pct,
    )


def parse_kv_or_json(obj: Any) -> Dict[str, Any]:
    """Parse params/features provided as dict, JSON string, or k=v pairs into a dict.

    - Dict: shallow-copied and returned
    - JSON-like string: parsed via json.loads (dict or list-of-pairs)
    - Plain string: split on whitespace/commas into k=v assignments
    """
    import json

    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return {}
        if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return dict(parsed)
                # Accept list-of-pairs JSON (e.g., [["k","v"],["k2","v2"]])
                if isinstance(parsed, list):
                    out_pairs: Dict[str, Any] = {}
                    ok = True
                    for item in parsed:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            out_pairs[str(item[0])] = item[1]
                        else:
                            ok = False
                            break
                    if ok:
                        return out_pairs
                # Non-dict JSON: fall back to token parsing for robustness
            except Exception:
                # Fallback to simple token parser inside braces; list-shaped JSON
                # should just fall through to return {}.
                if s.startswith('{') and s.endswith('}'):
                    s = s.strip().strip('{}').strip()
                else:
                    return {}
        # Parse k=v / k:v assignments. Commas split assignments only when a new key follows.
        import re
        out: Dict[str, Any] = {}
        pair_pattern = re.compile(
            r'(?:^|[\s,])([A-Za-z_][\w.\-]*)\s*([=:])\s*(.*?)\s*(?=(?:[\s,]+[A-Za-z_][\w.\-]*\s*[=:])|$)'
        )
        for m in pair_pattern.finditer(s):
            k = str(m.group(1) or '').strip()
            v = str(m.group(3) or '').strip().strip(',')
            # Avoid Windows drive paths like "C:\foo".
            if len(k) == 1 and v.startswith(("\\", "/")):
                continue
            out[k] = v
        if out:
            return out

        # Legacy token parser fallback for partially malformed inputs.
        toks = [tok for tok in s.split() if tok]
        i = 0
        while i < len(toks):
            tok = toks[i].strip().strip(',')
            if not tok:
                i += 1
                continue
            if '=' in tok:
                k, v = tok.split('=', 1)
                out[k.strip()] = v.strip().strip(',')
                i += 1
                continue
            # Support "k:v" tokens (avoid Windows drive paths like "C:\\foo")
            if ':' in tok and not tok.endswith(':') and tok.count(':') == 1:
                k, v = tok.split(':', 1)
                if len(k) == 1 and v.startswith(("\\", "/")):
                    i += 1
                    continue
                out[k.strip()] = v.strip().strip(',')
                i += 1
                continue
            if tok.endswith(':'):
                key = tok[:-1].strip()
                val = ''
                if i + 1 < len(toks):
                    val = toks[i + 1].strip().strip(',')
                    i += 2
                else:
                    i += 1
                out[key] = val
                continue
            i += 1
        return out
    return {}


def _format_float(v: float, d: int) -> str:
    return _format_float_shared(v, d)




def _format_numeric_rows_from_df(df: pd.DataFrame, headers: List[str]) -> List[List[str]]:
    # Precompute per-column decimals to trim numeric noise without losing precision.
    col_decimals: Dict[str, int] = {}
    for col in headers:
        if col == 'time' or col not in df.columns:
            continue
        try:
            series = pd.to_numeric(df[col], errors="coerce")
            values = [
                float(v)
                for v in series
                if v is not None and not pd.isna(v) and math.isfinite(v)
            ]
        except Exception:
            values = []
        if values:
            col_decimals[col] = _optimal_decimals(values)

    out_rows: List[List[str]] = []
    for _, row in df[headers].iterrows():
        out_row: List[str] = []
        for col in headers:
            val = row[col]
            if col == 'time':
                out_row.append(str(val))
            elif val is None or isinstance(val, bool):
                out_row.append(format_number(val))
            elif isinstance(val, Number):
                try:
                    num = float(val)
                except Exception:
                    out_row.append(str(val))
                    continue
                if not math.isfinite(num):
                    out_row.append(format_number(num))
                    continue
                decimals = col_decimals.get(col)
                if decimals is None:
                    out_row.append(format_number(num))
                else:
                    out_row.append(_format_float(num, decimals))
            else:
                out_row.append(str(val))
        out_rows.append(out_row)
    return out_rows

def to_float_np(
    values: Any,
    *,
    coerce: bool = True,
    drop_na: bool = False,
    finite_only: bool = False,
    return_mask: bool = False,
) -> "np.ndarray | Tuple[np.ndarray, 'np.ndarray']":
    """Convert a pandas Series/array-like to a float NumPy array.

    - coerce=True uses `pd.to_numeric(errors='coerce')` to convert invalids to NaN.
    - drop_na=True removes NaN entries from the returned array (mask applied).
    - finite_only=True removes non-finite entries (NaN, inf, -inf).
    - return_mask=True returns (array, mask) where mask marks kept elements.

    Notes: When both drop_na and finite_only are False, the original length is preserved.
    """
    import numpy as np  # local import

    try:
        # Normalize to pandas Series for robust conversion
        if hasattr(values, "to_numpy") and hasattr(values, "dtype"):
            ser = pd.Series(values)
        else:
            ser = pd.Series(values)

        arr = (
            pd.to_numeric(ser, errors="coerce").astype(float).to_numpy()
            if coerce
            else ser.astype(float).to_numpy()
        )

        mask = None
        if drop_na or finite_only:
            if finite_only:
                mask = np.isfinite(arr)
            else:
                mask = ~pd.isna(arr)
            arr = arr[mask]
        if return_mask:
            if mask is None:
                mask = np.ones(arr.shape, dtype=bool)
            return arr, mask
        return arr
    except Exception:
        # Fallbacks
        try:
            arr = np.asarray(values, dtype=float)
            if drop_na or finite_only:
                m = np.isfinite(arr) if finite_only else ~pd.isna(arr)
                arr = arr[m]
                if return_mask:
                    return arr, m
            elif return_mask:
                return arr, np.ones(arr.shape, dtype=bool)
            return arr
        except Exception:
            empty = np.asarray([], dtype=float)
            if return_mask:
                return empty, np.asarray([], dtype=bool)
            return empty


def align_finite(*arrays: Any) -> Tuple["np.ndarray", ...]:
    """Convert arrays to float and align them by keeping only rows where all are finite.

    Returns a tuple of filtered arrays, all of equal length.
    """
    import numpy as np
    conv = [to_float_np(a) for a in arrays]
    if not conv:
        return tuple()
    mask = np.ones_like(conv[0], dtype=bool)
    for a in conv:
        mask &= np.isfinite(a)
    return tuple(a[mask] for a in conv)


def _parse_start_datetime(value: str) -> Optional[datetime]:
    """Parse a date/time string via dateparser into UTC-naive datetime."""
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
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _utc_epoch_seconds(dt: datetime) -> float:
    """Convert a datetime to UTC epoch seconds, treating naive values as UTC.

    Python's `datetime.timestamp()` interprets naive datetimes as *local time*,
    which can silently shift values when the host isn't running in UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).timestamp()
    return dt.astimezone(timezone.utc).timestamp()

