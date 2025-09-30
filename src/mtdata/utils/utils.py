import io
import csv
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd
import dateparser

from .constants import PRECISION_ABS_TOL, PRECISION_MAX_DECIMALS, PRECISION_REL_TOL, TIME_DISPLAY_FORMAT


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


def _csv_from_rows(headers: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    """Build a normalized CSV payload for tabular results.

    Returns a dict with at least:
    - csv_header: comma-separated column names
    - csv_data: newline-separated CSV rows
    - success: True
    - count: number of data rows
    """
    data_buf = io.StringIO()
    writer = csv.writer(data_buf, lineterminator="\n")
    for row in rows:
        writer.writerow(row)
    return {
        "csv_header": ",".join(headers),
        "csv_data": data_buf.getvalue().rstrip("\n"),
        "success": True,
        "count": len(rows),
    }

# Backwards-compat alias for refactored imports
_csv_from_rows_util = _csv_from_rows


def _format_time_minimal(epoch_seconds: float) -> str:
    """Format epoch seconds into a normalized UTC datetime string.

    Normalized format everywhere: YYYY-MM-DD HH:MM:SS (UTC)
    """
    dt = datetime.utcfromtimestamp(epoch_seconds)
    return dt.strftime(TIME_DISPLAY_FORMAT)

# Backwards-compat alias
_format_time_minimal_util = _format_time_minimal


def _format_time_minimal_local(epoch_seconds: float) -> str:
    """Format epoch seconds into a normalized local/client datetime string.

    Normalized format everywhere: YYYY-MM-DD HH:MM:SS (local/client tz)
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

# Backwards-compat alias
_format_time_minimal_local_util = _format_time_minimal_local


def _use_client_tz(_: object = None) -> bool:
    """Return True when a client timezone is configured."""
    from ..core.config import mt5_config
    try:
        return mt5_config.get_client_tz() is not None
    except Exception:
        return False

# Backwards-compat alias
_use_client_tz_util = _use_client_tz


def _resolve_client_tz(_: object = None):
    """Return the configured client timezone, if any."""
    from ..core.config import mt5_config
    try:
        return mt5_config.get_client_tz()
    except Exception:
        return None

# Backwards-compat alias
_resolve_client_tz_util = _resolve_client_tz


def _time_format_from_epochs(epochs: List[float]) -> str:
    """Return the normalized display format regardless of epoch contents."""
    return TIME_DISPLAY_FORMAT

# Backwards-compat alias
_time_format_from_epochs_util = _time_format_from_epochs


def _maybe_strip_year(fmt: str, epochs: List[float]) -> str:
    """No-op when normalization is requested; keep full year for consistency."""
    return fmt

# Backwards-compat alias
_maybe_strip_year_util = _maybe_strip_year


def _style_time_format(fmt: str) -> str:
    """No special styling; keep normalized spacing."""
    try:
        if 'T' in fmt:
            return fmt.replace('T', ' ')
    except Exception:
        pass
    return fmt

# Backwards-compat alias
_style_time_format_util = _style_time_format


def _optimal_decimals(values: List[float], rel_tol: float = PRECISION_REL_TOL, abs_tol: float = PRECISION_ABS_TOL,
                      max_decimals: int = PRECISION_MAX_DECIMALS) -> int:
    if not values:
        return 0
    nums = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not nums:
        return 0
    scale = max(1.0, max(abs(v) for v in nums))
    tol = max(abs_tol, rel_tol * scale)
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


def parse_kv_or_json(obj: Any) -> Dict[str, Any]:
    """Parse params/features provided as dict, JSON string, or k=v pairs into a dict.

    - Dict: shallow-copied and returned
    - JSON-like string: parsed via json.loads (with simple fallback for colon/equals pairs)
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
        if (s.startswith('{') and s.endswith('}')):
            try:
                return json.loads(s)
            except Exception:
                # Fallback to simple token parser inside braces
                s = s.strip().strip('{}').strip()
        # Parse simple k=v tokens separated by whitespace/commas
        out: Dict[str, Any] = {}
        toks = [tok for tok in s.replace(',', ' ').split() if tok]
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
    s = f"{v:.{d}f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s


def _format_numeric_rows_from_df(df: pd.DataFrame, headers: List[str]) -> List[List[str]]:
    out_rows: List[List[str]] = []
    decimals_by_col: Dict[str, int] = {}
    for col in headers:
        if col not in df.columns or col == 'time':
            continue
        if pd.api.types.is_float_dtype(df[col]):
            decimals_by_col[col] = _optimal_decimals(df[col].tolist())
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

# Backwards-compat alias
_format_numeric_rows_from_df_util = _format_numeric_rows_from_df


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
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt

# Backwards-compat alias
_parse_start_datetime_util = _parse_start_datetime
