import io
import csv
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import dateparser

from ..core.config import mt5_config
from ..core.constants import PRECISION_ABS_TOL, PRECISION_MAX_DECIMALS, PRECISION_REL_TOL, TIME_DISPLAY_FORMAT


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


def _format_time_minimal(epoch_seconds: float) -> str:
    """Format epoch seconds into a normalized UTC datetime string.

    Normalized format everywhere: YYYY-MM-DD HH:MM:SS (UTC)
    """
    dt = datetime.utcfromtimestamp(epoch_seconds)
    return dt.strftime(TIME_DISPLAY_FORMAT)


def _format_time_minimal_local(epoch_seconds: float) -> str:
    """Format epoch seconds into a normalized local/client datetime string.

    Normalized format everywhere: YYYY-MM-DD HH:MM:SS (local/client tz)
    Falls back to UTC if tz resolution fails.
    """
    try:
        tz = mt5_config.get_client_tz()
        if tz is not None:
            dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).astimezone(tz)
        else:
            dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).astimezone()
        return dt.strftime(TIME_DISPLAY_FORMAT)
    except Exception:
        return _format_time_minimal(epoch_seconds)


def _use_client_tz(client_tz_param: object) -> bool:
    try:
        if isinstance(client_tz_param, str):
            v = client_tz_param.strip().lower()
            if v == 'utc':
                return False
            return True
        if isinstance(client_tz_param, bool):
            return bool(client_tz_param)
    except Exception:
        pass
    return False


def _resolve_client_tz(client_tz_param: object):
    try:
        if isinstance(client_tz_param, str):
            v = client_tz_param.strip()
            vlow = v.lower()
            if vlow == 'utc':
                return None
            if vlow in ('auto', 'client', ''):
                tz = None
                try:
                    tz = mt5_config.get_client_tz()
                except Exception:
                    tz = None
                return tz
            try:
                import pytz  # type: ignore
                return pytz.timezone(v)
            except Exception:
                try:
                    return mt5_config.get_client_tz()
                except Exception:
                    return None
        if isinstance(client_tz_param, bool):
            if client_tz_param:
                try:
                    return mt5_config.get_client_tz()
                except Exception:
                    return None
            return None
    except Exception:
        pass
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
