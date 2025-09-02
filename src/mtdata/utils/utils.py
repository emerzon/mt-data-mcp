import io
import csv
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import dateparser

from ..core.config import mt5_config
from ..core.constants import PRECISION_ABS_TOL, PRECISION_MAX_DECIMALS, PRECISION_REL_TOL


def _csv_from_rows(headers: List[str], rows: List[List[Any]]) -> Dict[str, str]:
    data_buf = io.StringIO()
    writer = csv.writer(data_buf, lineterminator="\n")
    for row in rows:
        writer.writerow(row)
    return {
        "csv_header": ",".join(headers),
        "csv_data": data_buf.getvalue().rstrip("\n"),
    }


def _format_time_minimal(epoch_seconds: float) -> str:
    dt = datetime.utcfromtimestamp(epoch_seconds)
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        return dt.strftime("%Y-%m-%d")
    if dt.minute == 0 and dt.second == 0:
        return dt.strftime("%Y-%m-%dT%H")
    if dt.second == 0:
        return dt.strftime("%Y-%m-%dT%H:%M")
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _format_time_minimal_local(epoch_seconds: float) -> str:
    try:
        tz = mt5_config.get_client_tz()
        if tz is not None:
            dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).astimezone(tz)
        else:
            dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).astimezone()
        if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
            return dt.strftime("%Y-%m-%d")
        if dt.minute == 0 and dt.second == 0:
            return dt.strftime("%Y-%m-%d %Hh")
        if dt.second == 0:
            return dt.strftime("%Y-%m-%d %H:%M")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
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
        return "%Y-%m-%d %H:%M:%S"
    if any_min:
        return "%Y-%m-%d %H:%M"
    if any_hour:
        return "%Y-%m-%d %H"
    return "%Y-%m-%d"


def _maybe_strip_year(fmt: str, epochs: List[float]) -> str:
    try:
        years = set(datetime.utcfromtimestamp(e).year for e in epochs)
        if len(years) == 1 and fmt.startswith("%Y-"):
            return fmt[3:]
    except Exception:
        pass
    return fmt


def _style_time_format(fmt: str) -> str:
    try:
        if 'T' in fmt:
            fmt = fmt.replace('T', ' ')
        if fmt.endswith('%H'):
            fmt = fmt + 'h'
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
