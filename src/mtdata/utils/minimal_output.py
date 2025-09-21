"""Utilities to render minimal, plain-text outputs from tool results.

This mirrors the CLI's compact formatting so API responses can avoid JSON
wrappers and return readable text (CSV-first when applicable).
"""

from __future__ import annotations

import io
import csv
import math
from typing import Any, Dict, List


def _is_scalar_value(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set)):
        return all(_is_empty_value(v) for v in value)
    if isinstance(value, dict):
        return all(_is_empty_value(v) for v in value.values())
    return False


def _minify_number(num: float) -> str:
    try:
        f = float(num)
    except Exception:
        return str(num)
    if not math.isfinite(f):
        return str(num)
    text = f"{f:.8f}".rstrip('0').rstrip('.')
    return text if text else '0'


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _minify_number(value)
    return str(value)


def _stringify_cell(value: Any) -> str:
    if _is_scalar_value(value):
        return _stringify_scalar(value)
    if isinstance(value, list):
        values = [v for v in value if not _is_empty_value(v)]
        if not values:
            return ""
        if all(_is_scalar_value(v) for v in values):
            return "|".join(_stringify_scalar(v) for v in values)
        return "; ".join(_stringify_cell(v) for v in values if not _is_empty_value(v))
    if isinstance(value, dict):
        parts = []
        for key, subval in value.items():
            if _is_empty_value(subval):
                continue
            parts.append(f"{key}={_stringify_cell(subval)}")
        return "; ".join(parts)
    return str(value)


def _indent_text(text: str, indent: str = "  ") -> str:
    return "\n".join(f"{indent}{line}" if line else indent.rstrip() for line in text.splitlines())


def _list_of_dicts_to_csv(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    headers: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in item.keys():
            if key not in headers:
                headers.append(key)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(headers)
    for item in items:
        row = [_stringify_cell(item.get(header)) for header in headers]
        writer.writerow(row)
    return buffer.getvalue().rstrip("\n")


def _format_complex_value(value: Any) -> str:
    if _is_scalar_value(value):
        return _stringify_scalar(value)
    if isinstance(value, list):
        values = [v for v in value if not _is_empty_value(v)]
        if not values:
            return ""
        if all(isinstance(v, dict) for v in values):
            return _list_of_dicts_to_csv(values)
        if all(_is_scalar_value(v) for v in values):
            return ", ".join(_stringify_scalar(v) for v in values)
        parts = []
        for entry in values:
            formatted = _format_complex_value(entry)
            if formatted:
                parts.append(formatted)
        return "\n".join(parts)
    if isinstance(value, dict):
        lines = []
        for key, subvalue in value.items():
            if _is_empty_value(subvalue):
                continue
            formatted = _format_complex_value(subvalue)
            if not formatted:
                continue
            if "\n" in formatted:
                lines.append(f"{key}:\n{_indent_text(formatted)}")
            else:
                lines.append(f"{key}: {formatted}")
        return "\n".join(lines)
    return _stringify_scalar(value)


def format_result_minimal(result: Any) -> str:
    """Render a compact, human-readable representation of tool results.

    - Strings are returned as-is (stripped).
    - Lists of dicts become CSV.
    - Dicts may emit known CSV blocks ('csv_header' + 'csv_data').
    - Otherwise, produce a small "key: value" block, omitting empty fields.
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, bool):
        return "true" if result else "false"
    if isinstance(result, int):
        return str(result)
    if isinstance(result, float):
        return _minify_number(result)
    if isinstance(result, list):
        if not result:
            return ""
        if all(isinstance(item, dict) for item in result):
            return _list_of_dicts_to_csv(result)  # type: ignore[arg-type]
        parts: List[str] = []
        for item in result:
            if _is_empty_value(item):
                continue
            if _is_scalar_value(item):
                parts.append(_stringify_scalar(item))
            else:
                formatted = _format_complex_value(item)
                if formatted:
                    parts.append(formatted)
        return "\n".join(parts)
    if isinstance(result, dict):
        # Special case: forecasting payload → essential meta + tabular forecast
        try:
            if isinstance(result.get('times'), list):
                times = list(result.get('times') or [])
                # Determine main forecast series key
                main_key = None
                for k in ('forecast_price', 'forecast_return', 'forecast_series'):
                    if isinstance(result.get(k), list):
                        main_key = k
                        break
                if main_key and times:
                    fvals = list(result.get(main_key) or [])
                    n = min(len(times), len(fvals))
                    # Optional bands
                    lower_key = 'lower_price' if main_key == 'forecast_price' else None
                    upper_key = 'upper_price' if main_key == 'forecast_price' else None
                    lower = list(result.get(lower_key) or []) if lower_key and isinstance(result.get(lower_key), list) else []
                    upper = list(result.get(upper_key) or []) if upper_key and isinstance(result.get(upper_key), list) else []
                    # Optional quantiles: dict[str -> list]
                    qmap = result.get('forecast_quantiles') if isinstance(result.get('forecast_quantiles'), dict) else None
                    qcols = []
                    if isinstance(qmap, dict):
                        try:
                            # Sort numeric by quantile value
                            qcols = sorted(qmap.keys(), key=lambda x: float(x))
                        except Exception:
                            qcols = list(qmap.keys())
                    # If q0.5 duplicates the main forecast, drop it to avoid redundancy
                    try:
                        if '0.5' in qcols:
                            q50 = qmap.get('0.5') if isinstance(qmap, dict) else None
                            if isinstance(q50, list) and len(q50) >= n and len(fvals) >= n:
                                same = True
                                for i in range(n):
                                    try:
                                        a = float(fvals[i])
                                        b = float(q50[i])
                                        if not (abs(a - b) <= 1e-9 or (math.isfinite(a) and math.isfinite(b) and abs(a - b) <= max(1e-9, 1e-8 * max(abs(a), abs(b))))):
                                            same = False
                                            break
                                    except Exception:
                                        same = False
                                        break
                                if same:
                                    qcols = [q for q in qcols if q != '0.5']
                    except Exception:
                        pass
                    # Build CSV header
                    headers = ['time', 'forecast']
                    if lower and upper:
                        headers += ['lower', 'upper']
                    for q in qcols:
                        headers.append(f"q{q}")
                    # Build CSV rows
                    buf = io.StringIO()
                    writer = csv.writer(buf, lineterminator="\n")
                    writer.writerow(headers)
                    for i in range(n):
                        row = [str(times[i]), _stringify_scalar(fvals[i])]
                        if lower and upper and i < len(lower) and i < len(upper):
                            row += [_stringify_scalar(lower[i]), _stringify_scalar(upper[i])]
                        for q in qcols:
                            qarr = qmap.get(q) if isinstance(qmap, dict) else None  # type: ignore[assignment]
                            row.append(_stringify_scalar(qarr[i]) if isinstance(qarr, list) and i < len(qarr) else '')
                        writer.writerow(row)
                    table = buf.getvalue().rstrip('\n')
                    # Essential meta block
                    essentials = []
                    for key in ('symbol', 'timeframe', 'method', 'horizon', 'lookback_used', 'forecast_trend'):
                        v = result.get(key)
                        if not _is_empty_value(v):
                            essentials.append(f"{key}: {_stringify_scalar(v)}")
                    # Include denoise method if present
                    try:
                        dn = result.get('denoise_used')
                        if isinstance(dn, dict):
                            m = dn.get('method')
                            if m:
                                essentials.append(f"denoise: {str(m)}")
                    except Exception:
                        pass
                    tz = result.get('timezone')
                    if isinstance(tz, str) and tz.strip():
                        essentials.append(f"timezone: {tz.strip()}")
                    meta_block = "\n".join(essentials)
                    return f"{meta_block}\n\n{table}" if meta_block else table
        except Exception:
            # Fallback to generic handling below
            pass
        # Special case: volatility payload → prepend essential meta with denoise
        try:
            if ('sigma_bar_return' in result) or ('sigma_annual_return' in result):
                essentials = []
                for key in ('symbol', 'timeframe', 'method', 'horizon'):
                    v = result.get(key)
                    if not _is_empty_value(v):
                        essentials.append(f"{key}: {_stringify_scalar(v)}")
                dn = result.get('denoise_used')
                if isinstance(dn, dict):
                    m = dn.get('method')
                    if m:
                        essentials.append(f"denoise: {str(m)}")
                header = "\n".join(essentials)
                # Append generic formatted block below header
                rest = _format_complex_value(result)
                return f"{header}\n\n{rest}" if header else rest
        except Exception:
            pass
        if 'error' in result and _is_scalar_value(result['error']):
            return f"error: {_stringify_scalar(result['error'])}"
        if 'markdown' in result and isinstance(result['markdown'], str):
            md = result['markdown'].strip()
            extras = {k: v for k, v in result.items() if k not in {'markdown'} and not _is_empty_value(v)}
            if extras:
                meta_block = _format_complex_value(extras)
                if meta_block:
                    return f"{md}\n\n{meta_block}" if md else meta_block
            return md
        header = result.get('csv_header') if isinstance(result, dict) else None
        data = result.get('csv_data') if isinstance(result, dict) else None
        sections: List[str] = []
        if header or data:
            csv_lines: List[str] = []
            if header and str(header).strip():
                csv_lines.append(str(header).strip())
            if data and str(data).strip():
                csv_lines.append(str(data).strip())
            if csv_lines:
                sections.append("\n".join(csv_lines))
        ignore_keys = {'csv_header', 'csv_data', 'markdown', 'success', 'count'}
        extras = {k: v for k, v in result.items() if k not in ignore_keys and not _is_empty_value(v)}
        if extras:
            meta_block = _format_complex_value(extras)
            if meta_block:
                sections.append(meta_block)
        return "\n\n".join(sections).strip()
    return _format_complex_value(result)


def to_methods_availability_csv(methods: List[Dict[str, Any]]) -> str:
    """Special-case helper: compact CSV for method availability.

    Expects entries like {'method': 'theta', 'available': True}.
    """
    rows = []
    for m in methods or []:
        if isinstance(m, dict):
            rows.append({'method': m.get('method'), 'available': m.get('available')})
    return _list_of_dicts_to_csv(rows) if rows else ""
