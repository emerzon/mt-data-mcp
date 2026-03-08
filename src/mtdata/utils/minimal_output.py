"""Utilities to render plain-text TOON output from tool results.

Outputs are normalized into the TOON v2.0 core profile so tools emit a single,
compact encoding instead of mixing ad-hoc tabular text and sparse JSON shapes.
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Dict, Iterable, List, Optional

from .constants import DISPLAY_MAX_DECIMALS
from .formatting import (
    format_number,
    optimal_decimals as _optimal_decimals,
    format_float as _format_float,
)


_INDENT = "  "
_DEFAULT_DELIMITER = ","


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
    return format_number(num)


def _format_number_fixed(num: float) -> str:
    return _format_float(num, DISPLAY_MAX_DECIMALS)


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return format_number(value)
    if isinstance(value, int):
        return format_number(value)
    if isinstance(value, float):
        return format_number(value)
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


def _quote_if_needed(text: str, delimiter: str = _DEFAULT_DELIMITER) -> str:
    """Quote TOON tokens that contain delimiters or surrounding whitespace."""
    if text is None:
        return ""
    raw = str(text)
    needs_quote = (
        raw.strip() != raw
        or any(ch in raw for ch in (delimiter, ":", "\n", "\r", '"', "|", "\t"))
    )
    if not needs_quote:
        return raw
    escaped = raw.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _quote_key(key: Any, delimiter: str = _DEFAULT_DELIMITER) -> str:
    """Quote object keys when necessary."""
    if key is None:
        return ""
    return _quote_if_needed(str(key), delimiter)


def _format_complex_value(value: Any) -> str:
    if _is_scalar_value(value):
        return _stringify_scalar(value)
    if isinstance(value, list):
        values = [v for v in value if not _is_empty_value(v)]
        if not values:
            return ""
        if all(isinstance(v, dict) for v in values):
            headers = _headers_from_dicts(values)
            return _encode_tabular("data", headers, values, indent=0) if headers else ""
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


def _headers_from_dicts(items: Iterable[Dict[str, Any]]) -> List[str]:
    headers: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            return []
        for key in item.keys():
            if key not in headers:
                headers.append(str(key))
    return headers


def _column_decimals(headers: List[str], rows: List[Dict[str, Any]]) -> Dict[str, int]:
    col_decimals: Dict[str, int] = {}
    for h in headers:
        values: List[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            val = row.get(h)
            if val is None or isinstance(val, bool):
                continue
            if isinstance(val, Number):
                try:
                    num = float(val)
                except Exception:
                    continue
                if math.isfinite(num):
                    values.append(num)
        if values:
            col_decimals[h] = _optimal_decimals(values)
    return col_decimals


def _stringify_for_toon(
    value: Any,
    delimiter: str = _DEFAULT_DELIMITER,
    *,
    simplify_numbers: bool = True,
) -> str:
    """Canonical scalar rendering for TOON with optional numeric simplification."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return format_number(value)
    if isinstance(value, int):
        return format_number(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return format_number(value)
        return format_number(value) if simplify_numbers else _format_number_fixed(value)
    return _quote_if_needed(str(value), delimiter)


def _stringify_for_toon_value(
    value: Any,
    decimals: Optional[int],
    delimiter: str,
    *,
    simplify_numbers: bool = True,
) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return format_number(value)
    if isinstance(value, Number):
        try:
            num = float(value)
        except Exception:
            return _quote_if_needed(str(value), delimiter)
        if not math.isfinite(num):
            return format_number(num)
        if not simplify_numbers:
            return _format_number_fixed(num)
        if decimals is None:
            return format_number(num)
        return _format_float(num, int(decimals))
    return _quote_if_needed(str(value), delimiter)


def _encode_tabular(
    key: str,
    headers: List[str],
    rows: List[Dict[str, Any]],
    indent: int = 0,
    delimiter: str = _DEFAULT_DELIMITER,
    *,
    simplify_numbers: bool = True,
) -> str:
    """Render a uniform list of dicts as a TOON tabular array."""
    ind = _INDENT * indent
    name = _quote_key(key, delimiter) or "items"
    header_line = delimiter.join(_quote_key(h, delimiter) for h in headers)
    col_decimals = _column_decimals(headers, rows) if simplify_numbers else {}
    lines = [f"{ind}{name}[{len(rows)}]{{{header_line}}}:"]
    row_indent = ind + _INDENT
    for row in rows:
        vals = [
            _stringify_for_toon_value(
                row.get(h),
                col_decimals.get(h),
                delimiter,
                simplify_numbers=simplify_numbers,
            )
            for h in headers
        ]
        lines.append(f"{row_indent}{delimiter.join(vals)}")
    return "\n".join(lines)


def format_table_toon(
    headers: List[str],
    rows: List[List[Optional[Any]]],
    name: str = "data",
    *,
    simplify_numbers: bool = True,
) -> List[str]:
    """Render a TOON table from headers and row values."""
    if not headers or not rows:
        return []
    cols = [str(h) for h in headers]
    items: List[Dict[str, Any]] = []
    for row in rows:
        item: Dict[str, Any] = {}
        for idx, col in enumerate(cols):
            item[col] = row[idx] if idx < len(row) else None
        items.append(item)
    return _encode_tabular(name, cols, items, indent=0, simplify_numbers=simplify_numbers).splitlines()


def _encode_inline_array(
    key: str,
    items: List[Any],
    indent: int = 0,
    delimiter: str = _DEFAULT_DELIMITER,
    *,
    simplify_numbers: bool = True,
) -> str:
    ind = _INDENT * indent
    name = _quote_key(key, delimiter) or "items"
    dec = None
    if not simplify_numbers:
        dec = DISPLAY_MAX_DECIMALS
    values: List[float] = []
    if simplify_numbers:
        for item in items:
            if item is None or isinstance(item, bool):
                continue
            if isinstance(item, Number):
                try:
                    num = float(item)
                except Exception:
                    continue
                if math.isfinite(num):
                    values.append(num)
        if values:
            dec = _optimal_decimals(values)
    vals = delimiter.join(
        _stringify_for_toon_value(v, dec, delimiter, simplify_numbers=simplify_numbers) for v in items
    )
    return f"{ind}{name}[{len(items)}]: {vals}"


def _encode_expanded_array(
    key: str,
    items: List[Any],
    indent: int = 0,
    delimiter: str = _DEFAULT_DELIMITER,
    *,
    simplify_numbers: bool = True,
) -> str:
    ind = _INDENT * indent
    name = _quote_key(key, delimiter) or "items"
    lines = [f"{ind}{name}[{len(items)}]:"]
    item_indent = _INDENT * (indent + 1)
    for item in items:
        rendered = _format_to_toon(
            item,
            key=None,
            indent=indent + 1,
            delimiter=delimiter,
            simplify_numbers=simplify_numbers,
        )
        if not rendered:
            continue
        sub_lines = rendered.splitlines()
        if not sub_lines:
            continue
        lines.append(f"{item_indent}- {sub_lines[0]}")
        for extra in sub_lines[1:]:
            lines.append(f"{item_indent}  {extra}")
    return "\n".join(lines)


def _normalize_forecast_payload(payload: Dict[str, Any], verbose: bool = True) -> Optional[Dict[str, Any]]:
    """Convert forecast payload into meta + tabular rows when possible."""
    try:
        # Detect time column
        times = None
        if isinstance(payload.get('times'), list):
            times = list(payload.get('times') or [])
        elif isinstance(payload.get('forecast_time'), list):
            times = list(payload.get('forecast_time') or [])
        elif isinstance(payload.get('forecast_epoch'), list):
            # Fallback to epochs if string times missing
            times = list(payload.get('forecast_epoch') or [])
        
        if not times:
            return None

        main_key = None
        for k in ('forecast_price', 'forecast_return', 'forecast_series', 'forecast'):
            if isinstance(payload.get(k), list):
                main_key = k
                break
        if not main_key:
            return None
            
        fvals = list(payload.get(main_key) or [])
        n = min(len(times), len(fvals))
        
        # Check for digits precision
        digits = payload.get('digits')
        if digits is not None:
            try:
                digits = int(digits)
            except Exception:
                digits = None
        
        if 'price' in main_key:
            lower_key, upper_key = 'lower_price', 'upper_price'
        elif 'return' in main_key:
            lower_key = 'lower_return' if isinstance(payload.get('lower_return'), list) else 'lower'
            upper_key = 'upper_return' if isinstance(payload.get('upper_return'), list) else 'upper'
        else:
            lower_key, upper_key = 'lower', 'upper'
        lower = list(payload.get(lower_key) or []) if isinstance(payload.get(lower_key), list) else []
        upper = list(payload.get(upper_key) or []) if isinstance(payload.get(upper_key), list) else []
        
        qmap = payload.get('forecast_quantiles') if isinstance(payload.get('forecast_quantiles'), dict) else None
        qcols: List[str] = []
        if isinstance(qmap, dict):
            try:
                qcols = sorted(qmap.keys(), key=lambda x: float(x))
            except Exception:
                qcols = list(qmap.keys())
        try:
            if '0.5' in qcols:
                q50 = qmap.get('0.5') if isinstance(qmap, dict) else None  # type: ignore[assignment]
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

        headers = ['time', 'forecast']
        if lower and upper:
            headers += ['lower', 'upper']
        for q in qcols:
            headers.append(f"q{q}")
        rows: List[Dict[str, Any]] = []
        for i in range(n):
            val = fvals[i]
            if digits is not None and isinstance(val, (int, float)):
                try:
                    val = f"{float(val):.{digits}f}"
                except Exception:
                    pass
                    
            row: Dict[str, Any] = {
                'time': times[i],
                'forecast': val,
            }
            if lower and upper and i < len(lower) and i < len(upper):
                low_val = lower[i]
                up_val = upper[i]
                if digits is not None:
                    try:
                        if isinstance(low_val, (int, float)): low_val = f"{float(low_val):.{digits}f}"
                        if isinstance(up_val, (int, float)): up_val = f"{float(up_val):.{digits}f}"
                    except Exception:
                        pass
                row['lower'] = low_val
                row['upper'] = up_val
            for q in qcols:
                qarr = qmap.get(q) if isinstance(qmap, dict) else None  # type: ignore[assignment]
                if isinstance(qarr, list) and i < len(qarr):
                    q_val = qarr[i]
                    if digits is not None and isinstance(q_val, (int, float)):
                        try:
                            q_val = f"{float(q_val):.{digits}f}"
                        except Exception:
                            pass
                    row[f"q{q}"] = q_val
            rows.append(row)

        out: Dict[str, Any] = {}
        if verbose:
            meta_block = _build_forecast_meta(payload)
            if meta_block:
                out['meta'] = meta_block

        ci_diag = _compact_forecast_ci(payload, lower=lower, upper=upper)
        if ci_diag:
            out['ci'] = ci_diag

        warnings_in = payload.get('warnings')
        if isinstance(warnings_in, list) and warnings_in:
            warnings_clean = [str(w).strip() for w in warnings_in if str(w).strip()]
            if warnings_clean:
                out['warnings'] = warnings_clean
                 
        out['forecast'] = rows
        return out
    except Exception:
        return None


def _compact_forecast_ci(
    payload: Dict[str, Any],
    *,
    lower: List[Any],
    upper: List[Any],
) -> Dict[str, Any]:
    """Emit CI diagnostics only when they add signal beyond the forecast table."""
    ci_requested = bool(payload.get('ci_requested'))
    ci_available = payload.get('ci_available')
    ci_status = payload.get('ci_status')
    ci_unavailable = bool(payload.get('ci_unavailable'))
    has_interval_columns = bool(lower and upper)

    alpha = None
    for key in ('ci_alpha', 'ci_alpha_requested'):
        value = payload.get(key)
        if not _is_empty_value(value):
            alpha = value
            break

    if ci_available is True and has_interval_columns:
        return {}

    if not ci_requested and ci_available not in (False, True) and _is_empty_value(ci_status) and not ci_unavailable:
        return {}

    out: Dict[str, Any] = {}
    if not _is_empty_value(ci_status):
        out['status'] = ci_status
    elif ci_unavailable:
        out['status'] = 'unavailable'
    elif ci_available is True:
        out['status'] = 'available'

    if alpha is not None:
        out['alpha'] = alpha

    if not out and ci_requested:
        out['requested'] = True

    return out


def _build_forecast_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Group verbose forecast metadata into stable sections."""
    domain: Dict[str, Any] = {}
    for key in ('symbol', 'timeframe', 'method', 'horizon', 'lookback_used', 'forecast_trend'):
        value = payload.get(key)
        if not _is_empty_value(value):
            domain[key] = value

    denoise_used = payload.get('denoise_used')
    if isinstance(denoise_used, dict) and denoise_used.get('method'):
        domain['denoise'] = denoise_used.get('method')
    elif payload.get('denoise_applied'):
        domain['denoise'] = 'applied'

    timezone = payload.get('timezone')
    if isinstance(timezone, str) and timezone.strip():
        domain['timezone'] = timezone.strip()

    params_used = payload.get('params_used')
    if not _is_empty_value(params_used):
        domain['params'] = params_used

    cli_meta_in = payload.get('cli_meta')
    cli_meta = dict(cli_meta_in) if isinstance(cli_meta_in, dict) else {}
    tool_name = str(cli_meta.pop('command', '')).strip()
    cli_timezone = cli_meta.pop('timezone', None)

    runtime: Dict[str, Any] = {}
    cli: Dict[str, Any] = {}

    runtime_timezone: Dict[str, Any] = {}
    timezone_source = cli_timezone if isinstance(cli_timezone, dict) else None
    if timezone_source is None:
        try:
            from ..core.runtime_metadata import build_runtime_timezone_meta

            generated_timezone = build_runtime_timezone_meta(payload)
            if isinstance(generated_timezone, dict):
                timezone_source = generated_timezone
        except Exception:
            timezone_source = None

    if isinstance(timezone_source, dict):
        local_meta = timezone_source.get('local')
        if isinstance(local_meta, dict):
            local_tz_meta = local_meta.get('tz')
            if isinstance(local_tz_meta, dict):
                local_tz = local_tz_meta.get('name')
                if not _is_empty_value(local_tz):
                    cli['local_tz'] = local_tz

        for key, value in timezone_source.items():
            if key == 'local' or _is_empty_value(value):
                continue
            runtime_timezone[str(key)] = _compact_timezone_display(value)

    if runtime_timezone:
        runtime['timezone'] = runtime_timezone

    if cli_meta:
        cli.update(cli_meta)

    meta: Dict[str, Any] = {}
    if tool_name:
        meta['tool'] = tool_name
    if domain:
        meta['domain'] = domain
    if runtime:
        meta['runtime'] = runtime
    if cli:
        meta['cli'] = cli
    return meta


def _compact_timezone_display(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, subval in value.items():
            compacted = _compact_timezone_display(subval)
            if _is_empty_value(compacted):
                continue
            out[str(key)] = compacted
        if 'configured' in out and 'resolved' in out and out.get('configured') == out.get('resolved'):
            out.pop('resolved', None)
        return out
    if isinstance(value, list):
        items = [_compact_timezone_display(item) for item in value]
        return [item for item in items if not _is_empty_value(item)]
    return value


def _collapse_single_key_path(key: str, value: Any) -> tuple[str, Any]:
    current_key = str(key)
    current_value = value
    while isinstance(current_value, dict):
        items = [(str(k), v) for k, v in current_value.items() if not _is_empty_value(v)]
        if len(items) != 1:
            break
        subkey, subval = items[0]
        current_key = f"{current_key}.{subkey}"
        current_value = subval
    return current_key, current_value


def _format_to_toon(
    value: Any,
    key: Optional[str] = None,
    indent: int = 0,
    delimiter: str = _DEFAULT_DELIMITER,
    *,
    simplify_numbers: bool = True,
) -> str:
    ind = _INDENT * indent
    if key is not None and isinstance(value, dict):
        key, value = _collapse_single_key_path(key, value)
    if _is_scalar_value(value):
        rendered = _stringify_for_toon(value, delimiter, simplify_numbers=simplify_numbers)
        if key is None:
            return f"{ind}{rendered}".rstrip()
        return f"{ind}{_quote_key(key, delimiter)}: {rendered}".rstrip()

    if isinstance(value, list):
        name = key or "items"
        if not value:
            return f"{ind}{_quote_key(name, delimiter)}[0]:"
        if all(isinstance(item, dict) for item in value):
            headers = _headers_from_dicts(value)  # type: ignore[arg-type]
            if headers:
                return _encode_tabular(
                    name,
                    headers,
                    value,  # type: ignore[arg-type]
                    indent,
                    delimiter,
                    simplify_numbers=simplify_numbers,
                )
        if all(_is_scalar_value(item) for item in value):
            return _encode_inline_array(name, value, indent, delimiter, simplify_numbers=simplify_numbers)
        return _encode_expanded_array(name, value, indent, delimiter, simplify_numbers=simplify_numbers)

    if isinstance(value, dict):
        if key is None and not value:
            return "{}"
        if not value:
            return f"{ind}{_quote_key(key, delimiter)}: {{}}" if key else "{}"
        lines: List[str] = []
        child_indent = indent if key is None else indent + 1
        for subkey, subval in value.items():
            if _is_empty_value(subval):
                continue
            chunk = _format_to_toon(
                subval,
                key=subkey,
                indent=child_indent,
                delimiter=delimiter,
                simplify_numbers=simplify_numbers,
            )
            if chunk:
                lines.append(chunk)
        if key is None:
            return "\n".join(lines)
        head = f"{ind}{_quote_key(key, delimiter)}:"
        if lines:
            return "\n".join([head, *lines])
        return head

    return f"{ind}{_stringify_for_toon(value, delimiter)}".rstrip()


def format_result_minimal(result: Any, verbose: bool = True, *, simplify_numbers: bool = True) -> str:
    """Render tool outputs as TOON text."""
    if result is None:
        return ""
    try:
        normalized = result
        if isinstance(result, dict):
            forecast_norm = _normalize_forecast_payload(result, verbose=verbose)
            if forecast_norm is not None:
                normalized = forecast_norm
        if isinstance(normalized, str):
            return normalized.strip()
        toon_text = _format_to_toon(normalized, simplify_numbers=simplify_numbers)
        return toon_text.strip()
    except Exception:
        try:
            return str(result) if result is not None else ""
        except Exception:
            return ""


def to_methods_availability_toon(methods: List[Dict[str, Any]]) -> str:
    """Special-case helper: compact TOON for method availability."""
    rows: List[Dict[str, Any]] = []
    for m in methods or []:
        if isinstance(m, dict):
            rows.append({'method': m.get('method'), 'available': m.get('available')})
    if not rows:
        return ""
    headers = _headers_from_dicts(rows)
    return _encode_tabular("methods", headers, rows, indent=0)
