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
    format_float as _format_float,
)
from .formatting import (
    format_number,
)
from .formatting import (
    optimal_decimals as _optimal_decimals,
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


def _stringify_nonfinite_number(num: float) -> str:
    if math.isnan(num):
        return "nan"
    if num > 0:
        return "inf"
    if num < 0:
        return "-inf"
    return "nan"


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


def _quote_always(text: Any) -> str:
    raw = str(text if text is not None else "")
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


def _render_news_bucket_toon(
    key: str,
    items: List[Any],
    *,
    indent: int = 0,
    delimiter: str = _DEFAULT_DELIMITER,
) -> str:
    ind = _INDENT * indent
    name = _quote_key(key, delimiter) or "items"
    if not items:
        return f"{ind}{name}[0]:"

    dict_rows = [row for row in items if isinstance(row, dict)]
    if len(dict_rows) != len(items):
        return _encode_expanded_array(key, items, indent, delimiter)

    include_kind = key not in {"upcoming_events", "recent_events"} and any(not _is_empty_value(row.get("kind")) for row in dict_rows)
    include_summary = any(not _is_empty_value(row.get("summary")) for row in dict_rows)
    headers = ["title", "time"]
    if include_kind:
        headers.append("kind")
    if include_summary:
        headers.append("summary")

    header_line = delimiter.join(_quote_key(h, delimiter) for h in headers)
    lines = [f"{ind}{name}[{len(items)}]{{{header_line}}}:"]
    row_indent = ind + _INDENT
    for row in dict_rows:
        values: List[str] = []
        title = row.get("title")
        if not _is_empty_value(title):
            values.append(_quote_always(title))

        display_time = row.get("time_utc")
        if _is_empty_value(display_time):
            display_time = row.get("relative_time")
        if not _is_empty_value(display_time):
            values.append(_stringify_for_toon_value(display_time, None, delimiter))

        kind = row.get("kind")
        if include_kind and not _is_empty_value(kind):
            values.append(_stringify_for_toon_value(kind, None, delimiter))

        summary = row.get("summary")
        if include_summary and not _is_empty_value(summary):
            values.append(_quote_always(summary))

        if values:
            lines.append(f"{row_indent}{delimiter.join(values)}")
    return "\n".join(lines)


def _render_news_payload(
    payload: Dict[str, Any],
    *,
    verbose: bool,
    tool_name: str,
    simplify_numbers: bool = True,
) -> Optional[str]:
    if tool_name != "news" or verbose:
        return None

    bucket_keys = {"general_news", "related_news", "impact_news", "upcoming_events", "recent_events"}
    if not any(isinstance(payload.get(key), list) for key in bucket_keys):
        return None

    lines: List[str] = []
    for key, value in payload.items():
        if _is_empty_value(value):
            continue
        if key in bucket_keys and isinstance(value, list):
            lines.append(_render_news_bucket_toon(key, value))
            continue
        chunk = _format_to_toon(
            value,
            key=key,
            indent=0,
            delimiter=_DEFAULT_DELIMITER,
            simplify_numbers=simplify_numbers,
        )
        if chunk:
            lines.append(chunk)
    return "\n".join(lines) if lines else None


def _dedupe_text_list(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _resolve_tool_name(result: Any, tool_name: Optional[str]) -> str:
    if isinstance(tool_name, str) and tool_name.strip():
        return tool_name.strip()
    if isinstance(result, dict):
        meta = result.get("meta")
        if isinstance(meta, dict):
            meta_tool = str(meta.get("tool") or "").strip()
            if meta_tool:
                return meta_tool
        operation = str(result.get("operation") or "").strip()
        if operation:
            return operation
    return ""


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
            return _stringify_nonfinite_number(value)
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
    if isinstance(value, (dict, list, tuple, set)):
        rendered = _stringify_cell(value)
        return _quote_if_needed(rendered, delimiter) if rendered else ""
    if isinstance(value, Number):
        try:
            num = float(value)
        except Exception:
            return _quote_if_needed(str(value), delimiter)
        if not math.isfinite(num):
            return _stringify_nonfinite_number(num)
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


def _normalize_forecast_payload(payload: Dict[str, Any], verbose: bool = True) -> Optional[Dict[str, Any]]:  # noqa: C901
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

        include_interval_columns = bool(lower and upper)
        usable_qcols: List[str] = []
        for q in qcols:
            qarr = qmap.get(q) if isinstance(qmap, dict) else None  # type: ignore[assignment]
            if isinstance(qarr, list) and qarr:
                usable_qcols.append(q)

        headers = ['time', 'forecast']
        if include_interval_columns:
            headers += ['lower', 'upper']
        for q in usable_qcols:
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
            if include_interval_columns:
                low_val = lower[i] if i < len(lower) else None
                up_val = upper[i] if i < len(upper) else None
                if digits is not None:
                    try:
                        if isinstance(low_val, (int, float)): low_val = f"{float(low_val):.{digits}f}"
                        if isinstance(up_val, (int, float)): up_val = f"{float(up_val):.{digits}f}"
                    except Exception:
                        pass
                row['lower'] = low_val
                row['upper'] = up_val
            for q in usable_qcols:
                qarr = qmap.get(q) if isinstance(qmap, dict) else None  # type: ignore[assignment]
                if not isinstance(qarr, list):
                    continue
                q_val = qarr[i] if i < len(qarr) else None
                if digits is not None and isinstance(q_val, (int, float)):
                    try:
                        q_val = f"{float(q_val):.{digits}f}"
                    except Exception:
                        pass
                row[f"q{q}"] = q_val
            rows.append(row)

        out: Dict[str, Any] = {}
        success_value = payload.get("success")
        if isinstance(success_value, bool):
            out["success"] = success_value

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


def _normalize_triple_barrier_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert triple-barrier column arrays into a single tabular block."""
    entries = payload.get("entries")
    labels = payload.get("labels")
    holding_bars = payload.get("holding_bars")
    if not isinstance(entries, list) or not isinstance(labels, list) or not isinstance(holding_bars, list):
        return None
    if any(isinstance(item, dict) for item in labels):
        return None

    n = min(len(entries), len(labels), len(holding_bars))
    if n <= 0:
        return None

    tp_times_raw = payload.get("tp_time")
    sl_times_raw = payload.get("sl_time")
    tp_times = list(tp_times_raw) if isinstance(tp_times_raw, list) else []
    sl_times = list(sl_times_raw) if isinstance(sl_times_raw, list) else []

    rows: List[Dict[str, Any]] = []
    for idx in range(n):
        row: Dict[str, Any] = {
            "entry": entries[idx],
            "label": labels[idx],
            "holding_bars": holding_bars[idx],
        }
        if "tp_time" in payload:
            row["tp_time"] = tp_times[idx] if idx < len(tp_times) else None
        if "sl_time" in payload:
            row["sl_time"] = sl_times[idx] if idx < len(sl_times) else None
        rows.append(row)

    out: Dict[str, Any] = {}
    for key in ("success", "symbol", "timeframe", "horizon"):
        value = payload.get(key)
        if not _is_empty_value(value):
            out[key] = value

    out["labels"] = rows

    summary = payload.get("summary")
    if isinstance(summary, dict) and summary:
        out["summary"] = summary

    for key in ("params_used", "meta"):
        value = payload.get(key)
        if not _is_empty_value(value):
            out[key] = value

    return out


def _extract_sl_tp_levels(levels: Any) -> tuple[Optional[Any], Optional[Any]]:
    if not isinstance(levels, dict):
        return None, None
    return levels.get("sl"), levels.get("tp")


def _is_informational_trade_warning(message: str) -> bool:
    lowered = str(message).strip().lower()
    if not lowered:
        return False
    return any(
        marker in lowered
        for marker in (
            "comment sanitized",
            "comment truncated",
            "broker rejected the comment field",
        )
    )


def _compact_trade_warnings(warnings: Any, *, verbose: bool) -> List[str]:
    if not isinstance(warnings, list):
        return []
    cleaned = _dedupe_text_list(warnings)
    if verbose:
        return cleaned
    actionable = [msg for msg in cleaned if not _is_informational_trade_warning(msg)]
    if not actionable:
        return []
    critical = [
        msg for msg in actionable
        if any(
            marker in msg.lower()
            for marker in (
                "critical:",
                "trade_modify",
                "close the position",
                "verify the live position is protected",
                "auto-close failed",
                "unprotected",
                "fallback modification",
            )
        )
    ]
    if critical:
        return critical[:1]
    return actionable[:1]


def _maybe_add_trade_key(
    out: Dict[str, Any],
    key: str,
    value: Any,
    *,
    skip_zero: bool = False,
) -> None:
    if _is_empty_value(value):
        return
    if skip_zero:
        try:
            if float(value) == 0.0:
                return
        except Exception:
            pass
    out[key] = value


def _compact_trade_row(row: Dict[str, Any], *, verbose: bool) -> Optional[Dict[str, Any]]:
    out: Dict[str, Any] = {}
    _maybe_add_trade_key(out, "ticket", row.get("ticket"), skip_zero=True)
    _maybe_add_trade_key(out, "error", row.get("error"))
    _maybe_add_trade_key(out, "retcode_name", row.get("retcode_name"))
    if "retcode_name" not in out:
        _maybe_add_trade_key(out, "retcode", row.get("retcode"))
    _maybe_add_trade_key(out, "order", row.get("order"), skip_zero=True)
    _maybe_add_trade_key(out, "deal", row.get("deal"), skip_zero=True)
    _maybe_add_trade_key(out, "volume", row.get("volume"))
    _maybe_add_trade_key(out, "requested_volume", row.get("requested_volume"))
    _maybe_add_trade_key(out, "price", row.get("close_price"))
    if "price" not in out:
        _maybe_add_trade_key(out, "price", row.get("applied_price"))
    if "price" not in out:
        _maybe_add_trade_key(out, "price", row.get("price"))
    _maybe_add_trade_key(out, "pnl", row.get("pnl"))
    _maybe_add_trade_key(out, "remaining_volume", row.get("position_volume_remaining_estimate"))
    _maybe_add_trade_key(out, "message", row.get("message"))

    if verbose:
        diagnostics: Dict[str, Any] = {}
        for key in ("comment", "attempts", "last_error", "open_price", "pnl_price_delta", "duration_seconds"):
            value = row.get(key)
            if not _is_empty_value(value):
                diagnostics[key] = value
        if diagnostics:
            out["diagnostics"] = diagnostics
    return out or None


def _trade_table_hidden_keys(tool_name: str) -> set[str]:
    if tool_name == "trade_get_open" or tool_name == "trade_get_pending":
        return {
            "Comment Length",
            "Comment Limit",
            "Comment May Be Truncated",
        }
    if tool_name == "trade_history":
        return {
            "timestamp_timezone",
            "comment_visible_length",
            "comment_max_length",
            "comment_may_be_truncated",
            "type_code",
            "state_code",
            "type_time_code",
            "type_filling_code",
            "entry_code",
            "reason_code",
            "time_setup_msc",
            "time_done_msc",
            "type_time",
            "type_filling",
            "position_by_id",
            "price_stoplimit",
            "external_id",
        }
    return set()


def _normalize_trade_table_payload(
    payload: Any,
    *,
    verbose: bool,
    tool_name: str,
) -> Optional[Any]:
    if tool_name not in {"trade_get_open", "trade_get_pending", "trade_history"}:
        return None
    if verbose or not isinstance(payload, list):
        return payload if isinstance(payload, list) else None

    hidden = _trade_table_hidden_keys(tool_name)
    if not hidden:
        return payload

    normalized_rows: List[Any] = []
    for row in payload:
        if not isinstance(row, dict):
            normalized_rows.append(row)
            continue
        normalized_rows.append({key: value for key, value in row.items() if key not in hidden})
    return normalized_rows


def _normalize_trade_payload(
    payload: Dict[str, Any],
    *,
    verbose: bool,
    tool_name: str,
) -> Optional[Dict[str, Any]]:
    trade_tools = {"trade_place", "trade_modify", "trade_close"}
    trade_markers = {
        "retcode_name",
        "checked_scopes",
        "protection_status",
        "sl_tp_result",
        "comment_sanitization",
        "comment_truncation",
        "comment_fallback",
        "fill_mode_attempts",
        "closed_count",
        "cancelled_count",
    }
    if tool_name not in trade_tools and not trade_markers.intersection(payload.keys()):
        return None

    out: Dict[str, Any] = {}
    if "error" in payload and not _is_empty_value(payload.get("error")):
        out["error"] = payload.get("error")
        _maybe_add_trade_key(out, "checked_scopes", payload.get("checked_scopes"))
        _maybe_add_trade_key(out, "message", payload.get("message"))
        warnings_out = _compact_trade_warnings(payload.get("warnings"), verbose=verbose)
        if warnings_out:
            out["warnings"] = warnings_out
        return out

    success_value = payload.get("success")
    if isinstance(success_value, bool):
        out["success"] = success_value
    elif "retcode_name" in payload or "retcode" in payload or "order" in payload or "deal" in payload:
        out["success"] = True

    if "results" in payload and isinstance(payload.get("results"), list):
        for key in ("closed_count", "cancelled_count", "attempted_count", "message", "no_action"):
            _maybe_add_trade_key(out, key, payload.get(key))
        rows: List[Dict[str, Any]] = []
        for row in payload.get("results", []):
            if not isinstance(row, dict):
                continue
            compacted = _compact_trade_row(row, verbose=verbose)
            if compacted:
                rows.append(compacted)
        if rows or payload.get("results") == []:
            out["results"] = rows
        return out

    resolved_ticket = payload.get("position_ticket")
    if _is_empty_value(resolved_ticket):
        resolved_ticket = payload.get("ticket")
    order_value = payload.get("order")

    _maybe_add_trade_key(out, "retcode_name", payload.get("retcode_name"))
    if "retcode_name" not in out:
        _maybe_add_trade_key(out, "retcode", payload.get("retcode"))
    _maybe_add_trade_key(out, "dry_run", payload.get("dry_run"))
    _maybe_add_trade_key(out, "trade_gate_passed", payload.get("trade_gate_passed"))
    _maybe_add_trade_key(out, "actionability", payload.get("actionability"))
    _maybe_add_trade_key(out, "symbol", payload.get("symbol"))
    _maybe_add_trade_key(out, "order_type", payload.get("order_type"))
    _maybe_add_trade_key(out, "pending", payload.get("pending"))
    _maybe_add_trade_key(out, "action", payload.get("action"))
    if verbose or str(order_value).strip() != str(resolved_ticket).strip():
        _maybe_add_trade_key(out, "order", order_value, skip_zero=True)
    _maybe_add_trade_key(out, "deal", payload.get("deal"), skip_zero=True)
    _maybe_add_trade_key(out, "ticket", resolved_ticket, skip_zero=True)
    _maybe_add_trade_key(out, "volume", payload.get("volume"))
    _maybe_add_trade_key(out, "price", payload.get("requested_price"))
    if "price" not in out:
        _maybe_add_trade_key(out, "price", payload.get("applied_price"))
    if "price" not in out:
        _maybe_add_trade_key(out, "price", payload.get("close_price"))
    if "price" not in out:
        _maybe_add_trade_key(out, "price", payload.get("price"), skip_zero=True)

    requested_sl = payload.get("requested_sl")
    requested_tp = payload.get("requested_tp")
    applied_sl = payload.get("applied_sl")
    applied_tp = payload.get("applied_tp")
    protection_error = None
    sl_tp_result = payload.get("sl_tp_result")
    if isinstance(sl_tp_result, dict):
        sl_req, tp_req = _extract_sl_tp_levels(sl_tp_result.get("requested"))
        sl_applied, tp_applied = _extract_sl_tp_levels(sl_tp_result.get("applied"))
        requested_sl = requested_sl if requested_sl is not None else sl_req
        requested_tp = requested_tp if requested_tp is not None else tp_req
        applied_sl = applied_sl if applied_sl is not None else sl_applied
        applied_tp = applied_tp if applied_tp is not None else tp_applied
        if str(sl_tp_result.get("status") or "").strip().lower() == "failed":
            protection_error = sl_tp_result.get("error")
    _maybe_add_trade_key(out, "requested_sl", requested_sl)
    _maybe_add_trade_key(out, "requested_tp", requested_tp)
    _maybe_add_trade_key(out, "applied_sl", applied_sl)
    _maybe_add_trade_key(out, "applied_tp", applied_tp)
    _maybe_add_trade_key(out, "expiration", payload.get("expiration"))
    _maybe_add_trade_key(out, "expiration_normalized", payload.get("expiration_normalized"))
    _maybe_add_trade_key(out, "requested_expiration", payload.get("requested_expiration"))
    _maybe_add_trade_key(out, "applied_expiration", payload.get("applied_expiration"))
    _maybe_add_trade_key(out, "protection_status", payload.get("protection_status"))
    _maybe_add_trade_key(out, "protection_error", protection_error)
    _maybe_add_trade_key(out, "validation_scope", payload.get("validation_scope"))
    _maybe_add_trade_key(out, "preview_scope_summary", payload.get("preview_scope_summary"))
    _maybe_add_trade_key(out, "require_sl_tp", payload.get("require_sl_tp"))
    _maybe_add_trade_key(out, "auto_close_on_sl_tp_fail", payload.get("auto_close_on_sl_tp_fail"))
    _maybe_add_trade_key(out, "pnl", payload.get("pnl"))
    _maybe_add_trade_key(out, "remaining_volume", payload.get("position_volume_remaining_estimate"))
    _maybe_add_trade_key(out, "no_action", payload.get("no_action"))
    _maybe_add_trade_key(out, "message", payload.get("message"))
    _maybe_add_trade_key(out, "actionability_reason", payload.get("actionability_reason"))

    warnings_out = _compact_trade_warnings(payload.get("warnings"), verbose=verbose)
    if warnings_out:
        out["warnings"] = warnings_out

    if verbose:
        diagnostics: Dict[str, Any] = {}
        for key in (
            "retcode",
            "comment",
            "request_id",
            "bid",
            "ask",
            "type_filling_used",
            "validation_passed",
            "validation_not_performed",
            "position_ticket_candidates",
            "position_ticket_resolution",
            "ticket_requested",
            "ticket_resolution",
            "comment_sanitization",
            "comment_truncation",
            "comment_fallback",
            "fill_mode_attempts",
            "sl_tp_result",
            "auto_close_result",
        ):
            value = payload.get(key)
            if not _is_empty_value(value):
                diagnostics[key] = value
        if diagnostics:
            out["diagnostics"] = diagnostics

    return out


def _normalize_market_ticker_payload(
    payload: Dict[str, Any],
    *,
    verbose: bool,
    tool_name: str,
) -> Optional[Dict[str, Any]]:
    if tool_name != "market_ticker":
        return None

    out: Dict[str, Any] = {}
    if "error" in payload and not _is_empty_value(payload.get("error")):
        out["error"] = payload.get("error")
        return out

    for key in (
        "success",
        "symbol",
        "type",
        "price_precision",
        "bid",
        "ask",
        "last",
        "spread",
        "spread_points",
        "spread_pct",
        "spread_usd",
    ):
        value = payload.get(key)
        if not _is_empty_value(value):
            out[key] = value

    display_time = payload.get("time_display")
    epoch_time = payload.get("time_epoch")
    if _is_empty_value(epoch_time):
        raw_time = payload.get("time")
        if isinstance(raw_time, (int, float)):
            epoch_time = raw_time
    canonical_time = display_time
    if _is_empty_value(canonical_time):
        canonical_time = payload.get("time")
    if not _is_empty_value(canonical_time):
        out["time"] = canonical_time
    if verbose and not _is_empty_value(epoch_time):
        out["time_epoch"] = epoch_time

    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict) and diagnostics:
        out["diagnostics"] = diagnostics

    timezone = payload.get("timezone")
    if not _is_empty_value(timezone):
        out["timezone"] = timezone

    meta = payload.get("meta")
    if isinstance(meta, dict) and meta:
        out["meta"] = meta

    return out


def _compact_barrier_rows(
    rows: Any,
    *,
    preferred_keys: List[str],
) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(rows, list):
        return None

    dict_rows = [row for row in rows if isinstance(row, dict)]
    if not dict_rows:
        return []

    active_keys = [
        key
        for key in preferred_keys
        if any(key in row and not _is_empty_value(row.get(key)) for row in dict_rows)
    ]
    compacted: List[Dict[str, Any]] = []
    for row in dict_rows:
        compacted.append(
            {
                key: row.get(key)
                for key in active_keys
                if key in row and not _is_empty_value(row.get(key))
            }
        )
    return compacted


def _normalize_barrier_optimize_payload(
    payload: Dict[str, Any],
    *,
    verbose: bool,
    tool_name: str,
) -> Optional[Dict[str, Any]]:
    if tool_name != "forecast_barrier_optimize" or verbose:
        return None

    out = dict(payload)
    best = payload.get("best")
    if isinstance(best, dict):
        best_keys = [
            "tp",
            "sl",
            "rr",
            "tp_price",
            "sl_price",
            "prob_win",
            "prob_loss",
            "prob_resolve",
            "ev",
            "profit_factor",
            "breakeven_win_rate",
            "edge_vs_breakeven",
            "phantom_profit_risk",
            "low_confidence",
        ]
        out["best"] = {
            key: best.get(key)
            for key in best_keys
            if key in best and not _is_empty_value(best.get(key))
        }

    row_keys = [
        "tp",
        "sl",
        "rr",
        "tp_price",
        "sl_price",
        "prob_win",
        "prob_loss",
        "prob_no_hit",
        "prob_resolve",
        "ev",
        "profit_factor",
        "t_hit_resolve_median",
        "low_confidence",
    ]
    compact_results = _compact_barrier_rows(payload.get("results"), preferred_keys=row_keys)
    if compact_results is not None:
        out["results"] = compact_results

    if "grid" in out and "results" in out:
        out.pop("grid", None)
    else:
        compact_grid = _compact_barrier_rows(payload.get("grid"), preferred_keys=row_keys)
        if compact_grid is not None:
            out["grid"] = compact_grid

    return out


def _normalize_analysis_legends_payload(
    payload: Dict[str, Any],
    *,
    verbose: bool,
    tool_name: str,
) -> Optional[Dict[str, Any]]:
    if verbose or tool_name not in {"causal_discover_signals", "correlation_matrix", "cointegration_test"}:
        return None
    if "legends" not in payload:
        return None
    out = dict(payload)
    out.pop("legends", None)
    return out


def _normalize_regime_all_payload(
    payload: Dict[str, Any],
    *,
    verbose: bool,
    tool_name: str,
) -> Optional[Dict[str, Any]]:
    if tool_name != "regime_detect" or verbose:
        return None
    if str(payload.get("method") or "").strip().lower() != "all":
        return None

    detail_value = str(payload.get("detail") or "compact").strip().lower()
    if detail_value == "full":
        return None

    comparison_in = payload.get("comparison")
    if not isinstance(comparison_in, dict):
        return None

    out: Dict[str, Any] = {}
    for key in ("success", "symbol", "timeframe", "method", "target", "detail"):
        value = payload.get(key)
        if not _is_empty_value(value):
            out[key] = value

    comparison_out: Dict[str, Any] = {}
    current_regimes_in = comparison_in.get("current_regimes")
    if isinstance(current_regimes_in, dict):
        rows: List[Dict[str, Any]] = []
        for method_name, value in current_regimes_in.items():
            if not isinstance(value, dict):
                rows.append({"method": method_name, "value": value})
                continue
            compact = {"method": method_name}
            for key in ("bias", "direction", "volatility"):
                if key in value and not _is_empty_value(value.get(key)):
                    compact[key] = value.get(key)
            summary_value = value.get("status")
            if _is_empty_value(summary_value):
                summary_value = value.get("label")
            if not _is_empty_value(summary_value):
                compact["summary"] = summary_value
            if "regime_confidence" in value and not _is_empty_value(value.get("regime_confidence")):
                compact["regime_confidence"] = value.get("regime_confidence")
            rows.append(compact)
        if rows:
            comparison_out["current_regimes"] = rows

    agreement_in = comparison_in.get("agreement")
    if isinstance(agreement_in, dict):
        agreement_out: Dict[str, Any] = {}
        for name, value in agreement_in.items():
            if isinstance(value, dict):
                compact = {
                    key: value.get(key)
                    for key in ("majority", "agreement_pct")
                    if key in value and not _is_empty_value(value.get(key))
                }
                if compact:
                    agreement_out[str(name)] = compact
            elif not _is_empty_value(value):
                agreement_out[str(name)] = value
        if agreement_out:
            comparison_out["agreement"] = agreement_out

    methods_failed = comparison_in.get("methods_failed")
    if isinstance(methods_failed, list) and methods_failed:
        comparison_out["methods_failed"] = methods_failed

    if comparison_out:
        out["comparison"] = comparison_out

    if detail_value != "full" and ("results" in payload or "params_used" in payload):
        out["show_all_hint"] = (
            "Use --detail summary for stats only, or --detail full / --verbose for per-method details."
        )

    return out


def _normalize_forecast_methods_payload(
    payload: Dict[str, Any],
    *,
    verbose: bool,
    tool_name: str,
) -> Optional[Dict[str, Any]]:
    if tool_name != "forecast_list_methods" or verbose:
        return None

    if "error" in payload and not _is_empty_value(payload.get("error")):
        return {"error": payload.get("error")}

    out: Dict[str, Any] = {
        "success": bool(payload.get("success")) if isinstance(payload.get("success"), bool) else True,
    }
    for key in ("detail", "total", "total_filtered", "available", "unavailable", "methods_shown", "methods_hidden"):
        value = payload.get(key)
        if not _is_empty_value(value):
            out[key] = value

    filters = payload.get("filters")
    if isinstance(filters, dict):
        filters_out = {str(key): value for key, value in filters.items() if not _is_empty_value(value)}
        if filters_out:
            out["filters"] = filters_out

    methods = payload.get("methods")
    if isinstance(methods, list):
        dict_rows = [row for row in methods if isinstance(row, dict)]
        if len(dict_rows) == len(methods):
            compact_rows: List[Dict[str, Any]] = []
            for row in dict_rows:
                compact = {
                    key: row.get(key)
                    for key in ("method", "category", "available")
                    if key in row and not _is_empty_value(row.get(key))
                }
                compact_rows.append(compact or dict(row))
            out["methods"] = compact_rows
        else:
            out["methods"] = methods

    hidden = payload.get("methods_hidden")
    try:
        if hidden is not None and int(hidden) > 0:
            out["show_all_hint"] = "Use --detail full to see all methods."
    except Exception:
        pass

    return out


def _normalize_library_models_payload(
    payload: Dict[str, Any],
    *,
    verbose: bool,
    tool_name: str,
) -> Optional[Dict[str, Any]]:
    if tool_name != "forecast_list_library_models" or verbose:
        return None

    if "error" in payload and not _is_empty_value(payload.get("error")):
        return {"error": payload.get("error")}

    out: Dict[str, Any] = {}
    library = payload.get("library")
    if not _is_empty_value(library):
        out["library"] = library

    capabilities = payload.get("capabilities")
    compact_rows: List[Dict[str, Any]] = []
    if isinstance(capabilities, list):
        dict_rows = [row for row in capabilities if isinstance(row, dict)]
        if len(dict_rows) == len(capabilities):
            for row in dict_rows:
                model_name = row.get("display_name")
                if _is_empty_value(model_name):
                    model_name = row.get("method")
                if _is_empty_value(model_name):
                    model_name = row.get("adapter_method")
                description = row.get("description")
                if isinstance(description, str):
                    description = description.splitlines()[0].strip()
                compact = {
                    "model": model_name,
                    "available": row.get("available"),
                    "description": description,
                }
                compact_rows.append(
                    {
                        key: value
                        for key, value in compact.items()
                        if not _is_empty_value(value)
                    }
                    or dict(row)
                )

    if not compact_rows:
        models = payload.get("models")
        if isinstance(models, list):
            for item in models:
                if isinstance(item, dict):
                    compact = {
                        "model": item.get("display_name") or item.get("method"),
                        "description": item.get("description") or item.get("notes"),
                    }
                    compact_rows.append(
                        {
                            key: value
                            for key, value in compact.items()
                            if not _is_empty_value(value)
                        }
                        or dict(item)
                    )
                elif not _is_empty_value(item):
                    compact_rows.append({"model": item})

    if compact_rows:
        out["models"] = compact_rows
        out["total_models"] = len(compact_rows)

    note = payload.get("note")
    if not _is_empty_value(note):
        out["note"] = note

    usage = payload.get("usage")
    if not _is_empty_value(usage):
        out["usage"] = usage

    if compact_rows and "capabilities" in payload:
        out["show_all_hint"] = "Use --json for full capability metadata and params."

    return out


def _compact_support_resistance_level(
    value: Any,
    *,
    preferred_keys: List[str],
) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    out = {
        key: value.get(key)
        for key in preferred_keys
        if key in value and not _is_empty_value(value.get(key))
    }
    return out or None


def _normalize_support_resistance_payload(
    payload: Dict[str, Any],
    *,
    verbose: bool,
    tool_name: str,
) -> Optional[Dict[str, Any]]:
    if tool_name != "support_resistance_levels" or verbose:
        return None

    if "error" in payload and not _is_empty_value(payload.get("error")):
        return {"error": payload.get("error")}

    out: Dict[str, Any] = {}
    for key in ("success", "symbol", "timeframe", "mode", "method", "current_price", "level_counts"):
        value = payload.get(key)
        if not _is_empty_value(value):
            out[key] = value

    nearest_in = payload.get("nearest")
    if isinstance(nearest_in, dict):
        nearest_out: Dict[str, Any] = {}
        for side in ("support", "resistance"):
            compact = _compact_support_resistance_level(
                nearest_in.get(side),
                preferred_keys=["value", "distance_pct", "touches", "status"],
            )
            if compact:
                nearest_out[side] = compact
        if nearest_out:
            out["nearest"] = nearest_out

    levels_in = payload.get("levels")
    if isinstance(levels_in, list):
        compact_levels = [
            compact
            for compact in (
                _compact_support_resistance_level(
                    row,
                    preferred_keys=["type", "value", "distance_pct", "touches", "status"],
                )
                for row in levels_in
            )
            if compact
        ]
        if compact_levels:
            out["levels"] = compact_levels

    fibonacci_in = payload.get("fibonacci")
    if isinstance(fibonacci_in, dict):
        fibonacci_out: Dict[str, Any] = {}
        fib_timeframe = fibonacci_in.get("selected_timeframe")
        if _is_empty_value(fib_timeframe):
            fib_timeframe = fibonacci_in.get("timeframe")
        if not _is_empty_value(fib_timeframe):
            fibonacci_out["timeframe"] = fib_timeframe

        fib_nearest_in = fibonacci_in.get("nearest")
        if isinstance(fib_nearest_in, dict):
            fib_nearest_out: Dict[str, Any] = {}
            for side in ("support", "resistance"):
                compact = _compact_support_resistance_level(
                    fib_nearest_in.get(side),
                    preferred_keys=["label", "value", "distance_pct"],
                )
                if compact:
                    fib_nearest_out[side] = compact
            if fib_nearest_out:
                fibonacci_out["nearest"] = fib_nearest_out

        fib_levels_in = fibonacci_in.get("levels")
        if isinstance(fib_levels_in, list):
            fib_levels = [
                compact
                for compact in (
                    _compact_support_resistance_level(
                        row,
                        preferred_keys=["label", "type", "value", "distance_pct"],
                    )
                    for row in fib_levels_in
                )
                if compact
            ]
            if fib_levels:
                fibonacci_out["levels"] = fib_levels

        if fibonacci_out:
            out["fibonacci"] = fibonacci_out

    warnings_out = payload.get("warnings")
    if isinstance(warnings_out, list) and warnings_out:
        out["warnings"] = warnings_out

    if any(key in payload for key in ("coverage_gaps", "zone_overlap", "meta", "timeframes_analyzed", "window")):
        out["show_all_hint"] = (
            "Use --detail full for timeframe selection rationale, zone widths, and coverage diagnostics."
        )

    return out


def _compact_forecast_ci(
    payload: Dict[str, Any],
    *,
    lower: List[Any],
    upper: List[Any],
) -> Dict[str, Any]:
    """Emit CI diagnostics only when they add signal beyond the forecast table."""
    ci_status = payload.get('ci_status')
    if _is_empty_value(ci_status):
        ci_available = payload.get('ci_available')
        if ci_available is True:
            ci_status = 'available'
        elif payload.get('ci_unavailable') or ci_available is False:
            ci_status = 'unavailable'
        elif bool(payload.get('ci_requested')):
            ci_status = 'requested'
    has_interval_columns = bool(lower and upper)

    alpha = None
    for key in ('ci_alpha', 'ci_alpha_requested'):
        value = payload.get(key)
        if not _is_empty_value(value):
            alpha = value
            break

    if ci_status == 'available' and has_interval_columns:
        return {}

    if _is_empty_value(ci_status) and alpha is None:
        return {}

    out: Dict[str, Any] = {}
    if not _is_empty_value(ci_status):
        out['status'] = ci_status

    if alpha is not None:
        out['ci_alpha'] = alpha

    return out


def _build_forecast_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Group verbose forecast metadata into stable sections."""
    meta_in = payload.get('meta')
    meta_existing = dict(meta_in) if isinstance(meta_in, dict) else {}

    domain_in = meta_existing.get('domain')
    domain: Dict[str, Any] = dict(domain_in) if isinstance(domain_in, dict) else {}
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

    tool_name = str(meta_existing.get('tool') or '').strip()

    runtime_in = meta_existing.get('runtime')
    runtime: Dict[str, Any] = dict(runtime_in) if isinstance(runtime_in, dict) else {}

    existing_runtime_timezone = runtime.get('timezone')
    timezone_source = existing_runtime_timezone if isinstance(existing_runtime_timezone, dict) else None
    if timezone_source is None:
        try:
            from ..core.runtime_metadata import build_runtime_timezone_meta

            generated_timezone = build_runtime_timezone_meta(
                payload,
                include_local=False,
                include_now=False,
            )
            if isinstance(generated_timezone, dict):
                timezone_source = generated_timezone
        except Exception:
            timezone_source = None

    runtime_timezone = _normalize_timezone_display_meta(timezone_source)
    if runtime_timezone:
        runtime['timezone'] = runtime_timezone
    elif 'timezone' in runtime and _is_empty_value(runtime.get('timezone')):
        runtime.pop('timezone', None)

    meta: Dict[str, Any] = {}
    for key, value in meta_existing.items():
        if key in {'tool', 'domain', 'runtime', 'cli'} or _is_empty_value(value):
            continue
        meta[str(key)] = value
    if tool_name:
        meta['tool'] = tool_name
    if domain:
        meta['domain'] = domain
    if runtime:
        meta['runtime'] = runtime
    return meta


def _timezone_value_from_any(value: Any) -> Any:
    if _is_empty_value(value):
        return None
    if isinstance(value, dict):
        if 'tz' in value:
            nested = _timezone_value_from_any(value.get('tz'))
            if not _is_empty_value(nested):
                return nested
        for key in ('resolved', 'configured', 'value', 'hint', 'name'):
            candidate = value.get(key)
            if not _is_empty_value(candidate):
                return candidate
        return None
    return value


def _normalize_timezone_display_meta(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    out: Dict[str, Any] = {}

    utc_meta = value.get('utc')
    if isinstance(utc_meta, dict) or not _is_empty_value(utc_meta):
        utc_out: Dict[str, Any] = {}
        if isinstance(utc_meta, dict):
            utc_out['tz'] = _timezone_value_from_any(utc_meta) or 'UTC'
            now = utc_meta.get('now')
            if not _is_empty_value(now):
                utc_out['now'] = now
        else:
            utc_out['tz'] = _timezone_value_from_any(utc_meta) or 'UTC'
        if utc_out:
            out['utc'] = utc_out

    server_meta = value.get('server')
    if isinstance(server_meta, dict) or not _is_empty_value(server_meta):
        server_out: Dict[str, Any] = {}
        if isinstance(server_meta, dict):
            source = server_meta.get('source')
            if not _is_empty_value(source):
                server_out['source'] = source
            tz_value = _timezone_value_from_any(server_meta)
            if not _is_empty_value(tz_value):
                server_out['tz'] = tz_value
            offset_seconds = server_meta.get('offset_seconds')
            if not _is_empty_value(offset_seconds):
                server_out['offset_seconds'] = offset_seconds
            now = server_meta.get('now')
            if not _is_empty_value(now):
                server_out['now'] = now
        else:
            tz_value = _timezone_value_from_any(server_meta)
            if not _is_empty_value(tz_value):
                server_out['tz'] = tz_value
        if server_out:
            out['server'] = server_out

    client_meta = value.get('client')
    output_meta = value.get('output')
    output_tz = _timezone_value_from_any(output_meta)
    if isinstance(client_meta, dict) or not _is_empty_value(client_meta) or not _is_empty_value(output_tz):
        client_out: Dict[str, Any] = {}
        if isinstance(client_meta, dict):
            tz_value = _timezone_value_from_any(client_meta)
            if _is_empty_value(tz_value):
                tz_value = output_tz
            if not _is_empty_value(tz_value):
                client_out['tz'] = tz_value
            now = client_meta.get('now')
            if not _is_empty_value(now):
                client_out['now'] = now
        else:
            tz_value = _timezone_value_from_any(client_meta)
            if _is_empty_value(tz_value):
                tz_value = output_tz
            if not _is_empty_value(tz_value):
                client_out['tz'] = tz_value
        if client_out:
            out['client'] = client_out

    return out


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


def format_result_minimal(
    result: Any,
    verbose: bool = True,
    *,
    simplify_numbers: bool = True,
    tool_name: Optional[str] = None,
) -> str:
    """Render tool outputs as TOON text."""
    if result is None:
        return ""
    try:
        normalized = result
        resolved_tool_name = _resolve_tool_name(result, tool_name)
        if isinstance(result, dict):
            news_rendered = _render_news_payload(
                result,
                verbose=verbose,
                tool_name=resolved_tool_name,
                simplify_numbers=simplify_numbers,
            )
            if news_rendered is not None:
                return news_rendered.strip()
        trade_table_norm = _normalize_trade_table_payload(
            result,
            verbose=verbose,
            tool_name=resolved_tool_name,
        )
        if trade_table_norm is not None:
            normalized = trade_table_norm
        if isinstance(result, dict):
            trade_norm = _normalize_trade_payload(
                result,
                verbose=verbose,
                tool_name=resolved_tool_name,
            )
            if trade_norm is not None:
                normalized = trade_norm
            else:
                market_ticker_norm = _normalize_market_ticker_payload(
                    result,
                    verbose=verbose,
                    tool_name=resolved_tool_name,
                )
                if market_ticker_norm is not None:
                    normalized = market_ticker_norm
                else:
                    barrier_optimize_norm = _normalize_barrier_optimize_payload(
                        result,
                        verbose=verbose,
                        tool_name=resolved_tool_name,
                    )
                    if barrier_optimize_norm is not None:
                        normalized = barrier_optimize_norm
                    else:
                        triple_barrier_norm = _normalize_triple_barrier_payload(result)
                        if triple_barrier_norm is not None:
                            normalized = triple_barrier_norm
                        else:
                            forecast_norm = _normalize_forecast_payload(result, verbose=verbose)
                            if forecast_norm is not None:
                                normalized = forecast_norm
        if isinstance(normalized, dict):
            analysis_legends_norm = _normalize_analysis_legends_payload(
                normalized,
                verbose=verbose,
                tool_name=resolved_tool_name,
            )
            if analysis_legends_norm is not None:
                normalized = analysis_legends_norm

            regime_all_norm = _normalize_regime_all_payload(
                normalized,
                verbose=verbose,
                tool_name=resolved_tool_name,
            )
            if regime_all_norm is not None:
                normalized = regime_all_norm

            forecast_methods_norm = _normalize_forecast_methods_payload(
                normalized,
                verbose=verbose,
                tool_name=resolved_tool_name,
            )
            if forecast_methods_norm is not None:
                normalized = forecast_methods_norm

            library_models_norm = _normalize_library_models_payload(
                normalized,
                verbose=verbose,
                tool_name=resolved_tool_name,
            )
            if library_models_norm is not None:
                normalized = library_models_norm

            support_resistance_norm = _normalize_support_resistance_payload(
                normalized,
                verbose=verbose,
                tool_name=resolved_tool_name,
            )
            if support_resistance_norm is not None:
                normalized = support_resistance_norm
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
