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
        cli_meta = result.get("cli_meta")
        if isinstance(cli_meta, dict):
            command = str(cli_meta.get("command") or "").strip()
            if command:
                return command
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

    _maybe_add_trade_key(out, "retcode_name", payload.get("retcode_name"))
    if "retcode_name" not in out:
        _maybe_add_trade_key(out, "retcode", payload.get("retcode"))
    _maybe_add_trade_key(out, "order", payload.get("order"), skip_zero=True)
    _maybe_add_trade_key(out, "deal", payload.get("deal"), skip_zero=True)
    _maybe_add_trade_key(out, "ticket", payload.get("position_ticket"), skip_zero=True)
    if "ticket" not in out:
        _maybe_add_trade_key(out, "ticket", payload.get("ticket"), skip_zero=True)
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
    _maybe_add_trade_key(out, "requested_expiration", payload.get("requested_expiration"))
    _maybe_add_trade_key(out, "applied_expiration", payload.get("applied_expiration"))
    _maybe_add_trade_key(out, "protection_status", payload.get("protection_status"))
    _maybe_add_trade_key(out, "protection_error", protection_error)
    _maybe_add_trade_key(out, "pnl", payload.get("pnl"))
    _maybe_add_trade_key(out, "remaining_volume", payload.get("position_volume_remaining_estimate"))
    _maybe_add_trade_key(out, "no_action", payload.get("no_action"))
    _maybe_add_trade_key(out, "message", payload.get("message"))

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
    cli_meta_in = payload.get('cli_meta')
    cli_meta = dict(cli_meta_in) if isinstance(cli_meta_in, dict) else {}
    if not tool_name:
        tool_name = str(cli_meta.pop('command', '')).strip()
    cli_timezone = cli_meta.pop('timezone', None)

    runtime_in = meta_existing.get('runtime')
    runtime: Dict[str, Any] = dict(runtime_in) if isinstance(runtime_in, dict) else {}

    existing_runtime_timezone = runtime.get('timezone')
    timezone_source = existing_runtime_timezone if isinstance(existing_runtime_timezone, dict) else None
    if timezone_source is None:
        timezone_source = cli_timezone if isinstance(cli_timezone, dict) else None
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
        if isinstance(result, dict):
            resolved_tool_name = _resolve_tool_name(result, tool_name)
            trade_norm = _normalize_trade_payload(
                result,
                verbose=verbose,
                tool_name=resolved_tool_name,
            )
            if trade_norm is not None:
                normalized = trade_norm
            else:
                triple_barrier_norm = _normalize_triple_barrier_payload(result)
                if triple_barrier_norm is not None:
                    normalized = triple_barrier_norm
                else:
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
