"""Generic TOON encoding helpers extracted from minimal_output."""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Dict, Iterable, List, Optional, cast

from ..shared.constants import DISPLAY_MAX_DECIMALS
from .formatting import format_float as _format_float
from .formatting import format_number
from .formatting import optimal_decimals as _optimal_decimals

_INDENT = "  "
_DEFAULT_DELIMITER = ","
_QUOTE_DECIMALS_BY_FIELD = {
    "bid": 8,
    "ask": 8,
    "mid": 8,
    "last": 8,
    "open": 8,
    "high": 8,
    "low": 8,
    "close": 8,
    "price": 8,
    "point": 8,
    "spread": 8,
    "spread_points": 4,
    "spread_pips": 4,
    "spread_pct": 6,
    "spread_usd": 6,
    "bidlow": 8,
    "bidhigh": 8,
    "asklow": 8,
    "askhigh": 8,
    "session_open": 8,
    "session_close": 8,
    "current_price": 8,
    "reference_price": 8,
    "level_price": 8,
    "target_price": 8,
    "entry_price": 8,
    "open_price": 8,
    "close_price": 8,
    "high_price": 8,
    "low_price": 8,
    "tp_price": 8,
    "sl_price": 8,
    "fast_ma": 8,
    "slow_ma": 8,
    "sma_value": 8,
    "ema_value": 8,
    "first_low": 8,
    "first_high": 8,
    "last_low": 8,
    "last_high": 8,
    "pivot": 8,
}


def _quote_decimals_for_field(field: Any) -> Optional[int]:
    text = str(field)
    forced = _QUOTE_DECIMALS_BY_FIELD.get(text)
    if forced is not None:
        return forced
    return _QUOTE_DECIMALS_BY_FIELD.get(text.lower())

_PRICE_CONTAINER_KEYS = {
    "levels",
    "nearest",
    "support",
    "resistance",
    "supports",
    "resistances",
    "active_levels",
}
_PRICE_LEVEL_KEYS = {
    "PP",
    *(f"R{idx}" for idx in range(1, 11)),
    *(f"S{idx}" for idx in range(1, 11)),
}

_QUOTE_STAT_DECIMAL_FIELDS = {
    "first",
    "last",
    "low",
    "high",
    "mean",
    "median",
    "q25",
    "q75",
    "std",
    "change",
}


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


def _format_number_fixed(num: float) -> str:
    return _format_float(num, DISPLAY_MAX_DECIMALS)


def _minify_number(num: float) -> str:
    return format_number(num)


def _stringify_nonfinite_number(num: float) -> str:
    if math.isnan(num):
        return "nan"
    if num > 0:
        return "inf"
    if num < 0:
        return "-inf"
    return "nan"


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


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


def _quote_if_needed(text: str, delimiter: str = _DEFAULT_DELIMITER) -> str:
    if text is None:
        return ""
    raw = str(text)
    needs_quote = raw.strip() != raw or any(
        ch in raw for ch in (delimiter, ":", "\n", "\r", '"', "|", "\t")
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
    if key is None:
        return ""
    return _quote_if_needed(str(key), delimiter)


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
    for header in headers:
        forced = _quote_decimals_for_field(header)
        if forced is not None:
            col_decimals[header] = int(forced)
            continue
        values: List[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            val = row.get(header)
            if val is None or isinstance(val, bool):
                continue
            if isinstance(val, Number):
                try:
                    num = float(cast(Any, val))
                except Exception:
                    continue
                if math.isfinite(num):
                    values.append(num)
        if values:
            col_decimals[header] = _optimal_decimals(values)
    return col_decimals


def _forced_scalar_decimals(
    key: Optional[str],
    *,
    parent_key: Optional[str] = None,
) -> Optional[int]:
    if key is None:
        return None
    forced = _quote_decimals_for_field(key)
    if forced is not None:
        return forced
    if "." in str(key):
        key_parts = str(key).split(".")
        parent, child = key_parts[0], key_parts[-1]
        if child in _QUOTE_STAT_DECIMAL_FIELDS:
            forced = _quote_decimals_for_field(parent)
            if forced is not None:
                return forced
        if any(part in _PRICE_CONTAINER_KEYS for part in key_parts[:-1]):
            if child in _PRICE_LEVEL_KEYS or child in {"value", "price"}:
                return _QUOTE_DECIMALS_BY_FIELD["price"]
    if parent_key is not None and str(key) in _QUOTE_STAT_DECIMAL_FIELDS:
        return _quote_decimals_for_field(parent_key)
    if parent_key is not None and str(parent_key) in _PRICE_CONTAINER_KEYS:
        key_text = str(key)
        if key_text in _PRICE_LEVEL_KEYS or key_text in {"value", "price"}:
            return _QUOTE_DECIMALS_BY_FIELD["price"]
    return None


def _collapse_single_key_path(key: str, value: Any) -> tuple[str, Any]:
    current_key = str(key)
    current_value = value
    while isinstance(current_value, dict):
        items = [
            (str(k), v) for k, v in current_value.items() if not _is_empty_value(v)
        ]
        if len(items) != 1:
            break
        subkey, subval = items[0]
        current_key = f"{current_key}.{subkey}"
        current_value = subval
    return current_key, current_value


def _stringify_for_toon(
    value: Any,
    delimiter: str = _DEFAULT_DELIMITER,
    *,
    simplify_numbers: bool = True,
) -> str:
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
        num = _coerce_float(value)
        if num is None:
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
    ind = _INDENT * indent
    name = _quote_key(key, delimiter) or "items"
    header_line = delimiter.join(_quote_key(h, delimiter) for h in headers)
    col_decimals = _column_decimals(headers, rows) if simplify_numbers else {}
    lines = [f"{ind}{name}[{len(rows)}]{{{header_line}}}:"]
    row_indent = ind + _INDENT
    for row in rows:
        vals = [
            _stringify_for_toon_value(
                row.get(header),
                col_decimals.get(header),
                delimiter,
                simplify_numbers=simplify_numbers,
            )
            for header in headers
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
    if not headers or not rows:
        return []
    cols = [str(h) for h in headers]
    items: List[Dict[str, Any]] = []
    for row in rows:
        item: Dict[str, Any] = {}
        for idx, col in enumerate(cols):
            item[col] = row[idx] if idx < len(row) else None
        items.append(item)
    return _encode_tabular(
        name, cols, items, indent=0, simplify_numbers=simplify_numbers
    ).splitlines()


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
    decimals = None
    if not simplify_numbers:
        decimals = DISPLAY_MAX_DECIMALS
    values: List[float] = []
    if simplify_numbers:
        for item in items:
            if item is None or isinstance(item, bool):
                continue
            if not isinstance(item, Number):
                continue
            try:
                num = float(cast(Any, item))
            except Exception:
                continue
            if math.isfinite(num):
                values.append(num)
        if values:
            decimals = _optimal_decimals(values)
    vals = delimiter.join(
        _stringify_for_toon_value(
            value,
            decimals,
            delimiter,
            simplify_numbers=simplify_numbers,
        )
        for value in items
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


def _format_to_toon(
    value: Any,
    *,
    key: Optional[str] = None,
    parent_key: Optional[str] = None,
    indent: int = 0,
    delimiter: str = _DEFAULT_DELIMITER,
    simplify_numbers: bool = True,
) -> str:
    ind = _INDENT * indent
    if key is not None and isinstance(value, dict):
        key, value = _collapse_single_key_path(key, value)

    if _is_scalar_value(value):
        forced_decimals = _forced_scalar_decimals(key, parent_key=parent_key)
        if (
            forced_decimals is not None
            and isinstance(value, Number)
            and not isinstance(value, bool)
        ):
            num = _coerce_float(value)
            if num is None:
                rendered = _stringify_for_toon(
                    value, delimiter, simplify_numbers=simplify_numbers
                )
            else:
                rendered = _stringify_for_toon_value(
                    num,
                    int(forced_decimals),
                    delimiter,
                    simplify_numbers=True,
                )
            if key is None:
                return f"{ind}{rendered}".rstrip()
            return f"{ind}{_quote_key(key, delimiter)}: {rendered}".rstrip()
        rendered = _stringify_for_toon(
            value, delimiter, simplify_numbers=simplify_numbers
        )
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
            return _encode_inline_array(
                name, value, indent, delimiter, simplify_numbers=simplify_numbers
            )
        return _encode_expanded_array(
            name, value, indent, delimiter, simplify_numbers=simplify_numbers
        )

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
                parent_key=str(key) if key is not None else parent_key,
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
