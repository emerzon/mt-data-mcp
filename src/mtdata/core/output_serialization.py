from __future__ import annotations

import math
import types
from datetime import datetime
from typing import Any

from ..utils.formatting import format_number

_JSON_UNSET = object()


def json_default(value: Any) -> Any:
    """Default JSON conversion shared by final presentation/transport layers."""
    return sanitize_json(value)


def _json_float(value: float, *, compact_numbers: bool) -> Any:
    if not math.isfinite(value):
        return None
    if not compact_numbers:
        return value
    try:
        return float(format_number(value))
    except Exception:
        return value


def _json_special_value(value: Any, *, compact_numbers: bool = False) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8", errors="replace")
        except Exception:
            return str(value)

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass

    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return [sanitize_json(v, compact_numbers=compact_numbers) for v in value.tolist()]
        if isinstance(value, np.integer):
            return int(value.item())
        if isinstance(value, np.bool_):
            return bool(value.item())
        if isinstance(value, np.floating):
            return _json_float(float(value.item()), compact_numbers=compact_numbers)
    except Exception:
        pass

    return _JSON_UNSET


def sanitize_json(value: Any, *, compact_numbers: bool = False) -> Any:
    """Return a JSON-compatible presentation copy without requiring CLI imports."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return _json_float(value, compact_numbers=compact_numbers)
    if isinstance(value, dict):
        return {str(k): sanitize_json(v, compact_numbers=compact_numbers) for k, v in value.items()}
    asdict = getattr(value, "_asdict", None)
    if callable(asdict):
        try:
            return sanitize_json(asdict(), compact_numbers=compact_numbers)
        except Exception:
            pass
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json(v, compact_numbers=compact_numbers) for v in value]
    if isinstance(value, types.GeneratorType):
        return [sanitize_json(v, compact_numbers=compact_numbers) for v in value]
    if isinstance(value, range):
        return [sanitize_json(v, compact_numbers=compact_numbers) for v in value]
    special_value = _json_special_value(value, compact_numbers=compact_numbers)
    if special_value is not _JSON_UNSET:
        return special_value

    return str(value)
