from __future__ import annotations

import math
import types
from datetime import datetime
from typing import Any

_JSON_UNSET = object()


def json_default(value: Any) -> Any:
    """Default JSON conversion shared by final presentation/transport layers."""
    return sanitize_json_compat(value)


def _json_special_value(value: Any) -> Any:
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
            return [sanitize_json_compat(v) for v in value.tolist()]
        if isinstance(value, np.integer):
            return int(value.item())
        if isinstance(value, np.bool_):
            return bool(value.item())
        if isinstance(value, np.floating):
            fv = float(value.item())
            return fv if math.isfinite(fv) else None
    except Exception:
        pass

    return _JSON_UNSET


def sanitize_json_compat(value: Any) -> Any:
    """Return a JSON-compatible presentation copy without requiring CLI imports."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): sanitize_json_compat(v) for k, v in value.items()}
    asdict = getattr(value, "_asdict", None)
    if callable(asdict):
        try:
            return sanitize_json_compat(asdict())
        except Exception:
            pass
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json_compat(v) for v in value]
    if isinstance(value, types.GeneratorType):
        return [sanitize_json_compat(v) for v in value]
    if isinstance(value, range):
        return [sanitize_json_compat(v) for v in value]
    special_value = _json_special_value(value)
    if special_value is not _JSON_UNSET:
        return special_value

    return str(value)
