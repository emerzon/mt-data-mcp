"""Trading input validation helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, Literal, Optional, Tuple, Union

from ..utils.mt5 import _auto_connect_wrapper, mt5_adapter
from ..utils.utils import _coerce_finite_float, _coerce_scalar


MarketOrderTypeLiteral = Literal["BUY", "SELL"]
OrderTypeLiteral = Literal[
    "BUY",
    "SELL",
    "BUY_LIMIT",
    "BUY_STOP",
    "SELL_LIMIT",
    "SELL_STOP",
]

MarketOrderTypeInput = Union[MarketOrderTypeLiteral, int, float, str]
OrderTypeInput = Union[OrderTypeLiteral, int, float, str]

_SUPPORTED_ORDER_TYPES = {
    "BUY",
    "SELL",
    "BUY_LIMIT",
    "BUY_STOP",
    "SELL_LIMIT",
    "SELL_STOP",
}
_ORDER_TYPE_NUMERIC_MAP = {
    0: "BUY",
    1: "SELL",
    2: "BUY_LIMIT",
    3: "SELL_LIMIT",
    4: "BUY_STOP",
    5: "SELL_STOP",
}
_ORDER_TYPE_ALIASES = {
    "LONG": "BUY",
    "SHORT": "SELL",
}


def _normalize_order_type_input(order_type: Any) -> Tuple[Optional[str], Optional[str]]:
    """Normalize order_type inputs from MCP clients into canonical MT5 order names."""
    if order_type is None:
        return None, "order_type is required."

    value = order_type
    if isinstance(value, bool):
        return None, f"Unsupported order_type '{order_type}'."

    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None, f"Unsupported order_type '{order_type}'."
        if float(value).is_integer():
            mapped = _ORDER_TYPE_NUMERIC_MAP.get(int(value))
            if mapped:
                return mapped, None
            return (
                None,
                (
                    f"Unsupported order_type '{order_type}'. "
                    "Numeric values must match MT5 constants 0..5."
                ),
            )
        return None, f"Unsupported order_type '{order_type}'."

    text = str(value).strip()
    if not text:
        return None, "order_type is required."

    scalar = _coerce_scalar(text)
    if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
        if isinstance(scalar, float) and (not math.isfinite(scalar) or not scalar.is_integer()):
            return None, f"Unsupported order_type '{order_type}'."
        mapped = _ORDER_TYPE_NUMERIC_MAP.get(int(scalar))
        if mapped:
            return mapped, None

    normalized = text.upper().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    if normalized.startswith("MT5."):
        normalized = normalized[4:]
    if normalized.startswith("ORDER_TYPE_"):
        normalized = normalized[len("ORDER_TYPE_") :]
    normalized = _ORDER_TYPE_ALIASES.get(normalized, normalized)
    if normalized in _SUPPORTED_ORDER_TYPES:
        return normalized, None

    return (
        None,
        (
            f"Unsupported order_type '{order_type}'. "
            "Use BUY/SELL or BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP."
        ),
    )


def _validate_volume(volume: Union[int, float], symbol_info: Any) -> Tuple[Optional[float], Optional[str]]:
    """Validate lot size against symbol constraints."""
    try:
        vol = float(volume)
    except (TypeError, ValueError):
        return None, "volume must be numeric"

    if not math.isfinite(vol) or vol <= 0:
        return None, "volume must be positive and finite"

    min_vol = getattr(symbol_info, "volume_min", None)
    max_vol = getattr(symbol_info, "volume_max", None)
    step = getattr(symbol_info, "volume_step", None)

    try:
        min_vol = float(min_vol) if min_vol is not None else None
    except (TypeError, ValueError):
        min_vol = None
    if min_vol is not None and min_vol <= 0:
        min_vol = None

    try:
        max_vol = float(max_vol) if max_vol is not None else None
    except (TypeError, ValueError):
        max_vol = None
    if max_vol is not None and max_vol <= 0:
        max_vol = None

    try:
        step = float(step) if step is not None else None
    except (TypeError, ValueError):
        step = None
    if step is not None and step <= 0:
        step = None

    if min_vol is not None and vol < (min_vol - 1e-12):
        return None, f"volume must be >= {min_vol}"
    if max_vol is not None and vol > (max_vol + 1e-12):
        return None, f"volume must be <= {max_vol}"

    if step is not None:
        normalized = round(vol / step) * step
        normalized = float(f"{normalized:.10f}")
        tol = step * 1e-6
        if abs(normalized - vol) > tol:
            return None, f"volume must align to step {step}. Try {normalized}"
        vol = normalized
        if min_vol is not None and vol < (min_vol - 1e-12):
            return None, f"volume must be >= {min_vol}"
        if max_vol is not None and vol > (max_vol + 1e-12):
            return None, f"volume must be <= {max_vol}"

    return vol, None


def _validate_deviation(deviation: Union[int, float]) -> Tuple[Optional[int], Optional[str]]:
    """Validate/normalize MT5 deviation in points."""
    try:
        dev = int(float(deviation))
    except (TypeError, ValueError):
        return None, "deviation must be numeric"
    if dev < 0:
        return None, "deviation must be >= 0"
    return dev, None


def _prevalidate_trade_place_market_input(symbol: str, volume: Any) -> Optional[Dict[str, Any]]:
    """Validate symbol and volume before market-order SL/TP enforcement returns."""
    mt5 = mt5_adapter

    @_auto_connect_wrapper
    def _prevalidate():
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {"error": f"Symbol {symbol} not found"}

        if not getattr(symbol_info, "visible", True):
            if not mt5.symbol_select(symbol, True):
                return {"error": f"Failed to select symbol {symbol}"}

        _, volume_error = _validate_volume(volume, symbol_info)
        if volume_error:
            return {"error": volume_error}
        return {"ok": True}

    result = _prevalidate()
    if isinstance(result, dict):
        err = result.get("error")
        if isinstance(err, str):
            if err.strip():
                return result
        elif err not in (None, False):
            return result
    return None


def _normalize_ticket_filter(ticket: Any, *, name: str) -> Tuple[Optional[int], Optional[str]]:
    if ticket in (None, ""):
        return None, None
    value = _coerce_finite_float(ticket)
    if value is None or not float(value).is_integer():
        return None, f"{name} must be an integer ticket."
    return int(value), None


def _normalize_minutes_back(minutes_back: Any) -> Tuple[Optional[int], Optional[str]]:
    if minutes_back in (None, ""):
        return None, None
    value = _coerce_finite_float(minutes_back)
    if value is None or not float(value).is_integer():
        return None, "minutes_back must be a positive integer."
    minutes = int(value)
    if minutes <= 0:
        return None, "minutes_back must be a positive integer."
    return minutes, None


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return None
        return bool(value)
    return None


def _safe_int_ticket(value: Any) -> Optional[int]:
    """Best-effort conversion for MT5 ticket-like values."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            fv = float(value)
        except Exception:
            return None
        if not math.isfinite(fv) or not fv.is_integer():
            return None
        iv = int(fv)
        return iv if iv > 0 else None
    try:
        scalar = _coerce_scalar(str(value).strip())
    except Exception:
        scalar = value
    if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
        try:
            fv = float(scalar)
        except Exception:
            return None
        if not math.isfinite(fv) or not fv.is_integer():
            return None
        iv = int(fv)
        return iv if iv > 0 else None
    return None
