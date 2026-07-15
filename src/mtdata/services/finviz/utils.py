"""Utility helpers for Finviz service."""
from typing import Any, Dict, List, Optional


def to_float_or_none(value: Any) -> Optional[float]:
    """Convert a value to float, returning None if conversion fails.

    Handles bool rejection, NaN, comma-separated numbers, and percentage
    strings like ``"12.34%"``.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            out = float(value)
            return out if out == out else None
        except Exception:
            return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        out = float(text)
        return out if out == out else None
    except Exception:
        return None


def values_equivalent(lhs: Any, rhs: Any) -> bool:
    """Compare two values for near-equality, preferring numeric comparison."""
    left_num = to_float_or_none(lhs)
    right_num = to_float_or_none(rhs)
    if left_num is not None and right_num is not None:
        scale = max(1.0, abs(left_num), abs(right_num))
        return abs(left_num - right_num) <= (1e-9 * scale)
    return lhs == rhs


def crypto_day_week_identical(
    rows: List[Dict[str, Any]],
    *,
    day_key: str = "Perf Day",
    week_key: str = "Perf Week",
) -> bool:
    """Return True when day/week performance columns match on every matched row.

    Requires at least one row that contains both keys.
    """
    matched = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if day_key not in row or week_key not in row:
            continue
        matched += 1
        if not values_equivalent(row.get(day_key), row.get(week_key)):
            return False
    return matched > 0


def crypto_price_display(value: Any) -> Optional[str]:
    """Format a crypto price value for compact display."""
    num = to_float_or_none(value)
    if num is None:
        return None
    abs_num = abs(num)
    if abs_num >= 1.0:
        decimals = 2
    elif abs_num >= 0.01:
        decimals = 4
    elif abs_num >= 0.0001:
        decimals = 6
    elif abs_num > 0.0 and abs_num < 0.00000001:
        return f"{num:.8g}"
    else:
        decimals = 8
    return f"{num:.{decimals}f}"


def load_finviz_attr(module_name: str, attr_name: str) -> Any:
    """Safely load an attribute from a finvizfinance module."""
    try:
        import importlib
        mod = importlib.import_module(f"finvizfinance.{module_name}")
        return getattr(mod, attr_name)
    except Exception:
        return None


def get_finviz_stock_quote(symbol: str) -> tuple[str, Any]:
    """Get a finvizfinance Stock quote object with normalized symbol."""
    try:
        import finvizfinance.quote as _fv_quote
    except Exception as e:
        raise RuntimeError(f"finvizfinance not installed: {e}")
    normalized = str(symbol or "").upper().strip()
    if not normalized:
        raise ValueError("Symbol is required")
    return normalized, _fv_quote.Stock(normalized)


def build_finviz_screener(view: str) -> Any:
    """Build a finvizfinance screener for the given view."""
    try:
        import finvizfinance.screener as _fv_screener
    except Exception as e:
        raise RuntimeError(f"finvizfinance not installed: {e}")
    view_map: Dict[str, str] = {
        "overview": "Overview",
        "valuation": "Valuation",
        "financial": "Financial",
        "ownership": "Ownership",
        "performance": "Performance",
        "technical": "Technical",
    }
    screener_view = view_map.get(str(view or "").lower().strip(), "Overview")
    return _fv_screener.Screener(view=screener_view)


def apply_finvizfinance_timeout_patch() -> None:
    """Patch finvizfinance's internal bare requests.get call to include timeout."""
    try:
        import finvizfinance.quote as _fv_quote
    except Exception:
        return

    # Avoid double-patch
    if getattr(_fv_quote, "_mtdata_timeout_patched", False):
        return

    import requests

    from .client import get_finviz_http_timeout

    _orig_get = requests.get

    def _patched_get(url: str, **kwargs: Any) -> Any:
        if "timeout" not in kwargs:
            kwargs["timeout"] = get_finviz_http_timeout()
        return _orig_get(url, **kwargs)

    try:
        import finvizfinance as _fv
        import finvizfinance.custom as _fv_custom
        if hasattr(_fv_custom, "_mtdata_timeout_patched"):
            return
        _fv.requester.session.get = _patched_get
        _fv_custom._mtdata_timeout_patched = True  # type: ignore[attr-defined]
        _fv_quote._mtdata_timeout_patched = True  # type: ignore[attr-defined]
    except Exception:
        pass


__all__ = [
    "to_float_or_none",
    "values_equivalent",
    "crypto_day_week_identical",
    "crypto_price_display",
    "load_finviz_attr",
    "get_finviz_stock_quote",
    "build_finviz_screener",
    "apply_finvizfinance_timeout_patch",
]
