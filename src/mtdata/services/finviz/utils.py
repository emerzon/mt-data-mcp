"""Utility helpers for Finviz service."""
from typing import Any, Dict, List, Optional


def to_float_or_none(value: Any) -> Optional[float]:
    """Convert a value to float, returning None if conversion fails."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        # Handle percentage strings like "12.34%"
        s = str(value).strip()
        if s.endswith("%"):
            s = s[:-1].strip()
        return float(s) if s else None
    except Exception:
        return None


def values_equivalent(lhs: Any, rhs: Any) -> bool:
    """Compare two values for equivalence, handling None."""
    if lhs is None and rhs is None:
        return True
    if lhs is None or rhs is None:
        return False
    try:
        return float(lhs) == float(rhs)
    except Exception:
        return str(lhs) == str(rhs)


def crypto_day_week_identical(rows: List[Dict[str, Any]]) -> bool:
    """Check if crypto performance rows have identical day and week values."""
    if not rows:
        return False
    for r in rows:
        day = r.get("day")
        week = r.get("week")
        if day is None and week is None:
            continue
        if day != week:
            return False
    return True


def crypto_price_display(value: Any) -> Optional[str]:
    """Format a crypto price value for display."""
    try:
        f = float(value)
        if abs(f) >= 1_000_000_000:
            return f"{f / 1_000_000_000:.2f}B"
        if abs(f) >= 1_000_000:
            return f"{f / 1_000_000:.2f}M"
        if abs(f) >= 1_000:
            return f"{f / 1_000:.2f}K"
        return f"{f:.4f}"
    except Exception:
        return None


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
