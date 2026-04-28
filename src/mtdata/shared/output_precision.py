"""Shared render-time numeric precision policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

PRECISION_AUTO = "auto"
PRECISION_COMPACT = "compact"
PRECISION_FULL = "full"

PRECISION_CHOICES = (PRECISION_AUTO, PRECISION_COMPACT, "display", PRECISION_FULL, "raw")

_COMPACT_ALIASES = {"compact", "display", "min", "minimal", "simplified"}
_FULL_ALIASES = {"full", "raw", "none", "off", "exact"}
_AUTO_ALIASES = {"auto", "default", ""}

_COMPACT_BY_DEFAULT_TOOLS = {
    "data_fetch_candles",
    "data_fetch_ticks",
    "forecast_list_methods",
    "forecast_list_library_models",
    "forecast_models_list",
    "market_scan",
    "mt5_symbols_get",
    "symbols_list",
    "symbols_search",
    "symbols_top_markets",
}

_FULL_BY_DEFAULT_PREFIXES = ("trade_",)

_FULL_BY_DEFAULT_TOOLS = {
    "forecast_barrier_optimize",
    "forecast_barrier_prob",
    "forecast_generate",
    "market_depth_fetch",
    "market_ticker",
    "support_resistance_find",
}


@dataclass(frozen=True)
class OutputPrecisionPolicy:
    """Resolved numeric precision behavior for display rendering only."""

    mode: str = PRECISION_AUTO
    simplify_numbers: bool = False

    @property
    def is_full(self) -> bool:
        return not self.simplify_numbers


def normalize_precision_mode(value: Any) -> str:
    """Normalize user-facing precision aliases into canonical modes."""
    raw = str(value or PRECISION_AUTO).strip().lower()
    if raw in _COMPACT_ALIASES:
        return PRECISION_COMPACT
    if raw in _FULL_ALIASES:
        return PRECISION_FULL
    if raw in _AUTO_ALIASES:
        return PRECISION_AUTO
    raise ValueError(
        "precision must be one of: auto, compact/display, full/raw"
    )


def _source_get(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def auto_simplifies_numbers(tool_name: Optional[str], *, fmt: Optional[str] = None) -> bool:
    """Return whether auto mode should compact numeric display for a tool."""
    if str(fmt or "").strip().lower() == "json":
        return False
    name = str(tool_name or "").strip().lower()
    if not name:
        return True
    if any(name.startswith(prefix) for prefix in _FULL_BY_DEFAULT_PREFIXES):
        return False
    if name in _FULL_BY_DEFAULT_TOOLS:
        return False
    return True


def resolve_output_precision(
    source: Any = None,
    *,
    tool_name: Optional[str] = None,
    fmt: Optional[str] = None,
    precision: Any = None,
    simplify_numbers: Optional[bool] = None,
) -> OutputPrecisionPolicy:
    """Resolve precision controls from args/kwargs for output rendering."""
    if precision is None:
        precision = _source_get(source, "precision")

    mode = normalize_precision_mode(precision)

    if mode == PRECISION_FULL:
        simplify = False
    elif mode == PRECISION_COMPACT:
        simplify = True
    elif simplify_numbers is not None:
        simplify = bool(simplify_numbers)
    else:
        simplify = auto_simplifies_numbers(tool_name, fmt=fmt)

    return OutputPrecisionPolicy(
        mode=mode,
        simplify_numbers=simplify,
    )

