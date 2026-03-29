"""
Lazy exports for selected forecast method helpers.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PRETRAINED_EXPORTS = {
    "forecast_chronos_bolt",
    "forecast_timesfm",
    "forecast_lag_llama",
}

__all__ = sorted(_PRETRAINED_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _PRETRAINED_EXPORTS:
        pretrained = import_module(f"{__name__}.pretrained")
        return getattr(pretrained, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _PRETRAINED_EXPORTS)
