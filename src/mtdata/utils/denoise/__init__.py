"""Canonical denoise package."""

from __future__ import annotations

from .api import (
    apply_denoise,
    consume_denoise_warnings,
    denoise_series,
    resolve_denoise_base_col,
    denoise_list_methods,
    get_denoise_methods_data,
    normalize_denoise_spec,
)
from .base import (
    _series_like,
    get_filter,
    list_filters,
    register_filter,
)

# Side-effect import: registers all filter implementations via @register_filter
from . import filters as _filters  # noqa: F401

__all__ = [
    "register_filter",
    "get_filter",
    "list_filters",
    "_series_like",
    "denoise_series",
    "apply_denoise",
    "consume_denoise_warnings",
    "resolve_denoise_base_col",
    "normalize_denoise_spec",
    "get_denoise_methods_data",
    "denoise_list_methods",
]

del _filters

