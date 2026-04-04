"""Canonical denoise package."""

from __future__ import annotations

from functools import update_wrapper
import inspect
import types

from . import api as _api
from . import base as _base
from . import filters as _filters  # noqa: F401

_REBOUND_CACHE: dict[int, types.FunctionType] = {}


def _rebind_function(func: types.FunctionType) -> types.FunctionType:
    cached = _REBOUND_CACHE.get(id(func))
    if cached is not None:
        return cached

    rebound = types.FunctionType(
        func.__code__,
        globals(),
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=func.__closure__,
    )
    _REBOUND_CACHE[id(func)] = rebound
    update_wrapper(rebound, func)
    rebound.__kwdefaults__ = getattr(func, "__kwdefaults__", None)
    rebound.__dict__.update(getattr(func, "__dict__", {}))
    rebound.__module__ = __name__
    signature = getattr(func, "__signature__", None)
    if signature is not None:
        rebound.__signature__ = signature
    wrapped = getattr(func, "__wrapped__", None)
    if inspect.isfunction(wrapped) and wrapped.__module__ == func.__module__:
        rebound.__wrapped__ = _rebind_function(wrapped)
    return rebound


def _copy_namespace(module) -> None:
    for name, value in vars(module).items():
        if name.startswith("__"):
            continue
        globals()[name] = value

    for name, value in list(vars(module).items()):
        if inspect.isfunction(value) and value.__module__ == module.__name__:
            globals()[name] = _rebind_function(value)


_copy_namespace(_base)
_copy_namespace(_api)

__all__ = [
    "register_filter",
    "get_filter",
    "list_filters",
    "_series_like",
    "_denoise_series",
    "_apply_denoise",
    "_resolve_denoise_base_col",
    "normalize_denoise_spec",
    "get_denoise_methods_data",
    "denoise_list_methods",
]

del _copy_namespace
del _rebind_function
del _api
del _base
del _filters
