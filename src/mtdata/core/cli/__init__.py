"""Canonical CLI package."""

from __future__ import annotations

from functools import update_wrapper
from pathlib import Path
import inspect
import types

from . import api as _api

_REBOUND_CACHE: dict[int, types.FunctionType] = {}


def _matches_module_file(func: object, module_file: str | None) -> bool:
    code = getattr(func, "__code__", None)
    filename = getattr(code, "co_filename", None)
    if not module_file or not filename:
        return False
    try:
        return Path(filename).resolve() == Path(module_file).resolve()
    except Exception:
        return filename == module_file


def _rebind_function(func: types.FunctionType, module_file: str | None) -> types.FunctionType:
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
    if inspect.isfunction(wrapped) and _matches_module_file(wrapped, module_file):
        rebound.__wrapped__ = _rebind_function(wrapped, module_file)
    return rebound


def _copy_namespace(module) -> None:
    module_file = getattr(module, "__file__", None)
    for name, value in vars(module).items():
        if name.startswith("__") and name not in {"__all__", "__doc__"}:
            continue
        globals()[name] = value

    for name, value in list(vars(module).items()):
        if inspect.isfunction(value) and _matches_module_file(value, module_file):
            globals()[name] = _rebind_function(value, module_file)


_copy_namespace(_api)

del _copy_namespace
del _rebind_function
del _api
