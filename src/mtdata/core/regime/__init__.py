"""Canonical regime package."""

from __future__ import annotations

import inspect
import types
from functools import update_wrapper
from pathlib import Path

from . import api as _api
from . import methods

_REBOUND_CACHE: dict[int, types.FunctionType] = {}
_SYNC_EXCLUDED = {
    "_api",
    "methods",
    "_REBOUND_CACHE",
    "_SYNC_EXCLUDED",
    "_matches_module_file",
    "_rebind_function",
    "_copy_namespace",
    "_unwrap_function",
    "_sync_api_namespace",
    "_ORIG_REGIME_CONNECTION_ERROR",
    "_REGIME_DETECT_IMPL",
    "_REGIME_DETECT_RAW_IMPL",
    "_regime_detect_raw",
    "_regime_connection_error",
    "regime_detect",
}


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
        if name.startswith("__"):
            continue
        globals()[name] = value

    for name, value in list(vars(module).items()):
        if inspect.isfunction(value) and _matches_module_file(value, module_file):
            globals()[name] = _rebind_function(value, module_file)


def _unwrap_function(func):
    raw = func
    while inspect.isfunction(getattr(raw, "__wrapped__", None)):
        raw = raw.__wrapped__
    return raw


def _sync_api_namespace() -> None:
    for name, value in list(globals().items()):
        if name.startswith("__") or name in _SYNC_EXCLUDED:
            continue
        if hasattr(_api, name):
            setattr(_api, name, value)
    _api._regime_connection_error = _regime_connection_error


_copy_namespace(_api)

_ORIG_REGIME_CONNECTION_ERROR = _api._regime_connection_error
_REGIME_DETECT_IMPL = _api.regime_detect
_REGIME_DETECT_RAW_IMPL = _unwrap_function(_REGIME_DETECT_IMPL)


def _regime_connection_error():
    _sync_api_namespace()
    return _ORIG_REGIME_CONNECTION_ERROR()


update_wrapper(_regime_connection_error, _ORIG_REGIME_CONNECTION_ERROR)
_regime_connection_error.__module__ = __name__


def _regime_detect_raw(*args, **kwargs):
    _sync_api_namespace()
    return _REGIME_DETECT_RAW_IMPL(*args, **kwargs)


update_wrapper(_regime_detect_raw, _REGIME_DETECT_RAW_IMPL)
_regime_detect_raw.__module__ = __name__
_regime_detect_raw.__signature__ = inspect.signature(_REGIME_DETECT_RAW_IMPL)
_regime_detect_raw.__dict__.pop("__wrapped__", None)
try:
    delattr(_regime_detect_raw, "__wrapped__")
except AttributeError:
    pass


def regime_detect(*args, **kwargs):
    _sync_api_namespace()
    return _REGIME_DETECT_IMPL(*args, **kwargs)


update_wrapper(regime_detect, _REGIME_DETECT_IMPL)
regime_detect.__module__ = __name__
regime_detect.__wrapped__ = _regime_detect_raw
regime_detect.__signature__ = inspect.signature(_REGIME_DETECT_IMPL)

__all__ = [
    "regime_detect",
    "_regime_connection_error",
    "ensure_mt5_connection_or_raise",
    "MT5ConnectionError",
    "_fetch_history",
    "extract_rolling_features",
    "_resolve_denoise_base_col",
    "_format_time_minimal",
    "get_mt5_gateway",
    "mt5_connection_error",
    "_count_state_transitions",
    "_state_runs",
    "_smooth_short_state_runs",
    "_normalize_state_probability_matrix",
    "_is_probably_crypto_symbol",
    "_CRYPTO_SYMBOL_HINTS",
    "_consolidate_payload",
    "_summary_only_payload",
    "_default_bocpd_hazard_lambda",
    "_default_bocpd_cp_threshold",
    "_auto_calibrate_bocpd_params",
    "_bocpd_reliability_score",
    "_walkforward_quantile_threshold_calibration",
    "_filter_bocpd_change_points",
    "_hmm_reliability_from_gamma",
    "_ms_ar_reliability_from_smoothed",
    "methods",
]

del _copy_namespace
del _unwrap_function
