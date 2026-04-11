"""Canonical regime package.

Copies the ``api`` namespace into the package so that tests can
monkeypatch internal helpers (e.g. ``_fetch_history``) at the
``mtdata.core.regime`` level.  A lightweight ``_sync_api_namespace()``
pushes those patches back to ``api`` before every public entry-point
call.
"""

from __future__ import annotations

import inspect
from functools import update_wrapper

from . import api as _api
from . import methods

# ---- namespace import (one-time at package load) ---------------------------

_SYNC_EXCLUDED = frozenset({
    "_api",
    "methods",
    "_SYNC_EXCLUDED",
    "_sync_api_namespace",
    "_ORIG_REGIME_CONNECTION_ERROR",
    "_REGIME_DETECT_IMPL",
    "_REGIME_DETECT_RAW_IMPL",
    "_regime_detect_raw",
    "_regime_connection_error",
    "regime_detect",
})

for _name, _value in vars(_api).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
del _name, _value


# ---- per-call sync (pushes monkeypatches back to _api) ---------------------

def _sync_api_namespace() -> None:
    for name, value in list(globals().items()):
        if name.startswith("__") or name in _SYNC_EXCLUDED:
            continue
        if hasattr(_api, name):
            setattr(_api, name, value)
    _api._regime_connection_error = _regime_connection_error


# ---- thin public wrappers --------------------------------------------------

_ORIG_REGIME_CONNECTION_ERROR = _api._regime_connection_error
_REGIME_DETECT_IMPL = _api.regime_detect

_REGIME_DETECT_RAW_IMPL = _REGIME_DETECT_IMPL
while hasattr(_REGIME_DETECT_RAW_IMPL, "__wrapped__"):
    _REGIME_DETECT_RAW_IMPL = _REGIME_DETECT_RAW_IMPL.__wrapped__


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
    "_canonicalize_regime_labels",
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
