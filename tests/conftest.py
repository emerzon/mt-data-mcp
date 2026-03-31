"""Shared fixtures to prevent sys.modules pollution between test files.

Several test files inject mock modules (MetaTrader5, torch, etc.) into
sys.modules at import time.  Without cleanup the mocks leak into later
test modules and cause spurious failures.
"""

import os
from pathlib import Path
import logging
import sys
from unittest.mock import MagicMock

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _path in (str(_SRC), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _make_mt5_stub() -> MagicMock:
    return MagicMock(name="MetaTrader5")


_DEFAULT_MT5_STUB = sys.modules.setdefault("MetaTrader5", _make_mt5_stub())


def _clear_scipy_lru_cache():
    """Clear scipy's _issubclass_fast LRU cache.

    test_pretrained_coverage.py injects a fake ``torch`` module.  scipy's
    ``_issubclass_fast`` is decorated with ``@lru_cache`` and caches lookups
    against ``sys.modules["torch"]``.  If the cache entry is created while
    the fake module is installed, later tests that pass real numpy arrays
    through scipy will get ``AttributeError: module 'torch' has no attribute
    'Tensor'``.
    """
    try:
        from scipy._lib.array_api_compat.common._helpers import _issubclass_fast
        _issubclass_fast.cache_clear()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _restore_mtdata_logging_state():
    """Keep mtdata logger mutations from leaking across tests."""
    logger_names = ("mtdata", "mtdata.bootstrap", "mtdata.bootstrap.settings")
    snapshots = {}
    for name in logger_names:
        logger = logging.getLogger(name)
        snapshots[name] = {
            "level": logger.level,
            "propagate": logger.propagate,
            "disabled": logger.disabled,
            "handlers": list(logger.handlers),
            "filters": list(logger.filters),
        }

    yield

    for name, snapshot in snapshots.items():
        logger = logging.getLogger(name)
        logger.setLevel(snapshot["level"])
        logger.propagate = snapshot["propagate"]
        logger.disabled = snapshot["disabled"]
        logger.handlers[:] = snapshot["handlers"]
        logger.filters[:] = snapshot["filters"]


def _snapshot_mt5_config_state():
    settings_mod = sys.modules.get("mtdata.bootstrap.settings")
    if settings_mod is None:
        return None

    config_obj = getattr(settings_mod, "mt5_config", None)
    if config_obj is None:
        return None

    return {
        "module": settings_mod,
        "config": config_obj,
        "news_embeddings_config": getattr(settings_mod, "news_embeddings_config", None),
        "env_loaded": getattr(settings_mod, "_ENV_LOADED", None),
        "warned_server_tz": getattr(settings_mod, "_WARNED_SERVER_TZ", None),
        "state": {
            "login": getattr(config_obj, "login", None),
            "_login_value": getattr(config_obj, "_login_value", None),
            "password": getattr(config_obj, "password", None),
            "server": getattr(config_obj, "server", None),
            "timeout": getattr(config_obj, "timeout", None),
            "server_tz_name": getattr(config_obj, "server_tz_name", None),
            "client_tz_name": getattr(config_obj, "client_tz_name", None),
            "time_offset_minutes": getattr(config_obj, "time_offset_minutes", None),
            "broker_time_check_enabled": getattr(config_obj, "broker_time_check_enabled", None),
            "broker_time_check_ttl_seconds": getattr(config_obj, "broker_time_check_ttl_seconds", None),
        },
        "news_embeddings_state": {
            "model_name": getattr(getattr(settings_mod, "news_embeddings_config", None), "model_name", None),
            "top_n": getattr(getattr(settings_mod, "news_embeddings_config", None), "top_n", None),
            "weight": getattr(getattr(settings_mod, "news_embeddings_config", None), "weight", None),
            "truncate_dim": getattr(getattr(settings_mod, "news_embeddings_config", None), "truncate_dim", None),
            "cache_size": getattr(getattr(settings_mod, "news_embeddings_config", None), "cache_size", None),
            "hf_token_env_var": getattr(getattr(settings_mod, "news_embeddings_config", None), "hf_token_env_var", None),
        },
    }


def _restore_mt5_config_state(snapshot) -> None:
    if snapshot is None:
        return

    module = snapshot["module"]
    config_obj = snapshot["config"]
    module.mt5_config = config_obj
    if snapshot.get("news_embeddings_config") is not None:
        module.news_embeddings_config = snapshot["news_embeddings_config"]
    for name, value in snapshot["state"].items():
        setattr(config_obj, name, value)
    news_embeddings_config = snapshot.get("news_embeddings_config")
    if news_embeddings_config is not None:
        for name, value in snapshot["news_embeddings_state"].items():
            setattr(news_embeddings_config, name, value)
    module._ENV_LOADED = snapshot["env_loaded"]
    module._WARNED_SERVER_TZ = snapshot["warned_server_tz"]


@pytest.fixture(autouse=True)
def _restore_mtdata_process_state():
    """Keep env and MT5 bootstrap mutations from leaking across tests."""
    env_snapshot = dict(os.environ)
    mt5_snapshot = _snapshot_mt5_config_state()

    yield

    os.environ.clear()
    os.environ.update(env_snapshot)
    _restore_mt5_config_state(mt5_snapshot)


@pytest.fixture
def mt5_module():
    """Install a temporary MetaTrader5 stub for the duration of a test."""
    prev = sys.modules.get("MetaTrader5")
    stub = _make_mt5_stub()
    sys.modules["MetaTrader5"] = stub
    try:
        yield stub
    finally:
        if prev is None:
            sys.modules["MetaTrader5"] = _DEFAULT_MT5_STUB
        else:
            sys.modules["MetaTrader5"] = prev


def pytest_runtest_setup(item):
    """Clear the cache before every test to guard against import-time pollution."""
    sys.modules.setdefault("MetaTrader5", _DEFAULT_MT5_STUB)
    _clear_scipy_lru_cache()


def pytest_runtest_teardown(item, nextitem):
    """Clear the cache after every test as well."""
    sys.modules.setdefault("MetaTrader5", _DEFAULT_MT5_STUB)
    _clear_scipy_lru_cache()
