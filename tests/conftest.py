"""Shared fixtures to prevent sys.modules pollution between test files.

Several test files inject mock modules (MetaTrader5, torch, etc.) into
sys.modules at import time.  Without cleanup the mocks leak into later
test modules and cause spurious failures.
"""

from pathlib import Path
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
