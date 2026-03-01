"""Shared fixtures to prevent sys.modules pollution between test files.

Several test files inject mock modules (MetaTrader5, torch, etc.) into
sys.modules at import time.  Without cleanup the mocks leak into later
test modules and cause spurious failures.
"""

import sys
import pytest


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


def pytest_runtest_setup(item):
    """Clear the cache before every test to guard against import-time pollution."""
    _clear_scipy_lru_cache()


def pytest_runtest_teardown(item, nextitem):
    """Clear the cache after every test as well."""
    _clear_scipy_lru_cache()
