from __future__ import annotations

import importlib
import sys
import types

import pytest


def _blocking_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)

    def _raise(attr: str):
        raise AssertionError(f"unexpected access to {name}.{attr}")

    module.__getattr__ = _raise  # type: ignore[attr-defined]
    return module


def test_importing_methods_package_does_not_touch_pretrained(monkeypatch):
    sys.modules.pop("mtdata.forecast.methods", None)
    monkeypatch.setitem(
        sys.modules,
        "mtdata.forecast.methods.pretrained",
        _blocking_module("mtdata.forecast.methods.pretrained"),
    )

    methods_pkg = importlib.import_module("mtdata.forecast.methods")

    assert methods_pkg.__name__ == "mtdata.forecast.methods"


def test_methods_package_no_longer_exposes_pretrained_wrapper_exports(monkeypatch):
    sys.modules.pop("mtdata.forecast.methods", None)
    monkeypatch.setitem(
        sys.modules,
        "mtdata.forecast.methods.pretrained",
        _blocking_module("mtdata.forecast.methods.pretrained"),
    )

    methods_pkg = importlib.import_module("mtdata.forecast.methods")

    assert methods_pkg.__all__ == []
    for name in ("forecast_timesfm", "forecast_chronos_bolt", "forecast_lag_llama"):
        with pytest.raises(AttributeError):
            getattr(methods_pkg, name)
