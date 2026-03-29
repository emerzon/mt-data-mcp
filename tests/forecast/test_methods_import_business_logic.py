from __future__ import annotations

import importlib
import sys
import types


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


def test_methods_package_lazily_resolves_pretrained_exports(monkeypatch):
    sys.modules.pop("mtdata.forecast.methods", None)
    fake_pretrained = types.ModuleType("mtdata.forecast.methods.pretrained")
    fake_pretrained.forecast_timesfm = object()
    fake_pretrained.forecast_chronos_bolt = object()
    fake_pretrained.forecast_lag_llama = object()
    monkeypatch.setitem(sys.modules, "mtdata.forecast.methods.pretrained", fake_pretrained)

    methods_pkg = importlib.import_module("mtdata.forecast.methods")

    assert methods_pkg.forecast_timesfm is fake_pretrained.forecast_timesfm
