import importlib
import sys
import types


def _blocking_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)

    def _raise(attr: str):
        raise AssertionError(f"unexpected import-time access to {name}.{attr}")

    module.__getattr__ = _raise  # type: ignore[attr-defined]
    return module


def test_core_forecast_import_does_not_touch_runtime_forecast_modules(monkeypatch):
    heavy_modules = (
        "mtdata.forecast.forecast",
        "mtdata.forecast.backtest",
        "mtdata.forecast.use_cases",
        "mtdata.forecast.volatility",
        "mtdata.forecast.tune",
    )

    sys.modules.pop("mtdata.core.forecast", None)
    for module_name in heavy_modules:
        sys.modules.pop(module_name, None)
        monkeypatch.setitem(sys.modules, module_name, _blocking_module(module_name))

    module = importlib.import_module("mtdata.core.forecast")

    assert callable(module.forecast_generate)
    assert callable(module.forecast_list_methods)
