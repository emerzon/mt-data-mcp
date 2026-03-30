import subprocess
import sys
from pathlib import Path
import textwrap


def test_core_forecast_import_does_not_touch_runtime_forecast_modules(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    script = textwrap.dedent(
        f"""
        import importlib
        import sys
        import types

        def _blocking_module(name: str) -> types.ModuleType:
            module = types.ModuleType(name)

            def _raise(attr: str):
                raise AssertionError(f"unexpected import-time access to {{name}}.{{attr}}")

            module.__getattr__ = _raise  # type: ignore[attr-defined]
            return module

        sys.path.insert(0, {str(src)!r})
        sys.path.insert(0, {str(root)!r})

        heavy_modules = (
            "mtdata.forecast.forecast",
            "mtdata.forecast.backtest",
            "mtdata.forecast.use_cases",
            "mtdata.forecast.volatility",
            "mtdata.forecast.tune",
        )

        for module_name in heavy_modules:
            sys.modules[module_name] = _blocking_module(module_name)

        module = importlib.import_module("mtdata.core.forecast")

        assert callable(module.forecast_generate)
        assert callable(module.forecast_list_methods)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
