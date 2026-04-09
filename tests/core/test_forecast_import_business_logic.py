import subprocess
import sys
import textwrap
from pathlib import Path


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


def test_bootstrap_tools_does_not_import_torch_or_tslearn_for_non_pattern_commands():
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    script = textwrap.dedent(
        f"""
        import builtins
        import sys

        sys.path.insert(0, {str(src)!r})
        sys.path.insert(0, {str(root)!r})

        blocked_prefixes = ("torch", "tslearn")
        original_import = builtins.__import__

        def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked_prefixes):
                raise AssertionError(f"unexpected import: {{name}}")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = _guarded_import
        try:
            from mtdata.bootstrap.tools import bootstrap_tools

            modules = bootstrap_tools()
        finally:
            builtins.__import__ = original_import

        assert any(getattr(module, "__name__", "") == "mtdata.core.data" for module in modules)
        assert any(getattr(module, "__name__", "") == "mtdata.core.patterns" for module in modules)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
