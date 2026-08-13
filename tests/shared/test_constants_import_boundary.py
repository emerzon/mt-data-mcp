"""Import-boundary checks for the cycle cuts in this change."""

from __future__ import annotations

import subprocess
import sys
from textwrap import dedent


def _run_isolated(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_constants_import_does_not_load_mt5_adapter():
    result = _run_isolated(
        """
        import sys
        import mtdata.shared.constants as constants
        assert "mtdata.utils.mt5" not in sys.modules, sorted(sys.modules)
        assert constants.TIMEFRAME_MAP["H1"] == 16385
        assert constants.TIMEFRAME_MAP["D1"] == 16408
        assert constants.TIMEFRAME_MAP["M1"] == 1
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_trading_validation_has_no_module_level_gateway_import():
    result = _run_isolated(
        """
        import ast
        from pathlib import Path
        import mtdata.core.trading.validation as validation

        source = Path(validation.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        module_imports = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                module_imports.add(node.module or "")
            elif isinstance(node, ast.Import):
                module_imports.update(alias.name for alias in node.names)
        assert "gateway" not in module_imports
        assert ".gateway" not in module_imports
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_report_utils_import_does_not_load_basic_template():
    result = _run_isolated(
        """
        import sys
        import mtdata.core.report.utils as report_utils
        assert "mtdata.core.report_templates.basic" not in sys.modules, sorted(sys.modules)
        assert report_utils._compute_compact_trend is not None
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
