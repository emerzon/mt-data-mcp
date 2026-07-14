from __future__ import annotations

import pytest

from mtdata.bootstrap.tools import cli_tool_module_names


@pytest.mark.parametrize(
    ("command", "module"),
    [
        ("market_ticker", "mtdata.core.market_depth"),
        ("data-fetch-candles", "mtdata.core.data"),
        ("forecast_generate", "mtdata.core.forecast"),
        ("forecast_task_status", "mtdata.core.forecast_tasks"),
        ("trade_session_context", "mtdata.core.trading"),
        ("support_resistance_levels", "mtdata.core.pivot"),
    ],
)
def test_cli_tool_module_names_routes_command_families(command: str, module: str) -> None:
    assert cli_tool_module_names(command) == (module,)


def test_cli_tool_module_names_uses_full_bootstrap_for_discovery() -> None:
    assert cli_tool_module_names("") is None
    assert cli_tool_module_names("tools_list") is None
    assert cli_tool_module_names("unknown_tool") is None
