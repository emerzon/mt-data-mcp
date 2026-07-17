import io
from unittest.mock import patch

import pytest


def test_version_path_does_not_import_cli_api(capsys):
    from mtdata.core.cli import main

    with (
        patch("mtdata.core.cli._installed_version", return_value="9.8.7"),
        patch.dict("sys.modules", {"mtdata.core.cli.api": None}),
    ):
        status = main(["--version"])

    assert status == 0
    assert capsys.readouterr().out.strip() == "mtdata-cli 9.8.7"


def test_root_help_path_does_not_import_cli_api(capsys):
    from mtdata.core.cli import main

    with patch.dict("sys.modules", {"mtdata.core.cli.api": None}):
        status = main(["--help"])

    output = capsys.readouterr().out
    assert status == 0
    assert "forecast_generate" in output
    assert "command-level" in output
    assert "--timeframe overrides it" in output


def test_unknown_command_path_does_not_import_cli_api(capsys):
    from mtdata.core.cli import main

    with patch.dict("sys.modules", {"mtdata.core.cli.api": None}):
        status = main(["market-tickr"])

    assert status == 2
    assert "market_ticker" in capsys.readouterr().err


def test_shell_reuses_process_and_runs_entered_commands(monkeypatch):
    from mtdata.core.cli import api

    commands = iter(["symbols_list --limit 2 --json", "quit"])
    observed = []
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    monkeypatch.setattr(api, "main", lambda: observed.append(list(api.sys.argv)) or 0)

    status = api.run_shell()

    assert status == 0
    assert observed == [[api.sys.argv[0], "symbols_list", "--limit", "2", "--json"]]


def test_shell_continues_after_argparse_system_exit(monkeypatch):
    from mtdata.core.cli import api

    commands = iter(["market_ticker EURUSD", "bad --flag", "market_ticker GBPUSD", "quit"])
    observed = []

    def _main():
        observed.append(list(api.sys.argv))
        if api.sys.argv[1] == "bad":
            raise SystemExit(2)
        return 0

    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    monkeypatch.setattr(api, "main", _main)

    assert api.run_shell() == 0
    assert [argv[1:] for argv in observed] == [
        ["market_ticker", "EURUSD"],
        ["bad", "--flag"],
        ["market_ticker", "GBPUSD"],
    ]


def test_noninteractive_shell_reads_batch_and_aggregates_failures(monkeypatch, capsys):
    from mtdata.core.cli import api

    batch = "# warm batch\nmarket_ticker EURUSD\n\nbad --flag\nmarket_ticker GBPUSD\n"
    observed = []

    def _main():
        observed.append(list(api.sys.argv[1:]))
        return 2 if api.sys.argv[1] == "bad" else 0

    monkeypatch.setattr(api.sys, "stdin", io.StringIO(batch))
    monkeypatch.setattr(api, "main", _main)

    assert api.run_shell(interactive=False) == 2
    assert observed == [
        ["market_ticker", "EURUSD"],
        ["bad", "--flag"],
        ["market_ticker", "GBPUSD"],
    ]
    assert capsys.readouterr().out == ""


def test_static_command_catalog_matches_registered_tools():
    from mtdata.bootstrap.tools import bootstrap_tools
    from mtdata.core.cli.api import discover_tools
    from mtdata.core.cli.catalog import CLI_COMMAND_NAMES

    bootstrap_tools()

    assert set(CLI_COMMAND_NAMES) == set(discover_tools())


def test_shell_is_registered_and_has_help(monkeypatch, capsys):
    from mtdata.core.cli import api

    monkeypatch.setattr(api, "load_environment", lambda: None)
    monkeypatch.setattr(api, "discover_tools", lambda *_args: {"sample": {
        "func": lambda: {},
        "meta": {"description": "Sample tool"},
    }})
    monkeypatch.setattr(api.sys, "argv", ["mtdata-cli", "shell", "--help"])

    with patch.object(api, "_cli_version", return_value="test"), pytest.raises(
        SystemExit
    ) as exc_info:
        api.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "Run an interactive mtdata-cli session" in output
    assert "batch from stdin" in output
    assert "exit or quit" in output
