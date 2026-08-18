import io
import json
import subprocess
import sys
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("module", ["mtdata", "mtdata.core.cli"])
def test_cli_module_execution_shows_root_help(module):
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
    assert "forecast_generate" in completed.stdout
    if module == "mtdata":
        assert "usage: python -m mtdata" in completed.stdout
        assert "__main__.py" not in completed.stdout
    assert completed.stderr == ""


def test_cli_runtime_import_does_not_register_data_tool_family():
    probe = (
        "import sys; import mtdata.core.cli.runtime.commands; "
        "assert 'mtdata.core.data' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_cli_formatting_import_does_not_load_trading_graph():
    probe = (
        "import sys; import mtdata.core.cli.formatting; "
        "assert 'mtdata.core.trading' not in sys.modules; "
        "assert 'mtdata.services.data_service' not in sys.modules; "
        "assert 'mtdata.utils.denoise' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_dynamic_tools_list_subprocess_has_clean_stderr():
    probe = (
        "from mtdata.core.cli import main; "
        "raise SystemExit(main(['tools_list', '--limit', '1', '--json']))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["success"] is True
    assert completed.stderr == ""


def test_version_path_does_not_import_cli_api(capsys):
    from mtdata.core.cli import main

    with (
        patch("mtdata.core.cli.cli_version", return_value="9.8.7"),
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
    assert "bare command = major-equity exchange calendar" in output
    assert "pass a broker symbol for MT5 tradability" in output


def test_unknown_command_path_does_not_import_cli_api(capsys):
    from mtdata.core.cli import main

    with patch.dict("sys.modules", {"mtdata.core.cli.api": None}):
        status = main(["market-tickr"])

    captured = capsys.readouterr()
    assert status == 2
    assert captured.err == ""
    assert "success: false" in captured.out
    assert "error_code: cli_unknown_command" in captured.out
    assert "market_ticker" in captured.out


def test_unknown_command_json_uses_standard_error_envelope(capsys):
    from mtdata.core.cli import main

    with patch.dict("sys.modules", {"mtdata.core.cli.api": None}):
        status = main(["no-such-command", "--json"])

    rendered = capsys.readouterr().out
    payload = json.loads(rendered)
    assert status == 2
    assert payload["success"] is False
    assert payload["error_code"] == "cli_unknown_command"
    assert payload["operation"] == "cli"
    assert payload["request_id"]
    assert payload["remediation"]
    assert payload["documentation"] == "docs/CLI.md"
    assert rendered.startswith("{\n  \"success\": false")


def test_module_unknown_command_uses_invocable_remediation():
    completed = subprocess.run(
        [sys.executable, "-m", "mtdata", "no-such-command", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert payload["remediation"] == "Run 'python -m mtdata --help' to list commands."
    assert "__main__.py" not in completed.stdout


def test_unknown_command_honors_json_output_environment(monkeypatch, capsys):
    from mtdata.core.cli import main

    monkeypatch.setenv("MTDATA_OUTPUT_FORMAT", "json")
    with patch.dict("sys.modules", {"mtdata.core.cli.api": None}):
        status = main(["no-such-command"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert status == 2
    assert captured.err == ""
    assert payload["error_code"] == "cli_unknown_command"


def test_invalid_output_format_fails_in_lightweight_entrypoint(monkeypatch, capsys):
    from mtdata.core.cli import main

    monkeypatch.setenv("MTDATA_OUTPUT_FORMAT", "jsoon")
    with patch.dict("sys.modules", {"mtdata.core.cli.api": None}):
        status = main(["no-such-command"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert status == 2
    assert captured.err == ""
    assert payload["error_code"] == "cli_invalid_output_format"
    assert payload["valid_values"] == {
        "MTDATA_OUTPUT_FORMAT": ["json", "toon"]
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["--json", "no-such-command"],
        ["--precision", "compact", "--json", "no-such-command"],
        ["--output-fields", "success", "--json", "no-such-command"],
        ["--timeframe", "H1", "--json", "no-such-command"],
    ],
)
def test_unknown_command_json_skips_leading_global_options(argv, capsys):
    from mtdata.core.cli import main

    with patch.dict("sys.modules", {"mtdata.core.cli.api": None}):
        status = main(argv)

    payload = json.loads(capsys.readouterr().out)
    assert status == 2
    assert payload["error_code"] == "cli_unknown_command"
    assert payload["documentation"] == "docs/CLI.md"


def test_json_without_command_returns_compact_error_envelope(monkeypatch, capsys):
    from mtdata.core.cli import api
    from mtdata.core.cli import main as entry_main

    def sample() -> dict:
        return {"success": True}

    monkeypatch.setattr(api, "load_environment", lambda: None)
    monkeypatch.setattr(
        api,
        "discover_tools",
        lambda *_args: {
            "sample": {"func": sample, "meta": {"description": "Sample tool"}}
        },
    )

    status = entry_main(["--json"])

    payload = json.loads(capsys.readouterr().out)
    assert status == 1
    assert payload["error_code"] == "cli_missing_command"
    assert payload["operation"] == "cli"


def test_shell_reuses_process_and_runs_entered_commands(monkeypatch):
    from mtdata.core.cli import api

    commands = iter(["symbols_list --limit 2 --json", "quit"])
    observed = []
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    monkeypatch.setattr(api, "main", lambda: observed.append(list(api.sys.argv)) or 0)

    status = api.run_shell()

    assert status == 0
    assert observed == [[api.sys.argv[0], "symbols_list", "--limit", "2", "--json"]]


def test_shell_inherits_json_for_child_commands(monkeypatch):
    from mtdata.core.cli import api

    commands = iter(["symbols_list --limit 2", "quit"])
    observed = []
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    monkeypatch.setattr(api, "main", lambda: observed.append(list(api.sys.argv)) or 0)

    status = api.run_shell(inherited_argv=["--json"])

    assert status == 0
    assert observed == [
        [api.sys.argv[0], "--json", "symbols_list", "--limit", "2"]
    ]


def test_shell_removes_syntactic_quotes_and_preserves_windows_paths(monkeypatch):
    from mtdata.core.cli import api

    commands = iter([
        'data_fetch_candles EURUSD --indicators "rsi(14)" '
        '--params \'{"period": 14}\' --path C:\\MT5\\profiles\\default.ini',
        "quit",
    ])
    observed = []
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    monkeypatch.setattr(api, "main", lambda: observed.append(list(api.sys.argv)) or 0)

    assert api.run_shell() == 0
    assert observed[0][1:] == [
        "data_fetch_candles",
        "EURUSD",
        "--indicators",
        "rsi(14)",
        "--params",
        '{"period": 14}',
        "--path",
        "C:\\MT5\\profiles\\default.ini",
    ]


def test_shell_continues_after_argparse_system_exit(monkeypatch):
    from mtdata.core.cli import api

    commands = iter(
        [
            "market_ticker EURUSD",
            "market_ticker BAD --flag",
            "market_ticker GBPUSD",
            "quit",
        ]
    )
    observed = []

    def _main():
        observed.append(list(api.sys.argv))
        if "--flag" in api.sys.argv:
            raise SystemExit(2)
        return 0

    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    monkeypatch.setattr(api, "main", _main)

    assert api.run_shell() == 0
    assert [argv[1:] for argv in observed] == [
        ["market_ticker", "EURUSD"],
        ["market_ticker", "BAD", "--flag"],
        ["market_ticker", "GBPUSD"],
    ]


def test_noninteractive_shell_reads_batch_and_aggregates_failures(monkeypatch, capsys):
    from mtdata.core.cli import api

    batch = (
        "# warm batch\nmarket_ticker EURUSD\n\n"
        "market_ticker BAD --flag\nmarket_ticker GBPUSD\n"
    )
    observed = []

    def _main():
        observed.append(list(api.sys.argv[1:]))
        if "--flag" in api.sys.argv:
            raise SystemExit(2)
        return 0

    monkeypatch.setattr(api.sys, "stdin", io.StringIO(batch))
    monkeypatch.setattr(api, "main", _main)

    assert api.run_shell(interactive=False) == 2
    assert observed == [
        ["market_ticker", "EURUSD"],
        ["market_ticker", "BAD", "--flag"],
        ["market_ticker", "GBPUSD"],
    ]
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records == [
        {
            "line": 2,
            "command": "market_ticker EURUSD",
            "success": True,
            "status": 0,
        },
        {
            "line": 4,
            "command": "market_ticker BAD --flag",
            "success": False,
            "status": 2,
        },
        {
            "line": 5,
            "command": "market_ticker GBPUSD",
            "success": True,
            "status": 0,
        },
    ]


def test_noninteractive_shell_unknown_command_is_short_and_actionable(
    monkeypatch, capsys
):
    from mtdata.core.cli import api

    monkeypatch.setattr(api.sys, "stdin", io.StringIO("echo\n"))
    monkeypatch.setattr(
        api,
        "main",
        lambda: pytest.fail("unknown shell commands must not reach argparse"),
    )

    assert api.run_shell(interactive=False) == 2
    record = json.loads(capsys.readouterr().out)
    payload = record["result"]
    assert payload["error_code"] == "cli_unknown_command"
    assert "choose from" not in payload["error"]
    assert "echo --help" not in payload["remediation"]
    assert payload["remediation"].endswith("--help' to list commands.")


def test_noninteractive_shell_frames_pretty_json_as_ndjson(monkeypatch, capsys):
    from mtdata.core.cli import api

    batch = "market_ticker EURUSD --json\nmarket_ticker GBPUSD --json\n"

    def _main():
        print(
            json.dumps(
                {"symbol": api.sys.argv[2], "prices": [1.1, 1.2]},
                indent=2,
            )
        )
        return 0

    monkeypatch.setattr(api.sys, "stdin", io.StringIO(batch))
    monkeypatch.setattr(api, "main", _main)

    assert api.run_shell(interactive=False) == 0
    output_lines = capsys.readouterr().out.splitlines()
    assert len(output_lines) == 2
    records = [json.loads(line) for line in output_lines]
    assert [record["line"] for record in records] == [1, 2]
    assert [record["result"]["symbol"] for record in records] == [
        "EURUSD",
        "GBPUSD",
    ]
    assert all(record["success"] for record in records)


def test_static_command_catalog_matches_registered_tools():
    from mtdata.bootstrap.tools import bootstrap_tools
    from mtdata.core.cli.api import discover_tools
    from mtdata.core.cli.catalog import available_command_names

    bootstrap_tools()

    assert set(available_command_names()) == set(discover_tools())


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
    assert "--json" in output
    assert "--precision" in output
    assert "--output-fields" in output
    assert "--timeframe" in output
    assert "exit or quit" in output
    assert "NDJSON" in output


@pytest.mark.parametrize(
    "argv",
    [
        ["shell", "--json"],
        ["--json", "shell"],
        ["--precision", "full", "shell", "--precision", "compact", "--json"],
    ],
)
def test_shell_accepts_shared_options_before_or_after_command(
    monkeypatch, capsys, argv
):
    from mtdata.core.cli import api

    def sample_tool(**_kwargs) -> dict:
        return {"success": True, "value": 7}

    monkeypatch.setattr(api, "load_environment", lambda: None)
    monkeypatch.setattr(
        api,
        "discover_tools",
        lambda *_args: {
            "sample_tool": {
                "func": sample_tool,
                "meta": {"description": "Sample tool"},
            }
        },
    )
    monkeypatch.setattr(api.sys, "stdin", io.StringIO("sample_tool\n"))
    monkeypatch.setattr(api.sys, "argv", ["mtdata-cli", *argv])

    status = api.main()

    assert status == 0
    record = json.loads(capsys.readouterr().out)
    assert record["result"] == {"success": True, "value": 7}


def test_shell_timeframe_only_reaches_compatible_children(monkeypatch, capsys):
    from mtdata.core.cli import api

    def candles(timeframe: str = "H1", **_kwargs) -> dict:
        return {"success": True, "timeframe": timeframe}

    def ticker(**_kwargs) -> dict:
        return {"success": True, "kind": "ticker"}

    monkeypatch.setattr(api, "load_environment", lambda: None)
    monkeypatch.setattr(
        api,
        "discover_tools",
        lambda *_args: {
            "candles": {
                "func": candles,
                "meta": {"description": "Candles"},
            },
            "ticker": {
                "func": ticker,
                "meta": {"description": "Ticker"},
            },
        },
    )
    monkeypatch.setattr(
        api.sys,
        "stdin",
        io.StringIO("candles\nticker\n"),
    )
    monkeypatch.setattr(
        api.sys,
        "argv",
        ["mtdata-cli", "shell", "--json", "--timeframe", "H4"],
    )

    status = api.main()

    assert status == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["result"]["timeframe"] == "H4"
    assert records[1]["result"]["kind"] == "ticker"
