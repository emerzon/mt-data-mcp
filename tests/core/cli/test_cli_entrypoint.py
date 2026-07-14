from unittest.mock import patch


def test_version_path_does_not_import_cli_api(capsys):
    from mtdata.core.cli import main

    with (
        patch("mtdata.core.cli._installed_version", return_value="9.8.7"),
        patch.dict("sys.modules", {"mtdata.core.cli.api": None}),
    ):
        status = main(["--version"])

    assert status == 0
    assert capsys.readouterr().out.strip() == "mtdata-cli 9.8.7"


def test_shell_reuses_process_and_runs_entered_commands(monkeypatch):
    from mtdata.core.cli import api

    commands = iter(["symbols_list --limit 2 --json", "quit"])
    observed = []
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    monkeypatch.setattr(api, "main", lambda: observed.append(list(api.sys.argv)) or 0)

    status = api.run_shell()

    assert status == 0
    assert observed == [[api.sys.argv[0], "symbols_list", "--limit", "2", "--json"]]
