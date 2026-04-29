from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_web_api_handlers_do_not_import_cli_formatting() -> None:
    source = _source("src/mtdata/core/web_api_handlers.py")

    assert "cli_formatting" not in source


def test_forecast_tools_do_not_import_cli_formatting() -> None:
    source = _source("src/mtdata/core/forecast.py")

    assert "cli_formatting" not in source


def test_tool_calling_exposes_transport_neutral_raw_invocation() -> None:
    source = _source("src/mtdata/core/tool_calling.py")

    assert "def call_tool_sync_structured" in source
    assert "raw_tool_output" in source
