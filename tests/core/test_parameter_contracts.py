import argparse

from mtdata.core.cli import _resolve_param_kwargs
from mtdata.core.cli.api import _add_forecast_generate_args
from mtdata.core.unified_params import add_global_args_to_parser
from mtdata.shared.parameter_contracts import PARAMETER_HELP
from mtdata.shared.schema import PARAM_HINTS


def test_param_hints_reuse_shared_help_text() -> None:
    for name, help_text in PARAMETER_HELP.items():
        assert PARAM_HINTS[name] == help_text


def test_dynamic_cli_param_help_reuses_shared_help_text() -> None:
    symbol_kwargs, _ = _resolve_param_kwargs(
        {"name": "symbol", "type": str, "required": True, "default": None},
        None,
    )
    timeframe_kwargs, _ = _resolve_param_kwargs(
        {"name": "timeframe", "type": str, "required": False, "default": "H1"},
        None,
    )

    assert symbol_kwargs["help"] == PARAMETER_HELP["symbol"]
    assert timeframe_kwargs["help"] == PARAMETER_HELP["timeframe"]


def test_cli_parsers_reuse_shared_help_text() -> None:
    parser = argparse.ArgumentParser()
    add_global_args_to_parser(parser)
    assert parser._option_string_actions["--timeframe"].help == PARAMETER_HELP["timeframe"]
    assert parser._option_string_actions["--verbose"].help == PARAMETER_HELP["verbose"]

    forecast_parser = argparse.ArgumentParser()
    _add_forecast_generate_args(forecast_parser)
    symbol_action = next(
        action
        for action in forecast_parser._actions
        if action.dest == "symbol" and not action.option_strings
    )
    assert symbol_action.help == PARAMETER_HELP["symbol"]
    assert forecast_parser._option_string_actions["--timeframe"].help == PARAMETER_HELP["timeframe"]
    assert forecast_parser._option_string_actions["--verbose"].help == PARAMETER_HELP["verbose"]
