from __future__ import annotations

import argparse

from mtdata.core.cli.api import (
    _CLI_DESCRIPTION,
    _build_epilog,
    _sort_subparser_help_choices,
    get_function_info,
)


def test_cli_description_promotes_warm_workflows() -> None:
    assert "One-shot commands initialize the full tool runtime" in _CLI_DESCRIPTION
    assert "mtdata-cli shell" in _CLI_DESCRIPTION
    assert "long-lived stdio or HTTP server" in _CLI_DESCRIPTION


def test_custom_subparser_help_choices_are_sorted() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser("forecast_volatility_estimate", help="Volatility")
    subparsers.add_parser("forecast_generate", help="Forecast")

    _sort_subparser_help_choices(subparsers)

    assert [action.dest for action in subparsers._choices_actions] == [
        "forecast_generate",
        "forecast_volatility_estimate",
    ]


def test_build_epilog_formats_required_args_like_runtime_parser() -> None:
    def trade_place_like(symbol: str, volume: float, order_type: str):
        """Place order."""

    info = get_function_info(trade_place_like)
    functions = {
        "trade_place_like": {
            "func": trade_place_like,
            "meta": {"description": "Place order"},
            "_cli_func_info": info,
        },
    }

    epilog = _build_epilog(functions)
    assert "trade_place_like: symbol<str> --volume<float> --order-type<str>" in epilog


def test_build_epilog_groups_commands_by_category() -> None:
    def data_fetch_candles(symbol: str):
        """Fetch candles."""

    def forecast_generate(symbol: str):
        """Generate forecast."""

    def trade_place(symbol: str, volume: float):
        """Place order."""

    functions = {}
    for func in (data_fetch_candles, forecast_generate, trade_place):
        info = get_function_info(func)
        functions[func.__name__] = {
            "func": func,
            "meta": {"description": info["doc"].splitlines()[0]},
            "_cli_func_info": info,
        }

    epilog = _build_epilog(functions)

    assert "Commands and Arguments by Category:" in epilog
    assert "DATA ACCESS:" in epilog
    assert "FORECASTING:" in epilog
    assert "TRADING:" in epilog
    assert epilog.index("DATA ACCESS:") < epilog.index("- data_fetch_candles:")
    assert epilog.index("FORECASTING:") < epilog.index("- forecast_generate:")
    assert epilog.index("TRADING:") < epilog.index("- trade_place:")
