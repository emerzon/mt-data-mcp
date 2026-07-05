from __future__ import annotations

from mtdata.core.cli import _build_epilog, get_function_info


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
