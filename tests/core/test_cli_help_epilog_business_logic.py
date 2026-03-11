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
