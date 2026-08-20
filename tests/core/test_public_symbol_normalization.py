from __future__ import annotations

from pydantic import BaseModel

from mtdata.core._mcp_tools import _prepare_public_tool_call


def test_public_tool_boundary_normalizes_flat_symbol() -> None:
    def example(symbol: str) -> dict[str, str]:
        return {"symbol": symbol}

    kwargs = {"symbol": " eurusd "}

    _prepare_public_tool_call(example, kwargs)

    assert kwargs["symbol"] == "EURUSD"


def test_public_tool_boundary_normalizes_request_model_symbol() -> None:
    class Request(BaseModel):
        symbol: str

    def example(request: Request) -> dict[str, str]:
        return {"symbol": request.symbol}

    kwargs = {"request": Request(symbol=" eurusd ")}

    _prepare_public_tool_call(example, kwargs)

    assert kwargs["request"].symbol == "EURUSD"
