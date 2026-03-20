from __future__ import annotations

from pydantic import BaseModel

from mtdata.core.tool_calling import call_tool_sync_raw


class _DummyRequest(BaseModel):
    symbol: str
    limit: int = 25


def test_call_tool_sync_raw_builds_request_model_from_keyword_fields() -> None:
    def _tool(request: _DummyRequest):
        return {"symbol": request.symbol, "limit": request.limit}

    out = call_tool_sync_raw(_tool, cli_raw=True, symbol="EURUSD", limit=10)

    assert out == {"symbol": "EURUSD", "limit": 10}
