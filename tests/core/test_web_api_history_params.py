from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from mtdata.core import web_api


def test_history_uses_start_end_ohlcv_and_drops_open_bar() -> None:
    payload = {
        "success": True,
        "data": [
            {"time": 1735689600.0, "close": 1.1},
            {"time": 1735693200.0, "close": 1.2},
            {"time": 1735696800.0, "close": 1.3},
        ],
        "last_candle_open": True,
    }
    with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), patch(
        "mtdata.core.web_api._fetch_candles_impl", return_value=payload
    ) as mock_fetch:
        res = web_api.get_history(
            symbol="EURUSD",
            timeframe="H1",
            limit=3,
            start="2025-01-01 00:00",
            end="2025-01-01 03:00",
            ohlcv="close",
            include_incomplete=False,
        )

    assert len(res["data"]) == 2
    assert res["candles"] == 2
    kwargs = mock_fetch.call_args.kwargs
    assert kwargs["start"] == "2025-01-01 00:00"
    assert kwargs["end"] == "2025-01-01 03:00"
    assert kwargs["ohlcv"] == "close"
    assert kwargs["include_incomplete"] is False
    assert kwargs["time_as_epoch"] is True
    assert all(isinstance(row["time"], (int, float)) for row in res["data"])


def test_history_passes_include_spread() -> None:
    payload = {
        "success": True,
        "data": [
            {"time": 1735689600.0, "close": 1.1, "spread": 12},
        ],
        "last_candle_open": False,
    }
    with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), patch(
        "mtdata.core.web_api._fetch_candles_impl", return_value=payload
    ) as mock_fetch:
        res = web_api.get_history(
            symbol="EURUSD",
            timeframe="H1",
            limit=1,
            ohlcv="close",
            include_spread=True,
        )

    assert res["data"][0]["spread"] == 12
    kwargs = mock_fetch.call_args.kwargs
    assert kwargs["include_spread"] is True
