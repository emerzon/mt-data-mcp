from types import SimpleNamespace

from mtdata.core.data_requests import DataFetchCandlesRequest, DataFetchTicksRequest
from mtdata.core import data as core_data
from mtdata.core.data_use_cases import run_data_fetch_candles, run_data_fetch_ticks
from mtdata.utils.mt5 import MT5ConnectionError


def test_run_data_fetch_candles_logs_finish_event(caplog):
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=10)

    with caplog.at_level("INFO", logger="mtdata.core.data_use_cases"):
        result = run_data_fetch_candles(
            request,
            gateway=SimpleNamespace(ensure_connection=lambda: None),
            fetch_candles_impl=lambda **kwargs: {"candles": [], "success": True},
        )

    assert result["success"] is True
    assert any(
        "event=finish operation=data_fetch_candles success=True" in record.message
        for record in caplog.records
    )


def test_run_data_fetch_ticks_logs_connection_error(caplog):
    request = DataFetchTicksRequest(symbol="EURUSD", limit=5)

    with caplog.at_level("INFO", logger="mtdata.core.data_use_cases"):
        result = run_data_fetch_ticks(
            request,
            gateway=SimpleNamespace(
                ensure_connection=lambda: (_ for _ in ()).throw(
                    MT5ConnectionError("no mt5")
                )
            ),
            fetch_ticks_impl=lambda **kwargs: {"ticks": []},
        )

    assert result == {"error": "no mt5"}
    assert any(
        "event=finish operation=data_fetch_ticks success=False" in record.message
        for record in caplog.records
    )


def test_data_fetch_candles_logs_finish_event(monkeypatch, caplog):
    monkeypatch.setattr(
        core_data,
        "run_data_fetch_candles",
        lambda request, gateway, fetch_candles_impl: {
            "success": True,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "data": [],
        },
    )

    raw = getattr(
        core_data.data_fetch_candles, "__wrapped__", core_data.data_fetch_candles
    )
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=10)
    with caplog.at_level("INFO", logger=core_data.logger.name):
        result = raw(request)

    assert result["success"] is True
    assert any(
        "event=finish operation=data_fetch_candles success=True" in record.message
        for record in caplog.records
    )


def test_data_fetch_candles_wrapper_and_use_case_emit_single_finish_event(
    monkeypatch, caplog
):
    monkeypatch.setattr(
        core_data,
        "run_data_fetch_candles",
        lambda request, gateway, fetch_candles_impl: {
            "success": True,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "data": [],
        },
    )

    raw = getattr(
        core_data.data_fetch_candles, "__wrapped__", core_data.data_fetch_candles
    )
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=10)
    with caplog.at_level("INFO"):
        result = raw(request)

    assert result["success"] is True
    finish_records = [
        record
        for record in caplog.records
        if "event=finish operation=data_fetch_candles success=True" in record.message
    ]
    assert len(finish_records) == 1
