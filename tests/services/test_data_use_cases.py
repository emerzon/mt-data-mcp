from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mtdata.core import data as core_data
from mtdata.core.data.requests import DataFetchCandlesRequest, DataFetchTicksRequest
from mtdata.core.data.use_cases import run_data_fetch_candles, run_data_fetch_ticks
from mtdata.utils.mt5 import MT5ConnectionError


def test_run_data_fetch_candles_logs_finish_event(caplog):
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=10)

    with caplog.at_level("INFO", logger="mtdata.core.data.use_cases"):
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


def test_run_data_fetch_candles_passes_allow_stale_to_service():
    captured = {}
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        allow_stale=True,
    )

    def _fetch(**kwargs):
        captured["kwargs"] = kwargs
        return {"success": True}

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=_fetch,
    )

    assert result["success"] is True
    assert captured["kwargs"]["allow_stale"] is True


def test_run_data_fetch_candles_passes_include_spread_to_service():
    captured = {}
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        include_spread=True,
    )

    def _fetch(**kwargs):
        captured["kwargs"] = kwargs
        return {"success": True}

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=_fetch,
    )

    assert result["success"] is True
    assert captured["kwargs"]["include_spread"] is True


def test_data_fetch_requests_accept_simplify_boolean_and_string_aliases():
    candles_on = DataFetchCandlesRequest(symbol="EURUSD", simplify=True)
    ticks_on = DataFetchTicksRequest(symbol="EURUSD", simplify="on")
    candles_off = DataFetchCandlesRequest(symbol="EURUSD", simplify="off")

    assert candles_on.simplify == {}
    assert ticks_on.simplify == {}
    assert candles_off.simplify is None


def test_data_fetch_requests_explain_invalid_simplify_string():
    with pytest.raises(ValidationError) as exc_info:
        DataFetchCandlesRequest(symbol="EURUSD", simplify="maybe")

    assert "{'method': 'lttb', 'points': 100}" in str(exc_info.value)


def test_data_fetch_candles_rejects_standard_detail_alias():
    with pytest.raises(ValidationError):
        DataFetchCandlesRequest(symbol="EURUSD", detail="standard")


def test_run_data_fetch_candles_omits_contract_metadata_in_compact_detail():
    rows = [{"time": 1.0, "close": 1.1}]
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=10)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {"success": True, "data": rows},
    )

    assert result["data"] == rows
    assert "series" not in result
    assert "collection_kind" not in result
    assert "collection_contract_version" not in result


def test_run_data_fetch_candles_adds_contract_metadata_in_full_detail():
    rows = [{"time": 1.0, "close": 1.1}]
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        detail="full",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {"success": True, "data": rows},
    )

    assert result["data"] == rows
    assert result["series"] == rows
    assert result["collection_kind"] == "time_series"
    assert result["collection_contract_version"] == "collection.v1"
    assert result["canonical_source"] == "series"


def test_run_data_fetch_ticks_logs_connection_error(caplog):
    request = DataFetchTicksRequest(symbol="EURUSD", limit=5)

    with caplog.at_level("INFO", logger="mtdata.core.data.use_cases"):
        result = run_data_fetch_ticks(
            request,
            gateway=SimpleNamespace(
                ensure_connection=lambda: (_ for _ in ()).throw(MT5ConnectionError("no mt5"))
            ),
            fetch_ticks_impl=lambda **kwargs: {"ticks": []},
        )

    assert result["error"] == "no mt5"
    assert result["success"] is False
    assert result["error_code"] == "mt5_connection_error"
    assert result["operation"] == "mt5_ensure_connection"
    assert isinstance(result.get("request_id"), str)
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

    raw = getattr(core_data.data_fetch_candles, "__wrapped__", core_data.data_fetch_candles)
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=10)
    with caplog.at_level("INFO", logger=core_data.logger.name):
        result = raw(request)

    assert result["success"] is True
    assert any(
        "event=finish operation=data_fetch_candles success=True" in record.message
        for record in caplog.records
    )


def test_data_fetch_candles_wrapper_and_use_case_emit_single_finish_event(monkeypatch, caplog):
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

    raw = getattr(core_data.data_fetch_candles, "__wrapped__", core_data.data_fetch_candles)
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


def test_data_fetch_candles_request_defaults_to_compact_detail():
    request = DataFetchCandlesRequest(symbol="EURUSD")

    assert request.detail == "compact"


def test_data_fetch_candles_wrapper_respects_detail_contract(monkeypatch):
    monkeypatch.setattr(
        core_data,
        "run_data_fetch_candles",
        lambda request, gateway, fetch_candles_impl: {
            "success": True,
            "symbol": request.symbol,
            "data": [],
            "meta": {"diagnostics": {"query": {"requested_bars": request.limit}}},
        },
    )

    compact = core_data.data_fetch_candles(
        request=DataFetchCandlesRequest(symbol="EURUSD", detail="compact"),
        __cli_raw=True,
    )
    full = core_data.data_fetch_candles(
        request=DataFetchCandlesRequest(symbol="EURUSD", detail="full"),
        __cli_raw=True,
    )

    assert "meta" not in compact
    assert full["meta"]["tool"] == "data_fetch_candles"
    assert full["meta"]["diagnostics"]["query"]["requested_bars"] == 200


def test_data_fetch_ticks_request_rejects_removed_output_field():
    with pytest.raises(ValidationError, match="output was removed; use output_mode"):
        DataFetchTicksRequest(symbol="EURUSD", output="rows")


def test_data_fetch_ticks_request_accepts_format_as_output_mode_alias():
    request = DataFetchTicksRequest(symbol="EURUSD", format="rows")

    assert request.output_mode == "rows"
    assert list(DataFetchTicksRequest.model_fields) == [
        "symbol",
        "limit",
        "start",
        "end",
        "simplify",
        "output_mode",
    ]


@pytest.mark.parametrize("raw_output_mode", ["compact", "full"])
def test_data_fetch_ticks_request_rejects_shared_detail_aliases_as_output_modes(
    raw_output_mode: str,
):
    with pytest.raises(ValidationError):
        DataFetchTicksRequest(symbol="EURUSD", output_mode=raw_output_mode)
