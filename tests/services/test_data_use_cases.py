from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mtdata.core import data as core_data
from mtdata.core.data.requests import DataFetchCandlesRequest, DataFetchTicksRequest
from mtdata.core.data.use_cases import run_data_fetch_candles, run_data_fetch_ticks
from mtdata.utils.mt5 import MT5ConnectionError


def test_run_data_fetch_candles_logs_finish_event(caplog):
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=10)

    with caplog.at_level("DEBUG", logger="mtdata.core.data.use_cases"):
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


def test_data_fetch_requests_accept_simplify_boolean_and_modes():
    candles_on = DataFetchCandlesRequest(symbol="EURUSD", simplify=True)
    ticks_on = DataFetchTicksRequest(symbol="EURUSD", simplify="auto")
    candles_off = DataFetchCandlesRequest(symbol="EURUSD", simplify="off")
    ticks_off = DataFetchTicksRequest(symbol="EURUSD", simplify=False)

    assert candles_on.simplify == {}
    assert ticks_on.simplify == {}
    assert candles_off.simplify is None
    assert ticks_off.simplify is None


def test_data_fetch_requests_explain_invalid_simplify_string():
    with pytest.raises(ValidationError) as exc_info:
        DataFetchCandlesRequest(symbol="EURUSD", simplify="maybe")

    message = str(exc_info.value)
    assert "{'method': 'lttb', 'points': 100}" in message
    assert "on/auto" in message


def test_data_fetch_candles_accepts_standard_detail_alias():
    assert DataFetchCandlesRequest(symbol="EURUSD", detail="standard").detail == "standard"


def test_data_fetch_candles_schema_documents_ohlcv():
    schema = DataFetchCandlesRequest.model_json_schema()
    ohlcv = schema["properties"]["ohlcv"]

    assert "Candle fields to include" in ohlcv["description"]
    assert "ohlcv" in ohlcv["examples"]
    assert "open,high,low,close,volume" in ohlcv["description"]


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


def test_run_data_fetch_candles_compact_omits_default_metadata():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=5)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "candles_requested": 5,
            "candles_excluded": 0,
            "last_candle_open": False,
            "incomplete_candles_skipped": 0,
            "has_forming_candle": False,
            "data": [],
        },
    )

    assert result == {"success": True, "candles": 5, "data": []}


def test_run_data_fetch_candles_compact_keeps_freshness_without_meta():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=5)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "candles": 5,
            "data": [],
            "meta": {
                "diagnostics": {
                    "query": {"latency_ms": 12.3, "warmup_bars": 0},
                    "freshness": {
                        "data_freshness_seconds": 60.0,
                        "last_bar_within_policy_window": True,
                    },
                },
            },
        },
    )

    assert "meta" not in result
    assert result["data_freshness_seconds"] == 60.0
    assert "latency_ms" not in result
    assert "last_bar_within_policy_window" not in result


def test_run_data_fetch_candles_standard_keeps_public_diagnostics_only():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        detail="standard",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "candles_requested": 5,
            "data": [],
            "meta": {
                "diagnostics": {
                    "query": {
                        "latency_ms": 12.3,
                        "warmup_retry": {"applied": False},
                        "cache_status": "unknown",
                    },
                    "freshness": {
                        "data_freshness_seconds": 60.0,
                        "last_bar_within_policy_window": True,
                    },
                },
            },
        },
    )

    assert "meta" not in result
    assert "symbol" not in result
    assert result["data_freshness_seconds"] == 60.0
    assert result["latency_ms"] == 12.3
    assert result["last_bar_within_policy_window"] is True
    assert "warmup_retry" not in result
    assert "cache_status" not in result


def test_run_data_fetch_candles_compact_keeps_anomaly_metadata():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=5)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 4,
            "candles_requested": 5,
            "candles_excluded": 1,
            "candle_counts": {
                "requested": 5,
                "returned": 4,
                "excluded": {
                    "forming_bar": 1,
                    "indicator_warmup": 0,
                    "quality_filtered": 0,
                    "window_or_source_shortfall": 0,
                    "total": 1,
                },
            },
            "last_candle_open": True,
            "incomplete_candles_skipped": 1,
            "has_forming_candle": True,
            "data": [],
        },
    )

    assert result["candles"] == 4
    assert "candle_counts" not in result
    assert "last_candle_open" not in result
    assert "hint" not in result
    assert "candles_excluded" not in result
    assert "incomplete_candles_skipped" not in result
    assert result["has_forming_candle"] is True
    assert "symbol" not in result
    assert "timeframe" not in result
    assert "candles_requested" not in result


def test_run_data_fetch_candles_full_omits_zero_exclusion_categories():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        detail="full",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 4,
            "candle_counts": {
                "requested": 5,
                "returned": 4,
                "excluded": {
                    "forming_bar": 1,
                    "indicator_warmup": 0,
                    "quality_filtered": 0,
                    "window_or_source_shortfall": 0,
                    "total": 1,
                },
            },
            "data": [],
        },
    )

    assert result["candle_counts"]["excluded"] == {"forming_bar": 1, "total": 1}


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
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles_requested": 10,
            "last_candle_open": False,
            "data": rows,
        },
    )

    assert result["data"] == rows
    assert result["symbol"] == "EURUSD"
    assert result["timeframe"] == "H1"
    assert result["candles_requested"] == 10
    assert result["last_candle_open"] is False
    assert "series" not in result
    assert result["collection_kind"] == "time_series"
    assert result["collection_contract_version"] == "collection.v1"
    assert "canonical_source" not in result


def test_run_data_fetch_ticks_logs_connection_error(caplog):
    request = DataFetchTicksRequest(symbol="EURUSD", limit=5)

    with caplog.at_level("DEBUG", logger="mtdata.core.data.use_cases"):
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
    with caplog.at_level("DEBUG", logger=core_data.logger.name):
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
    with caplog.at_level("DEBUG"):
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
    assert request.limit == 200


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
