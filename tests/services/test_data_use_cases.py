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


@pytest.mark.parametrize(
    ("message", "expected_code"),
    [
        (
            "Symbol 'EURUSD.bad' was not found or is not available in MT5.",
            "data_fetch_candles_symbol_unavailable",
        ),
        (
            "start_datetime must be before end_datetime",
            "data_fetch_candles_invalid_date_range",
        ),
        (
            "start datetime 2099-01-01 is in the future; no historical data is available for future dates.",
            "data_fetch_candles_future_date_range",
        ),
    ],
)
def test_run_data_fetch_candles_classifies_query_errors(message, expected_code):
    request = DataFetchCandlesRequest(
        symbol="EURUSD.bad",
        timeframe="H1",
        start="2026-01-02T00:00:00Z",
        end="2026-01-01T00:00:00Z",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {"error": message},
    )

    assert result["success"] is False
    assert result["error_code"] == expected_code
    assert result["operation"] == "data_fetch_candles"
    assert result["remediation"]
    assert result["details"]["symbol"] == "EURUSD.BAD"


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


def test_run_data_fetch_candles_expands_default_limit_for_indicators():
    captured = {}
    request = DataFetchCandlesRequest(symbol="EURUSD", indicators="rsi(14)")

    def _fetch(**kwargs):
        captured["kwargs"] = kwargs
        return {"success": True, "count": 100, "data": []}

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=_fetch,
    )

    assert result["success"] is True
    assert captured["kwargs"]["limit"] == 100


def test_run_data_fetch_candles_honors_explicit_indicator_limit():
    captured = {}
    request = DataFetchCandlesRequest(symbol="EURUSD", indicators="rsi(14)", limit=20)

    def _fetch(**kwargs):
        captured["kwargs"] = kwargs
        return {"success": True, "count": 20, "data": []}

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=_fetch,
    )

    assert result["success"] is True
    assert captured["kwargs"]["limit"] == 20


def test_run_data_fetch_candles_uses_compact_plain_default_limit():
    captured = {}
    request = DataFetchCandlesRequest(symbol="EURUSD")

    def _fetch(**kwargs):
        captured["kwargs"] = kwargs
        return {"success": True, "count": 20, "data": []}

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=_fetch,
    )

    assert result["success"] is True
    assert request.limit == 20
    assert captured["kwargs"]["limit"] == 20


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


def test_data_fetch_candles_accepts_summary_detail():
    assert DataFetchCandlesRequest(symbol="EURUSD", detail="summary").detail == "summary"


@pytest.mark.parametrize("request_cls", [DataFetchCandlesRequest, DataFetchTicksRequest])
def test_data_fetch_requests_normalize_detail_aliases(request_cls):
    assert request_cls(symbol="EURUSD", detail="summary_only").detail == "summary"
    assert request_cls(symbol="EURUSD", detail=" Full ").detail == "full"


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
            "incomplete_candles_skipped": 0,
            "has_forming_candle": False,
            "forming_candle_status": "none",
            "forming_candle_included": False,
            "forming_candle_skipped": False,
            "volume_note": "MT5 tick_volume is broker tick count.",
            "bar_time_convention": "bar_open_time",
            "data": [],
        },
    )

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "H1",
        "count": 5,
        "data": [],
    }


def test_run_data_fetch_candles_compact_omits_tick_volume_note():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=5)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "volume_type": "tick_count",
            "volume_note": "MT5 tick_volume is broker tick count.",
            "data": [{"time": 1, "close": 1.1, "tick_volume": 20}],
        },
    )

    assert result["volume_type"] == "tick_count"
    assert result["volume_semantics"] == "tick_volume_is_broker_tick_count_not_lots"
    assert "volume_note" not in result


def test_run_data_fetch_candles_compact_discloses_inplace_denoise():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        denoise={"method": "ema"},
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "denoise": {
                "applications": [
                    {
                        "method": "ema",
                        "when": "pre_ti",
                        "keep_original": False,
                        "columns": ["close"],
                        "added_columns": [],
                    }
                ]
            },
            "data": [{"time": 1, "close": 1.1}],
        },
    )

    assert result["denoise_applied"] is True
    assert result["denoise_method"] == "ema"
    assert result["denoise_overwrote_columns"] == ["close"]
    assert result["price_column"] == "close (ema-smoothed)"
    assert "denoise" not in result


def test_run_data_fetch_candles_projection_drops_hidden_volume_semantics():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=5, ohlcv="close")

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "ohlcv_filter_applied": True,
            "volume_type": "tick_count",
            "volume_unit": "broker_tick_count",
            "real_volume_type": "traded_volume",
            "real_volume_unit": "traded_volume",
            "units": {
                "open": "absolute_price",
                "close": "absolute_price",
                "tick_volume": "broker_tick_count",
                "real_volume": "traded_volume",
            },
            "data": [{"time": 1, "close": 1.1}],
        },
    )

    assert result["data"] == [{"time": 1, "close": 1.1}]
    assert "volume_type" not in result
    assert "volume_unit" not in result
    assert "volume_semantics" not in result
    assert "real_volume_type" not in result
    assert "real_volume_unit" not in result
    assert result["units"] == {"close": "absolute_price"}
    assert result["timestamp_format"] == "epoch_seconds"


def test_run_data_fetch_candles_compact_keeps_staleness_without_meta():
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
    assert result["freshness"] == "fresh, bar 1m 0s ago"
    assert result["data_stale"] is False
    assert result["data_age_anchor"] == "wall_clock"
    assert result["data_age_metric"] == "last_completed_bar_age_seconds"
    assert "freshness_basis" not in result
    assert "data_freshness_seconds" not in result
    assert result["data_age_seconds"] == 60.0
    assert "data_age" not in result
    assert "latency_ms" not in result
    assert "last_bar_within_policy_window" not in result


def test_run_data_fetch_candles_compact_flags_stale_latest_data():
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
                    "query": {"mode": "latest"},
                    "freshness": {
                        "data_freshness_seconds": 3661.0,
                        "last_bar_within_policy_window": False,
                    },
                },
            },
        },
    )

    assert result["freshness"] == "stale, bar 1h 1m ago"
    assert result["query_type"] == "latest"
    assert result["data_stale"] is True
    assert result["data_age_anchor"] == "wall_clock"
    assert result["data_age_metric"] == "last_completed_bar_age_seconds"
    assert "freshness_basis" not in result
    assert result["data_age_seconds"] == 3661.0
    assert "data_age" not in result
    assert "stale_warning" not in result


def test_run_data_fetch_candles_closed_market_relaxation_is_not_stale():
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
                    "query": {"mode": "latest"},
                    "freshness": {
                        "data_freshness_seconds": 149668.6,
                        "last_bar_within_policy_window": False,
                        "freshness_policy_relaxed": (
                            "latest_completed_bar_for_live_request"
                        ),
                        "market_session_status": "closed_or_idle",
                        "freshness_note": (
                            "Market appears closed or idle; showing the latest "
                            "completed bar."
                        ),
                    },
                },
            },
        },
    )

    assert result["freshness"].startswith("closed or idle, bar ")
    assert result["query_type"] == "latest"
    assert result["data_stale"] is False
    assert result["data_age_seconds"] == 149668.6
    assert result["data_age_anchor"] == "wall_clock"
    assert result["data_age_metric"] == "last_completed_bar_age_seconds"
    assert result["freshness_policy_relaxed"] is True
    assert result["market_status"] == "closed_or_idle"
    assert result["note"] == (
        "Market appears closed or idle; showing the latest completed bar."
    )
    assert "stale_warning" not in result


def test_run_data_fetch_candles_range_applies_limit_cap():
    rows = [{"time": f"t{i}", "close": i} for i in range(5)]
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=2,
        start="2026-01-01",
        end="2026-01-02",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "candles": 5,
            "data": rows,
            "meta": {"diagnostics": {"query": {"mode": "range"}}},
        },
    )

    assert result["data"] == rows[-2:]
    assert result["count"] == 2
    assert "candles" not in result
    assert result["available_count"] == 5
    assert result["limit_applied"] == 2
    assert result["truncated"] is True
    assert result["truncation"] == {
        "reason": "limit",
        "retained": "last",
        "excluded_count": 3,
    }
    assert result["warnings"] == [
        "Range contained 5 bars; returned the latest 2 because limit=2. "
        "Set limit>=5 to return the full range."
    ]
    assert result["query_type"] == "historical"


def test_run_data_fetch_candles_normalizes_count_metadata():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=2,
        start="2026-01-01",
        end="2026-01-02",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "candles": 5,
            "requested_limit": 2,
            "returned_count": 5,
            "data_window": {
                "start": "t1",
                "end": "t2",
                "requested_limit": 2,
                "returned_count": 5,
            },
            "meta": {"diagnostics": {"query": {"mode": "range"}}},
            "data": [{"time": f"t{index}"} for index in range(5)],
        },
    )

    assert result["count"] == 2
    assert result["requested_limit"] == 2
    assert "candles" not in result
    assert "returned_count" not in result
    assert result["data_window"] == {"start": "t1", "end": "t2"}


def test_run_data_fetch_candles_compact_keeps_spread_estimate_without_meta():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        include_spread=True,
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "candles": 1,
            "data": [{"time": 1.0, "close": 1.1, "spread": 0.00009}],
            "meta": {
                "diagnostics": {
                    "spread_estimate": {
                        "estimated_mean": 0.00009,
                        "source": "tick_stats",
                    },
                },
            },
        },
    )

    assert "meta" not in result
    assert result["spread_estimate"] == {
        "value": 0.00009,
        "source": "tick_stats",
    }


def test_run_data_fetch_candles_compact_exposes_range_gap_metadata():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        start="2 days ago",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "data": [],
            "meta": {
                "diagnostics": {
                    "query": {"mode": "range"},
                    "freshness": {
                        "data_freshness_seconds": -10.0,
                        "last_bar_within_policy_window": True,
                    },
                },
            },
        },
    )

    assert result["query_type"] == "historical"
    assert "data_freshness_seconds" not in result
    assert "data_age_seconds" not in result
    assert result["query_end_gap_seconds"] == 0.0
    assert result["query_end_gap_anchor"] == "query_expected_end"
    assert result["query_end_gap_metric"] == "requested_range_end_gap_seconds"
    assert result["query_end_gap"] == "0s"


def test_run_data_fetch_candles_standard_omits_verbose_diagnostics():
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
                        "mode": "latest",
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
    assert result["symbol"] == "EURUSD"
    assert result["timeframe"] == "H1"
    assert result["query_type"] == "latest"
    assert result["data_stale"] is False
    assert result["freshness"] == "fresh, bar 1m 0s ago"
    assert result["data_age_anchor"] == "wall_clock"
    assert result["data_age_metric"] == "last_completed_bar_age_seconds"
    assert "latency_ms" not in result
    assert "freshness_basis" not in result
    assert "data_freshness_seconds" not in result
    assert result["data_age_seconds"] == 60.0
    assert "data_age" not in result
    assert "last_bar_within_policy_window" not in result
    assert "warmup_retry" not in result
    assert "cache_status" not in result


def test_run_data_fetch_candles_standard_handles_bool_like_freshness_flags():
    class FalseLike:
        def __bool__(self):
            return False

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
            "data": [],
            "meta": {
                "diagnostics": {
                    "query": {"mode": "latest"},
                    "freshness": {
                        "data_freshness_seconds": 3661.0,
                        "last_bar_within_policy_window": FalseLike(),
                    },
                },
            },
        },
    )

    assert "last_bar_within_policy_window" not in result
    assert result["data_stale"] is True
    assert result["freshness"] == "stale, bar 1h 1m ago"


def test_run_data_fetch_candles_standard_surfaces_mt5_time_alignment_warning():
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
            "data": [],
            "meta": {
                "diagnostics": {
                    "query": {"mode": "latest"},
                    "freshness": {
                        "data_freshness_seconds": 60.0,
                        "last_bar_within_policy_window": True,
                    },
                    "mt5_time_alignment": {
                        "status": "stale",
                        "reason": "market_data_stale",
                        "warning": "MT5 broker-time sanity check could not confirm live alignment: market is closed",
                        "probe_timeframe": "M1",
                    },
                },
            },
        },
    )

    assert result["mt5_time_alignment"] == {
        "status": "stale",
        "reason": "market_data_stale",
        "warning": "MT5 broker-time sanity check could not confirm live alignment: market is closed",
        "probe_timeframe": "M1",
    }


def test_run_data_fetch_candles_summary_omits_rows_and_keeps_metadata():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        detail="summary",
    )

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
                "excluded": {"forming_bar": 1, "total": 1},
            },
            "incomplete_candles_skipped": 1,
            "has_forming_candle": True,
            "forming_candle_status": "skipped",
            "forming_candle_included": False,
            "forming_candle_skipped": True,
            "timezone": "UTC",
            "data": [
                {
                    "time": "2026-05-14 11:00",
                    "open": 1.0,
                    "high": 1.2,
                    "low": 0.9,
                    "close": 1.1,
                    "tick_volume": 10,
                },
                {
                    "time": "2026-05-14 12:00",
                    "open": 1.1,
                    "high": 1.3,
                    "low": 1.0,
                    "close": 1.2,
                    "tick_volume": 14,
                },
            ],
            "meta": {
                "diagnostics": {
                    "query": {"mode": "latest", "latency_ms": 12.3},
                    "freshness": {
                        "data_freshness_seconds": 60.0,
                        "last_bar_within_policy_window": True,
                    },
                },
            },
        },
    )

    assert result["output"] == "summary"
    assert result["symbol"] == "EURUSD"
    assert result["timeframe"] == "H1"
    assert result["count"] == 4
    assert "candles" not in result
    assert result["candles_requested"] == 5
    assert result["candles_excluded"] == 1
    assert result["candle_counts"]["excluded"] == {"forming_bar": 1, "total": 1}
    assert result["timezone"] == "UTC"
    assert result["query_type"] == "latest"
    assert result["latency_ms"] == 12.3
    assert result["data_age_seconds"] == 60.0
    assert result["data_age_anchor"] == "wall_clock"
    assert result["data_age_metric"] == "last_completed_bar_age_seconds"
    assert "data_freshness_seconds" not in result
    assert result["data_age"] == "1m 0s"
    assert result["data_stale"] is False
    assert "data" not in result
    assert result["latest_candle"] == {
        "time": "2026-05-14 12:00",
        "open": 1.1,
        "high": 1.3,
        "low": 1.0,
        "close": 1.2,
        "tick_volume": 14,
    }
    assert result["timestamp_format"] == "iso_utc"
    assert result["summary_statistics"]["close"] == {
        "min": 1.1,
        "max": 1.2,
        "mean": 1.15,
        "change": 0.1,
        "change_pct": pytest.approx(9.090909),
    }
    assert result["summary_statistics"]["range"]["mean"] == pytest.approx(0.3)
    assert result["summary_statistics"]["tick_volume"]["sum"] == 24.0
    assert "meta" not in result


def test_run_data_fetch_candles_compact_drops_redundant_session_gap_warnings():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=5)
    session_gap = {
        "from": "2026-05-01 20:00",
        "to": "2026-05-03 21:00",
        "gap_seconds": 176400.0,
        "expected_bar_seconds": 3600.0,
        "missing_bars_est": 48,
        "context": "weekend/session break",
    }

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "data": [],
            "session_gaps": [session_gap],
            "warnings": [
                "Detected session gaps larger than expected bar spacing (3600s).",
                "Example gap: 2026-05-01 20:00 -> 2026-05-03 21:00 (48 missing bars, likely weekend/session break).",
                "Other warning",
            ],
        },
    )

    assert result["session_gaps"] == [session_gap]
    assert result["warnings"] == ["Other warning"]


def test_run_data_fetch_candles_standard_keeps_session_gap_warnings():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        detail="standard",
    )
    warnings = [
        "Detected session gaps larger than expected bar spacing (3600s).",
        "Example gap: 2026-05-01 20:00 -> 2026-05-03 21:00 (48 missing bars, likely weekend/session break).",
    ]

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "data": [],
            "session_gaps": [{"missing_bars_est": 48}],
            "warnings": list(warnings),
        },
    )

    assert result["warnings"] == warnings


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
            "incomplete_candles_skipped": 1,
            "has_forming_candle": True,
            "forming_candle_status": "skipped",
            "forming_candle_included": False,
            "forming_candle_skipped": True,
            "data": [],
        },
    )

    assert result["count"] == 4
    assert "candles" not in result
    assert "candle_counts" not in result
    assert "last_candle_open" not in result
    assert "hint" not in result
    assert "candles_excluded" not in result
    assert "incomplete_candles_skipped" not in result
    assert result["forming_candle_status"] == "skipped"
    assert "has_forming_candle" not in result
    assert "forming_candle_included" not in result
    assert "forming_candle_skipped" not in result
    assert result["symbol"] == "EURUSD"
    assert result["timeframe"] == "H1"
    assert "candles_requested" not in result


def test_run_data_fetch_candles_compact_preserves_requested_rows():
    rows = [
        {"time": 1_700_000_000 + index * 60, "close": float(index)}
        for index in range(125)
    ]
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="M1", limit=125)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "M1",
            "candles": len(rows),
            "data": list(rows),
        },
    )

    assert result["count"] == 125
    assert len(result["data"]) == 125
    assert result["data"][0]["close"] == 0.0
    assert "data_truncated" not in result
    assert result["timestamp_format"] == "epoch_seconds"
    assert "timestamp_format=iso" in result["timestamp_format_hint"]


def test_run_data_fetch_candles_standard_keeps_forming_booleans():
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
            "candles": 4,
            "has_forming_candle": True,
            "forming_candle_status": "skipped",
            "forming_candle_included": False,
            "forming_candle_skipped": True,
            "data": [],
        },
    )

    assert result["has_forming_candle"] is True
    assert result["forming_candle_status"] == "skipped"
    assert result["forming_candle_included"] is False
    assert result["forming_candle_skipped"] is True


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
            "has_forming_candle": False,
            "data": rows,
        },
    )

    assert result["data"] == rows
    assert result["symbol"] == "EURUSD"
    assert result["timeframe"] == "H1"
    assert result["candles_requested"] == 10
    assert "last_candle_open" not in result
    assert result["has_forming_candle"] is False
    assert "series" not in result
    assert result["collection_kind"] == "time_series"
    assert result["collection_contract_version"] == "collection.v1"
    assert "canonical_source" not in result


def test_run_data_fetch_candles_full_keeps_forming_metadata_without_row_flag():
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
            "has_forming_candle": True,
            "forming_candle_status": "included",
            "forming_candle_included": True,
            "forming_candle_skipped": False,
            "ohlcv_filter_applied": True,
            "data": [{"time": 1.0, "close": 1.1}],
        },
    )

    assert result["has_forming_candle"] is True
    assert result["forming_candle_status"] == "included"
    assert result["forming_candle_included"] is True
    assert "is_forming" not in result["data"][0]


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


@pytest.mark.parametrize(
    ("detail", "expected_format"),
    [
        ("compact", "rows"),
        ("summary", "summary"),
        ("standard", "stats"),
        ("full", "full_rows"),
    ],
)
def test_run_data_fetch_ticks_maps_standard_detail_to_service_format(detail, expected_format):
    captured = {}

    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=5, detail=detail),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **kwargs: captured.update(kwargs) or {"success": True},
    )

    assert result["success"] is True
    assert captured["format"] == expected_format


def test_run_data_fetch_ticks_echoes_limit_and_truncation_signal():
    capped = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=2, detail="standard"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {"success": True, "count": 2, "data": []},
    )
    assert capped["requested_limit"] == 2
    assert capped["has_more"] is True

    partial = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=20, detail="standard"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {"success": True, "count": 5, "data": []},
    )
    assert partial["requested_limit"] == 20
    assert partial["has_more"] is False


def test_run_data_fetch_ticks_compact_prunes_row_diagnostics():
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=2, detail="compact"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "count": 2,
            "data": [
                {
                    "time": "2026-05-29 20:56",
                    "bid": 1.1659,
                    "ask": 1.16596,
                    "tick_volume": 3.0,
                    "real_volume": 1.25,
                    "flags": 1026,
                    "flags_decoded": ["bid", "volume_real"],
                },
                {
                    "time": "2026-05-29 20:57",
                    "bid": 1.16591,
                    "ask": 1.16599,
                    "tick_volume": 4.0,
                    "real_volume": 0.0,
                    "flags": 1030,
                    "flags_decoded": ["bid", "ask", "volume_real"],
                },
            ],
            "timezone": "UTC",
            "freshness": "stale, tick 10m 0s ago",
            "data_age_seconds": 600.0,
            "data_stale": True,
            "price_point": 0.00001,
            "units": {
                "tick_volume": "broker_tick_count",
                "real_volume": "traded_volume",
            },
            "stats": {"spread": {"low": 0.00006, "high": 0.00008}},
            "last_quote": {"bid": 1.16591, "ask": 1.16599},
            "flags_legend": {"1026": ["bid", "volume_real"]},
            "duration_seconds": 1,
            "tick_rate_per_second": 2,
            "price_precision": 5,
            "data_quality": {
                "one_sided_zero_spread_ticks": 1,
                "complete_ticks": 1,
                "incomplete_ticks": 1,
                "total_ticks": 2,
                "one_sided_zero_spread_ratio": 0.5,
                "spread_ticks_excluded": 1,
                "warning_ratio": 0.5,
                "quote_type_counts": {"bid_ask": 1, "bid_only": 1},
                "one_sided_zero_spread_status": "warning",
            },
            "last_unavailable": True,
            "warnings": [
                "Some ticks had identical bid/ask with one-sided quote flags.",
                "Broker tick data did not provide a usable last price; last is null.",
            ],
        },
    )

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "count": 2,
        "data": [
            {
                "time": "2026-05-29 20:56",
                "bid": 1.1659,
                "ask": 1.16596,
                "spread": 0.00006,
                "mid": 1.16593,
                "spread_points": 6.0,
                "spread_pct": 0.005146,
                "tick_volume": 3.0,
                "real_volume": 1.25,
            },
            {
                "time": "2026-05-29 20:57",
                "bid": 1.16591,
                "ask": 1.16599,
                "spread": 0.00008,
                "mid": 1.16595,
                "spread_points": 8.0,
                "spread_pct": 0.006861,
                "tick_volume": 4.0,
            },
        ],
        "timezone": "UTC",
        "price_precision": 5,
        "price_point": 0.00001,
        "freshness": "stale, tick 10m 0s ago",
        "data_age_seconds": 600.0,
        "data_age_anchor": "wall_clock",
        "data_age_metric": "last_tick_age_seconds",
        "data_stale": True,
        "units": {
            "bid": "absolute_price",
            "ask": "absolute_price",
            "spread": "absolute_price",
            "mid": "absolute_price",
            "spread_points": "broker_points",
            "spread_pct": "percentage_points (1.0 = 1%)",
            "tick_volume": "broker_tick_count",
            "real_volume": "traded_volume",
        },
        "volume_fields": ["tick_volume", "real_volume"],
        "quote_completeness_pct": 50.0,
        "quality": "partial_quotes=1/2; last=unavailable",
        "requested_limit": 2,
        "has_more": True,
    }


def test_run_data_fetch_ticks_compact_summarizes_quality_without_verbose_warnings():
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=5, detail="compact"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "count": 5,
            "data": [
                {"time": "t1", "bid": 1.1, "ask": None, "quote_type": "bid_only"},
                {"time": "t2", "bid": None, "ask": 1.1001, "quote_type": "ask_only"},
                {"time": "t3", "bid": 1.1, "ask": 1.1001},
                {"time": "t4", "bid": 1.10001, "ask": 1.10011},
                {"time": "t5", "bid": 1.10002, "ask": None, "quote_type": "bid_only"},
            ],
            "data_quality": {
                "one_sided_zero_spread_ticks": 3,
                "complete_ticks": 2,
                "incomplete_ticks": 3,
                "total_ticks": 5,
                "one_sided_zero_spread_ratio": 0.6,
                "spread_ticks_excluded": 3,
                "warning_ratio": 0.5,
                "quote_type_counts": {"ask_only": 1, "bid_ask": 2, "bid_only": 2},
                "one_sided_zero_spread_status": "warning",
            },
            "last_unavailable": True,
            "warnings": [
                "Some ticks had identical bid/ask with one-sided quote flags.",
                "Broker tick data did not provide a usable last price; last is null.",
            ],
        },
    )

    assert result["quality"] == "partial_quotes=3/5; last=unavailable"
    assert result["quote_completeness_pct"] == 40.0
    assert result["data"][3]["mid"] == 1.10006
    assert result["data"][4]["mid"] == 1.10007
    assert result["data"][4]["mid_inferred"] is True
    assert "data_quality" not in result
    assert "warnings" not in result


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
    assert request.limit == 20


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

    raw = core_data.data_fetch_candles(
        request=DataFetchCandlesRequest(symbol="EURUSD", detail="compact"),
        __cli_raw=True,
    )
    compact = core_data.data_fetch_candles(
        request=DataFetchCandlesRequest(symbol="EURUSD", detail="compact"),
        json=True,
    )
    full = core_data.data_fetch_candles(
        request=DataFetchCandlesRequest(symbol="EURUSD", detail="full"),
        json=True,
    )

    assert raw["meta"]["diagnostics"]["query"]["requested_bars"] == 20
    assert "meta" not in compact
    assert full["meta"]["tool"] == "data_fetch_candles"
    assert full["meta"]["diagnostics"]["query"]["requested_bars"] == 20


def test_data_fetch_ticks_request_rejects_removed_output_field():
    with pytest.raises(ValidationError, match="output was removed; use json"):
        DataFetchTicksRequest(symbol="EURUSD", output="rows")


def test_data_fetch_ticks_request_uses_detail_control():
    request = DataFetchTicksRequest(symbol="EURUSD", detail="full")

    assert request.detail == "full"
    assert list(DataFetchTicksRequest.model_fields) == [
        "symbol",
        "limit",
        "start",
        "end",
        "timestamp_format",
        "simplify",
        "detail",
    ]
    assert request.timestamp_format == "iso"


def test_data_fetch_ticks_request_rejects_removed_output_mode_field():
    with pytest.raises(ValidationError, match="output_mode was removed; use extras"):
        DataFetchTicksRequest(symbol="EURUSD", output_mode="rows")


def test_data_fetch_ticks_request_defaults_to_compact_detail():
    request = DataFetchTicksRequest(symbol="EURUSD")

    assert request.detail == "compact"


def test_data_fetch_ticks_request_rejects_excessive_limit():
    with pytest.raises(ValidationError, match="limit must be at most 10000"):
        DataFetchTicksRequest(symbol="EURUSD", limit=10_001)


@pytest.mark.parametrize("raw_detail", ["stats", "rows"])
def test_data_fetch_ticks_request_rejects_legacy_detail_values(raw_detail: str):
    with pytest.raises(ValidationError):
        DataFetchTicksRequest(symbol="EURUSD", detail=raw_detail)
