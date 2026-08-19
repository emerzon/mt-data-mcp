from mtdata.core._mcp_tools import _select_output_fields


def test_output_fields_supports_dotted_nested_paths() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "details": {"time": "2026-07-14T15:00Z", "digits": 5, "trade_mode": "full"},
    }

    result = _select_output_fields(payload, "details.time,details.digits")

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "details": {"time": "2026-07-14T15:00Z", "digits": 5},
    }


def test_output_fields_total_miss_preserves_tool_success() -> None:
    payload = {"success": True, "symbol": "EURUSD", "details": {"digits": 5}}

    result = _select_output_fields(payload, "symbol,details.missing")

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "unresolved_output_fields": ["details.missing"],
        "valid_output_fields": ["details"],
    }


def test_output_fields_resolves_canonical_forecast_arrays_from_compact_rows() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "quantity": "price",
        "forecast": [
            {"time": "2026-08-18T02:00Z", "value": 1.15782},
            {"time": "2026-08-18T03:00Z", "value": 1.15801},
        ],
    }

    result = _select_output_fields(payload, "forecast_time,forecast_price")

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "forecast_time": ["2026-08-18T02:00Z", "2026-08-18T03:00Z"],
        "forecast_price": [1.15782, 1.15801],
    }


def test_output_fields_allows_error_field_on_success() -> None:
    payload = {"success": True, "symbol": "EURUSD", "data": [1, 2]}

    result = _select_output_fields(payload, "success,data,error")

    assert result == {"success": True, "symbol": "EURUSD", "data": [1, 2]}


def test_output_fields_keeps_requested_query_context_on_error() -> None:
    payload = {
        "success": False,
        "error": "No data available",
        "error_code": "data_fetch_candles_no_data",
        "query_applied": {
            "resolved_start": "2026-08-12T00:00:00Z",
            "start_bound": "inclusive_day_start",
        },
    }

    result = _select_output_fields(payload, "success,query_applied")

    assert result == {
        "success": False,
        "error": "No data available",
        "error_code": "data_fetch_candles_no_data",
        "query_applied": payload["query_applied"],
    }


def test_output_fields_preserves_complete_error_recovery_envelope() -> None:
    payload = {
        "success": False,
        "error": "Unsupported date range",
        "error_code": "unsupported_date_range",
        "request_id": "abc123",
        "operation": "data_fetch_candles",
        "remediation": "Use a date on or after 1970-01-01.",
        "related_tools": ["data_fetch_candles"],
        "valid_values": {"end": ">= 1970-01-01"},
        "example": "--end 2024-01-01",
        "documentation": "docs/CLI.md",
        "details": {"end": "1960-01-01"},
        "data": [],
    }

    result = _select_output_fields(payload, "success,data")

    assert result == payload


def test_output_fields_does_not_inject_units_for_selected_values() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "bid": 1.1,
        "ask": 1.2,
        "units": {"bid": "price", "ask": "price"},
    }

    result = _select_output_fields(payload, "bid,ask")

    assert result == {"success": True, "symbol": "EURUSD", "bid": 1.1, "ask": 1.2}


def test_output_fields_prefers_top_level_quote_values_over_nested_diagnostics() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "bid": 1.1,
        "ask": 1.1002,
        "quote_source_conflict": {
            "symbol_info_tick": {"bid": 1.0999, "ask": 1.1001},
            "stream_tick": {"bid": 1.1, "ask": 1.1002},
        },
    }

    result = _select_output_fields(payload, "bid,ask")

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "bid": 1.1,
        "ask": 1.1002,
    }


def test_output_fields_projects_bare_fields_from_row_collections() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "row_key": "data",
        "count": 2,
        "data": [
            {"time": 1, "close": 1.1, "open": 1.0},
            {"time": 2, "close": 1.2, "open": 1.1},
        ],
    }

    result = _select_output_fields(payload, "time,close")

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "count": 2,
        "data": [{"time": 1, "close": 1.1}, {"time": 2, "close": 1.2}],
    }


def test_output_fields_uses_dotted_paths_for_row_collections() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "data": [{"time": 1, "close": 1.1}, {"time": 2, "close": 1.2}],
    }

    result = _select_output_fields(payload, "data.close")

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "data": [{"close": 1.1}, {"close": 1.2}],
    }


def test_output_fields_preserves_pagination_metadata() -> None:
    payload = {
        "success": True,
        "tools": [{"name": "forecast_generate", "description": "Forecast"}],
        "pagination": {"offset": 0, "limit": 1, "returned": 1, "total": 8},
    }

    result = _select_output_fields(payload, "tools.name")

    assert result == {
        "success": True,
        "tools": [{"name": "forecast_generate"}],
        "pagination": {"offset": 0, "limit": 1, "returned": 1, "total": 8},
    }


def test_output_fields_preserves_history_truncation_warnings() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "data": [{"time": 1, "bid": 1.1}],
        "history_window_truncated": True,
        "history_window_limit_days": 30,
        "history_window_floor": "2026-07-16T00:00Z",
        "effective_start": "2026-07-16T00:00Z",
        "warnings": ["Requested start was outside the tick-history budget."],
    }

    result = _select_output_fields(payload, "data")

    assert result == payload
