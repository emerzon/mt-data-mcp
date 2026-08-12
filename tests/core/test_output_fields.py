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


def test_output_fields_ignores_partially_unresolved_projection() -> None:
    payload = {"success": True, "symbol": "EURUSD", "details": {"digits": 5}}

    result = _select_output_fields(payload, "symbol,details.missing")

    assert result == {"success": True, "symbol": "EURUSD"}


def test_output_fields_allows_error_field_on_success() -> None:
    payload = {"success": True, "symbol": "EURUSD", "data": [1, 2]}

    result = _select_output_fields(payload, "success,data,error")

    assert result == {"success": True, "symbol": "EURUSD", "data": [1, 2]}


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


def test_output_fields_still_projects_bare_fields_from_row_collections() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "data": [{"time": 1, "close": 1.1}, {"time": 2, "close": 1.2}],
    }

    result = _select_output_fields(payload, "close")

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

    result = _select_output_fields(payload, "name")

    assert result == {
        "success": True,
        "tools": [{"name": "forecast_generate"}],
        "pagination": {"offset": 0, "limit": 1, "returned": 1, "total": 8},
    }
