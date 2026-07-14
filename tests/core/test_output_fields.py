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


def test_output_fields_rejects_partially_unresolved_projection() -> None:
    payload = {"success": True, "symbol": "EURUSD", "details": {"digits": 5}}

    result = _select_output_fields(payload, "symbol,details.missing")

    assert result["success"] is False
    assert result["error_code"] == "invalid_output_fields"
    assert result["unresolved_fields"] == ["details.missing"]


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
