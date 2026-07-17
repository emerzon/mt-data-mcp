from mtdata.core.error_envelope import build_error_payload, normalize_error_payload


def test_build_error_payload_adds_common_remediation():
    out = build_error_payload(
        "MT5 connection failed",
        code="mt5_connection_error",
        operation="data_fetch_candles",
        request_id="req123",
    )

    assert out["request_id"] == "req123"
    assert "MetaTrader 5 is running" in out["remediation"]
    assert out["related_tools"] == ["symbols_list"]


def test_build_error_payload_keeps_explicit_guidance():
    out = build_error_payload(
        "No such method",
        code="forecast_generate_error",
        operation="forecast_generate",
        request_id="req123",
        remediation="Choose theta.",
        related_tools=["forecast_list_methods"],
        valid_values={"method": ["theta"]},
        example="mtdata-cli forecast_generate EURUSD --method theta",
    )

    assert out["remediation"] == "Choose theta."
    assert out["valid_values"] == {"method": ["theta"]}
    assert out["example"].endswith("--method theta")


def test_normalize_error_payload_adds_symbol_lookup_guidance():
    out = normalize_error_payload(
        {
            "error": "Symbol not found",
            "error_code": "symbol_not_found",
            "request_id": "req123",
        },
        operation="symbols_describe",
    )

    assert out["remediation"].startswith("Use symbols_list")
    assert out["related_tools"] == ["symbols_list"]


def test_normalize_error_payload_classifies_symbol_root_cause_from_warnings():
    out = normalize_error_payload(
        {
            "error": "Not enough valid symbol data fetched.",
            "error_code": "insufficient_symbols",
            "warnings": [
                "Symbol NOTAREALSYM was not found in MT5.",
                "Symbol NOTAREALSYM was not found in MT5.",
            ],
            "remediation": "Increase the lookback.",
        },
        operation="correlation_matrix",
    )

    assert out["error_code"] == "symbol_not_found"
    assert out["warnings"] == ["Symbol NOTAREALSYM was not found in MT5."]
    assert out["remediation"].startswith("Use symbols_list")
    assert out["related_tools"] == ["symbols_list"]


def test_normalize_error_payload_canonicalizes_date_ranges_and_details():
    out = normalize_error_payload(
        {
            "error": "Error detecting regimes: start_datetime must be before end_datetime",
            "error_code": "tool_error",
            "details": [
                "start_datetime must be before end_datetime",
                "start_datetime must be before end_datetime",
            ],
            "remediation": "Run forecast_list_methods.",
        },
        operation="regime_detect",
    )

    assert out["error_code"] == "invalid_date_range"
    assert out["error"] == "start must be before or equal to end."
    assert out["details"] == ["start_datetime must be before end_datetime"]
    assert out["remediation"] == "Set start to a timestamp earlier than or equal to end."
