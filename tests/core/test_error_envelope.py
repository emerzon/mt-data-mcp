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
