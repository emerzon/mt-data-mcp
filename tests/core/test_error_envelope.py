from mtdata.core.error_envelope import build_error_payload, normalize_error_payload
from mtdata.core.request_context import (
    current_request_id,
    ensure_request_id_scope,
    request_id_scope,
)


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


def test_build_error_payload_uses_bound_request_id():
    with request_id_scope("bound-request-7"):
        out = build_error_payload("broken", code="test_error")

    assert out["request_id"] == "bound-request-7"


def test_ensure_request_id_scope_generates_and_cleans_up_identifier():
    assert current_request_id() is None

    with ensure_request_id_scope() as request_id:
        assert len(request_id) == 12
        assert current_request_id() == request_id

    assert current_request_id() is None


def test_ensure_request_id_scope_preserves_transport_identifier():
    with request_id_scope("transport-request-9"):
        with ensure_request_id_scope() as request_id:
            assert request_id == "transport-request-9"
            assert current_request_id() == request_id


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


def test_forecast_train_errors_point_to_trainable_method_discovery():
    out = build_error_payload(
        "Method 'ets' does not support separate training.",
        code="tool_error",
        operation="forecast_train",
    )

    assert out["remediation"] == (
        "Choose a trainable method with forecast_list_methods "
        "--supports-training true, then retry forecast_train."
    )
    assert out["related_tools"] == ["forecast_list_methods"]


def test_generic_method_errors_use_the_failing_operation_help():
    out = build_error_payload(
        "Invalid method. Valid options: pearson, spearman",
        code="invalid_method",
        operation="correlation_matrix",
    )

    assert out["remediation"] == (
        "Use this operation's --help and choose one of the listed method values."
    )
    assert "related_tools" not in out


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


def test_normalize_error_payload_preserves_specific_code_from_warnings():
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

    assert out["error_code"] == "insufficient_symbols"
    assert out["warnings"] == ["Symbol NOTAREALSYM was not found in MT5."]
    assert out["remediation"] == "Increase the lookback."
    assert "related_tools" not in out


def test_normalize_error_payload_classifies_generic_code_from_warnings():
    out = normalize_error_payload(
        {
            "error": "Not enough valid symbol data fetched.",
            "error_code": "tool_error",
            "warnings": ["Symbol NOTAREALSYM was not found in MT5."],
        },
        operation="correlation_matrix",
    )

    assert out["error_code"] == "symbol_not_found"
    assert out["remediation"].startswith("Use symbols_list")
    assert out["related_tools"] == ["symbols_list"]


def test_normalize_error_payload_does_not_override_dependency_code():
    out = normalize_error_payload(
        build_error_payload(
            "finviz: symbol XYZ not found in screener",
            code="dependency_missing",
            operation="news_fetch",
            details={"provider": "finviz"},
        )
    )

    assert out["error_code"] == "dependency_missing"
    assert "optional dependency group" in out["remediation"]


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
