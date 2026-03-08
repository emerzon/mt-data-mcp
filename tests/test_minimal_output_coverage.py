"""Tests for utils/minimal_output.py — TOON formatting helpers."""
import pytest

from mtdata.utils.minimal_output import (
    _is_scalar_value,
    _is_empty_value,
    _stringify_scalar,
    _stringify_cell,
    _compact_forecast_ci,
    _normalize_forecast_payload,
    format_table_toon,
    _encode_inline_array,
    _encode_expanded_array,
    _format_to_toon,
    _format_complex_value,
)


class TestIsScalarValue:
    def test_string(self):
        assert _is_scalar_value("hello")

    def test_int(self):
        assert _is_scalar_value(42)

    def test_float(self):
        assert _is_scalar_value(3.14)

    def test_bool(self):
        assert _is_scalar_value(True)

    def test_none(self):
        assert _is_scalar_value(None)

    def test_list_not_scalar(self):
        assert not _is_scalar_value([1, 2])

    def test_dict_not_scalar(self):
        assert not _is_scalar_value({"a": 1})


class TestIsEmptyValue:
    def test_none(self):
        assert _is_empty_value(None)

    def test_empty_string(self):
        assert _is_empty_value("")
        assert _is_empty_value("  ")

    def test_nonempty_string(self):
        assert not _is_empty_value("hello")

    def test_empty_list(self):
        assert _is_empty_value([])
        assert _is_empty_value([None, None])

    def test_nonempty_list(self):
        assert not _is_empty_value([1])

    def test_empty_dict(self):
        assert _is_empty_value({})
        assert _is_empty_value({"a": None})

    def test_nonempty_dict(self):
        assert not _is_empty_value({"a": 1})

    def test_number_not_empty(self):
        assert not _is_empty_value(0)
        assert not _is_empty_value(0.0)


class TestStringifyScalar:
    def test_none(self):
        assert _stringify_scalar(None) == "null"

    def test_int(self):
        result = _stringify_scalar(42)
        assert "42" in result

    def test_float(self):
        result = _stringify_scalar(3.14)
        assert "3.14" in result

    def test_string(self):
        assert _stringify_scalar("hello") == "hello"


class TestStringifyCell:
    def test_scalar(self):
        assert _stringify_cell("hello") == "hello"

    def test_list_of_scalars(self):
        result = _stringify_cell([1, 2, 3])
        assert "|" in result or "," in result

    def test_dict(self):
        result = _stringify_cell({"a": 1, "b": 2})
        assert "a=" in result
        assert "b=" in result

    def test_empty_list(self):
        assert _stringify_cell([None]) == ""

    def test_nested_list(self):
        result = _stringify_cell([[1, 2], [3, 4]])
        assert len(result) > 0


class TestNormalizeForecastPayload:
    def test_basic_forecast(self):
        payload = {
            "times": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "forecast_price": [100.0, 101.0, 102.0],
            "symbol": "EURUSD",
            "method": "arima",
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert "forecast" in result
        assert len(result["forecast"]) == 3
        assert result["forecast"][0]["time"] == "2024-01-01"
        assert result["forecast"][0]["forecast"] == 100.0

    def test_with_ci_bounds(self):
        payload = {
            "times": ["t1", "t2"],
            "forecast_price": [100.0, 101.0],
            "lower_price": [98.0, 99.0],
            "upper_price": [102.0, 103.0],
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert "lower" in result["forecast"][0]
        assert "upper" in result["forecast"][0]

    def test_with_digits(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [1.23456],
            "digits": 2,
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert result["forecast"][0]["forecast"] == "1.23"

    def test_with_quantiles(self):
        payload = {
            "times": ["t1", "t2"],
            "forecast_price": [100.0, 101.0],
            "forecast_quantiles": {
                "0.1": [95.0, 96.0],
                "0.9": [105.0, 106.0],
            },
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert "q0.1" in result["forecast"][0]
        assert "q0.9" in result["forecast"][0]

    def test_no_times_returns_none(self):
        assert _normalize_forecast_payload({"forecast_price": [1.0]}) is None

    def test_no_forecast_returns_none(self):
        assert _normalize_forecast_payload({"times": ["t1"]}) is None

    def test_forecast_return_key(self):
        payload = {
            "times": ["t1"],
            "forecast_return": [0.01],
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None

    def test_verbose_meta(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "arima",
            "horizon": 5,
            "timezone": "UTC",
        }
        result = _normalize_forecast_payload(payload, verbose=True)
        assert "meta" in result
        assert result["meta"]["symbol"] == "EURUSD"
        assert result["meta"]["timezone"] == "UTC"

    def test_non_verbose_no_meta(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "symbol": "EURUSD",
        }
        result = _normalize_forecast_payload(payload, verbose=False)
        assert "meta" not in result

    def test_q50_dedup(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "forecast_quantiles": {
                "0.1": [95.0],
                "0.5": [100.0],
                "0.9": [105.0],
            },
        }
        result = _normalize_forecast_payload(payload)
        headers_present = set(result["forecast"][0].keys())
        # q0.5 should be deduplicated since it matches forecast_price
        assert "q0.5" not in headers_present

    def test_forecast_epoch_fallback(self):
        payload = {
            "forecast_epoch": [1000, 2000],
            "forecast_price": [100.0, 101.0],
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert result["forecast"][0]["time"] == 1000

    def test_denoise_in_meta(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "denoise_used": {"method": "wavelet"},
        }
        result = _normalize_forecast_payload(payload, verbose=True)
        assert result["meta"]["denoise"] == "wavelet"

    def test_denoise_applied_flag(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "denoise_applied": True,
        }
        result = _normalize_forecast_payload(payload, verbose=True)
        assert result["meta"]["denoise"] == "applied"

    def test_ci_warnings_preserved_in_non_verbose_output(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "ci_requested": True,
            "ci_alpha_requested": 0.05,
            "ci_available": False,
            "ci_status": "unavailable",
            "warnings": [
                "Point forecast only for method 'theta'; confidence intervals are unavailable."
            ],
        }
        result = _normalize_forecast_payload(payload, verbose=False)
        assert "meta" not in result
        assert result["ci"] == {"status": "unavailable", "alpha": 0.05}
        assert result["warnings"][0].startswith("Point forecast only")

    def test_ci_diag_omitted_when_bounds_already_rendered(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "lower_price": [98.0],
            "upper_price": [102.0],
            "ci_requested": True,
            "ci_alpha_requested": 0.05,
            "ci_available": True,
            "ci_status": "available",
            "ci_alpha": 0.05,
        }
        result = _normalize_forecast_payload(payload, verbose=False)
        assert "ci" not in result


class TestCompactForecastCi:
    def test_omits_available_ci_when_bounds_exist(self):
        payload = {
            "ci_requested": True,
            "ci_available": True,
            "ci_status": "available",
            "ci_alpha_requested": 0.05,
            "ci_alpha": 0.05,
        }
        assert _compact_forecast_ci(payload, lower=[1.0], upper=[2.0]) == {}

    def test_compacts_unavailable_ci_to_status_and_alpha(self):
        payload = {
            "ci_requested": True,
            "ci_available": False,
            "ci_status": "unavailable",
            "ci_alpha_requested": 0.1,
            "ci_unavailable": True,
        }
        assert _compact_forecast_ci(payload, lower=[], upper=[]) == {
            "status": "unavailable",
            "alpha": 0.1,
        }


class TestFormatTableToon:
    def test_basic(self):
        lines = format_table_toon(["name", "value"], [["a", 1], ["b", 2]])
        assert len(lines) > 0

    def test_empty(self):
        assert format_table_toon([], []) == []


class TestEncodeInlineArray:
    def test_basic(self):
        result = _encode_inline_array("prices", [1.0, 2.0, 3.0])
        assert "prices" in result
        assert "3" in result  # length indicator


class TestEncodeExpandedArray:
    def test_basic(self):
        result = _encode_expanded_array("items", [{"a": 1}, {"a": 2}])
        assert "items" in result
        assert "2" in result  # length indicator


class TestFormatToToon:
    def test_scalar(self):
        result = _format_to_toon(42, key="count")
        assert "42" in result

    def test_string(self):
        result = _format_to_toon("hello", key="msg")
        assert "hello" in result

    def test_dict(self):
        result = _format_to_toon({"a": 1, "b": 2}, key="data")
        assert "a" in result

    def test_list(self):
        result = _format_to_toon([1, 2, 3], key="items")
        assert len(result) > 0

    def test_none(self):
        result = _format_to_toon(None, key="empty")
        assert result == "" or "null" in result.lower() or result is None or not result


class TestFormatComplexValue:
    def test_dict(self):
        result = _format_complex_value({"key": "val"})
        assert "key" in result

    def test_list(self):
        result = _format_complex_value([1, 2, 3])
        assert len(result) > 0

    def test_scalar(self):
        result = _format_complex_value(42)
        assert "42" in result
