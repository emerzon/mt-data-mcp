"""Tests for src/mtdata/utils/minimal_output.py — TOON formatting helpers."""
import math
from mtdata.utils.minimal_output import (
    _is_scalar_value,
    _is_empty_value,
    _minify_number,
    _stringify_scalar,
    _stringify_cell,
    _indent_text,
    _quote_if_needed,
    _quote_key,
    _format_complex_value,
    _headers_from_dicts,
    _column_decimals,
    _encode_tabular,
    _stringify_for_toon,
    _stringify_for_toon_value,
)


class TestIsScalarValue:
    def test_string(self):
        assert _is_scalar_value("hello") is True

    def test_int(self):
        assert _is_scalar_value(42) is True

    def test_float(self):
        assert _is_scalar_value(3.14) is True

    def test_bool(self):
        assert _is_scalar_value(True) is True

    def test_none(self):
        assert _is_scalar_value(None) is True

    def test_list(self):
        assert _is_scalar_value([1, 2]) is False

    def test_dict(self):
        assert _is_scalar_value({"a": 1}) is False


class TestIsEmptyValue:
    def test_none(self):
        assert _is_empty_value(None) is True

    def test_empty_string(self):
        assert _is_empty_value("") is True

    def test_whitespace_string(self):
        assert _is_empty_value("   ") is True

    def test_non_empty_string(self):
        assert _is_empty_value("hello") is False

    def test_empty_list(self):
        assert _is_empty_value([]) is True

    def test_list_of_nones(self):
        assert _is_empty_value([None, None]) is True

    def test_nonempty_list(self):
        assert _is_empty_value([1]) is False

    def test_empty_dict(self):
        assert _is_empty_value({}) is True

    def test_dict_with_empty_values(self):
        assert _is_empty_value({"a": None, "b": ""}) is True

    def test_dict_with_values(self):
        assert _is_empty_value({"a": 1}) is False

    def test_number(self):
        assert _is_empty_value(0) is False


class TestStringifyScalar:
    def test_none(self):
        assert _stringify_scalar(None) == "null"

    def test_bool_true(self):
        assert _stringify_scalar(True) == "true"

    def test_bool_false(self):
        assert _stringify_scalar(False) == "false"

    def test_int(self):
        assert _stringify_scalar(42) == "42"

    def test_float(self):
        result = _stringify_scalar(1.5)
        assert "1.5" in result

    def test_string(self):
        assert _stringify_scalar("hello") == "hello"


class TestStringifyCell:
    def test_scalar(self):
        assert _stringify_cell(42) == "42"

    def test_empty_list(self):
        assert _stringify_cell([None, None]) == ""

    def test_scalar_list(self):
        result = _stringify_cell([1, 2, 3])
        assert "|" in result

    def test_dict(self):
        result = _stringify_cell({"a": 1, "b": 2})
        assert "a=" in result
        assert "b=" in result

    def test_nested_dict_empty_values_skipped(self):
        result = _stringify_cell({"a": None, "b": 1})
        assert "a=" not in result
        assert "b=" in result


class TestIndentText:
    def test_single_line(self):
        assert _indent_text("hello") == "  hello"

    def test_multi_line(self):
        result = _indent_text("a\nb")
        lines = result.split("\n")
        assert lines[0] == "  a"
        assert lines[1] == "  b"

    def test_custom_indent(self):
        assert _indent_text("x", indent=">>") == ">>x"


class TestQuoteIfNeeded:
    def test_simple_string(self):
        assert _quote_if_needed("hello") == "hello"

    def test_string_with_comma(self):
        result = _quote_if_needed("a,b")
        assert result.startswith('"')
        assert result.endswith('"')

    def test_string_with_leading_space(self):
        result = _quote_if_needed(" hello")
        assert result.startswith('"')

    def test_string_with_colon(self):
        result = _quote_if_needed("a:b")
        assert result.startswith('"')

    def test_none_returns_empty(self):
        assert _quote_if_needed(None) == ""


class TestQuoteKey:
    def test_simple_key(self):
        assert _quote_key("name") == "name"

    def test_none_key(self):
        assert _quote_key(None) == ""

    def test_key_with_special_chars(self):
        result = _quote_key("a,b")
        assert '"' in result


class TestHeadersFromDicts:
    def test_basic(self):
        items = [{"a": 1, "b": 2}, {"a": 3, "c": 4}]
        headers = _headers_from_dicts(items)
        assert headers == ["a", "b", "c"]

    def test_empty_list(self):
        assert _headers_from_dicts([]) == []

    def test_non_dict_returns_empty(self):
        assert _headers_from_dicts([1, 2, 3]) == []


class TestColumnDecimals:
    def test_int_columns(self):
        result = _column_decimals(["x"], [{"x": 1}, {"x": 2}])
        assert result["x"] == 0

    def test_float_columns(self):
        result = _column_decimals(["x"], [{"x": 1.001}, {"x": 1.002}])
        assert result["x"] >= 2

    def test_none_values_skipped(self):
        result = _column_decimals(["x"], [{"x": None}, {"x": 1.0}])
        assert "x" in result

    def test_bool_values_skipped(self):
        result = _column_decimals(["x"], [{"x": True}, {"x": 1.0}])
        assert "x" in result


class TestEncodeTabular:
    def test_basic_table(self):
        result = _encode_tabular(
            "data",
            ["name", "value"],
            [{"name": "a", "value": 1}, {"name": "b", "value": 2}],
        )
        assert "data[2]" in result
        assert "name" in result
        assert "value" in result

    def test_empty_rows(self):
        result = _encode_tabular("data", ["x"], [])
        assert "data[0]" in result


class TestStringifyForToon:
    def test_none(self):
        assert _stringify_for_toon(None) == "null"

    def test_bool(self):
        assert _stringify_for_toon(True) == "true"

    def test_int(self):
        assert _stringify_for_toon(42) == "42"

    def test_float(self):
        result = _stringify_for_toon(3.14)
        assert "3.14" in result

    def test_inf(self):
        result = _stringify_for_toon(float('inf'))
        assert "inf" in result.lower()

    def test_string(self):
        assert _stringify_for_toon("hello") == "hello"

    def test_string_with_comma_quoted(self):
        result = _stringify_for_toon("a,b")
        assert '"' in result


class TestStringifyForToonValue:
    def test_none(self):
        assert _stringify_for_toon_value(None, None, ",") == "null"

    def test_bool(self):
        assert _stringify_for_toon_value(True, None, ",") == "true"

    def test_number_with_decimals(self):
        result = _stringify_for_toon_value(1.23456, 2, ",")
        assert result == "1.23"

    def test_number_no_decimals(self):
        result = _stringify_for_toon_value(42, None, ",")
        assert result == "42"

    def test_inf(self):
        result = _stringify_for_toon_value(float('inf'), None, ",")
        assert "inf" in result.lower()


class TestFormatComplexValue:
    def test_scalar(self):
        assert _format_complex_value(42) == "42"

    def test_list_of_scalars(self):
        result = _format_complex_value([1, 2, 3])
        assert "1" in result
        assert "2" in result

    def test_list_of_dicts(self):
        result = _format_complex_value([{"a": 1}, {"a": 2}])
        assert "a" in result

    def test_dict(self):
        result = _format_complex_value({"key": "value"})
        assert "key: value" in result

    def test_nested_dict_with_multiline(self):
        result = _format_complex_value({"outer": {"a": 1, "b": 2}})
        assert "outer:" in result

    def test_empty_values_skipped(self):
        result = _format_complex_value({"a": None, "b": 1})
        assert "a:" not in result
        assert "b: 1" in result
