"""Tests for src/mtdata/utils/minimal_output.py — TOON formatting helpers."""
import math

import pytest

from mtdata.utils.minimal_output import (
    _encode_tabular,
    _format_complex_value,
    _headers_from_dicts,
    _indent_text,
    _quote_key,
    _stringify_for_toon_value,
    _stringify_scalar,
)
from mtdata.utils.minimal_output_toon import (
    _column_decimals,
    _minify_number,
    _quote_if_needed,
    _stringify_cell,
    _stringify_for_toon,
)


class TestStringifyScalar:
    @pytest.mark.parametrize(("value", "expected"), [(True, "true"), (False, "false")])
    def test_bools(self, value, expected):
        assert _stringify_scalar(value) == expected


class TestStringifyCell:
    def test_nested_dict_empty_values_skipped(self):
        result = _stringify_cell({"a": None, "b": 1})
        assert "a=" not in result
        assert "b=" in result


class TestIndentText:
    @pytest.mark.parametrize(
        ("text", "indent", "expected"),
        [
            ("hello", "  ", "  hello"),
            ("a\nb", "  ", "  a\n  b"),
            ("x", ">>", ">>x"),
        ],
    )
    def test_indent_variants(self, text, indent, expected):
        assert _indent_text(text, indent=indent) == expected


class TestQuoteIfNeeded:
    @pytest.mark.parametrize(
        ("value", "must_quote"),
        [
            ("hello", False),
            ("a,b", True),
            (" hello", True),
            ("a:b", True),
        ],
    )
    def test_quotes_special_strings(self, value, must_quote):
        result = _quote_if_needed(value)
        if must_quote:
            assert result.startswith('"') and result.endswith('"')
        else:
            assert result == value

    def test_none_returns_empty(self):
        assert _quote_if_needed(None) == ""


class TestQuoteKey:
    @pytest.mark.parametrize(
        ("value", "check"),
        [
            ("name", lambda r: r == "name"),
            (None, lambda r: r == ""),
            ("a,b", lambda r: '"' in r),
        ],
    )
    def test_quote_key_variants(self, value, check):
        assert check(_quote_key(value))


class TestHeadersFromDicts:
    def test_basic(self):
        items = [{"a": 1, "b": 2}, {"a": 3, "c": 4}]
        headers = _headers_from_dicts(items)
        assert headers == ["a", "b", "c"]

    @pytest.mark.parametrize("items", [[], [1, 2, 3]])
    def test_empty_or_non_dict_returns_empty(self, items):
        assert _headers_from_dicts(items) == []


class TestColumnDecimals:
    def test_int_columns(self):
        result = _column_decimals(["x"], [{"x": 1}, {"x": 2}])
        assert result["x"] == 0

    def test_float_columns(self):
        result = _column_decimals(["x"], [{"x": 1.001}, {"x": 1.002}])
        assert result["x"] >= 2

    def test_quote_decimal_columns_are_case_insensitive(self):
        result = _column_decimals(["Price"], [{"Price": 77000.0}, {"Price": 0.00001234}])
        assert result["Price"] == 8

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
    @pytest.mark.parametrize(
        ("value", "predicate"),
        [
            (None, lambda r: r == "null"),
            (True, lambda r: r == "true"),
            (42, lambda r: r == "42"),
            (3.14, lambda r: "3.14" in r),
            (float("inf"), lambda r: "inf" in r.lower()),
            ("hello", lambda r: r == "hello"),
            ("a,b", lambda r: '"' in r),
        ],
    )
    def test_scalar_matrix(self, value, predicate):
        assert predicate(_stringify_for_toon(value))


class TestStringifyForToonValue:
    @pytest.mark.parametrize(
        ("value", "decimals", "expected"),
        [
            (None, None, "null"),
            (True, None, "true"),
            (1.23456, 2, "1.23"),
            (42, None, "42"),
        ],
    )
    def test_scalar_matrix(self, value, decimals, expected):
        assert _stringify_for_toon_value(value, decimals, ",") == expected

    def test_inf(self):
        result = _stringify_for_toon_value(float("inf"), None, ",")
        assert "inf" in result.lower()

    def test_dict_uses_compact_key_value_text_instead_of_repr(self):
        result = _stringify_for_toon_value({"low": 0.1, "high": 0.2}, None, ",")
        assert result == "low=0.1; high=0.2"
        assert "{" not in result

    def test_list_uses_compact_scalar_joining(self):
        result = _stringify_for_toon_value([1, 2, 3], None, ",")
        assert result == '"1|2|3"'


class TestFormatComplexValue:
    def test_list_of_dicts(self):
        result = _format_complex_value([{"a": 1}, {"a": 2}])
        assert "a" in result

    def test_nested_dict_with_multiline(self):
        result = _format_complex_value({"outer": {"a": 1, "b": 2}})
        assert "outer:" in result

    def test_empty_values_skipped(self):
        result = _format_complex_value({"a": None, "b": 1})
        assert "a:" not in result
        assert "b: 1" in result


class TestEncodeTabularNestedCells:
    def test_nested_cells_do_not_render_python_repr(self):
        result = _encode_tabular(
            "results",
            ["method", "prob_win_ci95"],
            [{"method": "theta", "prob_win_ci95": {"low": 0.3, "high": 0.5}}],
        )

        assert "prob_win_ci95" in result
        assert "low=0.3; high=0.5" in result
        assert "{'low': 0.3, 'high': 0.5}" not in result
