"""Tests for src/mtdata/utils/formatting.py"""
import math
import pytest
from mtdata.utils.formatting import (
    _adaptive_decimals,
    format_float,
    optimal_decimals,
    format_number,
)


class TestFormatFloat:
    def test_basic_integer(self):
        assert format_float(3.0, 0) == "3"

    def test_trailing_zeros_trimmed(self):
        assert format_float(1.50, 2) == "1.5"

    def test_all_zeros_trimmed(self):
        assert format_float(2.0, 5) == "2"

    def test_negative_zero(self):
        assert format_float(-0.0, 2) == "0"

    def test_precision_preserved(self):
        assert format_float(1.234, 3) == "1.234"

    def test_zero_decimals(self):
        assert format_float(42.9, 0) == "43"

    def test_large_number(self):
        assert format_float(123456.789, 2) == "123456.79"

    def test_small_number(self):
        assert format_float(0.00123, 5) == "0.00123"


class TestFormatNumber:
    def test_none_returns_null(self):
        assert format_number(None) == "null"

    def test_bool_true(self):
        assert format_number(True) == "true"

    def test_bool_false(self):
        assert format_number(False) == "false"

    def test_int_value(self):
        result = format_number(42)
        assert result == "42"

    def test_float_value(self):
        result = format_number(3.14)
        assert "3.14" in result

    def test_non_numeric_string(self):
        assert format_number("hello") == "hello"

    def test_inf_returns_str(self):
        result = format_number(float('inf'))
        assert "inf" in result.lower()

    def test_nan_returns_str(self):
        result = format_number(float('nan'))
        assert "nan" in result.lower()

    def test_explicit_decimals(self):
        assert format_number(1.23456, decimals=2) == "1.23"

    def test_zero(self):
        assert format_number(0) == "0"

    def test_negative(self):
        result = format_number(-5.5, decimals=1)
        assert result == "-5.5"


class TestOptimalDecimals:
    def test_empty_list(self):
        assert optimal_decimals([]) == 0

    def test_single_integer(self):
        assert optimal_decimals([100.0]) == 0

    def test_values_need_decimals(self):
        result = optimal_decimals([1.001, 1.002, 1.003])
        assert result >= 2

    def test_non_finite_filtered(self):
        result = optimal_decimals([1.0, float('nan'), float('inf'), 2.0])
        assert isinstance(result, int)

    def test_non_numeric_filtered(self):
        result = optimal_decimals(["abc", 1.0, 2.0])
        assert isinstance(result, int)

    def test_identical_values(self):
        result = optimal_decimals([5.0, 5.0, 5.0])
        assert result == 0

    def test_max_decimals_respected(self):
        result = optimal_decimals([0.000000001, 0.000000002], max_decimals=4)
        assert result <= 4


class TestAdaptiveDecimals:
    def test_integer(self):
        assert _adaptive_decimals(100.0) == 0

    def test_small_fraction(self):
        result = _adaptive_decimals(0.001)
        assert result >= 2

    def test_large_number(self):
        result = _adaptive_decimals(100000.0)
        assert result == 0

    def test_max_decimals_cap(self):
        result = _adaptive_decimals(0.000000001, max_decimals=3)
        assert result <= 3
