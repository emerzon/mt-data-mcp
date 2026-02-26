"""Tests for src/mtdata/utils/utils.py — pure utility functions."""
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from mtdata.utils.utils import (
    _coerce_scalar,
    _normalize_ohlcv_arg,
    _normalize_limit,
    _table_from_rows,
    _format_time_minimal,
    _time_format_from_epochs,
    _maybe_strip_year,
    _style_time_format,
    to_float_np,
    align_finite,
    _utc_epoch_seconds,
)


class TestCoerceScalar:
    def test_none(self):
        assert _coerce_scalar(None) is None

    def test_empty_string(self):
        assert _coerce_scalar("") == ""

    def test_int_string(self):
        assert _coerce_scalar("42") == 42

    def test_negative_int_string(self):
        assert _coerce_scalar("-7") == -7

    def test_float_string(self):
        result = _coerce_scalar("3.14")
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-9

    def test_non_numeric(self):
        assert _coerce_scalar("hello") == "hello"

    def test_whitespace_padded(self):
        assert _coerce_scalar("  5  ") == 5

    def test_zero(self):
        assert _coerce_scalar("0") == 0


class TestNormalizeOhlcvArg:
    def test_none_returns_none(self):
        assert _normalize_ohlcv_arg(None) is None

    def test_empty_returns_none(self):
        assert _normalize_ohlcv_arg("") is None

    def test_all(self):
        assert _normalize_ohlcv_arg("all") == {"O", "H", "L", "C", "V"}

    def test_ohlcv(self):
        assert _normalize_ohlcv_arg("ohlcv") == {"O", "H", "L", "C", "V"}

    def test_ohlc(self):
        assert _normalize_ohlcv_arg("ohlc") == {"O", "H", "L", "C"}

    def test_price(self):
        assert _normalize_ohlcv_arg("price") == {"C"}

    def test_close(self):
        assert _normalize_ohlcv_arg("close") == {"C"}

    def test_compact_letters(self):
        assert _normalize_ohlcv_arg("cl") == {"C", "L"}

    def test_comma_separated_names(self):
        result = _normalize_ohlcv_arg("open,high,close")
        assert result == {"O", "H", "C"}

    def test_semicolon_separated(self):
        result = _normalize_ohlcv_arg("open;volume")
        assert result == {"O", "V"}

    def test_unknown_names_return_none(self):
        assert _normalize_ohlcv_arg("foo,bar") is None


class TestNormalizeLimit:
    def test_none_returns_none(self):
        assert _normalize_limit(None) is None

    def test_positive_int(self):
        assert _normalize_limit(10) == 10

    def test_zero_returns_none(self):
        assert _normalize_limit(0) is None

    def test_negative_returns_none(self):
        assert _normalize_limit(-5) is None

    def test_float_truncated(self):
        assert _normalize_limit(3.9) == 3

    def test_string_number(self):
        assert _normalize_limit("7") == 7

    def test_empty_string(self):
        assert _normalize_limit("") is None

    def test_invalid_string(self):
        assert _normalize_limit("abc") is None


class TestTableFromRows:
    def test_basic_table(self):
        result = _table_from_rows(["a", "b"], [[1, 2], [3, 4]])
        assert result["success"] is True
        assert result["count"] == 2
        assert result["data"][0] == {"a": 1, "b": 2}

    def test_empty_rows(self):
        result = _table_from_rows(["x"], [])
        assert result["count"] == 0
        assert result["data"] == []

    def test_short_row_padded_with_none(self):
        result = _table_from_rows(["a", "b", "c"], [[1]])
        assert result["data"][0] == {"a": 1, "b": None, "c": None}


class TestFormatTimeMinimal:
    def test_epoch_zero(self):
        result = _format_time_minimal(0)
        assert result == "1970-01-01 00:00"

    def test_known_timestamp(self):
        result = _format_time_minimal(1704067200)  # 2024-01-01 00:00 UTC
        assert "2024-01-01" in result


class TestTimeFormatHelpers:
    def test_time_format_from_epochs(self):
        result = _time_format_from_epochs([1.0, 2.0])
        assert "%" in result

    def test_maybe_strip_year_noop(self):
        fmt = "%Y-%m-%d %H:%M"
        assert _maybe_strip_year(fmt, [1.0]) == fmt

    def test_style_time_format_with_t(self):
        assert _style_time_format("%Y-%m-%dT%H:%M") == "%Y-%m-%d %H:%M"

    def test_style_time_format_without_t(self):
        fmt = "%Y-%m-%d %H:%M"
        assert _style_time_format(fmt) == fmt


class TestToFloatNp:
    def test_list_of_ints(self):
        result = to_float_np([1, 2, 3])
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_series(self):
        s = pd.Series([1.5, 2.5])
        result = to_float_np(s)
        np.testing.assert_array_almost_equal(result, [1.5, 2.5])

    def test_drop_na(self):
        result = to_float_np([1.0, float('nan'), 3.0], drop_na=True)
        np.testing.assert_array_equal(result, [1.0, 3.0])

    def test_finite_only(self):
        result = to_float_np([1.0, float('inf'), 3.0], finite_only=True)
        np.testing.assert_array_equal(result, [1.0, 3.0])

    def test_return_mask(self):
        arr, mask = to_float_np([1.0, 2.0], return_mask=True)
        assert len(arr) == 2
        assert all(mask)

    def test_coerce_strings(self):
        result = to_float_np(["1", "abc", "3"], coerce=True, drop_na=True)
        np.testing.assert_array_equal(result, [1.0, 3.0])


class TestAlignFinite:
    def test_basic_alignment(self):
        a, b = align_finite([1, float('nan'), 3], [4, 5, 6])
        np.testing.assert_array_equal(a, [1.0, 3.0])
        np.testing.assert_array_equal(b, [4.0, 6.0])

    def test_empty_input(self):
        result = align_finite()
        assert result == ()


class TestUtcEpochSeconds:
    def test_naive_datetime(self):
        dt = datetime(2024, 1, 1, 0, 0, 0)
        result = _utc_epoch_seconds(dt)
        assert abs(result - 1704067200.0) < 1.0

    def test_aware_datetime(self):
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = _utc_epoch_seconds(dt)
        assert abs(result - 1704067200.0) < 1.0
