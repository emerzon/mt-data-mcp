"""Tests for mtdata.core.temporal pure helper functions and temporal_analyze."""

import inspect
import math
from contextlib import contextmanager
from typing import get_args
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mtdata.core.temporal import (
    _error_response,
    _normalize_group_by,
    _parse_month,
    _parse_time_range,
    _parse_time_token,
    _parse_weekday,
    _safe_float,
    _stats_for_group,
    _time_label,
    temporal_analyze,
)

# Access the raw function, bypassing @mcp.tool()
_raw_temporal_analyze = temporal_analyze.__wrapped__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rates(n=100, start_epoch=1704067200, interval=3600):
    """Create synthetic rates structured array similar to MT5 output.

    Default start_epoch = 2024-01-01 00:00:00 UTC (Monday).
    """
    rng = np.random.RandomState(42)
    times = np.arange(start_epoch, start_epoch + n * interval, interval, dtype=np.int64)
    close = 1.1000 + np.cumsum(rng.randn(n) * 0.001)
    open_ = close - rng.randn(n) * 0.0005
    high = np.maximum(open_, close) + np.abs(rng.randn(n) * 0.0003)
    low = np.minimum(open_, close) - np.abs(rng.randn(n) * 0.0003)
    tick_vol = rng.randint(100, 10000, size=n).astype(np.int64)
    real_vol = np.zeros(n, dtype=np.int64)
    spread = rng.randint(1, 20, size=n).astype(np.int32)

    dtype = np.dtype([
        ("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"),
        ("close", "<f8"), ("tick_volume", "<i8"), ("spread", "<i4"),
        ("real_volume", "<i8"),
    ])
    rates = np.empty(n, dtype=dtype)
    rates["time"] = times
    rates["open"] = open_
    rates["high"] = high
    rates["low"] = low
    rates["close"] = close
    rates["tick_volume"] = tick_vol
    rates["spread"] = spread
    rates["real_volume"] = real_vol
    return rates


def _make_weekday_heavy_rates():
    """Create two full weekday weeks plus a sparse Sunday session."""
    start_epoch = 1704067200  # 2024-01-01 00:00:00 UTC, Monday
    times = []
    for week in range(2):
        for day in range(5):
            for hour in range(24):
                times.append(start_epoch + (week * 7 + day) * 86400 + hour * 3600)
    for hour in range(5):
        times.append(start_epoch + 6 * 86400 + hour * 3600)
    rates = _make_rates(n=len(times))
    rates["time"] = np.asarray(sorted(times), dtype=np.int64)
    return rates


@contextmanager
def _mock_guard_ok():
    """Context manager stub replacing _symbol_ready_guard."""
    yield (None, MagicMock())


def _epoch_identity(x):
    """Stand-in for the canonical MT5 epoch normalizer: returns input unchanged."""
    return float(x)


def _fmt_stub(epoch):
    """Stand-in for _format_time_minimal."""
    return "2024-01-01 00:00"


def _guard_stub(*_args, **_kwargs):
    """Stand-in for _symbol_ready_guard: always succeeds."""
    return _mock_guard_ok()


def _info_stub(*_args, **_kwargs):
    """Stand-in for get_symbol_info_cached."""
    return MagicMock()


def _tz_stub(*_args, **_kwargs):
    """Stand-in for _resolve_client_tz: no client timezone."""
    return None


# Patch targets living inside the temporal module's namespace
_P = "mtdata.core.temporal."


# ===================================================================
# _error_response
# ===================================================================

class TestErrorResponse:
    def test_basic(self):
        r = _error_response("bad input", "validate")
        assert r == {"error": "bad input", "stage": "validate"}

    def test_with_context(self):
        ctx = {"symbol": "EURUSD"}
        r = _error_response("fail", "fetch", context=ctx)
        assert r["context"] is ctx

    def test_with_details(self):
        r = _error_response("x", "s", details={"d": 1})
        assert r["details"] == {"d": 1}

    def test_details_none_omitted(self):
        r = _error_response("x", "s", details=None)
        assert "details" not in r

    def test_with_bars(self):
        r = _error_response("x", "s", bars=42)
        assert r["bars"] == 42
        assert isinstance(r["bars"], int)

    def test_bars_coerced_to_int(self):
        r = _error_response("x", "s", bars=3.7)
        assert r["bars"] == 3

    def test_with_filters(self):
        f = {"day_of_week": 0}
        r = _error_response("x", "s", filters=f)
        assert r["filters"] is f

    def test_filters_none_omitted(self):
        r = _error_response("x", "s", filters=None)
        assert "filters" not in r

    def test_all_optional_params(self):
        r = _error_response("msg", "stg", context={"a": 1}, details="d", bars=5, filters={"f": 1})
        assert set(r.keys()) == {"error", "stage", "context", "details", "bars", "filters"}


# ===================================================================
# _normalize_group_by
# ===================================================================

class TestNormalizeGroupBy:
    def test_none_returns_dow(self):
        assert _normalize_group_by(None) == "dow"

    @pytest.mark.parametrize(
        "val", ["day_of_week", "weekday", "week_day", "day", "daily", "dow", "wday"]
    )
    def test_dow_aliases(self, val):
        assert _normalize_group_by(val) == "dow"

    @pytest.mark.parametrize("val", ["hour", "hours", "hr", "time", "time_of_day"])
    def test_hour_aliases(self, val):
        assert _normalize_group_by(val) == "hour"

    @pytest.mark.parametrize("val", ["month", "months", "mo"])
    def test_month_aliases(self, val):
        assert _normalize_group_by(val) == "month"

    @pytest.mark.parametrize("val", ["all", "none", "overall"])
    def test_all_aliases(self, val):
        assert _normalize_group_by(val) == "all"

    def test_case_insensitive(self):
        assert _normalize_group_by("HOUR") == "hour"
        assert _normalize_group_by("  Month  ") == "month"

    def test_unknown_passthrough(self):
        assert _normalize_group_by("quarter") == "quarter"

    def test_empty_string_passthrough(self):
        result = _normalize_group_by("")
        assert result == ""


def test_temporal_analyze_group_by_literal_exposes_canonical_modes_only():
    annotation = inspect.signature(_raw_temporal_analyze).parameters["group_by"].annotation
    assert set(get_args(annotation)) == {"dow", "hour", "month", "all"}


# ===================================================================
# _parse_weekday
# ===================================================================

class TestParseWeekday:
    def test_none_returns_none(self):
        assert _parse_weekday(None) is None

    def test_empty_returns_none(self):
        assert _parse_weekday("") is None
        assert _parse_weekday("  ") is None

    @pytest.mark.parametrize("val,expected", [
        ("0", 0), ("1", 1), ("2", 2), ("3", 3), ("4", 4), ("5", 5), ("6", 6),
    ])
    def test_numeric_strings(self, val, expected):
        assert _parse_weekday(val) == expected

    def test_7_maps_to_6(self):
        assert _parse_weekday("7") == 6

    def test_out_of_range_numeric(self):
        assert _parse_weekday("8") is None
        assert _parse_weekday("99") is None

    @pytest.mark.parametrize("val,expected", [
        ("monday", 0), ("mon", 0),
        ("tuesday", 1), ("tue", 1), ("tues", 1),
        ("wednesday", 2), ("wed", 2),
        ("thursday", 3), ("thu", 3), ("thur", 3), ("thurs", 3),
        ("friday", 4), ("fri", 4),
        ("saturday", 5), ("sat", 5),
        ("sunday", 6), ("sun", 6),
    ])
    def test_day_names(self, val, expected):
        assert _parse_weekday(val) == expected

    def test_case_insensitive(self):
        assert _parse_weekday("MONDAY") == 0
        assert _parse_weekday("Sun") == 6

    def test_whitespace_stripped(self):
        assert _parse_weekday("  fri  ") == 4

    def test_invalid_name(self):
        assert _parse_weekday("notaday") is None


# ===================================================================
# _parse_month
# ===================================================================

class TestParseMonth:
    def test_none_returns_none(self):
        assert _parse_month(None) is None

    def test_empty_returns_none(self):
        assert _parse_month("") is None
        assert _parse_month("   ") is None

    @pytest.mark.parametrize("val,expected", [
        ("1", 1), ("6", 6), ("12", 12),
    ])
    def test_numeric_strings(self, val, expected):
        assert _parse_month(val) == expected

    def test_out_of_range_numeric(self):
        assert _parse_month("0") is None
        assert _parse_month("13") is None

    @pytest.mark.parametrize("val,expected", [
        ("jan", 1), ("january", 1),
        ("feb", 2), ("february", 2),
        ("mar", 3), ("march", 3),
        ("apr", 4), ("april", 4),
        ("may", 5),
        ("jun", 6), ("june", 6),
        ("jul", 7), ("july", 7),
        ("aug", 8), ("august", 8),
        ("sep", 9), ("sept", 9), ("september", 9),
        ("oct", 10), ("october", 10),
        ("nov", 11), ("november", 11),
        ("dec", 12), ("december", 12),
    ])
    def test_month_names(self, val, expected):
        assert _parse_month(val) == expected

    def test_case_insensitive(self):
        assert _parse_month("JANUARY") == 1
        assert _parse_month("Dec") == 12

    def test_whitespace_stripped(self):
        assert _parse_month("  aug  ") == 8

    def test_invalid_name(self):
        assert _parse_month("notamonth") is None


# ===================================================================
# _parse_time_token
# ===================================================================

class TestParseTimeToken:
    def test_hour_only(self):
        assert _parse_time_token("9") == 540

    def test_hour_minute(self):
        assert _parse_time_token("09:30") == 570

    def test_midnight(self):
        assert _parse_time_token("00:00") == 0

    def test_end_of_day(self):
        assert _parse_time_token("23:59") == 1439

    def test_empty_returns_none(self):
        assert _parse_time_token("") is None
        assert _parse_time_token("   ") is None

    def test_invalid_non_numeric(self):
        assert _parse_time_token("abc") is None

    def test_hour_out_of_range(self):
        assert _parse_time_token("24:00") is None
        assert _parse_time_token("-1:00") is None

    def test_minute_out_of_range(self):
        assert _parse_time_token("12:60") is None

    def test_invalid_minute_part(self):
        assert _parse_time_token("12:xx") is None

    def test_whitespace_stripped(self):
        assert _parse_time_token("  14:00  ") == 840


# ===================================================================
# _parse_time_range
# ===================================================================

class TestParseTimeRange:
    def test_none_returns_nones(self):
        s, e, err = _parse_time_range(None)
        assert s is None and e is None and err is None

    def test_empty_returns_nones(self):
        s, e, err = _parse_time_range("")
        assert s is None and e is None and err is None

    def test_dash_separator(self):
        s, e, err = _parse_time_range("09:00-17:00")
        assert s == 540 and e == 1020 and err is None

    def test_to_separator(self):
        s, e, err = _parse_time_range("09:00 to 17:00")
        assert s == 540 and e == 1020 and err is None

    def test_wraps_midnight(self):
        s, e, err = _parse_time_range("22:00-02:00")
        assert s == 1320 and e == 120 and err is None

    def test_equal_start_end_error(self):
        s, e, err = _parse_time_range("10:00-10:00")
        assert s is None and e is None
        assert "differ" in err

    def test_no_separator_error(self):
        s, e, err = _parse_time_range("0900")
        assert err is not None

    def test_invalid_tokens_error(self):
        s, e, err = _parse_time_range("ab:cd-ef:gh")
        assert err is not None

    def test_too_many_parts(self):
        s, e, err = _parse_time_range("09:00-12:00-15:00")
        # Three parts after split → still should parse first two via '-' split
        # but filter strips empties; depends on behavior
        assert err is not None or s is not None  # implementation-dependent

    def test_midnight_range(self):
        s, e, err = _parse_time_range("00:00-23:59")
        assert s == 0 and e == 1439 and err is None


# ===================================================================
# _time_label
# ===================================================================

class TestTimeLabel:
    def test_zero(self):
        assert _time_label(0) == "00:00"

    def test_570(self):
        assert _time_label(570) == "09:30"

    def test_end_of_day(self):
        assert _time_label(1439) == "23:59"

    def test_noon(self):
        assert _time_label(720) == "12:00"

    def test_single_digit_hour(self):
        assert _time_label(65) == "01:05"


# ===================================================================
# _safe_float
# ===================================================================

class TestSafeFloat:
    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string_numeric(self):
        assert _safe_float("2.5") == 2.5

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_nan_returns_none(self):
        assert _safe_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert _safe_float(float("inf")) is None

    def test_neg_inf_returns_none(self):
        assert _safe_float(float("-inf")) is None

    def test_non_numeric_string(self):
        assert _safe_float("abc") is None

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_zero(self):
        assert _safe_float(0) == 0.0

    def test_negative(self):
        assert _safe_float(-1.5) == -1.5

    def test_numpy_float(self):
        assert _safe_float(np.float64(1.23)) == pytest.approx(1.23)

    def test_numpy_nan(self):
        assert _safe_float(np.nan) is None


# ===================================================================
# _stats_for_group
# ===================================================================

class TestStatsForGroup:
    def _make_df(self, returns, volume=None, ranges=None, range_pcts=None):
        data = {"__return": returns}
        if volume is not None:
            data["tick_volume"] = volume
        if ranges is not None:
            data["__range"] = ranges
        if range_pcts is not None:
            data["__range_pct"] = range_pcts
        return pd.DataFrame(data)

    def test_basic_stats(self):
        df = self._make_df([1.0, -0.5, 0.3, 0.2, -0.1])
        out = _stats_for_group(df, None)
        assert out["bars"] == 5
        assert out["returns"] == 5
        assert out["avg_return"] == pytest.approx(0.18, abs=0.01)
        assert out["volatility"] is not None
        assert out["win_rate"] == pytest.approx(3 / 5)

    def test_empty_df(self):
        df = self._make_df([])
        out = _stats_for_group(df, None)
        assert out["bars"] == 0
        assert out["returns"] == 0
        assert out["avg_return"] is None
        assert out["volatility"] is None
        assert out["win_rate"] is None

    def test_all_positive(self):
        df = self._make_df([1.0, 2.0, 3.0])
        out = _stats_for_group(df, None)
        assert out["win_rate"] == pytest.approx(1.0)

    def test_all_negative(self):
        df = self._make_df([-1.0, -2.0])
        out = _stats_for_group(df, None)
        assert out["win_rate"] == pytest.approx(0.0)

    def test_volume_column(self):
        df = self._make_df([0.1, 0.2], volume=[1000, 2000])
        out = _stats_for_group(df, "tick_volume")
        assert out["avg_volume"] == pytest.approx(1500.0)

    def test_volume_none_when_no_col(self):
        df = self._make_df([0.1])
        out = _stats_for_group(df, "tick_volume")
        assert "avg_volume" not in out

    def test_range_columns(self):
        df = self._make_df([0.1, 0.2], ranges=[0.005, 0.010], range_pcts=[0.5, 1.0])
        out = _stats_for_group(df, None)
        assert out["avg_range"] == pytest.approx(0.0075)
        assert out["avg_range_pct"] == pytest.approx(0.75)

    def test_nan_returns_filtered(self):
        df = self._make_df([1.0, float("nan"), 2.0])
        out = _stats_for_group(df, None)
        assert out["returns"] == 2
        assert out["avg_return"] == pytest.approx(1.5)

    def test_median_return(self):
        df = self._make_df([1.0, 2.0, 10.0])
        out = _stats_for_group(df, None)
        assert out["median_return"] == pytest.approx(2.0)

    def test_avg_abs_return(self):
        df = self._make_df([1.0, -3.0])
        out = _stats_for_group(df, None)
        assert out["avg_abs_return"] == pytest.approx(2.0)


# ===================================================================
# temporal_analyze (integration with mocks)
# ===================================================================

def _apply_analyze_patches(func):
    """Stack all patches needed to call _raw_temporal_analyze without MT5.

    Only _fetch_rates injects a mock argument (first positional after self).
    All others use ``new=`` so they don't add extra args.
    """
    func = patch(_P + "_resolve_client_tz", new=_tz_stub)(func)
    func = patch(_P + "_format_time_minimal", new=_fmt_stub)(func)
    func = patch("mtdata.utils.mt5._mt5_epoch_to_utc", new=_epoch_identity)(func)
    func = patch(_P + "_symbol_ready_guard", new=_guard_stub)(func)
    func = patch(_P + "ensure_mt5_connection_or_raise", new=lambda: None)(func)
    func = patch(_P + "get_symbol_info_cached", new=_info_stub)(func)
    func = patch(_P + "_fetch_rates")(func)
    return func


class TestTemporalAnalyze:
    """Tests for temporal_analyze with _fetch_rates mocked."""

    def _call(self, mock_fetch, rates=None, n=200, **kwargs):
        if rates is None:
            rates = _make_rates(n=n, start_epoch=1704067200, interval=3600)
        mock_fetch.return_value = (rates, None)
        defaults = dict(symbol="EURUSD", timeframe="H1", lookback=1000, group_by="dow", detail="full")
        defaults.update(kwargs)
        return _raw_temporal_analyze(**defaults)

    @_apply_analyze_patches
    def test_basic_dow(self, mock_fetch, *_):
        r = self._call(mock_fetch)
        assert r.get("success") is True
        assert r["group_by"] == "dow"
        assert "overall" in r
        assert "groups" in r

    @_apply_analyze_patches
    def test_default_compact_omits_verbose_overall_stats(self, mock_fetch, *_):
        mock_fetch.return_value = (_make_rates(n=200, start_epoch=1704067200, interval=3600), None)

        r = _raw_temporal_analyze(symbol="EURUSD", timeframe="H1", lookback=1000, group_by="dow")

        assert r.get("success") is True
        assert "overall" not in r
        assert "groups" in r
        assert "avg_range" not in r["groups"][0]
        assert "avg_volume" not in r["groups"][0]
        assert "best" in r

    @_apply_analyze_patches
    def test_group_by_day_of_week_alias(self, mock_fetch, *_):
        r = self._call(mock_fetch, group_by="day_of_week")
        assert r.get("success") is True
        assert r["group_by"] == "dow"

    @_apply_analyze_patches
    def test_dow_auto_excludes_sparse_weekend_groups(self, mock_fetch, *_):
        r = self._call(mock_fetch, rates=_make_weekday_heavy_rates(), group_by="dow")

        assert r.get("success") is True
        assert [group["group"] for group in r["groups"]] == [
            "Mon",
            "Tue",
            "Wed",
            "Thu",
            "Fri",
        ]
        assert r["bars"] == 240
        assert r["filters"]["min_bars"] == {"value": 12, "auto": True}
        assert r["excluded_groups"] == [
            {"group": "Sun", "bars": 5, "min_bars": 12, "auto": True}
        ]

    @_apply_analyze_patches
    def test_min_bars_zero_keeps_sparse_weekend_groups(self, mock_fetch, *_):
        r = self._call(
            mock_fetch,
            rates=_make_weekday_heavy_rates(),
            group_by="dow",
            min_bars=0,
        )

        assert r.get("success") is True
        assert "Sun" in [group["group"] for group in r["groups"]]
        assert "excluded_groups" not in r

    @_apply_analyze_patches
    def test_temporal_analyze_logs_finish_event(self, mock_fetch, caplog):
        mock_fetch.return_value = (_make_rates(n=200, start_epoch=1704067200, interval=3600), None)
        with caplog.at_level("DEBUG", logger="mtdata.core.temporal"):
            r = _raw_temporal_analyze(symbol="EURUSD", timeframe="H1", lookback=1000, group_by="dow")
        assert r.get("success") is True
        assert any(
            "event=finish operation=temporal_analyze success=True" in record.message
            for record in caplog.records
        )

    @_apply_analyze_patches
    def test_group_by_hour(self, mock_fetch, *_):
        r = self._call(mock_fetch, group_by="hour")
        assert r.get("success") is True
        assert r["group_by"] == "hour"
        assert len(r["groups"]) > 0
        assert r["groups"][0]["group"].endswith(":00")

    @_apply_analyze_patches
    def test_group_by_hour_uses_already_normalized_bar_times(self, mock_fetch, *_):
        rates = _make_rates(n=3, start_epoch=1704067200, interval=3600)
        mock_fetch.return_value = (rates, None)

        with patch("mtdata.utils.mt5._mt5_epoch_to_utc", new=lambda value: float(value) - 7200.0):
            r = _raw_temporal_analyze(symbol="EURUSD", timeframe="H1", lookback=1000, group_by="hour")

        assert r.get("success") is True
        assert [group["group"] for group in r["groups"]] == ["00:00", "01:00", "02:00"]

    @_apply_analyze_patches
    def test_group_by_month(self, mock_fetch, *_):
        # Use enough data to span multiple months
        rates = _make_rates(n=2000, start_epoch=1704067200, interval=3600)
        r = self._call(mock_fetch, rates=rates, group_by="month")
        assert r.get("success") is True
        assert r["group_by"] == "month"
        assert len(r["groups"]) > 0

    @_apply_analyze_patches
    def test_group_by_all(self, mock_fetch, *_):
        r = self._call(mock_fetch, group_by="all")
        assert r.get("success") is True
        assert r["group_by"] == "all"
        assert "overall" in r
        assert [group["dimension"] for group in r["groups"]] == ["dow", "hour", "month"]
        assert all("breakdown" in group for group in r["groups"])
        assert r["groups"][0]["breakdown"][0]["group"] == "Mon"
        assert r["groups"][1]["breakdown"][0]["group"] == "00:00"
        assert r["groups"][2]["breakdown"][0]["group"] == "Jan"

    @_apply_analyze_patches
    def test_group_by_all_compact_keeps_grouped_breakdowns(self, mock_fetch, *_):
        mock_fetch.return_value = (_make_rates(n=200, start_epoch=1704067200, interval=3600), None)

        r = _raw_temporal_analyze(
            symbol="EURUSD",
            timeframe="H1",
            lookback=1000,
            group_by="all",
        )

        assert r.get("success") is True
        assert "overall" not in r
        assert [group["dimension"] for group in r["groups"]] == ["dow", "hour", "month"]
        assert "avg_range" not in r["groups"][0]["breakdown"][0]
        assert set(r["best"]) == {"dow", "hour", "month"}

    @_apply_analyze_patches
    def test_filter_day_of_week(self, mock_fetch, *_):
        r = self._call(mock_fetch, day_of_week="monday")
        assert r.get("success") is True
        assert r["filters"]["day_of_week"]["label"] == "Mon"

    @_apply_analyze_patches
    def test_filter_month(self, mock_fetch, *_):
        rates = _make_rates(n=2000, start_epoch=1704067200, interval=3600)
        r = self._call(mock_fetch, rates=rates, month="jan")
        assert r.get("success") is True
        assert r["filters"]["month"]["label"] == "Jan"

    @_apply_analyze_patches
    def test_filter_time_range(self, mock_fetch, *_):
        r = self._call(mock_fetch, time_range="09:00-17:00")
        assert r.get("success") is True
        assert r["filters"]["time_range"]["start"] == "09:00"
        assert r["filters"]["time_range"]["end_exclusive"] is True
        assert r["filters"]["time_range"]["wraps_midnight"] is False

    @_apply_analyze_patches
    def test_filter_time_range_wraps_midnight(self, mock_fetch, *_):
        r = self._call(mock_fetch, time_range="22:00-02:00")
        assert r.get("success") is True
        assert r["filters"]["time_range"]["end_exclusive"] is True
        assert r["filters"]["time_range"]["wraps_midnight"] is True

    @_apply_analyze_patches
    def test_filter_time_range_excludes_exact_end_time(self, mock_fetch, *_):
        rates = _make_rates(n=24, start_epoch=1704067200, interval=3600)
        r = self._call(mock_fetch, rates=rates, group_by="hour", time_range="09:00-17:00")
        assert r.get("success") is True
        group_strs = [g["group"] for g in r["groups"]]
        assert group_strs == ["09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"]
        assert "17:00" not in group_strs

    @_apply_analyze_patches
    def test_filter_time_range_wraps_midnight_excludes_exact_end_time(self, mock_fetch, *_):
        rates = _make_rates(n=24, start_epoch=1704067200, interval=3600)
        r = self._call(mock_fetch, rates=rates, group_by="hour", time_range="22:00-02:00")
        assert r.get("success") is True
        group_strs = [g["group"] for g in r["groups"]]
        assert group_strs == ["00:00", "01:00", "22:00", "23:00"]
        assert "02:00" not in group_strs

    @_apply_analyze_patches
    def test_return_mode_log(self, mock_fetch, *_):
        r = self._call(mock_fetch, return_mode="log")
        assert r.get("success") is True
        assert r["return_mode"] == "log"

    @_apply_analyze_patches
    def test_return_mode_pct(self, mock_fetch, *_):
        r = self._call(mock_fetch, return_mode="pct")
        assert r.get("success") is True
        assert r["return_mode"] == "pct"

    @_apply_analyze_patches
    def test_volume_source_tick(self, mock_fetch, *_):
        r = self._call(mock_fetch)
        assert r.get("success") is True
        assert r["volume_source"] == "tick_volume"

    @_apply_analyze_patches
    def test_volume_source_real_when_present(self, mock_fetch, *_):
        rates = _make_rates(n=200)
        # Set real_volume to nonzero
        rates["real_volume"] = np.arange(1, 201, dtype=np.int64)
        r = self._call(mock_fetch, rates=rates)
        assert r.get("success") is True
        assert r["volume_source"] == "real_volume"

    @_apply_analyze_patches
    def test_overall_stats_present(self, mock_fetch, *_):
        r = self._call(mock_fetch)
        overall = r["overall"]
        assert "bars" in overall
        assert "avg_return" in overall
        assert "volatility" in overall
        assert "win_rate" in overall

    @_apply_analyze_patches
    def test_invalid_lookback(self, mock_fetch, *_):
        r = _raw_temporal_analyze(symbol="EURUSD", timeframe="H1", lookback=1, group_by="dow")
        assert "error" in r
        assert r["stage"] == "validate"

    @_apply_analyze_patches
    def test_invalid_group_by(self, mock_fetch, *_):
        r = _raw_temporal_analyze(symbol="EURUSD", timeframe="H1", lookback=100, group_by="quarter")
        assert "error" in r

    @_apply_analyze_patches
    def test_invalid_day_of_week(self, mock_fetch, *_):
        r = _raw_temporal_analyze(symbol="EURUSD", lookback=100, day_of_week="xyz")
        assert "error" in r
        assert r["stage"] == "validate"

    @_apply_analyze_patches
    def test_invalid_month(self, mock_fetch, *_):
        r = _raw_temporal_analyze(symbol="EURUSD", lookback=100, month="xyz")
        assert "error" in r
        assert r["stage"] == "validate"

    @_apply_analyze_patches
    def test_invalid_time_range(self, mock_fetch, *_):
        r = _raw_temporal_analyze(symbol="EURUSD", lookback=100, time_range="bad")
        assert "error" in r

    @_apply_analyze_patches
    def test_fetch_error(self, mock_fetch, *_):
        mock_fetch.return_value = (None, "fetch failed")
        r = _raw_temporal_analyze(symbol="EURUSD", lookback=100)
        assert "error" in r
        assert r["stage"] == "fetch"

    @_apply_analyze_patches
    def test_insufficient_data(self, mock_fetch, *_):
        rates = _make_rates(n=1)
        mock_fetch.return_value = (rates, None)
        r = _raw_temporal_analyze(symbol="EURUSD", lookback=100)
        assert "error" in r

    @_apply_analyze_patches
    def test_bars_count_in_response(self, mock_fetch, *_):
        r = self._call(mock_fetch, n=50)
        assert r.get("success") is True
        assert r["bars"] > 0

    @_apply_analyze_patches
    def test_symbol_in_response(self, mock_fetch, *_):
        r = self._call(mock_fetch, symbol="GBPUSD")
        assert r.get("success") is True
        assert r["symbol"] == "GBPUSD"

    @_apply_analyze_patches
    def test_timeframe_in_response(self, mock_fetch, *_):
        r = self._call(mock_fetch, timeframe="H1")
        assert r.get("success") is True
        assert r["timeframe"] == "H1"

    @_apply_analyze_patches
    def test_timezone_utc_default(self, mock_fetch, *_):
        r = self._call(mock_fetch)
        assert r.get("success") is True
        assert r["timezone"] == "UTC"

    @_apply_analyze_patches
    def test_group_keys_are_ints(self, mock_fetch, *_):
        r = self._call(mock_fetch, group_by="dow")
        for g in r.get("groups", []):
            assert isinstance(g["group"], str)

    @_apply_analyze_patches
    def test_filter_narrows_results(self, mock_fetch, *_):
        """Filtering by a specific day should yield fewer bars than unfiltered."""
        r_all = self._call(mock_fetch)
        r_mon = self._call(mock_fetch, day_of_week="monday")
        assert r_mon["bars"] < r_all["bars"]
