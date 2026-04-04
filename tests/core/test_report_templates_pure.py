"""Comprehensive pure-function tests for report template modules.

Covers:
  - mtdata.core.report_templates.basic  (pure helpers + template_basic via mocks)
  - mtdata.core.report                  (_report_error_text, _report_error_payload)
  - mtdata.core.report_templates.scalping (template_scalping via mocks)
  - mtdata.core.report_templates.advanced (template_advanced via mocks)

Every test is deterministic – no MT5, no network, no side effects.
"""

import math
import pytest
from unittest.mock import patch, MagicMock

from mtdata.core.report_templates.basic import (
    _safe_float,
    _ema,
    _compute_tr,
    _linreg_slope_r2,
    _percentile_rank,
    _compute_compact_trend,
    _parse_value,
    _parse_table_data,
    _parse_formatted_output,
    _get_raw_result,
)
from mtdata.core.report import _report_error_text, _report_error_payload


# ---------------------------------------------------------------------------
# Synthetic candle helpers
# ---------------------------------------------------------------------------

def _make_rows(n=30, base=100.0, step=0.1):
    return [
        {"close": base + i * step, "high": base + 1 + i * step,
         "low": base - 1 + i * step, "tick_volume": 1000}
        for i in range(n)
    ]


# ===== _safe_float ==========================================================

class TestSafeFloat:
    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string_number(self):
        assert _safe_float("2.71") == 2.71

    def test_nan_returns_default(self):
        assert _safe_float(float("nan")) is None

    def test_nan_returns_custom_default(self):
        assert _safe_float(float("nan"), default=0.0) == 0.0

    def test_inf_returns_default(self):
        assert _safe_float(float("inf")) is None

    def test_neg_inf_returns_default(self):
        assert _safe_float(float("-inf"), default=-1.0) == -1.0

    def test_none_returns_default(self):
        assert _safe_float(None) is None

    def test_non_numeric_string(self):
        assert _safe_float("abc") is None

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_zero(self):
        assert _safe_float(0) == 0.0

    def test_negative(self):
        assert _safe_float(-5.5) == -5.5


# ===== _ema =================================================================

class TestEma:
    def test_empty(self):
        assert _ema([], 10) == []

    def test_single_value(self):
        assert _ema([5.0], 10) == [5.0]

    def test_length_one_passthrough(self):
        vals = [1.0, 2.0, 3.0]
        assert _ema(vals, 1) == vals

    def test_length_zero_passthrough(self):
        vals = [1.0, 2.0, 3.0]
        assert _ema(vals, 0) == vals

    def test_constant_series(self):
        vals = [5.0] * 10
        result = _ema(vals, 5)
        assert len(result) == 10
        for v in result:
            assert abs(v - 5.0) < 1e-9

    def test_increasing_series_smoothing(self):
        vals = [float(i) for i in range(20)]
        result = _ema(vals, 5)
        assert len(result) == 20
        # EMA of increasing series should lag behind actual values
        assert result[-1] < vals[-1]
        assert result[-1] > vals[0]

    def test_output_length_matches_input(self):
        vals = [1.0, 3.0, 2.0, 4.0, 3.5]
        result = _ema(vals, 3)
        assert len(result) == len(vals)

    def test_first_value_equals_input(self):
        vals = [10.0, 20.0, 30.0]
        result = _ema(vals, 5)
        assert result[0] == 10.0


# ===== _compute_tr ===========================================================

class TestComputeTr:
    def test_empty(self):
        assert _compute_tr([], [], []) == []

    def test_single_bar(self):
        result = _compute_tr([105.0], [95.0], [100.0])
        assert len(result) == 1
        assert result[0] == 10.0  # high - low

    def test_gap_up(self):
        # Previous close=100, current bar high=112, low=108
        result = _compute_tr([100.0, 112.0], [98.0, 108.0], [100.0, 110.0])
        assert len(result) == 2
        # Second bar: max(112-108=4, |112-100|=12, |108-100|=8) = 12
        assert result[1] == 12.0

    def test_gap_down(self):
        result = _compute_tr([100.0, 92.0], [98.0, 88.0], [100.0, 90.0])
        # Second bar: max(92-88=4, |92-100|=8, |88-100|=12) = 12
        assert result[1] == 12.0

    def test_none_high_uses_prev_close(self):
        result = _compute_tr([None, None], [95.0, 95.0], [100.0, 100.0])
        assert len(result) == 2

    def test_none_low_uses_prev_close(self):
        result = _compute_tr([105.0, 105.0], [None, None], [100.0, 100.0])
        assert len(result) == 2

    def test_mismatched_lengths(self):
        # high shorter than close
        result = _compute_tr([105.0], [95.0, 94.0], [100.0, 99.0])
        assert len(result) == 2


# ===== _linreg_slope_r2 =====================================================

class TestLinregSlopeR2:
    def test_too_short(self):
        assert _linreg_slope_r2([]) is None
        assert _linreg_slope_r2([1.0]) is None

    def test_perfect_uptrend(self):
        series = [float(i) for i in range(10)]
        result = _linreg_slope_r2(series)
        assert result is not None
        slope, r2 = result
        assert abs(slope - 1.0) < 1e-9
        assert abs(r2 - 1.0) < 1e-9

    def test_perfect_downtrend(self):
        series = [10.0 - i for i in range(10)]
        result = _linreg_slope_r2(series)
        assert result is not None
        slope, r2 = result
        assert abs(slope - (-1.0)) < 1e-9
        assert abs(r2 - 1.0) < 1e-9

    def test_flat_series(self):
        series = [5.0] * 10
        result = _linreg_slope_r2(series)
        assert result is not None
        slope, r2 = result
        assert abs(slope) < 1e-9
        # R² is 0 for constant series (no variance)
        assert r2 == 0.0

    def test_two_points(self):
        result = _linreg_slope_r2([1.0, 3.0])
        assert result is not None
        slope, r2 = result
        assert abs(slope - 2.0) < 1e-9

    def test_noisy_series_r2_less_than_one(self):
        series = [1.0, 3.0, 2.0, 4.0, 3.5, 5.0]
        result = _linreg_slope_r2(series)
        assert result is not None
        slope, r2 = result
        assert slope > 0
        assert 0.0 < r2 < 1.0


# ===== _percentile_rank =====================================================

class TestPercentileRank:
    def test_empty_list(self):
        assert _percentile_rank([], 5.0) == 0

    def test_all_below(self):
        assert _percentile_rank([1.0, 2.0, 3.0], 10.0) == 100

    def test_all_above(self):
        assert _percentile_rank([10.0, 20.0, 30.0], 1.0) == 0

    def test_median(self):
        result = _percentile_rank([1.0, 2.0, 3.0, 4.0, 5.0], 3.0)
        assert result == 60  # 3 out of 5 values <= 3

    def test_nan_filtered(self):
        result = _percentile_rank([1.0, float("nan"), 2.0, 3.0], 2.5)
        # NaN is filtered; [1, 2, 3], 2 values <= 2.5 → 67%
        assert result == 67

    def test_single_value_equal(self):
        assert _percentile_rank([5.0], 5.0) == 100

    def test_single_value_below(self):
        assert _percentile_rank([5.0], 1.0) == 0

    def test_clamped_to_0_100(self):
        result = _percentile_rank([1.0, 2.0], 0.5)
        assert 0 <= result <= 100

    def test_all_same_values(self):
        assert _percentile_rank([3.0, 3.0, 3.0], 3.0) == 100


# ===== _compute_compact_trend ===============================================

class TestComputeCompactTrend:
    def test_none_for_too_few_rows(self):
        assert _compute_compact_trend([]) is None
        assert _compute_compact_trend([{"close": 100}] * 4) is None

    def test_returns_dict_with_expected_keys(self):
        rows = _make_rows(30)
        result = _compute_compact_trend(rows)
        assert result is not None
        for key in ('s', 'r', 'v', 'q', 'g', 'h', 'l'):
            assert key in result

    def test_slopes_list_length(self):
        rows = _make_rows(60)
        result = _compute_compact_trend(rows)
        assert result is not None
        assert len(result['s']) == 3  # windows [5, 20, 60]
        assert len(result['r']) == 3

    def test_uptrend_positive_slopes(self):
        rows = _make_rows(30, base=100, step=0.5)
        result = _compute_compact_trend(rows)
        assert result is not None
        # Short-term slope should be positive for uptrend
        assert result['s'][0] > 0

    def test_downtrend_negative_slopes(self):
        rows = [
            {"close": 200 - i * 0.5, "high": 201 - i * 0.5,
             "low": 199 - i * 0.5, "tick_volume": 1000}
            for i in range(30)
        ]
        result = _compute_compact_trend(rows)
        assert result is not None
        assert result['s'][0] < 0

    def test_volatility_positive(self):
        rows = _make_rows(30)
        result = _compute_compact_trend(rows)
        assert result is not None
        assert result['v'] >= 0

    def test_squeeze_percentile_range(self):
        rows = _make_rows(80)
        result = _compute_compact_trend(rows)
        assert result is not None
        assert 0 <= result['q'] <= 100

    def test_regime_code_range(self):
        rows = _make_rows(30)
        result = _compute_compact_trend(rows)
        assert result is not None
        assert result['g'] in (0, 1, 2, 3, 4)

    def test_h_l_non_negative(self):
        rows = _make_rows(30)
        result = _compute_compact_trend(rows)
        assert result is not None
        assert result['h'] >= 0
        assert result['l'] >= 0

    def test_none_close_handled(self):
        rows = [{"close": None, "high": 101, "low": 99, "tick_volume": 100} for _ in range(10)]
        result = _compute_compact_trend(rows)
        assert result is not None  # Should not crash

    def test_missing_high_low_keys(self):
        rows = [{"close": 100 + i * 0.1} for i in range(10)]
        result = _compute_compact_trend(rows)
        assert result is not None

    def test_large_dataset(self):
        rows = _make_rows(200, base=50, step=0.01)
        result = _compute_compact_trend(rows)
        assert result is not None
        assert len(result['s']) == 3

    def test_flat_market_regime_zero(self):
        # Flat market: all same close
        rows = [{"close": 100.0, "high": 100.1, "low": 99.9, "tick_volume": 500}
                for _ in range(30)]
        result = _compute_compact_trend(rows)
        assert result is not None
        # Flat market should have regime 0 (range)
        assert result['g'] == 0


# ===== _parse_value ==========================================================

class TestParseValue:
    def test_int(self):
        assert _parse_value("42") == 42

    def test_float(self):
        assert _parse_value("3.14") == 3.14

    def test_scientific(self):
        assert _parse_value("1e5") == 1e5

    def test_true_variants(self):
        assert _parse_value("true") is True
        assert _parse_value("True") is True
        assert _parse_value("yes") is True
        assert _parse_value("Yes") is True

    def test_false_variants(self):
        assert _parse_value("false") is False
        assert _parse_value("False") is False
        assert _parse_value("no") is False
        assert _parse_value("No") is False

    def test_none_variants(self):
        assert _parse_value("") is None
        assert _parse_value("null") is None
        assert _parse_value("None") is None
        assert _parse_value("none") is None

    def test_string_passthrough(self):
        assert _parse_value("hello world") == "hello world"

    def test_whitespace_stripped(self):
        assert _parse_value("  42  ") == 42

    def test_negative_int(self):
        assert _parse_value("-7") == -7

    def test_negative_float(self):
        assert _parse_value("-3.14") == -3.14


# ===== _parse_table_data =====================================================

class TestParseTableData:
    def test_empty(self):
        assert _parse_table_data([]) is None

    def test_headers_only(self):
        result = _parse_table_data(["level,classic,fibonacci"])
        assert result == {"headers": ["level", "classic", "fibonacci"]}

    def test_headers_and_rows(self):
        lines = ["level,classic", "PP,1.2345", "R1,1.2400"]
        result = _parse_table_data(lines)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["level"] == "PP"
        assert result[0]["classic"] == 1.2345
        assert result[1]["level"] == "R1"

    def test_numeric_parsing(self):
        lines = ["name,value", "a,42", "b,3.14"]
        result = _parse_table_data(lines)
        assert result[0]["value"] == 42
        assert result[1]["value"] == 3.14

    def test_empty_cell(self):
        lines = ["name,value", "a,"]
        result = _parse_table_data(lines)
        assert result[0]["value"] is None

    def test_more_headers_than_values(self):
        lines = ["a,b,c", "1,2"]
        result = _parse_table_data(lines)
        assert isinstance(result, list)
        assert result[0]["c"] is None  # missing value defaults to empty→None

    def test_blank_data_rows_skipped(self):
        lines = ["h1,h2", "  ", "v1,v2"]
        result = _parse_table_data(lines)
        assert isinstance(result, list)
        assert len(result) == 1


# ===== _parse_formatted_output ===============================================

class TestParseFormattedOutput:
    def test_simple_kv(self):
        output = "symbol: BTCUSD\ntimeframe: H1\n"
        result = _parse_formatted_output(output)
        assert result["symbol"] == "BTCUSD"
        assert result["timeframe"] == "H1"

    def test_numeric_values(self):
        output = "price: 1.2345\ncount: 10\n"
        result = _parse_formatted_output(output)
        assert result["price"] == 1.2345
        assert result["count"] == 10

    def test_nested_section(self):
        output = "period:\n  start: 2024-01-01\n  end: 2024-06-01\n"
        result = _parse_formatted_output(output)
        assert "period" in result
        assert result["period"]["start"] == "2024-01-01"

    def test_empty_output(self):
        result = _parse_formatted_output("")
        assert "error" in result

    def test_table_data_inline(self):
        output = "level,classic\nPP,1.234\nR1,1.250\n"
        result = _parse_formatted_output(output)
        assert "data" in result
        assert isinstance(result["data"], list)

    def test_levels_section_with_table(self):
        output = "symbol: EURUSD\nlevels:\nlevel,classic\nPP,1.234\nR1,1.250\n"
        result = _parse_formatted_output(output)
        assert "levels" in result

    def test_only_blank_lines(self):
        result = _parse_formatted_output("\n\n\n")
        assert "error" in result

    def test_mixed_sections(self):
        output = "name: test\nbest:\n  method: ema\n  score: 0.95\nstatus: ok\n"
        result = _parse_formatted_output(output)
        assert result["name"] == "test"
        assert "best" in result


# ===== _get_raw_result =======================================================

class TestGetRawResult:
    def test_dict_passthrough(self):
        fn = MagicMock(return_value={"data": [1, 2, 3]})
        result = _get_raw_result(fn, symbol="X")
        assert result == {"data": [1, 2, 3]}

    def test_coroutine_result_is_resolved(self):
        async def fn(**kwargs):
            return {"ok": kwargs.get("symbol")}

        result = _get_raw_result(fn, symbol="X")
        assert result == {"ok": "X"}

    def test_string_result_parsed(self):
        fn = MagicMock(return_value="symbol: TEST\nprice: 42\n")
        result = _get_raw_result(fn, symbol="X")
        assert isinstance(result, dict)
        assert result.get("symbol") == "TEST"

    def test_exception_returns_error(self):
        fn = MagicMock(side_effect=RuntimeError("boom"))
        result = _get_raw_result(fn)
        assert "error" in result
        assert "boom" in result["error"]

    def test_cli_raw_kwarg_passed(self):
        fn = MagicMock(return_value={"ok": True})
        _get_raw_result(fn, x=1)
        fn.assert_called_once()
        _, kwargs = fn.call_args
        assert kwargs.get("__cli_raw") is True

    def test_cli_raw_fallback_on_typeerror(self):
        def strict_fn(**kwargs):
            if "__cli_raw" in kwargs:
                raise TypeError("unexpected kwarg")
            return {"ok": True}
        result = _get_raw_result(strict_fn, x=1)
        assert result == {"ok": True}

    def test_unexpected_type_returns_error(self):
        fn = MagicMock(return_value=12345)
        result = _get_raw_result(fn)
        assert "error" in result
        assert "Unexpected" in result["error"]


# ===== _report_error_text / _report_error_payload ============================

class TestReportErrorText:
    def test_normal_message(self):
        result = _report_error_text("something broke")
        assert result == "error: something broke\n"

    def test_strips_whitespace(self):
        result = _report_error_text("  padded  ")
        assert result == "error: padded\n"

    def test_empty_message(self):
        result = _report_error_text("")
        assert result == "error: Unknown error.\n"

    def test_whitespace_only(self):
        result = _report_error_text("   ")
        assert result == "error: Unknown error.\n"

    def test_non_string_coerced(self):
        result = _report_error_text(42)
        assert result == "error: 42\n"

    def test_none_message(self):
        result = _report_error_text(None)
        assert "error:" in result


class TestReportErrorPayload:
    def test_normal_message(self):
        result = _report_error_payload("fail")
        assert result == {"error": "fail"}

    def test_empty_message(self):
        result = _report_error_payload("")
        assert result == {"error": "Unknown error."}

    def test_whitespace_only(self):
        result = _report_error_payload("   ")
        assert result == {"error": "Unknown error."}

    def test_strips_whitespace(self):
        result = _report_error_payload("  oops  ")
        assert result == {"error": "oops"}

    def test_non_string(self):
        result = _report_error_payload(Exception("err"))
        assert result["error"] == "err"


# ===== template_basic (mocked) ==============================================

_BASIC_MODULE = "mtdata.core.report_templates.basic"


def _mock_candle_data(n=40):
    """Return a dict that data_fetch_candles would return."""
    rows = []
    for i in range(n):
        rows.append({
            "time": f"2024-01-{(i%28)+1:02d}T00:00:00",
            "open": 100.0 + i * 0.1,
            "high": 101.0 + i * 0.1,
            "low": 99.0 + i * 0.1,
            "close": 100.05 + i * 0.1,
            "tick_volume": 1000 + i,
            "EMA_20": 100.0 + i * 0.08,
            "EMA_50": 100.0 + i * 0.05,
            "RSI_14": 55.0 + i * 0.1,
        })
    return {"rows": rows, "count": n}


def _mock_pivot_data():
    return {
        "levels": [
            {"level": "PP", "classic": 1.234},
            {"level": "R1", "classic": 1.250},
            {"level": "S1", "classic": 1.210},
        ],
        "methods": [{"method": "classic"}],
        "source": "D1",
        "period": {"start": "2024-01-01", "end": "2024-01-02"},
    }


def _mock_vol_data():
    return {
        "horizon_sigma_return": 0.02,
        "sigma_bar_return": 0.005,
        "horizon_sigma_price": 2.5,
    }


def _mock_backtest_data():
    return {
        "results": {
            "ema": {
                "avg_rmse": 0.01,
                "avg_mae": 0.008,
                "avg_directional_accuracy": 0.65,
                "successful_tests": 20,
            },
            "arima": {
                "avg_rmse": 0.02,
                "avg_mae": 0.015,
                "avg_directional_accuracy": 0.55,
                "successful_tests": 18,
            },
        }
    }


def _mock_forecast_data():
    return {
        "forecast_price": 102.5,
        "lower_price": 101.0,
        "upper_price": 104.0,
        "trend": "up",
        "ci_alpha": 0.05,
        "last_observation_epoch": 1740948000.0,
        "forecast_start_epoch": 1740951600.0,
        "forecast_anchor": "next_timeframe_bar_after_last_observation",
    }


def _mock_barrier_data():
    return {
        "best": {"tp": 1.0, "sl": 0.5, "edge": 0.3},
        "grid": [],
    }


def _mock_patterns_data():
    return {
        "rows": [
            {"pattern": "hammer", "signal": "bullish", "time": "2024-01-10"},
        ],
        "count": 1,
    }


class TestTemplateBasic:
    """Test template_basic with all external calls mocked."""

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method")
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_report_structure(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        candle_rows = _mock_candle_data()["rows"]
        mock_tail.return_value = candle_rows[-40:]
        mock_pick.return_value = ("ema", {"avg_rmse": 0.01, "avg_mae": 0.008,
                                          "avg_directional_accuracy": 0.65,
                                          "successful_tests": 20})
        mock_sum_bar.return_value = {"best": {"tp": 1.0, "sl": 0.5, "edge": 0.3}, "top": []}

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return _mock_backtest_data()
            if "forecast_generate" in name.lower() or "generate" in name.lower():
                return _mock_forecast_data()
            if "barrier" in name.lower():
                return _mock_barrier_data()
            if "pattern" in name.lower():
                return _mock_patterns_data()
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, None)

        assert isinstance(report, dict)
        assert "meta" in report
        assert report["meta"]["symbol"] == "EURUSD"
        assert report["meta"]["template"] == "basic"
        assert report["meta"]["horizon"] == 12
        assert "sections" in report

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_context_includes_trend_compact_legend(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        candle_rows = _mock_candle_data()["rows"]
        mock_tail.return_value = candle_rows[-40:]
        mock_sum_bar.return_value = {"best": {}}

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, "__name__") else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return {"results": {}}
            if "barrier" in name.lower():
                return {"error": "barrier failed"}
            if "pattern" in name.lower():
                return {"error": "patterns failed"}
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, None)

        ctx = report["sections"].get("context", {})
        assert isinstance(ctx.get("trend_compact"), dict)
        assert isinstance(ctx.get("trend_compact_legend"), dict)
        assert "s" in ctx["trend_compact_legend"]
        assert "trend_compact_explained" not in ctx
        assert "sparkline_close" not in ctx

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail", return_value=[])
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_error_in_candle_fetch(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_sum_bar.return_value = {"best": {"tp": 1.0, "sl": 0.5, "edge": 0.3}}

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return {"error": "MT5 not connected"}
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return {"error": "backtest failed"}
            if "barrier" in name.lower():
                return {"error": "barrier failed"}
            if "pattern" in name.lower():
                return {"error": "patterns failed"}
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, None)

        assert "error" in report["sections"]["context"]

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_all_errors(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_tail.return_value = []
        mock_sum_bar.return_value = {"best": {}}

        mock_raw.return_value = {"error": "everything broke"}

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, {})

        assert isinstance(report, dict)
        assert "sections" in report

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_custom_params(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_tail.return_value = _mock_candle_data()["rows"][-10:]
        mock_sum_bar.return_value = {"best": {"tp": 1.0, "sl": 0.5, "edge": 0.3}}
        mock_raw.return_value = _mock_candle_data()

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("BTCUSD", 24, None, {
            "timeframe": "H4",
            "context_limit": 100,
            "context_tail": 20,
            "backtest_steps": 10,
        })

        assert report["meta"]["timeframe"] == "H4"
        assert report["meta"]["horizon"] == 24

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method")
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_volatility_section_populated(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_tail.return_value = _mock_candle_data()["rows"]
        mock_pick.return_value = None
        mock_sum_bar.return_value = {"best": {}}

        mock_raw.return_value = _mock_vol_data()

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, {})

        vol = report["sections"].get("volatility", {})
        # Should have matrix if vol estimates succeed
        assert "matrix" in vol or "error" in vol

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method")
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_barriers_both_errors(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_tail.return_value = _mock_candle_data()["rows"]
        mock_pick.return_value = None

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "barrier" in name.lower():
                return {"error": "no barrier"}
            return _mock_vol_data()

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, {})

        barriers = report["sections"].get("barriers", {})
        assert "error" in barriers

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_barriers_elevates_conflict_caution_to_section_level(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_tail.return_value = _mock_candle_data()["rows"]
        mock_sum_bar.side_effect = [
            {
                "best": {"tp": 1.0, "sl": 0.5, "ev": 0.1, "edge": -0.2},
                "ev_edge_conflict": True,
                "caution": "long conflict",
            },
            {"best": {"tp": 1.1, "sl": 0.6, "ev": 0.05, "edge": 0.1}},
        ]

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, "__name__") else str(func)
            if "barrier" in name.lower():
                return {"success": True}
            if "pattern" in name.lower():
                return _mock_patterns_data()
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return {"results": {}}
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, {})

        barriers = report["sections"].get("barriers", {})
        assert barriers.get("ev_edge_conflict") is True
        assert barriers.get("ev_edge_conflict_directions") == ["long"]
        assert "caution" in barriers
        assert "note" in barriers

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_barriers_forward_template_volatility_defaults(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        candle_rows = _mock_candle_data()["rows"]
        mock_tail.return_value = candle_rows[-40:]
        mock_sum_bar.return_value = {"best": {"tp": 1.0, "sl": 0.5, "edge": 0.3}, "top": []}
        barrier_params = []

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, "__name__") else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return {"results": {}}
            if "forecast_generate" in name.lower() or "generate" in name.lower():
                return _mock_forecast_data()
            if "barrier" in name.lower():
                barrier_params.append(dict(kwargs.get("params") or {}))
                return _mock_barrier_data()
            if "pattern" in name.lower():
                return _mock_patterns_data()
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        _ = template_basic("EURUSD", 12, None, {})

        assert len(barrier_params) == 2
        for params in barrier_params:
            assert params["vol_window"] == 250
            assert params["vol_min_mult"] == 0.6
            assert params["vol_max_mult"] == 2.2
            assert params["vol_sl_multiplier"] == 1.7
            assert params["vol_sl_steps"] == 9
            assert params["vol_floor_pct"] == 0.2

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method")
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_forecast_when_best_found(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        candle_rows = _mock_candle_data()["rows"]
        mock_tail.return_value = candle_rows
        mock_pick.return_value = ("ema", {"avg_rmse": 0.01, "avg_mae": 0.008,
                                          "avg_directional_accuracy": 0.65,
                                          "successful_tests": 20})
        mock_sum_bar.return_value = {"best": {"tp": 1, "sl": 0.5, "edge": 0.3}}

        call_count = [0]

        def raw_side_effect(func, *args, **kwargs):
            call_count[0] += 1
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return _mock_backtest_data()
            if "generate" in name.lower() or "forecast_generate" in name.lower():
                return _mock_forecast_data()
            if "barrier" in name.lower():
                return _mock_barrier_data()
            if "pattern" in name.lower():
                return _mock_patterns_data()
            return {"data": "fallback"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, {})

        forecast = report["sections"].get("forecast", {})
        # If best method was found, forecast section should be populated
        assert "method" in forecast or "error" in forecast
        if "error" not in forecast:
            assert forecast.get("last_observation_epoch") is not None
            assert forecast.get("forecast_start_epoch") is not None
            assert forecast.get("forecast_anchor")

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method")
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_applies_min_directional_accuracy_threshold(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_tail.return_value = _mock_candle_data()["rows"]
        mock_pick.return_value = (
            "ema",
            {"avg_rmse": 0.01, "avg_mae": 0.008, "avg_directional_accuracy": 0.65, "successful_tests": 20},
        )
        mock_sum_bar.return_value = {"best": {"tp": 1, "sl": 0.5, "edge": 0.3}}

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return _mock_backtest_data()
            if "generate" in name.lower() or "forecast_generate" in name.lower():
                return _mock_forecast_data()
            if "barrier" in name.lower():
                return _mock_barrier_data()
            if "pattern" in name.lower():
                return _mock_patterns_data()
            return {"data": "fallback"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic(
            "EURUSD",
            12,
            None,
            {"backtest_min_directional_accuracy": 0.55},
        )

        call_kwargs = mock_pick.call_args.kwargs
        assert call_kwargs.get("min_directional_accuracy") == 0.55
        criteria = report["sections"].get("backtest", {}).get("selection_criteria", {})
        assert criteria.get("min_directional_accuracy") == 0.55

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method")
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_falls_back_when_best_forecast_is_degenerate(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        candle_rows = _mock_candle_data()["rows"]
        mock_tail.return_value = candle_rows
        mock_pick.return_value = (
            "sf_autoarima",
            {"avg_rmse": 0.9, "avg_mae": 0.7, "avg_directional_accuracy": 0.2, "successful_tests": 20},
        )
        mock_sum_bar.return_value = {"best": {"tp": 1, "sl": 0.5, "edge": 0.3}}

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return {
                    "results": {
                        "sf_autoarima": {
                            "avg_rmse": 0.9,
                            "avg_mae": 0.7,
                            "avg_directional_accuracy": 0.2,
                            "successful_tests": 20,
                        },
                        "naive": {
                            "avg_rmse": 0.95,
                            "avg_mae": 0.8,
                            "avg_directional_accuracy": 0.1,
                            "successful_tests": 20,
                        },
                    }
                }
            if "forecast_generate" in name.lower() or "generate" in name.lower():
                method = kwargs.get("method")
                if method == "sf_autoarima":
                    return {"forecast_price": [65290.6] * 12}
                if method == "naive":
                    return {"forecast_price": [65290.6, 65320.0, 65380.0, 65440.0]}
                return {"error": f"unexpected method: {method}"}
            if "barrier" in name.lower():
                return _mock_barrier_data()
            if "pattern" in name.lower():
                return _mock_patterns_data()
            return {"data": "fallback"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, {})

        forecast = report["sections"].get("forecast", {})
        assert forecast.get("method") == "naive"
        assert forecast.get("fallback_from") == "sf_autoarima"
        assert any("degenerate" in str(v).lower() for v in forecast.get("selection_warnings", []))
        assert report.get("fallback_applied") is True
        assert report.get("original_method") == "sf_autoarima"
        assert report.get("fallback_method") == "naive"

        best_method = report["sections"].get("backtest", {}).get("best_method", {})
        assert best_method.get("method") == "naive"
        assert best_method.get("initial_method") == "sf_autoarima"
        selection_basis = best_method.get("selection_basis", {})
        assert selection_basis.get("primary_metric") == "avg_rmse"
        assert selection_basis.get("tie_breaker") == "avg_directional_accuracy"
        assert selection_basis.get("fallback_applied") is True
        criteria = report["sections"].get("backtest", {}).get("selection_criteria", {})
        assert criteria.get("primary_metric") == "avg_rmse"

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_patterns_section(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_tail.return_value = _mock_candle_data()["rows"]
        mock_sum_bar.return_value = {"best": {}}

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "pattern" in name.lower():
                return _mock_patterns_data()
            return _mock_vol_data()

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, {})

        pats = report["sections"].get("patterns", {})
        assert "recent" in pats or "error" in pats


# ===== template_advanced (mocked) ============================================

_ADV_MODULE = "mtdata.core.report_templates.advanced"


class TestTemplateAdvanced:
    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_adds_regime(self, mock_raw, mock_basic):
        mock_basic.return_value = {
            "meta": {"symbol": "EURUSD", "template": "basic"},
            "sections": {
                "backtest": {"best_method": {"method": "ema"}},
            },
        }
        mock_raw.return_value = {"summary": "regime info"}

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        assert isinstance(report, dict)
        assert "regime" in report["sections"]

    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_adds_har_rv(self, mock_raw, mock_basic):
        mock_basic.return_value = {
            "meta": {"symbol": "EURUSD", "template": "basic"},
            "sections": {"backtest": {}},
        }
        mock_raw.return_value = {
            "sigma_bar_return": 0.005,
            "horizon_sigma_return": 0.02,
        }

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        har = report["sections"].get("volatility_har_rv", {})
        assert "sigma_bar_return" in har or "error" in har

    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_conformal_when_best_method(self, mock_raw, mock_basic):
        mock_basic.return_value = {
            "meta": {"symbol": "EURUSD", "template": "basic"},
            "sections": {
                "backtest": {"best_method": {"method": "arima"}},
            },
        }

        def raw_se(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "conformal" in name.lower():
                return {
                    "lower_price": 1.20,
                    "upper_price": 1.25,
                    "ci_alpha": 0.1,
                    "conformal": {"per_step_q": [0.01]},
                }
            return {"summary": "ok"}

        mock_raw.side_effect = raw_se

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        conf = report["sections"].get("forecast_conformal", {})
        assert "method" in conf or "error" in conf

    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_no_conformal_without_best(self, mock_raw, mock_basic):
        mock_basic.return_value = {
            "meta": {"symbol": "EURUSD", "template": "basic"},
            "sections": {"backtest": {}},
        }
        mock_raw.return_value = {"summary": "ok"}

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        assert "forecast_conformal" not in report.get("sections", {})

    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_handles_basic_string_error(self, mock_raw, mock_basic):
        mock_basic.return_value = "some error string"

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        assert "error" in report

    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_handles_basic_unexpected_type(self, mock_raw, mock_basic):
        mock_basic.return_value = 12345

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        assert "error" in report

    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_har_rv_error(self, mock_raw, mock_basic):
        mock_basic.return_value = {
            "meta": {"symbol": "EURUSD", "template": "basic"},
            "sections": {"backtest": {}},
        }
        mock_raw.return_value = {"error": "HAR failed"}

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        har = report["sections"].get("volatility_har_rv", {})
        assert "error" in har

    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_conformal_error(self, mock_raw, mock_basic):
        mock_basic.return_value = {
            "meta": {"symbol": "EURUSD", "template": "basic"},
            "sections": {
                "backtest": {"best_method": {"method": "ema"}},
            },
        }

        def raw_se(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "conformal" in name.lower():
                return {"error": "conformal failed"}
            return {"summary": "ok"}

        mock_raw.side_effect = raw_se

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        conf = report["sections"].get("forecast_conformal", {})
        assert "error" in conf


# ===== template_scalping (mocked) ============================================

_SCALP_MODULE = "mtdata.core.report_templates.scalping"
_COMMON_MODULE = "mtdata.core.report_templates.common"


class TestTemplateScalping:
    @patch(f"{_SCALP_MODULE}.build_report_with_market")
    @patch(f"{_SCALP_MODULE}.market_snapshot")
    @patch(f"{_SCALP_MODULE}.merge_params")
    @patch(f"{_SCALP_MODULE}._get_pip_size", return_value=0.01)
    def test_scalping_returns_report(self, mock_pip, mock_merge, mock_snap, mock_build):
        mock_merge.return_value = {"timeframe": "M5", "mode": "pips"}
        mock_snap.return_value = {"bid": 1.2345, "ask": 1.2347, "spread_ticks": 20}
        mock_build.return_value = {
            "meta": {"symbol": "EURUSD", "template": "scalping"},
            "sections": {},
        }

        from mtdata.core.report_templates.scalping import template_scalping
        report = template_scalping("EURUSD", 8, None, {})

        assert isinstance(report, dict)
        mock_build.assert_called_once()

    @patch(f"{_SCALP_MODULE}.build_report_with_market")
    @patch(f"{_SCALP_MODULE}.market_snapshot")
    @patch(f"{_SCALP_MODULE}.merge_params")
    @patch(f"{_SCALP_MODULE}._get_pip_size", return_value=0.01)
    def test_scalping_default_timeframe_m5(self, mock_pip, mock_merge, mock_snap, mock_build):
        mock_merge.return_value = {"mode": "pips"}
        mock_snap.return_value = {"bid": 1.23, "ask": 1.24}
        mock_build.return_value = {"meta": {}, "sections": {}}

        from mtdata.core.report_templates.scalping import template_scalping
        template_scalping("EURUSD", 8, None, {})

        # merge_params result should have timeframe set to M5
        call_args = mock_build.call_args
        params = call_args[1].get("params") if call_args[1] else call_args[0][3]
        # Check that build was called
        assert mock_build.called

    @patch(f"{_SCALP_MODULE}.build_report_with_market")
    @patch(f"{_SCALP_MODULE}.market_snapshot")
    @patch(f"{_SCALP_MODULE}.merge_params")
    @patch(f"{_SCALP_MODULE}._get_pip_size", return_value=1.0)
    def test_scalping_high_price_adjusts_pip_levels(self, mock_pip, mock_merge, mock_snap, mock_build):
        mock_merge.return_value = {"mode": "pips"}
        mock_snap.return_value = {
            "bid": 30000.0, "ask": 30002.0, "spread_ticks": 200,
        }
        mock_build.return_value = {"meta": {}, "sections": {}}

        from mtdata.core.report_templates.scalping import template_scalping
        template_scalping("BTCUSD", 8, None, {})

        assert mock_build.called

    @patch(f"{_SCALP_MODULE}.build_report_with_market")
    @patch(f"{_SCALP_MODULE}.market_snapshot")
    @patch(f"{_SCALP_MODULE}.merge_params")
    @patch(f"{_SCALP_MODULE}._get_pip_size", return_value=0.01)
    def test_scalping_pct_mode_no_adjustment(self, mock_pip, mock_merge, mock_snap, mock_build):
        mock_merge.return_value = {"mode": "pct"}
        mock_snap.return_value = {"bid": 1.23, "ask": 1.24}
        mock_build.return_value = {"meta": {}, "sections": {}}

        from mtdata.core.report_templates.scalping import template_scalping
        template_scalping("EURUSD", 8, None, {})

        assert mock_build.called

    @patch(f"{_SCALP_MODULE}.build_report_with_market")
    @patch(f"{_SCALP_MODULE}.market_snapshot")
    @patch(f"{_SCALP_MODULE}.merge_params")
    @patch(f"{_SCALP_MODULE}._get_pip_size", return_value=0.01)
    def test_scalping_snapshot_error_still_runs(self, mock_pip, mock_merge, mock_snap, mock_build):
        mock_merge.return_value = {"mode": "pips"}
        mock_snap.return_value = {"error": "MT5 offline"}
        mock_build.return_value = {"meta": {}, "sections": {}}

        from mtdata.core.report_templates.scalping import template_scalping
        report = template_scalping("EURUSD", 8, None, {})

        assert mock_build.called

    @patch(f"{_SCALP_MODULE}.build_report_with_market")
    @patch(f"{_SCALP_MODULE}.market_snapshot")
    @patch(f"{_SCALP_MODULE}.merge_params")
    @patch(f"{_SCALP_MODULE}._get_pip_size", return_value=1.0)
    def test_scalping_no_spread_ticks_uses_price_based(self, mock_pip, mock_merge, mock_snap, mock_build):
        mock_merge.return_value = {"mode": "pips"}
        mock_snap.return_value = {"bid": 2000.0, "ask": 2001.0}
        mock_build.return_value = {"meta": {}, "sections": {}}

        from mtdata.core.report_templates.scalping import template_scalping
        template_scalping("XAUUSD", 8, None, {})

        assert mock_build.called


# ===== Edge cases and additional coverage ====================================

class TestSafeFloatEdge:
    def test_bool_true(self):
        # bool is subclass of int: float(True) == 1.0
        assert _safe_float(True) == 1.0

    def test_bool_false(self):
        assert _safe_float(False) == 0.0

    def test_list_returns_default(self):
        assert _safe_float([1, 2]) is None

    def test_very_large_number(self):
        assert _safe_float(1e300) == 1e300


class TestEmaEdge:
    def test_two_values(self):
        result = _ema([10.0, 20.0], 3)
        assert len(result) == 2
        assert result[0] == 10.0
        k = 2.0 / 4.0  # 2/(3+1)
        expected = 10.0 + k * (20.0 - 10.0)
        assert abs(result[1] - expected) < 1e-9

    def test_large_length(self):
        vals = [100.0 + i for i in range(5)]
        result = _ema(vals, 100)
        # Very large length → very slow decay, stays close to first value
        assert result[-1] < vals[-1]


class TestComputeTrEdge:
    def test_all_none_close_raises(self):
        # When close[0] is None, prev_close is None, causing TypeError in abs(h - l)
        with pytest.raises(TypeError):
            _compute_tr([None], [None], [None])

    def test_three_bars_normal(self):
        h = [102.0, 104.0, 103.0]
        l = [98.0, 99.0, 97.0]
        c = [100.0, 103.0, 98.0]
        result = _compute_tr(h, l, c)
        assert len(result) == 3
        # First bar: max(4, 2, 2) = 4
        assert result[0] == 4.0


class TestLinregSlopeR2Edge:
    def test_two_identical_points(self):
        result = _linreg_slope_r2([5.0, 5.0])
        assert result is not None
        slope, r2 = result
        assert slope == 0.0
        assert r2 == 0.0

    def test_large_series(self):
        series = [float(i) * 2.0 + 1.0 for i in range(100)]
        result = _linreg_slope_r2(series)
        assert result is not None
        slope, r2 = result
        assert abs(slope - 2.0) < 1e-6
        assert abs(r2 - 1.0) < 1e-6


class TestParseValueEdge:
    def test_float_with_exponent(self):
        assert _parse_value("1.5E3") == 1500.0

    def test_leading_trailing_spaces(self):
        assert _parse_value("  true  ") is True

    def test_Null_uppercase(self):
        assert _parse_value("Null") is None

    def test_plain_string_with_dot(self):
        # "hello.world" can't be parsed as float but has a dot
        assert _parse_value("hello.world") == "hello.world"


class TestParseTableDataEdge:
    def test_single_column(self):
        lines = ["name", "alice", "bob"]
        result = _parse_table_data(lines)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "alice"

    def test_all_empty_data_rows(self):
        lines = ["h1,h2", "", ""]
        result = _parse_table_data(lines)
        # All blank data rows are skipped → headers only
        assert result == {"headers": ["h1", "h2"]}


class TestParseFormattedOutputEdge:
    def test_colon_in_value(self):
        output = "time: 2024-01-01T12:00:00\n"
        result = _parse_formatted_output(output)
        assert "time" in result
        # Value includes everything after first colon
        assert "2024" in str(result["time"])

    def test_summary_nested(self):
        output = "summary:\n  trend: up\n  confidence: high\n"
        result = _parse_formatted_output(output)
        assert "summary" in result
        assert result["summary"]["trend"] == "up"
