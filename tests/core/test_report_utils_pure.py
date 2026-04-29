"""Comprehensive pure-function tests for mtdata.core.report.utils.

Every test is deterministic – no MT5, no network, no side effects.
"""

import math
import re
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from mtdata.core.report.utils import (
    _as_float,
    _extract_base_timeframe,
    _format_decimal,
    _format_probability,
    _format_series_preview,
    _format_signed,
    _format_state_shares,
    _get_indicator_value,
    _indicator_key_variants,
    apply_market_gates,
    attach_candle_freshness_diagnostics,
    attach_multi_timeframes,
    context_for_tf,
    extract_candle_freshness_diagnostics,
    format_number,
    market_snapshot,
    merge_params,
    now_utc_iso,
    parse_table_tail,
    pick_best_forecast_method,
    summarize_barrier_grid,
)
from mtdata.utils.formatting import format_number as util_format_number


# ---------------------------------------------------------------------------
# 1. now_utc_iso
# ---------------------------------------------------------------------------
class TestNowUtcIso:
    def test_returns_string(self):
        result = now_utc_iso()
        assert isinstance(result, str)

    def test_format_matches_display(self):
        result = now_utc_iso()
        # Should match YYYY-MM-DD HH:MM
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", result)

    def test_approximately_now(self):
        before = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:")
        result = now_utc_iso()
        assert result.startswith(before[:11])  # at least same date


# ---------------------------------------------------------------------------
# 2. parse_table_tail
# ---------------------------------------------------------------------------
class TestParseTableTail:
    def test_basic_list(self):
        data = [{"a": "1", "b": "2.5"}]
        result = parse_table_tail(data, tail=1)
        assert result == [{"a": 1, "b": 2.5}]

    def test_dict_with_data_key(self):
        data = {"data": [{"x": "10"}, {"x": "20"}]}
        result = parse_table_tail(data, tail=1)
        assert len(result) == 1
        assert result[0]["x"] == 20

    def test_dict_with_bars_key(self):
        data = {"bars": [{"x": "10"}, {"x": "20"}]}
        result = parse_table_tail(data, tail=1)
        assert len(result) == 1
        assert result[0]["x"] == 20

    def test_tail_zero_returns_all(self):
        rows = [{"v": str(i)} for i in range(5)]
        result = parse_table_tail(rows, tail=0)
        assert len(result) == 5

    def test_tail_larger_than_rows(self):
        rows = [{"v": "1"}]
        result = parse_table_tail(rows, tail=100)
        assert len(result) == 1

    def test_coerces_int_string(self):
        result = parse_table_tail([{"k": "42"}])
        assert result[0]["k"] == 42

    def test_coerces_float_string(self):
        result = parse_table_tail([{"k": "3.14"}])
        assert isinstance(result[0]["k"], float)

    def test_coerces_scientific(self):
        result = parse_table_tail([{"k": "1e3"}])
        assert result[0]["k"] == 1000.0

    def test_coerces_nan(self):
        result = parse_table_tail([{"k": "nan"}])
        assert math.isnan(result[0]["k"])

    def test_coerces_inf(self):
        result = parse_table_tail([{"k": "inf"}])
        assert result[0]["k"] == float("inf")

    def test_coerces_negative_int(self):
        result = parse_table_tail([{"k": "-7"}])
        assert result[0]["k"] == -7

    def test_preserves_bool(self):
        result = parse_table_tail([{"k": True}])
        assert result[0]["k"] is True

    def test_preserves_none(self):
        result = parse_table_tail([{"k": None}])
        assert result[0]["k"] is None

    def test_empty_string_unchanged(self):
        result = parse_table_tail([{"k": ""}])
        assert result[0]["k"] == ""

    def test_non_numeric_string(self):
        result = parse_table_tail([{"k": "hello"}])
        assert result[0]["k"] == "hello"

    def test_non_list_returns_empty(self):
        assert parse_table_tail("not a list") == []

    def test_none_input(self):
        assert parse_table_tail(None) == []

    def test_skips_non_dict_rows(self):
        # tail=1 default, so only last dict row is returned
        result = parse_table_tail([{"a": "1"}, "bad", {"b": "2"}])
        assert len(result) == 1
        assert "b" in result[0]

    def test_tail_none(self):
        result = parse_table_tail([{"a": "1"}], tail=None)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 2a. candle freshness diagnostics
# ---------------------------------------------------------------------------
class TestCandleFreshnessDiagnostics:
    def test_extracts_from_meta(self):
        freshness = {"data_freshness_seconds": 3600.0}
        result = extract_candle_freshness_diagnostics(
            {"meta": {"diagnostics": {"freshness": freshness}}}
        )
        assert result == freshness
        assert result is not freshness

    def test_extracts_from_stale_error_details(self):
        freshness = {"data_freshness_seconds": 7200.0}
        result = extract_candle_freshness_diagnostics(
            {
                "error": "Data remained stale",
                "details": {"diagnostics": {"freshness": freshness}},
            }
        )
        assert result == freshness
        assert result is not freshness

    def test_meta_takes_priority_over_details(self):
        result = extract_candle_freshness_diagnostics(
            {
                "meta": {"diagnostics": {"freshness": {"source": "meta"}}},
                "details": {"diagnostics": {"freshness": {"source": "details"}}},
            }
        )
        assert result == {"source": "meta"}

    def test_attach_uses_stale_error_details(self):
        freshness = {"last_bar_within_policy_window": False}
        attached = attach_candle_freshness_diagnostics(
            {"error": "Data remained stale"},
            {
                "error": "Data remained stale",
                "details": {"diagnostics": {"freshness": freshness}},
            },
        )
        assert attached == {
            "error": "Data remained stale",
            "freshness": freshness,
        }


# ---------------------------------------------------------------------------
# 3. _indicator_key_variants
# ---------------------------------------------------------------------------
class TestIndicatorKeyVariants:
    def test_basic(self):
        variants = _indicator_key_variants("EMA_20")
        assert "EMA_20" in variants
        assert "ema_20" in variants

    def test_digit_expansion(self):
        variants = _indicator_key_variants("RSI_14")
        assert "RSI_14.0" in variants
        assert "rsi_14.0" in variants

    def test_no_digit_part(self):
        variants = _indicator_key_variants("MACD")
        assert "MACD" in variants
        assert "macd" in variants
        assert len(variants) == 2

    def test_empty_string(self):
        assert _indicator_key_variants("") == []

    def test_none_input(self):
        assert _indicator_key_variants(None) == []

    def test_multiple_digit_parts(self):
        variants = _indicator_key_variants("MACD_12_26_9")
        assert "MACD_12.0_26.0_9.0" in variants


# ---------------------------------------------------------------------------
# 4. _get_indicator_value
# ---------------------------------------------------------------------------
class TestGetIndicatorValue:
    def test_exact_match(self):
        assert _get_indicator_value({"EMA_20": 1.5}, "EMA_20") == 1.5

    def test_lowercase_fallback(self):
        assert _get_indicator_value({"ema_20": 2.0}, "EMA_20") == 2.0

    def test_digit_expansion_fallback(self):
        assert _get_indicator_value({"RSI_14.0": 55}, "RSI_14") == 55

    def test_none_value_skipped(self):
        row = {"EMA_20": None, "ema_20": 3.0}
        assert _get_indicator_value(row, "EMA_20") == 3.0

    def test_empty_string_skipped(self):
        row = {"EMA_20": "", "ema_20": 4.0}
        assert _get_indicator_value(row, "EMA_20") == 4.0

    def test_missing_key_returns_none(self):
        assert _get_indicator_value({"a": 1}, "EMA_20") is None

    def test_none_row(self):
        assert _get_indicator_value(None, "EMA_20") is None

    def test_non_dict_row(self):
        assert _get_indicator_value("string", "EMA_20") is None

    def test_empty_base_key(self):
        assert _get_indicator_value({"a": 1}, "") is None


# ---------------------------------------------------------------------------
# 5. _format_series_preview
# ---------------------------------------------------------------------------
class TestFormatSeriesPreview:
    def test_empty_list(self):
        assert _format_series_preview([]) == "n=0 []"

    def test_not_list(self):
        assert _format_series_preview("string") is None
        assert _format_series_preview(None) is None

    def test_all_numeric(self):
        result = _format_series_preview([1.0, 2.0, 3.0])
        assert "n=3" in result
        assert "start=" in result
        assert "end=" in result
        assert "min=" in result
        assert "max=" in result

    def test_non_numeric_items(self):
        result = _format_series_preview(["a", "b", "c"])
        assert "n=3" in result
        assert "[" in result

    def test_mixed_with_nan_falls_to_string(self):
        result = _format_series_preview([1.0, float("nan"), 3.0])
        assert result is not None
        # nan breaks numeric path, falls to string-style
        assert "n=3" in result

    def test_long_list_ellipsis(self):
        result = _format_series_preview(["a"] * 10, head=3, tail=3)
        assert "..." in result

    def test_short_list_no_ellipsis(self):
        result = _format_series_preview(["x", "y"], head=3, tail=3)
        assert "..." not in result

    def test_custom_decimals(self):
        result = _format_series_preview([1.123456789], decimals=2)
        assert "n=1" in result


# ---------------------------------------------------------------------------
# 6. _format_state_shares
# ---------------------------------------------------------------------------
class TestFormatStateShares:
    def test_basic(self):
        result = _format_state_shares({"0": 0.5, "1": 0.5})
        assert "0:50.0%" in result
        assert "1:50.0%" in result

    def test_sorted_numeric_keys(self):
        result = _format_state_shares({"2": 0.3, "1": 0.7})
        assert result.index("1:") < result.index("2:")

    def test_empty_dict(self):
        assert _format_state_shares({}) is None

    def test_none_input(self):
        assert _format_state_shares(None) is None

    def test_non_dict(self):
        assert _format_state_shares("nope") is None

    def test_non_numeric_value(self):
        result = _format_state_shares({"a": "bad"})
        assert "a:bad" in result


# ---------------------------------------------------------------------------
# 7. pick_best_forecast_method
# ---------------------------------------------------------------------------
class TestPickBestForecastMethod:
    def _bt(self, results):
        return {"results": results}

    def test_single_method(self):
        bt = self._bt({"ets": {"success": True, "avg_rmse": 0.5, "successful_tests": 3}})
        name, res = pick_best_forecast_method(bt)
        assert name == "ets"

    def test_picks_lowest_rmse(self):
        bt = self._bt({
            "a": {"success": True, "avg_rmse": 1.0, "successful_tests": 1},
            "b": {"success": True, "avg_rmse": 0.5, "successful_tests": 1},
        })
        name, _ = pick_best_forecast_method(bt)
        assert name == "b"

    def test_prefers_directional_accuracy(self):
        bt = self._bt({
            "a": {"success": True, "avg_rmse": 1.0, "avg_directional_accuracy": 0.8, "successful_tests": 1},
            "b": {"success": True, "avg_rmse": 1.02, "avg_directional_accuracy": 0.9, "successful_tests": 1},
        })
        name, _ = pick_best_forecast_method(bt)
        assert name == "b"

    def test_ignores_failed(self):
        bt = self._bt({
            "good": {"success": True, "avg_rmse": 10.0, "successful_tests": 1},
            "bad": {"success": False, "avg_rmse": 0.1, "successful_tests": 0},
        })
        name, _ = pick_best_forecast_method(bt)
        assert name == "good"

    def test_empty_results(self):
        assert pick_best_forecast_method({"results": {}}) is None

    def test_no_results_key(self):
        assert pick_best_forecast_method({}) is None

    def test_none_input(self):
        assert pick_best_forecast_method(None) is None

    def test_non_dict_input(self):
        assert pick_best_forecast_method("bad") is None

    def test_nan_rmse_skipped(self):
        bt = self._bt({
            "ok": {"success": True, "avg_rmse": 1.0, "successful_tests": 1},
            "bad": {"success": True, "avg_rmse": float("nan"), "successful_tests": 1},
        })
        name, _ = pick_best_forecast_method(bt)
        assert name == "ok"

    def test_tolerance_zero(self):
        bt = self._bt({
            "a": {"success": True, "avg_rmse": 1.0, "avg_directional_accuracy": 0.5, "successful_tests": 1},
            "b": {"success": True, "avg_rmse": 1.01, "avg_directional_accuracy": 0.9, "successful_tests": 1},
        })
        name, _ = pick_best_forecast_method(bt, rmse_tolerance=0.0)
        assert name == "a"

    def test_min_directional_accuracy_filters_candidates(self):
        bt = self._bt({
            "a": {"success": True, "avg_rmse": 0.9, "avg_directional_accuracy": 0.45, "successful_tests": 5},
            "b": {"success": True, "avg_rmse": 1.4, "avg_directional_accuracy": 0.60, "successful_tests": 5},
        })
        name, _ = pick_best_forecast_method(bt, min_directional_accuracy=0.5)
        assert name == "b"

    def test_min_directional_accuracy_returns_none_when_no_qualifying_methods(self):
        bt = self._bt({
            "a": {"success": True, "avg_rmse": 0.9, "avg_directional_accuracy": 0.45, "successful_tests": 5},
            "b": {"success": True, "avg_rmse": 1.1, "avg_directional_accuracy": 0.49, "successful_tests": 5},
        })
        assert pick_best_forecast_method(bt, min_directional_accuracy=0.5) is None


# ---------------------------------------------------------------------------
# 8. summarize_barrier_grid
# ---------------------------------------------------------------------------
class TestSummarizeBarrierGrid:
    def test_with_best_and_top(self):
        grid = {
            "best": {"tp": 1.0, "sl": 0.5, "edge": 0.1, "kelly": 0.2, "ev": 0.05,
                      "prob_tp_first": 0.6, "prob_sl_first": 0.3, "prob_no_hit": 0.1,
                      "median_time_to_tp": 5, "tp_price": 1.1, "sl_price": 0.9},
            "top": [
                {"tp": 1.0, "sl": 0.5, "edge": 0.1, "kelly": 0.2, "ev": 0.05,
                 "prob_tp_first": 0.6, "prob_sl_first": 0.3, "prob_no_hit": 0.1,
                 "tp_price": 1.1, "sl_price": 0.9},
            ],
        }
        result = summarize_barrier_grid(grid)
        assert "best" in result
        assert result["best"]["tp"] == 1.0

    def test_with_results_list(self):
        grid = {
            "results": [
                {"score": 10, "tp": 1, "sl": 0.5},
                {"score": 20, "tp": 2, "sl": 1.0},
            ]
        }
        result = summarize_barrier_grid(grid, top_k=1)
        assert "top" in result
        assert len(result["top"]) == 1

    def test_empty_grid(self):
        result = summarize_barrier_grid({})
        assert result == {"note": "no grid summary"}

    def test_none_input(self):
        result = summarize_barrier_grid(None)
        assert result == {"note": "no grid summary"}

    def test_direction_preserved(self):
        grid = {
            "best": {"tp": 1, "sl": 0.5},
            "direction": "long",
        }
        result = summarize_barrier_grid(grid)
        assert result.get("direction") == "long"

    def test_top_k_limits(self):
        grid = {"top": [{"tp": i, "sl": i} for i in range(10)]}
        result = summarize_barrier_grid(grid, top_k=2)
        assert len(result["top"]) == 2

    def test_top_deduplicates_near_identical_rows(self):
        grid = {
            "top": [
                {"tp": 1.0, "sl": 0.5, "ev": 0.12, "edge": -0.02},
                {"tp": 1.000001, "sl": 0.5000004, "ev": 0.1200001, "edge": -0.0199999},
                {"tp": 1.2, "sl": 0.6, "ev": 0.08, "edge": 0.01},
            ]
        }
        result = summarize_barrier_grid(grid, top_k=5)
        assert len(result["top"]) == 2

    def test_best_flags_ev_edge_conflict(self):
        grid = {"best": {"tp": 1.0, "sl": 0.5, "ev": 0.1, "edge": -0.2}}
        result = summarize_barrier_grid(grid)
        assert result["best"]["ev_edge_conflict"] is True
        assert result["ev_edge_conflict"] is True
        assert "caution" in result

    def test_copies_optimizer_level_caution_fields(self):
        grid = {
            "best": {"tp": 1.0, "sl": 0.5, "ev": 0.1, "edge": -0.2},
            "caution": "conflict warning",
            "selection_warnings": ["w1"],
        }
        result = summarize_barrier_grid(grid)
        assert result.get("caution") == "conflict warning"
        assert result.get("selection_warnings") == ["w1"]


# ---------------------------------------------------------------------------
# 9. merge_params
# ---------------------------------------------------------------------------
class TestMergeParams:
    def test_basic_merge(self):
        assert merge_params({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_no_override(self):
        assert merge_params({"a": 1}, {"a": 99}) == {"a": 1}

    def test_with_override(self):
        assert merge_params({"a": 1}, {"a": 99}, override=True) == {"a": 99}

    def test_none_base(self):
        assert merge_params(None, {"x": 1}) == {"x": 1}

    def test_empty_extra(self):
        assert merge_params({"a": 1}, {}) == {"a": 1}


# ---------------------------------------------------------------------------
# 10. apply_market_gates
# ---------------------------------------------------------------------------
class TestApplyMarketGates:
    def test_spread_ok(self):
        section = {"spread_ticks": 2.0}
        params = {"spread_max_ticks": 5.0}
        result = apply_market_gates(section, params)
        assert result["spread_ok"] is True

    def test_spread_not_ok(self):
        section = {"spread_ticks": 10.0}
        params = {"spread_max_ticks": 5.0}
        result = apply_market_gates(section, params)
        assert result["spread_ok"] is False

    def test_pips_fallback(self):
        section = {"spread_pips": 3.0}
        params = {"spread_max_pips": 5.0}
        result = apply_market_gates(section, params)
        assert result["spread_ok"] is True

    def test_no_params(self):
        result = apply_market_gates({"spread_ticks": 1.0}, {})
        assert result == {}

    def test_no_spread_data(self):
        result = apply_market_gates({}, {"spread_max_ticks": 5.0})
        assert result == {}


class TestMarketSnapshot:
    @patch("mtdata.core.report.utils._get_pip_size", return_value=0.00001)
    def test_spread_pips_uses_true_pip_units(self, mock_pip):
        with patch(
            "mtdata.core.market_depth.market_depth_fetch",
            new=lambda *args, **kwargs: {
                "success": True,
                "type": "tick_data",
                "data": {"bid": 1.23450, "ask": 1.23465},
            },
        ):
            snap = market_snapshot("EURUSD")

        assert snap["spread_ticks"] == pytest.approx(15.0)
        assert snap["spread_pips"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 11. _extract_base_timeframe
# ---------------------------------------------------------------------------
class TestExtractBaseTimeframe:
    def test_from_meta(self):
        report = {"meta": {"timeframe": "h1"}}
        assert _extract_base_timeframe(report) == "H1"

    def test_from_context_section(self):
        report = {"sections": {"context": {"timeframe": "m15"}}}
        assert _extract_base_timeframe(report) == "M15"

    def test_meta_takes_priority(self):
        report = {"meta": {"timeframe": "h4"}, "sections": {"context": {"timeframe": "m1"}}}
        assert _extract_base_timeframe(report) == "H4"

    def test_missing(self):
        assert _extract_base_timeframe({}) is None

    def test_non_dict(self):
        assert _extract_base_timeframe("bad") is None

    def test_none_input(self):
        assert _extract_base_timeframe(None) is None


# ---------------------------------------------------------------------------
# 12. format_number
# ---------------------------------------------------------------------------
class TestFormatNumber:
    def test_uses_shared_utils_formatter(self):
        assert format_number is util_format_number

    def test_none(self):
        assert format_number(None) == "null"

    def test_bool_true(self):
        assert format_number(True) == "true"

    def test_bool_false(self):
        assert format_number(False) == "false"

    def test_int(self):
        assert format_number(42) == "42"

    def test_float(self):
        result = format_number(3.14)
        assert "3.14" in result

    def test_string_passthrough(self):
        assert format_number("hello") == "hello"


# ---------------------------------------------------------------------------
# 18. _format_signed
# ---------------------------------------------------------------------------
class TestFormatSigned:
    def test_positive(self):
        assert _format_signed(1.5) == "+1.5"

    def test_negative(self):
        assert _format_signed(-0.5) == "-0.5"

    def test_zero(self):
        assert _format_signed(0.0) == "+0.0"

    def test_none(self):
        assert _format_signed(None) == "n/a"


# ---------------------------------------------------------------------------
# 19. _format_decimal
# ---------------------------------------------------------------------------
class TestFormatDecimal:
    def test_basic(self):
        result = _format_decimal(1.23456, 2)
        assert result is not None
        assert "1.23" in result

    def test_none_input(self):
        assert _format_decimal(None) is None

    def test_non_numeric(self):
        assert _format_decimal("abc") is None

    def test_inf(self):
        assert _format_decimal(float("inf")) is None

    def test_nan(self):
        assert _format_decimal(float("nan")) is None

    def test_zero_decimals(self):
        result = _format_decimal(3.7, 0)
        assert result is not None


# ---------------------------------------------------------------------------
# 20. _format_probability
# ---------------------------------------------------------------------------
class TestFormatProbability:
    def test_half(self):
        assert _format_probability(0.5) == "50.0%"

    def test_one(self):
        assert _format_probability(1.0) == "100.0%"

    def test_zero(self):
        assert _format_probability(0.0) == "0.0%"

    def test_none(self):
        assert _format_probability(None) == "n/a"

    def test_non_numeric(self):
        assert _format_probability("bad") == "n/a"

    def test_inf(self):
        assert _format_probability(float("inf")) == "n/a"


# ---------------------------------------------------------------------------
# 21. _as_float
# ---------------------------------------------------------------------------
class TestAsFloat:
    def test_int(self):
        assert _as_float(42) == 42.0

    def test_float(self):
        assert _as_float(3.14) == 3.14

    def test_string_number(self):
        assert _as_float("2.5") == 2.5

    def test_none(self):
        assert _as_float(None) is None

    def test_non_numeric_string(self):
        assert _as_float("abc") is None

    def test_inf(self):
        assert _as_float(float("inf")) is None

    def test_neg_inf(self):
        assert _as_float(float("-inf")) is None

    def test_nan(self):
        assert _as_float(float("nan")) is None

    def test_bool_true(self):
        assert _as_float(True) == 1.0

    def test_bool_false(self):
        assert _as_float(False) == 0.0


class TestAttachMultiTimeframes:
    def test_contexts_multi_omits_trend_compact_payload(self, monkeypatch):
        snap = {
            "close": 100.0,
            "ema20": 99.5,
            "trend_compact": {"slope_atr_scores": [10], "volatility_bps": 120, "squeeze_percentile": 5},
            "trend_compact_legend": {"slope_atr_scores": "slope"},
            "trend_compact_explained": {"slope_5": 0.1},
        }

        monkeypatch.setattr(
            "mtdata.core.report.utils.context_for_tf",
            lambda *args, **kwargs: dict(snap),
        )
        monkeypatch.setattr(
            "mtdata.core.report.utils._extract_base_timeframe",
            lambda report: None,
        )

        report = {"sections": {"context": {}}}
        attach_multi_timeframes(report, "EURUSD", None, extra_timeframes=["H1"], pivot_timeframes=None)

        contexts = report["sections"]["contexts_multi"]["H1"]
        assert "trend_compact" not in contexts
        assert "trend_compact_legend" not in contexts
        assert "trend_compact_explained" not in contexts
        assert contexts["close"] == 100.0
        assert contexts["ema20"] == 99.5

    def test_trend_mtf_keeps_compact_only(self, monkeypatch):
        snap = {
            "close": 100.0,
            "ema20": 99.5,
            "rsi": 55.0,
            "trend_compact": {"slope_atr_scores": [10], "volatility_bps": 120, "squeeze_percentile": 5},
        }

        monkeypatch.setattr(
            "mtdata.core.report.utils.context_for_tf",
            lambda *args, **kwargs: dict(snap),
        )
        monkeypatch.setattr(
            "mtdata.core.report.utils._extract_base_timeframe",
            lambda report: None,
        )

        report = {"sections": {"context": {}}}
        attach_multi_timeframes(report, "EURUSD", None, extra_timeframes=["H1"], pivot_timeframes=None)

        trend_mtf = report["sections"]["context"]["trend_mtf"]["H1"]
        assert trend_mtf == {"slope_atr_scores": [10], "volatility_bps": 120, "squeeze_percentile": 5}

    def test_attach_multi_timeframes_keeps_freshness_only_error_snapshots(self, monkeypatch):
        freshness = {
            "data_freshness_seconds": 7200.0,
            "last_bar_within_policy_window": False,
        }

        monkeypatch.setattr(
            "mtdata.core.data.data_fetch_candles",
            lambda **kwargs: {
                "error": "Data remained stale",
                "details": {"diagnostics": {"freshness": dict(freshness)}},
            },
        )
        monkeypatch.setattr(
            "mtdata.core.report.utils._extract_base_timeframe",
            lambda report: None,
        )

        report = {"sections": {"context": {}}}
        attach_multi_timeframes(report, "EURUSD", None, extra_timeframes=["H1"], pivot_timeframes=None)

        assert report["sections"]["contexts_multi"]["H1"] == {
            "error": "Data remained stale",
            "freshness": freshness,
        }


class TestContextForTf:
    def test_uses_unwrapped_tool_when_wrapper_is_async(self, monkeypatch):
        rows = [
            {
                "close": 101.25,
                "EMA_20": 100.5,
                "EMA_50": 99.5,
                "RSI_14": 57.0,
                "MACD_12_26_9": 0.12,
            }
        ]

        def _raw_fetch(**kwargs):
            assert kwargs.get("__cli_raw") is True
            return {"data": rows}

        async def _wrapped_fetch(**kwargs):
            return {"error": f"wrapped call should not be used: {kwargs}"}

        _wrapped_fetch.__wrapped__ = _raw_fetch

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _wrapped_fetch)
        monkeypatch.setattr(
            "mtdata.core.report_templates.basic._compute_compact_trend",
            lambda _rows: {"slope_atr_scores": [12], "volatility_bps": 45, "squeeze_percentile": 60},
        )

        result = context_for_tf("EURUSD", "H1", None, limit=20, tail=1)
        assert result is not None
        assert result["close"] == 101.25
        assert result["EMA_20"] == 100.5
        assert result["EMA_50"] == 99.5
        assert result["RSI_14"] == 57.0
        assert result["MACD"] == 0.12
        assert result["trend_compact"] == {"slope_atr_scores": [12], "volatility_bps": 45, "squeeze_percentile": 60}

    def test_preserves_freshness_on_error_payload(self, monkeypatch):
        freshness = {
            "data_freshness_seconds": 7200.0,
            "last_bar_within_policy_window": False,
        }

        monkeypatch.setattr(
            "mtdata.core.data.data_fetch_candles",
            lambda **kwargs: {
                "error": "Data remained stale",
                "details": {"diagnostics": {"freshness": dict(freshness)}},
            },
        )

        result = context_for_tf("EURUSD", "H1", None, limit=20, tail=1)

        assert result == {
            "error": "Data remained stale",
            "freshness": freshness,
        }


