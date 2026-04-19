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
    _compact_scalar,
    _compact_table_value,
    _compact_yaml,
    _escape_yaml_string,
    _extract_base_timeframe,
    _format_decimal,
    _format_probability,
    _format_series_preview,
    _format_signed,
    _format_state_shares,
    _get_indicator_value,
    _indicator_key_variants,
    _needs_yaml_quotes,
    _render_backtest_section,
    _render_barriers_section,
    _render_context_section,
    _render_contexts_multi_section,
    _render_execution_gates_section,
    _render_forecast_conformal_section,
    _render_forecast_section,
    _render_generic_section,
    _render_market_section,
    _render_patterns_section,
    _render_pivot_multi_section,
    _render_pivot_section,
    _render_regime_section,
    _render_volatility_har_section,
    _render_volatility_section,
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
    render_enhanced_report,
    summarize_barrier_grid,
)


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
# 13. _needs_yaml_quotes
# ---------------------------------------------------------------------------
class TestNeedsYamlQuotes:
    def test_empty_string(self):
        assert _needs_yaml_quotes("") is True

    def test_leading_space(self):
        assert _needs_yaml_quotes(" hello") is True

    def test_trailing_space(self):
        assert _needs_yaml_quotes("hello ") is True

    def test_starts_with_dash(self):
        assert _needs_yaml_quotes("-item") is True

    def test_contains_colon(self):
        assert _needs_yaml_quotes("key: value") is True

    def test_contains_comma(self):
        assert _needs_yaml_quotes("a, b") is True

    def test_contains_hash(self):
        assert _needs_yaml_quotes("# comment") is True

    def test_newline(self):
        assert _needs_yaml_quotes("line\nbreak") is True

    def test_tab(self):
        assert _needs_yaml_quotes("a\tb") is True

    def test_safe_string(self):
        assert _needs_yaml_quotes("hello") is False

    def test_starts_with_bracket(self):
        assert _needs_yaml_quotes("{block}") is True

    def test_starts_with_star(self):
        assert _needs_yaml_quotes("*anchor") is True

    def test_starts_with_ampersand(self):
        assert _needs_yaml_quotes("&ref") is True


# ---------------------------------------------------------------------------
# 14. _escape_yaml_string
# ---------------------------------------------------------------------------
class TestEscapeYamlString:
    def test_plain(self):
        assert _escape_yaml_string("hello") == "hello"

    def test_quotes(self):
        assert _escape_yaml_string('say "hi"') == 'say \\"hi\\"'

    def test_backslash(self):
        assert _escape_yaml_string("a\\b") == "a\\\\b"

    def test_newline_replaced(self):
        assert "\n" not in _escape_yaml_string("a\nb")

    def test_carriage_return_replaced(self):
        assert "\r" not in _escape_yaml_string("a\rb")


# ---------------------------------------------------------------------------
# 15. _compact_scalar
# ---------------------------------------------------------------------------
class TestCompactScalar:
    def test_none(self):
        assert _compact_scalar(None) == "null"

    def test_bool_true(self):
        assert _compact_scalar(True) == "true"

    def test_bool_false(self):
        assert _compact_scalar(False) == "false"

    def test_int(self):
        assert _compact_scalar(42) == "42"

    def test_float(self):
        result = _compact_scalar(1.5)
        assert "1.5" in result

    def test_string_safe(self):
        assert _compact_scalar("hello") == "hello"

    def test_string_needs_quotes(self):
        result = _compact_scalar("key: value")
        assert result.startswith('"')
        assert result.endswith('"')

    def test_empty_string_quoted(self):
        result = _compact_scalar("")
        assert result == '""'


# ---------------------------------------------------------------------------
# 16. _compact_yaml
# ---------------------------------------------------------------------------
class TestCompactYaml:
    def test_empty_dict(self):
        assert _compact_yaml({}) == "{}"

    def test_empty_list(self):
        assert _compact_yaml([]) == "[]"

    def test_scalar(self):
        assert _compact_yaml(42) == "42"

    def test_dict_with_values(self):
        result = _compact_yaml({"a": 1, "b": 2})
        assert "a: 1" in result
        assert "b: 2" in result

    def test_nested_dict(self):
        result = _compact_yaml({"outer": {"inner": 1}})
        assert "outer:" in result
        assert "inner: 1" in result

    def test_list_of_scalars(self):
        result = _compact_yaml([1, 2, 3])
        assert "- 1" in result

    def test_list_of_dicts(self):
        result = _compact_yaml([{"a": 1}])
        assert "-" in result
        assert "a: 1" in result

    def test_indent(self):
        result = _compact_yaml({"x": 1}, indent=2)
        assert result.startswith("    ")  # 2*2 spaces

    def test_none_value(self):
        assert "null" in _compact_yaml(None)


# ---------------------------------------------------------------------------
# 17. _compact_table_value
# ---------------------------------------------------------------------------
class TestCompactTableValue:
    def test_none(self):
        assert _compact_table_value(None) == "null"

    def test_bool(self):
        assert _compact_table_value(True) == "true"

    def test_int(self):
        assert _compact_table_value(42) == "42"

    def test_string(self):
        assert _compact_table_value("hello") == "hello"

    def test_string_with_comma(self):
        result = _compact_table_value("a, b")
        assert result.startswith('"')

    def test_string_with_quote(self):
        result = _compact_table_value('say "hi"')
        assert '""' in result

    def test_newline_replaced(self):
        result = _compact_table_value("a\nb")
        assert "\n" not in result

    def test_float(self):
        result = _compact_table_value(3.14)
        assert "3.14" in result


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


# ---------------------------------------------------------------------------
# 22. render_enhanced_report
# ---------------------------------------------------------------------------
class TestRenderEnhancedReport:
    def test_non_dict(self):
        assert "error" in render_enhanced_report("bad")

    def test_empty_report(self):
        assert render_enhanced_report({}) == ""

    def test_empty_sections(self):
        assert render_enhanced_report({"sections": {}}) == ""

    def test_with_forecast_section(self):
        report = {"sections": {"forecast": {"method": "ets", "forecast_price": [1.0, 2.0]}}}
        result = render_enhanced_report(report)
        assert "Forecast" in result
        assert "ets" in result

    def test_summary_block_renders_before_section_status(self):
        report = {
            "summary": ["first takeaway", "second takeaway"],
            "sections_status": {
                "summary": {"ok": 1, "partial": 0, "error": 0},
                "sections": {"forecast": "ok"},
            },
            "sections": {"forecast": {"method": "ets"}},
        }
        result = render_enhanced_report(report)
        assert "## Summary" in result
        assert result.index("## Summary") < result.index("## Section Status")
        assert "- first takeaway" in result

    def test_generic_section_rendered(self):
        report = {"sections": {"custom_thing": {"key1": "val1"}}}
        result = render_enhanced_report(report)
        assert "Custom Thing" in result
        assert "key1" in result

    def test_error_only_section_skipped(self):
        report = {"sections": {"bad_section": {"error": "something"}}}
        result = render_enhanced_report(report)
        assert "Bad Section" not in result

    def test_multiple_sections(self):
        report = {"sections": {
            "market": {"bid": 1.0, "ask": 1.1, "spread": 0.1},
            "forecast": {"method": "arima"},
        }}
        result = render_enhanced_report(report)
        assert "Market Snapshot" in result
        assert "Forecast" in result

    def test_none_payload_skipped(self):
        report = {"sections": {"forecast": None}}
        result = render_enhanced_report(report)
        assert result == ""


# ---------------------------------------------------------------------------
# 23. _render_context_section
# ---------------------------------------------------------------------------
class TestRenderContextSection:
    def test_non_dict(self):
        assert _render_context_section("bad") == []

    def test_error_key(self):
        lines = _render_context_section({"error": "fail"})
        assert any("error" in line.lower() for line in lines)

    def test_with_indicators(self):
        data = {
            "timeframe": "H1",
            "last_snapshot": {
                "close": 1.2345,
                "EMA_20": 1.2300,
                "EMA_50": 1.2200,
                "RSI_14": 65.0,
            },
        }
        lines = _render_context_section(data)
        text = "\n".join(lines)
        assert "Market Context" in text
        assert "H1" in text

    def test_rsi_overbought(self):
        data = {"last_snapshot": {"RSI_14": 75.0}}
        text = "\n".join(_render_context_section(data))
        assert "overbought" in text

    def test_rsi_oversold(self):
        data = {"last_snapshot": {"RSI_14": 25.0}}
        text = "\n".join(_render_context_section(data))
        assert "oversold" in text

    def test_rsi_neutral(self):
        data = {"last_snapshot": {"RSI_14": 50.0}}
        text = "\n".join(_render_context_section(data))
        assert "neutral" in text

    def test_with_trend_compact(self):
        data = {
            "trend_compact": {
                "s": [100, 200, 300],
                "r": [80, 90, 95],
                "v": 150,
                "q": 60,
                "g": 1,
                "h": 5,
                "l": 10,
            }
        }
        lines = _render_context_section(data)
        text = "\n".join(lines)
        assert "Slope" in text
        assert "uptrend" in text

    def test_empty_dict(self):
        lines = _render_context_section({})
        assert lines == ["## Market Context"]


# ---------------------------------------------------------------------------
# 24. _render_contexts_multi_section
# ---------------------------------------------------------------------------
class TestRenderContextsMultiSection:
    def test_non_dict(self):
        assert _render_contexts_multi_section("bad") == []

    def test_basic(self):
        data = {
            "H4": {"close": 1.23, "ema20": 1.22, "RSI_14": 55},
        }
        lines = _render_contexts_multi_section(data)
        text = "\n".join(lines)
        assert "Multi-Timeframe" in text
        assert "H4" in text

    def test_empty_data_rows_skipped(self):
        data = {"H4": {"no_useful_keys": True}}
        lines = _render_contexts_multi_section(data)
        # all cells are None, row should be dropped
        assert lines == []

    def test_with_trend_compact(self):
        data = {
            "D1": {
                "close": 100.0,
                "trend_compact": {"s": [50], "v": 200},
            }
        }
        lines = _render_contexts_multi_section(data)
        text = "\n".join(lines)
        assert "D1" in text


class TestAttachMultiTimeframes:
    def test_contexts_multi_omits_trend_compact_payload(self, monkeypatch):
        snap = {
            "close": 100.0,
            "ema20": 99.5,
            "trend_compact": {"s": [10], "v": 120, "q": 5},
            "trend_compact_legend": {"s": "slope"},
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
            "trend_compact": {"s": [10], "v": 120, "q": 5},
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
        assert trend_mtf == {"s": [10], "v": 120, "q": 5}


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
            lambda _rows: {"s": [12], "v": 45, "q": 60},
        )

        result = context_for_tf("EURUSD", "H1", None, limit=20, tail=1)
        assert result is not None
        assert result["close"] == 101.25
        assert result["EMA_20"] == 100.5
        assert result["EMA_50"] == 99.5
        assert result["RSI_14"] == 57.0
        assert result["MACD"] == 0.12
        assert result["trend_compact"] == {"s": [12], "v": 45, "q": 60}


# ---------------------------------------------------------------------------
# 25. _render_pivot_section
# ---------------------------------------------------------------------------
class TestRenderPivotSection:
    def test_non_dict(self):
        assert _render_pivot_section("bad") == []

    def test_no_levels(self):
        assert _render_pivot_section({"levels": []}) == []

    def test_basic(self):
        data = {
            "timeframe": "D1",
            "period": {"start": "2024-01-01", "end": "2024-01-02"},
            "calculation_basis": {
                "source_bar": "last completed D1 bar",
                "session_boundary": "MT5 broker/session calendar",
                "display_timezone": "UTC",
            },
            "methods": [{"method": "classic"}],
            "levels": [
                {"level": "R1", "classic": 1.25},
                {"level": "S1", "classic": 1.20},
            ],
        }
        lines = _render_pivot_section(data)
        text = "\n".join(lines)
        assert "Pivot Levels" in text
        assert "D1" in text
        assert "R1" in text
        assert "Context:" in text
        assert "session=MT5 broker/session calendar" in text
        assert "display_tz=UTC" in text

    def test_infers_methods_from_keys(self):
        data = {
            "levels": [{"level": "PP", "fibonacci": 1.23}],
        }
        lines = _render_pivot_section(data)
        text = "\n".join(lines)
        assert "Fibonacci" in text


# ---------------------------------------------------------------------------
# 26. _render_pivot_multi_section
# ---------------------------------------------------------------------------
class TestRenderPivotMultiSection:
    def test_non_dict(self):
        assert _render_pivot_multi_section("bad") == []

    def test_basic(self):
        data = {
            "H4": {
                "calculation_basis": {
                    "session_boundary": "MT5 broker/session calendar",
                    "display_timezone": "UTC",
                },
                "levels": [{"level": "R1", "classic": 1.25}],
            },
        }
        lines = _render_pivot_multi_section(data)
        text = "\n".join(lines)
        assert "Multi-Timeframe Pivots" in text
        assert "H4" in text
        assert "Context:" in text

    def test_skips_base_timeframe(self):
        data = {
            "__base_timeframe__": "H4",
            "H4": {"levels": [{"level": "PP", "classic": 1.0}]},
            "D1": {"levels": [{"level": "PP", "classic": 2.0}]},
        }
        lines = _render_pivot_multi_section(data)
        text = "\n".join(lines)
        assert "D1" in text
        # H4 should be skipped as it matches base
        found_h4 = any("### H4" in line for line in lines)
        assert not found_h4

    def test_empty_levels(self):
        data = {"H1": {"levels": []}}
        lines = _render_pivot_multi_section(data)
        assert lines == ["## Multi-Timeframe Pivots"]

    def test_renders_all_configured_methods(self):
        data = {
            "H4": {
                "methods": [{"method": "classic"}, {"method": "fibonacci"}],
                "levels": [{"level": "R1", "classic": 1.25, "fibonacci": 1.30}],
            },
        }

        lines = _render_pivot_multi_section(data)
        text = "\n".join(lines)

        assert "Classic" in text
        assert "Fibonacci" in text
        assert "1.3" in text


# ---------------------------------------------------------------------------
# 27. _render_volatility_section
# ---------------------------------------------------------------------------
class TestRenderVolatilitySection:
    def test_non_dict(self):
        assert _render_volatility_section("bad") == []

    def test_basic(self):
        data = {
            "methods": ["garch", "ewma"],
            "matrix": [
                {"horizon": 5, "garch": 0.01, "ewma": 0.012, "avg": 0.011},
            ],
        }
        lines = _render_volatility_section(data)
        text = "\n".join(lines)
        assert "Volatility Snapshot" in text
        assert "GARCH" in text

    def test_empty_matrix(self):
        assert _render_volatility_section({"matrix": [], "methods": ["a"]}) == []

    def test_auto_detect_methods(self):
        data = {
            "matrix": [{"horizon": 1, "garch": 0.01}],
        }
        lines = _render_volatility_section(data)
        text = "\n".join(lines)
        assert "GARCH" in text


# ---------------------------------------------------------------------------
# 28. _render_forecast_section
# ---------------------------------------------------------------------------
class TestRenderForecastSection:
    def test_non_dict(self):
        assert _render_forecast_section("bad") == []

    def test_basic(self):
        data = {"method": "arima", "forecast_price": [1.0, 1.1, 1.2]}
        lines = _render_forecast_section(data)
        text = "\n".join(lines)
        assert "Forecast" in text
        assert "arima" in text

    def test_with_interval(self):
        data = {"lower_price": 0.9, "upper_price": 1.1}
        lines = _render_forecast_section(data)
        text = "\n".join(lines)
        assert "Interval" in text

    def test_return_mode(self):
        data = {"quantity": "return", "lower_return": [0.01], "upper_return": [0.02]}
        lines = _render_forecast_section(data)
        text = "\n".join(lines)
        assert "Return interval" in text

    def test_with_trend(self):
        data = {"trend": "bullish"}
        lines = _render_forecast_section(data)
        text = "\n".join(lines)
        assert "bullish" in text

    def test_with_ci_alpha(self):
        data = {"ci_alpha": 0.05}
        lines = _render_forecast_section(data)
        text = "\n".join(lines)
        assert "CI alpha" in text

    def test_scalar_forecast_price(self):
        data = {"forecast_price": 1.2345}
        lines = _render_forecast_section(data)
        text = "\n".join(lines)
        assert "Forecast price" in text

    def test_with_forecast_timing_context(self):
        data = {
            "method": "theta",
            "last_observation_epoch": 1740948000.0,
            "forecast_start_epoch": 1740951600.0,
            "forecast_anchor": "next_timeframe_bar_after_last_observation",
            "forecast_start_gap_bars": 1.0,
            "forecast_step_seconds": 3600,
        }
        lines = _render_forecast_section(data)
        text = "\n".join(lines)
        assert "Last observation" in text
        assert "Forecast start" in text
        assert "Forecast anchor" in text
        assert "Forecast start gap (bars)" in text
        assert "Forecast step seconds" in text


# ---------------------------------------------------------------------------
# 29. _render_barriers_section
# ---------------------------------------------------------------------------
class TestRenderBarriersSection:
    def test_non_dict(self):
        assert _render_barriers_section("bad") == []

    def test_single_direction(self):
        data = {
            "direction": "long",
            "best": {
                "tp": 1.0, "sl": 0.5, "tp_price": 1.1, "sl_price": 0.9,
                "edge": 0.05, "kelly": 0.1, "ev": 0.03,
                "prob_tp_first": 0.6, "prob_sl_first": 0.3, "prob_no_hit": 0.1,
            },
        }
        lines = _render_barriers_section(data)
        text = "\n".join(lines)
        assert "Barrier Analytics" in text
        assert "long" in text

    def test_long_short(self):
        best = {
            "tp": 1.0, "sl": 0.5, "tp_price": 1.1, "sl_price": 0.9,
            "edge": 0.05, "kelly": 0.1, "ev": 0.03,
            "prob_tp_first": 0.6, "prob_sl_first": 0.3, "prob_no_hit": 0.1,
        }
        data = {"long": {"best": best}, "short": {"best": best}}
        lines = _render_barriers_section(data)
        text = "\n".join(lines)
        assert "Barrier Analytics" in text
        assert "long" in text
        assert "short" in text

    def test_note_is_rendered(self):
        data = {
            "best": {
                "tp": 1.0, "sl": 0.5, "edge": 0.1, "kelly": 0.2, "ev": 0.05,
                "prob_tp_first": 0.6, "prob_sl_first": 0.3, "prob_no_hit": 0.1,
            },
            "note": "Independent run note",
        }
        lines = _render_barriers_section(data)
        text = "\n".join(lines)
        assert "Note: Independent run note" in text

    def test_negative_edge_warning(self):
        data = {
            "best": {
                "tp": 1.0, "sl": 0.5, "edge": -0.1, "kelly": 0.0, "ev": -0.05,
                "prob_tp_first": 0.4, "prob_sl_first": 0.5, "prob_no_hit": 0.1,
            },
        }
        lines = _render_barriers_section(data)
        text = "\n".join(lines)
        assert "Warning" in text

    def test_ev_edge_conflict_caution(self):
        data = {
            "best": {
                "tp": 1.0,
                "sl": 0.5,
                "edge": -0.2,
                "kelly": 0.05,
                "ev": 0.1,
                "prob_tp_first": 0.4,
                "prob_sl_first": 0.5,
                "prob_no_hit": 0.1,
            },
        }
        lines = _render_barriers_section(data)
        text = "\n".join(lines)
        assert "CAUTION" in text

    def test_with_top_runners(self):
        data = {
            "best": {"tp": 1.0, "sl": 0.5, "edge": 0.1, "kelly": 0.2, "ev": 0.05,
                      "prob_tp_first": 0.6, "prob_sl_first": 0.3, "prob_no_hit": 0.1},
            "top": [
                {"tp": 1.0, "sl": 0.5, "edge": 0.1, "kelly": 0.2, "ev": 0.05,
                 "prob_tp_first": 0.6, "prob_sl_first": 0.3, "prob_no_hit": 0.1},
            ],
        }
        lines = _render_barriers_section(data)
        text = "\n".join(lines)
        assert "Barrier Analytics" in text

    def test_pips_mode(self):
        best = {"tp": 50, "sl": 25, "edge": 0.1, "kelly": 0.1, "ev": 0.05,
                "prob_tp_first": 0.6, "prob_sl_first": 0.3, "prob_no_hit": 0.1}
        data = {"mode": "pips", "long": {"best": best}}
        lines = _render_barriers_section(data)
        text = "\n".join(lines)
        assert "ticks" in text

    def test_empty_best_returns_empty(self):
        data = {"long": {"best": None}, "short": {}}
        lines = _render_barriers_section(data)
        assert lines == []


# ---------------------------------------------------------------------------
# 30. _render_market_section
# ---------------------------------------------------------------------------
class TestRenderMarketSection:
    def test_non_dict(self):
        assert _render_market_section("bad") == []

    def test_basic(self):
        data = {"bid": 1.0, "ask": 1.1, "spread": 0.1}
        lines = _render_market_section(data)
        text = "\n".join(lines)
        assert "Market Snapshot" in text
        assert "Bid" in text

    def test_error(self):
        data = {"error": "timeout"}
        lines = _render_market_section(data)
        text = "\n".join(lines)
        assert "Error" in text

    def test_with_tick_size(self):
        data = {"bid": 1.0, "tick_size": 0.0001}
        lines = _render_market_section(data)
        text = "\n".join(lines)
        assert "Tick size" in text

    def test_with_spread_pips(self):
        data = {"bid": 1.0, "spread_pips": 2.5}
        lines = _render_market_section(data)
        text = "\n".join(lines)
        assert "Spread (pips)" in text

    def test_with_depth(self):
        data = {"bid": 1.0, "depth": {"total_buy": 100, "total_sell": 200}}
        lines = _render_market_section(data)
        text = "\n".join(lines)
        assert "DOM volume" in text

    def test_empty_data(self):
        assert _render_market_section({}) == []


# ---------------------------------------------------------------------------
# 31. _render_backtest_section
# ---------------------------------------------------------------------------
class TestRenderBacktestSection:
    def test_non_dict(self):
        assert _render_backtest_section("bad") == []

    def test_no_ranking(self):
        assert _render_backtest_section({"ranking": []}) == []

    def test_basic(self):
        data = {"ranking": [
            {"method": "ets", "avg_rmse": 0.01, "avg_mae": 0.005,
             "avg_directional_accuracy": 0.7, "successful_tests": 5},
        ]}
        lines = _render_backtest_section(data)
        text = "\n".join(lines)
        assert "Backtest Ranking" in text
        assert "ets" in text

    def test_multiple_methods(self):
        data = {"ranking": [
            {"method": "ets", "avg_rmse": 0.01, "avg_mae": 0.005,
             "avg_directional_accuracy": 0.7, "successful_tests": 5},
            {"method": "arima", "avg_rmse": 0.02, "avg_mae": 0.01,
             "avg_directional_accuracy": 0.6, "successful_tests": 3},
        ]}
        lines = _render_backtest_section(data)
        text = "\n".join(lines)
        assert "ets" in text
        assert "arima" in text


# ---------------------------------------------------------------------------
# 32. _render_patterns_section
# ---------------------------------------------------------------------------
class TestRenderPatternsSection:
    def test_non_dict(self):
        assert _render_patterns_section("bad") == []

    def test_no_recent(self):
        assert _render_patterns_section({"recent": []}) == []

    def test_basic(self):
        data = {"recent": [
            {"pattern": "Doji", "direction": "neutral", "time": "2024-01-01 12:00"},
        ]}
        lines = _render_patterns_section(data)
        text = "\n".join(lines)
        assert "Recent Patterns" in text
        assert "Doji" in text

    def test_alternative_keys(self):
        data = {"recent": [
            {"Pattern": "Hammer", "Direction": "bullish", "Time": "2024-01-01"},
        ]}
        lines = _render_patterns_section(data)
        text = "\n".join(lines)
        assert "Hammer" in text

    def test_name_key(self):
        data = {"recent": [{"name": "Engulfing"}]}
        lines = _render_patterns_section(data)
        text = "\n".join(lines)
        assert "Engulfing" in text


# ---------------------------------------------------------------------------
# 33. _render_regime_section
# ---------------------------------------------------------------------------
class TestRenderRegimeSection:
    def test_non_dict(self):
        assert _render_regime_section("bad") == []

    def test_bocpd(self):
        data = {"bocpd": {"summary": {
            "last_cp_prob": 0.1, "max_cp_prob": 0.5,
            "mean_cp_prob": 0.2, "change_points_count": 3,
        }}}
        lines = _render_regime_section(data)
        text = "\n".join(lines)
        assert "Regime Signals" in text
        assert "BOCPD" in text
        assert "change_points=3" in text

    def test_hmm(self):
        data = {"hmm": {"summary": {
            "last_state": 2,
            "state_shares": {"0": 0.3, "1": 0.7},
            "state_order_by_sigma": {"0": "low", "1": "high"},
        }}}
        lines = _render_regime_section(data)
        text = "\n".join(lines)
        assert "HMM" in text
        assert "last_state=2" in text

    def test_bocpd_error(self):
        data = {"bocpd": {"error": "convergence failed"}}
        lines = _render_regime_section(data)
        text = "\n".join(lines)
        assert "BOCPD error" in text

    def test_empty_section(self):
        lines = _render_regime_section({})
        assert lines == []

    def test_hmm_string_summary(self):
        data = {"hmm": {"summary": "some raw text"}}
        lines = _render_regime_section(data)
        text = "\n".join(lines)
        assert "some raw text" in text


# ---------------------------------------------------------------------------
# 34. _render_execution_gates_section
# ---------------------------------------------------------------------------
class TestRenderExecutionGatesSection:
    def test_non_dict(self):
        assert _render_execution_gates_section("bad") == []

    def test_empty(self):
        assert _render_execution_gates_section({}) == []

    def test_bool_values(self):
        data = {"spread_ok": True}
        lines = _render_execution_gates_section(data)
        text = "\n".join(lines)
        assert "Execution Gates" in text
        assert "yes" in text

    def test_numeric_values(self):
        data = {"spread_ticks": 2.5}
        lines = _render_execution_gates_section(data)
        text = "\n".join(lines)
        assert "Spread Ticks" in text

    def test_false_bool(self):
        data = {"spread_ok": False}
        lines = _render_execution_gates_section(data)
        text = "\n".join(lines)
        assert "no" in text


# ---------------------------------------------------------------------------
# 35. _render_volatility_har_section
# ---------------------------------------------------------------------------
class TestRenderVolatilityHarSection:
    def test_non_dict(self):
        assert _render_volatility_har_section("bad") == []

    def test_basic(self):
        data = {"sigma_bar_return": 0.01, "horizon_sigma_return": 0.05}
        lines = _render_volatility_har_section(data)
        text = "\n".join(lines)
        assert "HAR-RV" in text
        assert "Bar sigma" in text
        assert "Horizon sigma" in text

    def test_partial(self):
        data = {"sigma_bar_return": 0.01}
        lines = _render_volatility_har_section(data)
        text = "\n".join(lines)
        assert "Bar sigma" in text
        assert "Horizon" not in text

    def test_empty(self):
        lines = _render_volatility_har_section({})
        assert lines == ["## HAR-RV Volatility"]


# ---------------------------------------------------------------------------
# 36. _render_forecast_conformal_section
# ---------------------------------------------------------------------------
class TestRenderForecastConformalSection:
    def test_non_dict(self):
        assert _render_forecast_conformal_section("bad") == []

    def test_basic(self):
        data = {
            "method": "conformal",
            "lower_price": 1.0,
            "upper_price": 1.5,
            "ci_alpha": 0.1,
        }
        lines = _render_forecast_conformal_section(data)
        text = "\n".join(lines)
        assert "Conformal Intervals" in text
        assert "conformal" in text
        assert "CI alpha" in text

    def test_with_quantiles(self):
        data = {"per_step_q": [0.1, 0.2, 0.3]}
        lines = _render_forecast_conformal_section(data)
        text = "\n".join(lines)
        assert "q1=" in text
        assert "q2=" in text
        assert "q3=" in text

    def test_return_mode(self):
        data = {"quantity": "return", "lower_return": 0.01, "upper_return": 0.02}
        lines = _render_forecast_conformal_section(data)
        text = "\n".join(lines)
        assert "Return interval" in text

    def test_list_intervals(self):
        data = {"lower_price": [1.0, 1.1], "upper_price": [1.5, 1.6]}
        lines = _render_forecast_conformal_section(data)
        text = "\n".join(lines)
        assert "Interval" in text

    def test_quantile_null_values(self):
        data = {"per_step_q": [None, float("nan"), "inf"]}
        lines = _render_forecast_conformal_section(data)
        text = "\n".join(lines)
        assert "null" in text

    def test_quantile_truncated_to_five(self):
        data = {"per_step_q": list(range(10))}
        lines = _render_forecast_conformal_section(data)
        text = "\n".join(lines)
        assert "q5=" in text
        assert "q6=" not in text


# ---------------------------------------------------------------------------
# 37. _render_generic_section
# ---------------------------------------------------------------------------
class TestRenderGenericSection:
    def test_empty_payload(self):
        assert _render_generic_section("test", {}) == []
        assert _render_generic_section("test", None) == []
        assert _render_generic_section("test", []) == []

    def test_dict_payload(self):
        lines = _render_generic_section("my_data", {"key1": "val1", "key2": 42})
        text = "\n".join(lines)
        assert "My Data" in text
        assert "key1" in text

    def test_list_payload(self):
        lines = _render_generic_section("items", ["a", "b", "c"])
        text = "\n".join(lines)
        assert "Items" in text
        assert "- a" in text

    def test_long_list_truncated(self):
        lines = _render_generic_section("big", list(range(25)))
        text = "\n".join(lines)
        assert "..." in text

    def test_scalar_payload(self):
        lines = _render_generic_section("note", "some text")
        text = "\n".join(lines)
        assert "Note" in text
        assert "some text" in text

    def test_nested_dict_value(self):
        lines = _render_generic_section("data", {"nested": {"a": 1, "b": 2}})
        text = "\n".join(lines)
        assert "nested" in text
        assert "a=1" in text

    def test_list_value_in_dict(self):
        lines = _render_generic_section("data", {"items": [1, 2, 3, 4, 5, 6]})
        text = "\n".join(lines)
        assert "..." in text

    def test_title_formatting(self):
        lines = _render_generic_section("my_fancy_name", {"x": 1})
        assert lines[0] == "## My Fancy Name"


# ---------------------------------------------------------------------------
# Integration: full report round-trip
# ---------------------------------------------------------------------------
class TestFullReportRoundTrip:
    def _make_full_report(self):
        return {
            "meta": {"symbol": "EURUSD", "timeframe": "H1"},
            "sections": {
                "context": {
                    "timeframe": "H1",
                    "last_snapshot": {
                        "close": 1.08500,
                        "EMA_20": 1.08400,
                        "EMA_50": 1.08300,
                        "RSI_14": 55.0,
                    },
                },
                "forecast": {
                    "method": "ets",
                    "forecast_price": [1.085, 1.086, 1.087],
                    "trend": "bullish",
                },
                "barriers": {
                    "direction": "long",
                    "best": {
                        "tp": 0.5, "sl": 0.3, "tp_price": 1.09, "sl_price": 1.08,
                        "edge": 0.12, "kelly": 0.15, "ev": 0.06,
                        "prob_tp_first": 0.55, "prob_sl_first": 0.35, "prob_no_hit": 0.1,
                    },
                },
                "backtest": {
                    "ranking": [
                        {"method": "ets", "avg_rmse": 0.001, "avg_mae": 0.0005,
                         "avg_directional_accuracy": 0.72, "successful_tests": 10},
                    ],
                },
                "market": {
                    "bid": 1.085, "ask": 1.0851, "spread": 0.0001,
                    "tick_size": 0.00001, "spread_ticks": 10.0,
                },
                "execution_gates": {"spread_ok": True, "spread_ticks": 10.0},
                "volatility": {
                    "methods": ["garch"],
                    "matrix": [{"horizon": 5, "garch": 0.01}],
                },
                "patterns": {
                    "recent": [{"pattern": "Doji", "direction": "neutral", "time": "2024-01-01"}],
                },
                "regime": {
                    "bocpd": {"summary": {"last_cp_prob": 0.05, "change_points_count": 1}},
                },
                "pivot": {
                    "timeframe": "D1",
                    "levels": [{"level": "PP", "classic": 1.085}],
                    "methods": [{"method": "classic"}],
                },
                "volatility_har_rv": {
                    "sigma_bar_return": 0.005,
                    "horizon_sigma_return": 0.02,
                },
                "forecast_conformal": {
                    "method": "conformal",
                    "lower_price": 1.08,
                    "upper_price": 1.09,
                    "ci_alpha": 0.1,
                },
            },
        }

    def test_full_report_renders(self):
        report = self._make_full_report()
        result = render_enhanced_report(report)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_all_sections_present(self):
        report = self._make_full_report()
        result = render_enhanced_report(report)
        for heading in [
            "Market Context", "Forecast", "Barrier Analytics",
            "Market Snapshot", "Backtest Ranking", "Recent Patterns",
            "Regime Signals", "Pivot Levels", "Volatility Snapshot",
            "Execution Gates", "HAR-RV", "Conformal Intervals",
        ]:
            assert heading in result, f"Missing section: {heading}"

    def test_ends_with_newline(self):
        report = self._make_full_report()
        result = render_enhanced_report(report)
        assert result.endswith("\n")

    def test_no_trailing_whitespace_on_lines(self):
        report = self._make_full_report()
        result = render_enhanced_report(report)
        for line in result.split("\n"):
            assert line == line.rstrip(), f"Trailing whitespace: {line!r}"

    def test_section_status_block_renders_when_present(self):
        report = self._make_full_report()
        report["sections_status"] = {
            "summary": {"ok": 3, "partial": 1, "error": 0},
            "sections": {"context": "ok", "forecast": "ok", "barriers": "partial"},
        }
        result = render_enhanced_report(report)
        assert "Section Status" in result
        assert "Partial" in result
        assert "barriers" in result
