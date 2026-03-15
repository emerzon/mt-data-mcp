"""Tests for mtdata.core.patterns — pattern detection helpers and tool wrappers."""

import copy
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest
from mtdata.core.patterns_requests import PatternsDetectRequest


# ---------------------------------------------------------------------------
# Helpers to build mock data
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n: int = 200, *, with_time: bool = True, with_volume: bool = True) -> pd.DataFrame:
    """Build a synthetic OHLCV dataframe."""
    rng = np.random.RandomState(42)
    base = 1.1000 + np.cumsum(rng.randn(n) * 0.0005)
    data: Dict[str, Any] = {
        "open": base,
        "high": base + rng.uniform(0.0001, 0.001, n),
        "low": base - rng.uniform(0.0001, 0.001, n),
        "close": base + rng.randn(n) * 0.0002,
    }
    if with_time:
        start = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
        data["time"] = np.arange(start, start + n * 3600, 3600)[:n]
    if with_volume:
        data["tick_volume"] = rng.randint(100, 5000, n)
    return pd.DataFrame(data)


def _make_rates_array(n: int = 200) -> np.ndarray:
    """Simulate the structured array returned by MT5 copy_rates_from."""
    df = _make_ohlcv_df(n, with_time=True, with_volume=True)
    return df.to_records(index=False)


def _mock_pattern_result(**overrides):
    """Build a SimpleNamespace mimicking ClassicPatternResult / ElliottWaveResult."""
    defaults = dict(
        name="Triangle",
        wave_type="Impulse",
        status="forming",
        confidence=0.85,
        start_index=10,
        end_index=50,
        start_time=1704067200.0,
        end_time=1704110400.0,
        details={"some_key": 1.23456789012},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── _round_value ──────────────────────────────────────────────────────────

class TestRoundValue:

    def _call(self, x):
        from mtdata.core.patterns import _round_value
        return _round_value(x)

    def test_rounds_float(self):
        assert self._call(1.123456789012) == pytest.approx(1.12345679, abs=1e-8)

    def test_rounds_int(self):
        assert self._call(5) == 5.0

    def test_rounds_numpy_float(self):
        assert self._call(np.float64(2.999999999)) == pytest.approx(3.0)

    def test_non_numeric_passthrough(self):
        assert self._call("hello") == "hello"

    def test_none_passthrough(self):
        assert self._call(None) is None


# ── _normalize_engine_name ────────────────────────────────────────────────

class TestNormalizeEngineName:

    def _call(self, value):
        from mtdata.core.patterns import _normalize_engine_name
        return _normalize_engine_name(value)

    def test_lowercase(self):
        assert self._call("Native") == "native"

    def test_strip(self):
        assert self._call("  native  ") == "native"

    def test_hyphens_to_underscores(self):
        assert self._call("stock-pattern") == "stock_pattern"

    def test_none(self):
        assert self._call(None) == ""

    def test_empty(self):
        assert self._call("") == ""


# ── _parse_engine_list ────────────────────────────────────────────────────

class TestParseEngineList:

    def _call(self, value):
        from mtdata.core.patterns import _parse_engine_list
        return _parse_engine_list(value)

    def test_none(self):
        assert self._call(None) == []

    def test_single_string(self):
        assert self._call("native") == ["native"]

    def test_comma_separated(self):
        assert self._call("native,stock_pattern") == ["native", "stock_pattern"]

    def test_semicolon_separated(self):
        assert self._call("native;stock_pattern") == ["native", "stock_pattern"]

    def test_list_input(self):
        assert self._call(["native", "stock_pattern"]) == ["native", "stock_pattern"]

    def test_tuple_input(self):
        assert self._call(("native",)) == ["native"]

    def test_set_input(self):
        result = self._call({"native"})
        assert result == ["native"]

    def test_whitespace_trimming(self):
        assert self._call(" native , stock_pattern ") == ["native", "stock_pattern"]

    def test_scalar_fallback(self):
        assert self._call(42) == ["42"]


# ── _select_classic_engines ───────────────────────────────────────────────

class TestSelectClassicEngines:

    def _call(self, engine, ensemble):
        from mtdata.core.patterns import _select_classic_engines
        return _select_classic_engines(engine, ensemble)

    def test_default_native(self):
        engines, invalid = self._call("", False)
        assert "native" in engines
        assert invalid == []

    def test_explicit_native(self):
        engines, invalid = self._call("native", False)
        assert engines == ["native"]
        assert invalid == []

    def test_invalid_engine(self):
        engines, invalid = self._call("nonexistent_engine_xyz", False)
        assert "nonexistent_engine_xyz" in invalid

    def test_ensemble_expands(self):
        engines, _ = self._call("native", True)
        assert len(engines) >= 1
        assert engines[0] == "native"

    def test_ensemble_adds_native(self):
        engines, _ = self._call("stock_pattern", True)
        assert "native" in engines

    def test_dedup(self):
        engines, _ = self._call("native,native", False)
        assert engines.count("native") == 1


# ── _to_jsonable ──────────────────────────────────────────────────────────

class TestToJsonable:

    def _call(self, value):
        from mtdata.core.patterns import _to_jsonable
        return _to_jsonable(value)

    def test_numpy_float(self):
        result = self._call(np.float64(1.5))
        assert result == 1.5 and isinstance(result, float)

    def test_numpy_int(self):
        result = self._call(np.int64(42))
        assert result == 42 and isinstance(result, int)

    def test_pd_timestamp(self):
        ts = pd.Timestamp("2024-01-01 12:30")
        assert self._call(ts) == "2024-01-01 12:30"

    def test_dict(self):
        result = self._call({"a": np.float64(1.5)})
        assert result == {"a": 1.5}

    def test_list(self):
        result = self._call([np.int64(1), np.int64(2)])
        assert result == [1, 2]

    def test_tuple(self):
        result = self._call((np.int64(1),))
        assert result == [1]

    def test_set(self):
        result = self._call({np.int64(1)})
        assert isinstance(result, list)

    def test_plain_value(self):
        assert self._call("hello") == "hello"
        assert self._call(42) == 42


# ── _timestamp_to_label ──────────────────────────────────────────────────

class TestTimestampToLabel:

    def _call(self, ts):
        from mtdata.core.patterns import _timestamp_to_label
        return _timestamp_to_label(ts)

    def test_pd_timestamp(self):
        ts = pd.Timestamp("2024-06-15 09:30")
        assert self._call(ts) == "2024-06-15 09:30"

    def test_non_timestamp_returns_none(self):
        assert self._call(12345) is None

    def test_none_returns_none(self):
        assert self._call(None) is None

    def test_string_returns_none(self):
        assert self._call("2024-01-01") is None


# ── _to_float_safe ───────────────────────────────────────────────────────

class TestToFloatSafe:

    def _call(self, value, default=0.6):
        from mtdata.core.patterns import _to_float_safe
        return _to_float_safe(value, default=default)

    def test_valid_float(self):
        assert self._call(1.5) == 1.5

    def test_valid_int(self):
        assert self._call(3) == 3.0

    def test_nan_returns_default(self):
        assert self._call(float("nan")) == 0.6

    def test_inf_returns_default(self):
        assert self._call(float("inf")) == 0.6

    def test_string_returns_default(self):
        assert self._call("abc") == 0.6

    def test_none_returns_default(self):
        assert self._call(None) == 0.6

    def test_custom_default(self):
        assert self._call("bad", default=0.9) == 0.9


# ── _infer_stock_pattern_confidence ──────────────────────────────────────

class TestInferStockPatternConfidence:

    def _call(self, row):
        from mtdata.core.patterns import _infer_stock_pattern_confidence
        return _infer_stock_pattern_confidence(row)

    def test_explicit_confidence(self):
        assert self._call({"confidence": 0.75}) == pytest.approx(0.75)

    def test_confidence_clamped_high(self):
        assert self._call({"confidence": 1.5}) == pytest.approx(1.0)

    def test_confidence_clamped_low(self):
        assert self._call({"confidence": -0.5}) == pytest.approx(0.0)

    def test_touches_based(self):
        result = self._call({"touches": 5})
        assert 0.35 <= result <= 0.95

    def test_no_data_default(self):
        assert self._call({}) == 0.6


# ── _map_stock_pattern_name ──────────────────────────────────────────────

class TestMapStockPatternName:

    def _call(self, row):
        from mtdata.core.patterns import _map_stock_pattern_name
        return _map_stock_pattern_name(row)

    def test_trng_with_alt(self):
        assert self._call({"pattern": "TRNG", "alt_name": "Ascending"}) == "Ascending Triangle"

    def test_dtop(self):
        assert self._call({"pattern": "DTOP", "alt_name": ""}) == "Double Top"

    def test_dbot(self):
        assert self._call({"pattern": "DBOT", "alt_name": ""}) == "Double Bottom"

    def test_hnsd(self):
        assert self._call({"pattern": "HNSD", "alt_name": ""}) == "Head and Shoulders"

    def test_with_alt_name(self):
        assert self._call({"pattern": "FLAGU", "alt_name": "Custom"}) == "Custom"

    def test_unknown_code(self):
        assert self._call({"pattern": "XXXX", "alt_name": ""}) == "XXXX"

    def test_empty(self):
        result = self._call({})
        assert isinstance(result, str)


# ── _parse_native_scale_factors ──────────────────────────────────────────

class TestParseNativeScaleFactors:

    def _call(self, config):
        from mtdata.core.patterns import _parse_native_scale_factors
        return _parse_native_scale_factors(config)

    def test_none_returns_defaults(self):
        result = self._call(None)
        assert 1.0 in result

    def test_empty_dict(self):
        result = self._call({})
        assert 1.0 in result

    def test_string_factors(self):
        result = self._call({"native_scale_factors": "0.5, 1.0, 2.0"})
        assert len(result) == 3
        assert 1.0 in result

    def test_list_factors(self):
        result = self._call({"native_scale_factors": [0.5, 1.0, 1.5]})
        assert len(result) == 3

    def test_dedup(self):
        result = self._call({"native_scale_factors": [1.0, 1.0, 1.0]})
        assert result.count(1.0) == 1

    def test_clamps_extremes(self):
        result = self._call({"native_scale_factors": [0.1, 10.0]})
        for v in result:
            assert 0.3 <= v <= 3.0

    def test_inserts_1_if_missing(self):
        result = self._call({"native_scale_factors": [0.5, 2.0]})
        assert any(round(v, 4) == 1.0 for v in result)

    def test_filters_non_positive(self):
        result = self._call({"native_scale_factors": [-1, 0, 1.0]})
        assert all(v > 0 for v in result)

    def test_native_scales_alias(self):
        result = self._call({"native_scales": "0.8, 1.0, 1.2"})
        assert len(result) == 3

    def test_semicolon_separator(self):
        result = self._call({"native_scale_factors": "0.8;1.0;1.5"})
        assert len(result) == 3


# ── _interval_overlap_ratio ──────────────────────────────────────────────

class TestIntervalOverlapRatio:

    def _call(self, a_start, a_end, b_start, b_end):
        from mtdata.core.patterns import _interval_overlap_ratio
        return _interval_overlap_ratio(a_start, a_end, b_start, b_end)

    def test_full_overlap(self):
        assert self._call(0, 10, 0, 10) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert self._call(0, 5, 10, 15) == pytest.approx(0.0)

    def test_partial_overlap(self):
        ratio = self._call(0, 10, 5, 15)
        assert 0 < ratio < 1

    def test_contained(self):
        ratio = self._call(0, 20, 5, 10)
        assert 0 < ratio < 1

    def test_zero_width(self):
        ratio = self._call(5, 5, 5, 5)
        assert ratio == pytest.approx(1.0)


# ── _format_pattern_dates ────────────────────────────────────────────────

class TestFormatPatternDates:

    def _call(self, start, end):
        from mtdata.core.patterns import _format_pattern_dates
        return _format_pattern_dates(start, end)

    def test_both_none(self):
        s, e = self._call(None, None)
        assert s is None and e is None

    def test_valid_epochs(self):
        s, e = self._call(1704067200.0, 1704153600.0)
        assert s is not None and e is not None
        assert isinstance(s, str) and isinstance(e, str)

    def test_start_only(self):
        s, e = self._call(1704067200.0, None)
        assert s is not None and e is None


# ── _apply_config_to_obj ────────────────────────────────────────────────

class TestApplyConfigToObj:

    def _call(self, cfg, config):
        from mtdata.core.patterns import _apply_config_to_obj
        return _apply_config_to_obj(cfg, config)

    def test_sets_float_attr(self):
        obj = SimpleNamespace(min_prominence_pct=0.5)
        unknown = self._call(obj, {"min_prominence_pct": 0.8})
        assert obj.min_prominence_pct == pytest.approx(0.8)
        assert unknown == []

    def test_sets_int_attr(self):
        obj = SimpleNamespace(min_distance=5)
        unknown = self._call(obj, {"min_distance": 10})
        assert obj.min_distance == 10
        assert unknown == []

    def test_sets_bool_attr_from_string(self):
        obj = SimpleNamespace(use_robust_fit=False)
        unknown = self._call(obj, {"use_robust_fit": "true"})
        assert obj.use_robust_fit is True
        assert unknown == []

    def test_sets_bool_false_from_string(self):
        obj = SimpleNamespace(use_robust_fit=True)
        unknown = self._call(obj, {"use_robust_fit": "false"})
        assert obj.use_robust_fit is False
        assert unknown == []

    def test_sets_list_from_string(self):
        obj = SimpleNamespace(pattern_types=["impulse"])
        unknown = self._call(obj, {"pattern_types": "impulse,correction"})
        assert obj.pattern_types == ["impulse", "correction"]
        assert unknown == []

    def test_sets_list_from_list(self):
        obj = SimpleNamespace(pattern_types=["impulse"])
        unknown = self._call(obj, {"pattern_types": ["a", "b"]})
        assert obj.pattern_types == ["a", "b"]
        assert unknown == []

    def test_ignores_unknown_keys(self):
        obj = SimpleNamespace(x=1)
        unknown = self._call(obj, {"unknown_key": 99})
        assert not hasattr(obj, "unknown_key")
        assert unknown == ["unknown_key"]

    def test_none_config_noop(self):
        obj = SimpleNamespace(x=1)
        unknown = self._call(obj, None)
        assert obj.x == 1
        assert unknown == []

    def test_non_dict_config_noop(self):
        obj = SimpleNamespace(x=1)
        unknown = self._call(obj, "not a dict")
        assert obj.x == 1
        assert unknown == []


# ── _resolve_engine_weights ──────────────────────────────────────────────

class TestResolveEngineWeights:

    def _call(self, engines, weights):
        from mtdata.core.patterns import _resolve_engine_weights
        return _resolve_engine_weights(engines, weights)

    def test_defaults_to_1(self):
        result = self._call(["native", "stock_pattern"], None)
        assert result == {"native": 1.0, "stock_pattern": 1.0}

    def test_custom_weights(self):
        result = self._call(["native", "stock_pattern"], {"native": 2.0, "stock_pattern": 0.5})
        assert result["native"] == 2.0
        assert result["stock_pattern"] == 0.5

    def test_ignores_unknown_engines(self):
        result = self._call(["native"], {"native": 1.5, "unknown": 3.0})
        assert "unknown" not in result

    def test_ignores_invalid_weight(self):
        result = self._call(["native"], {"native": "bad"})
        assert result["native"] == 1.0

    def test_ignores_zero_weight(self):
        result = self._call(["native"], {"native": 0.0})
        assert result["native"] == 1.0

    def test_ignores_negative_weight(self):
        result = self._call(["native"], {"native": -1.0})
        assert result["native"] == 1.0


# ── _merge_classic_ensemble ──────────────────────────────────────────────

class TestMergeClassicEnsemble:

    def _call(self, engine_patterns, weights, overlap=0.5):
        from mtdata.core.patterns import _merge_classic_ensemble
        return _merge_classic_ensemble(engine_patterns, weights, overlap_threshold=overlap)

    def test_empty(self):
        assert self._call({}, {}) == []

    def test_single_engine(self):
        pats = {"native": [{"name": "Triangle", "start_index": 0, "end_index": 10, "confidence": 0.8, "status": "forming"}]}
        result = self._call(pats, {"native": 1.0})
        assert len(result) == 1

    def test_merges_overlapping(self):
        pats = {
            "native": [{"name": "triangle", "start_index": 0, "end_index": 10, "confidence": 0.8, "status": "forming"}],
            "stock": [{"name": "triangle", "start_index": 2, "end_index": 10, "confidence": 0.7, "status": "forming"}],
        }
        result = self._call(pats, {"native": 1.0, "stock": 1.0})
        assert len(result) == 1
        assert result[0]["support_count"] == 2

    def test_does_not_merge_non_overlapping(self):
        pats = {
            "native": [{"name": "triangle", "start_index": 0, "end_index": 5, "confidence": 0.8, "status": "forming"}],
            "stock": [{"name": "triangle", "start_index": 50, "end_index": 60, "confidence": 0.7, "status": "forming"}],
        }
        result = self._call(pats, {"native": 1.0, "stock": 1.0})
        assert len(result) == 2

    def test_weighted_confidence(self):
        pats = {
            "native": [{"name": "triangle", "start_index": 0, "end_index": 10, "confidence": 1.0, "status": "forming"}],
            "stock": [{"name": "triangle", "start_index": 0, "end_index": 10, "confidence": 0.0, "status": "forming"}],
        }
        result = self._call(pats, {"native": 3.0, "stock": 1.0})
        assert result[0]["confidence"] == pytest.approx(0.75)

    def test_completed_status(self):
        pats = {
            "native": [{"name": "triangle", "start_index": 0, "end_index": 10, "confidence": 0.8, "status": "completed"}],
            "stock": [{"name": "triangle", "start_index": 0, "end_index": 10, "confidence": 0.7, "status": "completed"}],
        }
        result = self._call(pats, {"native": 1.0, "stock": 1.0})
        assert result[0]["status"] == "completed"

    def test_any_completed_status_promotes_merged_pattern(self):
        pats = {
            "native": [{"name": "triangle", "start_index": 0, "end_index": 10, "confidence": 0.8, "status": "completed"}],
            "stock": [{"name": "triangle", "start_index": 0, "end_index": 10, "confidence": 0.7, "status": "forming"}],
        }
        result = self._call(pats, {"native": 1.0, "stock": 1.0})
        assert result[0]["status"] == "completed"


# ── _estimate_classic_bars_to_completion ─────────────────────────────────

class TestEstimateClassicBarsToCompletion:

    def _call(self, name, details, start, end, n_bars):
        from mtdata.core.patterns import _estimate_classic_bars_to_completion
        return _estimate_classic_bars_to_completion(name, details, start, end, n_bars)

    def test_with_slopes(self):
        details = {"top_slope": -0.01, "top_intercept": 1.2, "bottom_slope": 0.01, "bottom_intercept": 1.0}
        result = self._call("Triangle", details, 0, 50, 100)
        assert result is None or isinstance(result, int)

    def test_with_upper_lower(self):
        details = {"upper_slope": -0.01, "upper_intercept": 1.2, "lower_slope": 0.01, "lower_intercept": 1.0}
        result = self._call("Triangle", details, 0, 50, 100)
        assert result is None or isinstance(result, int)

    def test_pennant(self):
        result = self._call("Bull Pennants", {}, 0, 20, 100)
        assert isinstance(result, int) and result >= 1

    def test_flag(self):
        result = self._call("Bull Flag", {}, 0, 20, 100)
        assert isinstance(result, int)

    def test_unknown_pattern(self):
        result = self._call("Unknown Pattern", {}, 0, 20, 100)
        assert result is None

    def test_zero_denom(self):
        details = {"top_slope": 0.01, "top_intercept": 1.0, "bottom_slope": 0.01, "bottom_intercept": 1.0}
        result = self._call("Triangle", details, 0, 50, 100)
        assert result is None


# ── _enrich_classic_patterns / current level projection ──────────────────

class TestEnrichClassicPatterns:

    def test_forming_patterns_project_line_levels_to_current_bar(self):
        from mtdata.core.patterns_support import _enrich_classic_patterns

        df = _make_ohlcv_df(6)
        rows = [
            {
                "name": "Ascending Triangle",
                "status": "forming",
                "start_index": 0,
                "end_index": 2,
                "details": {
                    "top_slope": 1.0,
                    "top_intercept": 100.0,
                    "bottom_slope": 0.5,
                    "bottom_intercept": 95.0,
                },
            }
        ]

        enriched = _enrich_classic_patterns(rows, df)

        assert enriched[0]["price_levels"]["resistance"] == pytest.approx(105.0)
        assert enriched[0]["price_levels"]["support"] == pytest.approx(97.5)


# ── _build_pattern_response ──────────────────────────────────────────────

class TestBuildPatternResponse:

    def _call(self, **kwargs):
        from mtdata.core.patterns import _build_pattern_response
        defaults = dict(
            symbol="EURUSD",
            timeframe="H1",
            limit=500,
            mode="classic",
            patterns=[{"status": "forming", "confidence": 0.8}],
            include_completed=False,
            include_series=False,
            series_time="string",
            df=_make_ohlcv_df(100),
        )
        defaults.update(kwargs)
        return _build_pattern_response(**defaults)

    def test_basic_response(self):
        resp = self._call()
        assert resp["success"] is True
        assert resp["symbol"] == "EURUSD"
        assert resp["timeframe"] == "H1"
        assert resp["mode"] == "classic"

    def test_filters_completed(self):
        patterns = [{"status": "forming"}, {"status": "completed"}]
        resp = self._call(patterns=patterns, include_completed=False)
        assert resp["n_patterns"] == 1

    def test_includes_completed(self):
        patterns = [{"status": "forming"}, {"status": "completed"}]
        resp = self._call(patterns=patterns, include_completed=True)
        assert resp["n_patterns"] == 2

    def test_include_series_string(self):
        resp = self._call(include_series=True, series_time="string")
        assert "series_close" in resp
        assert "series_time" in resp

    def test_include_series_epoch(self):
        resp = self._call(include_series=True, series_time="epoch")
        assert "series_close" in resp
        assert "series_epoch" in resp


# ── _build_stock_pattern_frame ───────────────────────────────────────────

class TestBuildStockPatternFrame:

    def _call(self, df):
        from mtdata.core.patterns import _build_stock_pattern_frame
        return _build_stock_pattern_frame(df)

    def test_basic(self):
        df = _make_ohlcv_df(50)
        result = self._call(df)
        assert "Open" in result.columns
        assert "Close" in result.columns
        assert len(result) == 50

    def test_no_time_column(self):
        df = _make_ohlcv_df(50, with_time=False)
        result = self._call(df)
        assert isinstance(result.index, pd.RangeIndex)

    def test_uppercase_columns(self):
        df = pd.DataFrame({
            "Open": [1.0], "High": [1.1], "Low": [0.9], "Close": [1.05], "Volume": [100]
        })
        result = self._call(df)
        assert len(result) == 1


# ── _index_pos_for_timestamp ─────────────────────────────────────────────

class TestIndexPosForTimestamp:

    def _call(self, index, ts):
        from mtdata.core.patterns import _index_pos_for_timestamp
        return _index_pos_for_timestamp(index, ts)

    def test_found(self):
        idx = pd.DatetimeIndex(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
        result = self._call(idx, "2024-01-02")
        assert result == 1

    def test_not_found(self):
        idx = pd.DatetimeIndex(pd.to_datetime(["2024-01-01", "2024-01-02"]))
        result = self._call(idx, "2025-06-01")
        assert result is None


# ── _available_classic_engines ───────────────────────────────────────────

class TestAvailableClassicEngines:

    def test_returns_tuple(self):
        from mtdata.core.patterns import _available_classic_engines
        result = _available_classic_engines()
        assert isinstance(result, tuple)
        assert "native" in result


# ── _register_classic_engine ─────────────────────────────────────────────

class TestRegisterClassicEngine:

    def test_registers(self):
        from mtdata.core.patterns import _register_classic_engine, _CLASSIC_ENGINE_REGISTRY
        @_register_classic_engine("__test_engine__")
        def dummy(symbol, df, cfg, config):
            return [], None
        assert "__test_engine__" in _CLASSIC_ENGINE_REGISTRY
        del _CLASSIC_ENGINE_REGISTRY["__test_engine__"]


# ── _run_classic_engine ──────────────────────────────────────────────────

class TestRunClassicEngine:

    def _call(self, engine, symbol, df, cfg, config):
        from mtdata.core.patterns import _run_classic_engine
        return _run_classic_engine(engine, symbol, df, cfg, config)

    def test_unknown_engine(self):
        pats, err = self._call("nonexistent_xyz", "EURUSD", _make_ohlcv_df(), None, None)
        assert pats == []
        assert "Unsupported" in err

    @patch("mtdata.core.patterns._detect_classic_patterns", return_value=[])
    def test_native_engine(self, mock_detect):
        from mtdata.patterns.classic import ClassicDetectorConfig
        pats, err = self._call("native", "EURUSD", _make_ohlcv_df(), ClassicDetectorConfig(), None)
        assert pats == []
        assert err is None


# ── _load_stock_pattern_utils ────────────────────────────────────────────

class TestLoadStockPatternUtils:

    def _call(self, config=None):
        from mtdata.core.patterns import _load_stock_pattern_utils, _STOCK_PATTERN_UTILS_CACHE
        _STOCK_PATTERN_UTILS_CACHE.clear()
        return _load_stock_pattern_utils(config)

    @patch("importlib.import_module", side_effect=ImportError("no module"))
    def test_import_error(self, mock_import):
        mod, err = self._call()
        assert mod is None
        assert err is not None and "unavailable" in err

    @patch("importlib.import_module")
    def test_missing_functions(self, mock_import):
        mock_mod = MagicMock(spec=[])
        mock_import.return_value = mock_mod
        mod, err = self._call()
        assert mod is None
        assert "missing" in (err or "")

    @patch("importlib.import_module")
    def test_success(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.get_max_min = MagicMock()
        mock_mod.find_double_top = MagicMock()
        mock_mod.find_double_bottom = MagicMock()
        mock_import.return_value = mock_mod
        mod, err = self._call()
        assert mod is not None
        assert err is None


# ── _fetch_pattern_data ──────────────────────────────────────────────────

class TestFetchPatternData:

    def _call(self, symbol, timeframe, limit, denoise=None):
        from mtdata.core.patterns import _fetch_pattern_data
        return _fetch_pattern_data(symbol, timeframe, limit, denoise)

    def test_invalid_timeframe(self):
        df, err = self._call("EURUSD", "INVALID", 500)
        assert df is None
        assert "Invalid timeframe" in err["error"]

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from", return_value=None)
    def test_no_rates(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        df, err = self._call("EURUSD", "H1", 500)
        assert df is None
        assert "Failed to fetch" in err["error"]

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_success(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        mock_rates.return_value = _make_rates_array(200)
        df, err = self._call("EURUSD", "H1", 100)
        assert err is None
        assert df is not None
        assert len(df) <= 100

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_tick_volume_renamed(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        mock_rates.return_value = _make_rates_array(200)
        df, err = self._call("EURUSD", "H1", 100)
        assert err is None
        assert "volume" in df.columns or "tick_volume" in df.columns

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_invisible_symbol_selected(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=False)
        mock_rates.return_value = _make_rates_array(200)
        self._call("EURUSD", "H1", 500)
        mock_mt5.symbol_select.assert_called_once_with("EURUSD", True)

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_symbol_info_none(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = None
        mock_rates.return_value = _make_rates_array(200)
        df, err = self._call("EURUSD", "H1", 100)
        assert err is None  # should still succeed


# ── _format_elliott_patterns ─────────────────────────────────────────────

class TestFormatElliottPatterns:

    def _call(self, df, cfg):
        from mtdata.core.patterns import _format_elliott_patterns
        return _format_elliott_patterns(df, cfg)

    @patch("mtdata.core.patterns._detect_elliott_waves")
    def test_basic(self, mock_detect):
        mock_detect.return_value = [
            _mock_pattern_result(wave_type="Impulse", start_index=0, end_index=10,
                                 start_time=1704067200.0, end_time=1704110400.0,
                                 confidence=0.9, details={"key": 1.5}),
        ]
        df = _make_ohlcv_df(50)
        result = self._call(df, MagicMock())
        assert len(result) == 1
        assert result[0]["wave_type"] == "Impulse"

    @patch("mtdata.core.patterns._detect_elliott_waves")
    def test_forming_status(self, mock_detect):
        df = _make_ohlcv_df(50)
        mock_detect.return_value = [
            _mock_pattern_result(end_index=48),  # near end
        ]
        result = self._call(df, MagicMock())
        assert result[0]["status"] == "forming"

    @patch("mtdata.core.patterns._detect_elliott_waves")
    def test_completed_status(self, mock_detect):
        df = _make_ohlcv_df(100)
        mock_detect.return_value = [
            _mock_pattern_result(end_index=10),  # far from end
        ]
        result = self._call(df, MagicMock())
        assert result[0]["status"] == "completed"

    @patch("mtdata.core.patterns._detect_elliott_waves")
    def test_exception_skipped(self, mock_detect):
        bad = _mock_pattern_result()
        bad.start_time = "not-a-number"
        bad.end_time = "not-a-number"
        bad.confidence = "bad"
        mock_detect.return_value = [bad]
        df = _make_ohlcv_df(50)
        result = self._call(df, MagicMock())
        # Should either skip or handle gracefully
        assert isinstance(result, list)


# ── patterns_detect (main tool) ──────────────────────────────────────────

def _fully_unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _call_patterns_detect(**kwargs):
    from mtdata.core.patterns import patterns_detect

    inner = _fully_unwrap(patterns_detect)
    with patch("mtdata.core.patterns.ensure_mt5_connection_or_raise", return_value=None):
        return inner(request=PatternsDetectRequest(**kwargs))

class TestPatternsDetect:

    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_candlestick_mode(self, mock_detect):
        mock_detect.return_value = {"success": True, "patterns": []}
        result = _call_patterns_detect(symbol="EURUSD", mode="candlestick")
        mock_detect.assert_called_once()

    def test_unknown_mode(self):
        result = _call_patterns_detect(symbol="EURUSD", mode="unknown_mode")
        assert "error" in result

    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_classic_mode_fetch_error(self, mock_fetch):
        mock_fetch.return_value = (None, {"error": "Failed to fetch"})
        result = _call_patterns_detect(symbol="EURUSD", mode="classic", timeframe="H1")
        assert "error" in result

    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_classic_mode_success(self, mock_fetch, mock_engine):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([{"name": "Triangle", "status": "forming", "confidence": 0.8,
                                       "start_index": 0, "end_index": 10}], None)
        result = _call_patterns_detect(symbol="EURUSD", mode="classic", timeframe="H1")
        assert result.get("success") is True

    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_classic_invalid_config_key_returns_error_before_fetch(self, mock_fetch):
        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="classic",
            timeframe="H1",
            config={"min_prominance_pct": 0.3},
        )
        assert result == {"error": "Invalid config key(s): ['min_prominance_pct']"}
        mock_fetch.assert_not_called()

    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_classic_mode_allows_engine_extra_config_keys(self, mock_fetch, mock_engine):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([{
            "name": "Triangle",
            "status": "forming",
            "confidence": 0.8,
            "start_index": 0,
            "end_index": 10,
        }], None)
        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="classic",
            timeframe="H1",
            config={"native_multiscale": True},
        )
        assert result.get("success") is True
        mock_fetch.assert_called_once()
        mock_engine.assert_called_once()

    @patch("mtdata.core.patterns._fetch_pattern_data", side_effect=RuntimeError("boom"))
    def test_classic_fetch_exception_propagates(self, mock_fetch):
        with pytest.raises(RuntimeError, match="boom"):
            _call_patterns_detect(symbol="EURUSD", mode="classic", timeframe="H1")
        mock_fetch.assert_called_once()

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_elliott_mode_single_tf(self, mock_fetch, mock_format):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_format.return_value = [{"wave_type": "Impulse", "status": "forming"}]
        result = _call_patterns_detect(symbol="EURUSD", mode="elliott", timeframe="H1")
        assert result.get("success") is True

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_elliott_mode_single_tf_zero_patterns_includes_diagnostic(self, mock_fetch, mock_format):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_format.return_value = []
        result = _call_patterns_detect(symbol="EURUSD", mode="elliott", timeframe="H1")
        assert result.get("success") is True
        assert "diagnostic" in result
        assert "No valid Elliott Wave structures detected" in str(result.get("diagnostic"))

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_elliott_mode_diagnostic_does_not_repeat_current_timeframe(self, mock_fetch, mock_format):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_format.return_value = []
        result = _call_patterns_detect(symbol="EURUSD", mode="elliott", timeframe="D1")
        diagnostic = str(result.get("diagnostic") or "")
        assert result.get("success") is True
        assert "--timeframe D1 or --timeframe D1" not in diagnostic
        assert "Try --timeframe H4 or --timeframe W1." in diagnostic

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_elliott_mode_all_tf(self, mock_fetch, mock_format):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_format.return_value = [{"wave_type": "Impulse", "status": "forming"}]
        result = _call_patterns_detect(symbol="EURUSD", mode="elliott", timeframe=None)
        assert result.get("success") is True
        assert "findings" in result

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_elliott_mode_all_tf_zero_patterns_includes_diagnostic(self, mock_fetch, mock_format):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_format.return_value = []
        result = _call_patterns_detect(symbol="EURUSD", mode="elliott", timeframe=None)
        assert result.get("success") is True
        assert result.get("n_patterns") == 0
        assert "diagnostic" in result

    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_elliott_all_tf_all_fail(self, mock_fetch):
        mock_fetch.return_value = (None, {"error": "No data"})
        result = _call_patterns_detect(symbol="EURUSD", mode="elliott", timeframe=None)
        assert "error" in result

    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_classic_invalid_engine(self, mock_fetch, mock_engine):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        result = _call_patterns_detect(symbol="EURUSD", mode="classic", timeframe="H1", engine="totally_fake_engine_xyz")
        assert "error" in result

    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_classic_all_engines_error(self, mock_fetch, mock_engine):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([], "engine error")
        result = _call_patterns_detect(symbol="EURUSD", mode="classic", timeframe="H1")
        # Should return error or empty response
        assert isinstance(result, dict)
