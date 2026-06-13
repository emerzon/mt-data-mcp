"""Core patterns detection and response building tests."""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.mtdata.core.patterns_support as patterns_support_mod
import src.mtdata.patterns.candlestick as candlestick_mod
import src.mtdata.patterns.classic as classic_mod
import src.mtdata.services.data_service as data_service_mod
from src.mtdata.core import patterns as core_patterns
from src.mtdata.core.patterns import _apply_config_to_obj, _build_pattern_response
from src.mtdata.core.patterns_requests import PatternsDetectRequest
from src.mtdata.patterns.common import data_quality_warnings
from src.mtdata.patterns.classic import (
    ClassicDetectorConfig,
    ClassicPatternResult,
    _count_recent_touches,
    _fit_lines_and_arrays,
)
from src.mtdata.utils.mt5 import MT5ConnectionError


def patterns_detect(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", True))
    request = kwargs.pop("request", None)
    if request is None:
        request = PatternsDetectRequest(**kwargs)
    return core_patterns.patterns_detect(request=request, __cli_raw=raw_output)


def test_data_quality_uses_tick_volume_when_real_volume_is_structural_zero():
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "open": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "high": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "low": [0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            "close": [1.05, 1.15, 1.25, 1.35, 1.45, 1.55],
            "real_volume": [0, 0, 0, 0, 0, 0],
            "tick_volume": [120, 135, 142, 128, 151, 149],
        }
    )

    warnings = data_quality_warnings(df, symbol="EURUSD", timeframe_seconds=1)

    assert not any("zero-volume bars dominate" in warning for warning in warnings)


def test_data_quality_suppresses_expected_fx_weekend_gap():
    friday = pd.Timestamp("2026-05-29 20:00:00", tz="UTC").timestamp()
    sunday = pd.Timestamp("2026-05-31 21:00:00", tz="UTC").timestamp()
    df = pd.DataFrame(
        {
            "time": [friday - 3600, friday, sunday, sunday + 3600],
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.05, 1.15, 1.25, 1.35],
            "tick_volume": [120, 135, 142, 128],
        }
    )

    warnings = data_quality_warnings(df, symbol="EURUSD", timeframe_seconds=3600)

    assert not any("time gap" in warning for warning in warnings)


def test_patterns_detect_returns_connection_error_payload(monkeypatch):
    def fail_connection():
        raise MT5ConnectionError(
            "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
        )

    monkeypatch.setattr(
        core_patterns, "ensure_mt5_connection_or_raise", fail_connection
    )
    monkeypatch.setattr(
        core_patterns,
        "_fetch_pattern_data",
        lambda *args, **kwargs: pytest.fail("fetch should not run"),
    )

    out = patterns_detect(symbol="EURUSD", mode="classic", timeframe="H1")

    assert (
        out["error"]
        == "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
    )
    assert out["success"] is False
    assert out["error_code"] == "mt5_connection_error"
    assert out["operation"] == "mt5_ensure_connection"
    assert isinstance(out.get("request_id"), str)


def test_patterns_detect_public_default_is_compact_for_classic_mode(monkeypatch):
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.05, 1.15, 1.25, 1.35],
            "tick_volume": [10, 11, 12, 13],
        }
    )

    monkeypatch.setattr(core_patterns, "_patterns_connection_error", lambda: None)
    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", lambda *args, **kwargs: (df, None))
    monkeypatch.setattr(
        core_patterns,
        "_select_classic_engines",
        lambda engine, ensemble: (["native"], []),
    )
    monkeypatch.setattr(core_patterns, "_enrich_classic_patterns", lambda rows, *_: rows)
    monkeypatch.setattr(
        core_patterns,
        "_run_classic_engine",
        lambda *args, **kwargs: (
            [
                {
                    "name": "Ascending Triangle",
                    "status": "forming",
                    "confidence": 0.81,
                    "start_index": 0,
                    "end_index": 3,
                    "details": {"bias": "bullish"},
                }
            ],
            None,
        ),
    )

    out = patterns_detect(symbol="EURUSD", mode="classic", timeframe="H1")

    assert out["n_patterns"] == 1
    assert out["suggested_review"] == "long_setup"
    assert out["bias"] == "bullish"
    assert out["review_recommended"] is True
    assert out["pattern_confidence"] == pytest.approx(0.81)
    assert out["is_signal"] is False
    assert out["usage"] == "information_only"
    assert out["top_patterns"][0]["name"] == "Ascending Triangle"
    assert "recent_patterns" not in out
    assert "patterns" not in out


def test_patterns_detect_standard_detail_preserves_full_payload(monkeypatch):
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.05, 1.15, 1.25, 1.35],
            "tick_volume": [10, 11, 12, 13],
        }
    )

    monkeypatch.setattr(core_patterns, "_patterns_connection_error", lambda: None)
    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", lambda *args, **kwargs: (df, None))
    monkeypatch.setattr(core_patterns, "_select_classic_engines", lambda engine, ensemble: (["native"], []))
    monkeypatch.setattr(core_patterns, "_enrich_classic_patterns", lambda rows, *_: rows)
    monkeypatch.setattr(
        core_patterns,
        "_run_classic_engine",
        lambda *args, **kwargs: (
            [
                {
                    "name": "Ascending Triangle",
                    "status": "forming",
                    "confidence": 0.81,
                    "start_index": 0,
                    "end_index": 3,
                    "details": {"bias": "bullish"},
                }
            ],
            None,
        ),
    )

    out = patterns_detect(symbol="EURUSD", mode="classic", timeframe="H1", detail="standard")

    assert out["n_patterns"] == 1
    assert "patterns" in out
    assert "recent_patterns" not in out


def test_fit_lines_and_arrays_uses_cfg_for_robust_fit(monkeypatch):
    calls = []

    def _fake_robust(x, y, cfg):
        calls.append(bool(cfg.use_robust_fit))
        # slope, intercept, r2
        return 1.0, 2.0, 0.9

    # Patch the implementation module since we refactored logic into classic_impl
    from src.mtdata.patterns.classic_impl import utils as impl_utils

    monkeypatch.setattr(impl_utils, "_fit_line_robust", _fake_robust)

    ih = np.array([1, 5, 9], dtype=int)
    il = np.array([2, 6, 10], dtype=int)
    c = np.linspace(10.0, 20.0, 12)
    cfg = ClassicDetectorConfig(use_robust_fit=True)

    sh, bh, r2h, sl, bl, r2l, upper, lower = _fit_lines_and_arrays(
        ih, il, c, len(c), cfg
    )

    assert calls == [True, True]
    assert (sh, bh, r2h) == (1.0, 2.0, 0.9)
    assert (sl, bl, r2l) == (1.0, 2.0, 0.9)
    assert upper.shape[0] == len(c)
    assert lower.shape[0] == len(c)


def test_patterns_detect_candlestick_passes_last_n_bars(monkeypatch):
    captured = {}

    def _fake_detect(**kwargs):
        captured.update(kwargs)
        return {"success": True, "patterns": []}

    monkeypatch.setattr(core_patterns, "_detect_candlestick_patterns", _fake_detect)

    res = patterns_detect(
        symbol="EURUSD",
        timeframe="H1",
        mode="candlestick",
        detail="full",
        last_n_bars=8,
    )

    assert res.get("success") is True
    assert captured.get("last_n_bars") == 8


def test_patterns_detect_candlestick_passes_config(monkeypatch):
    captured = {}

    def _fake_detect(**kwargs):
        captured.update(kwargs)
        return {"success": True, "patterns": []}

    monkeypatch.setattr(core_patterns, "_detect_candlestick_patterns", _fake_detect)

    patterns_detect(
        symbol="EURUSD",
        timeframe="H1",
        mode="candlestick",
        detail="full",
        config={"use_volume_confirmation": False},
        __cli_raw=True,
    )

    assert captured["config"] == {"use_volume_confirmation": False}


def test_patterns_detect_passes_date_range_to_candlestick(monkeypatch):
    captured = {}

    def _fake_detect(**kwargs):
        captured.update(kwargs)
        return {"success": True, "patterns": []}

    monkeypatch.setattr(core_patterns, "_detect_candlestick_patterns", _fake_detect)

    patterns_detect(
        symbol="EURUSD",
        timeframe="H1",
        mode="candlestick",
        start="2023-01-01",
        end="2023-02-01",
        detail="full",
    )

    assert captured["start"] == "2023-01-01"
    assert captured["end"] == "2023-02-01"


def test_patterns_support_config_helpers_read_dict_values():
    config = {
        "use_volume_confirmation": False,
        "volume_confirm_breakout_bars": 4,
        "volume_confirm_min_ratio": 1.35,
    }

    assert (
        patterns_support_mod._config_bool(config, "use_volume_confirmation", True)
        is False
    )
    assert (
        patterns_support_mod._config_int(config, "volume_confirm_breakout_bars", 2) == 4
    )
    assert patterns_support_mod._config_float(
        config, "volume_confirm_min_ratio", 1.1
    ) == pytest.approx(1.35)


def test_patterns_detect_candlestick_rejects_non_positive_last_n_bars():
    res = patterns_detect(
        symbol="EURUSD",
        timeframe="H1",
        mode="candlestick",
        detail="full",
        last_n_bars=0,
    )
    assert "error" in res
    assert "last_n_bars" in str(res["error"])


def test_select_classic_engines_uses_registry(monkeypatch):
    monkeypatch.setitem(
        core_patterns._CLASSIC_ENGINE_REGISTRY,
        "unit_test",
        lambda symbol, df, cfg, config: ([{"name": "Unit Test"}], None),
    )

    engines, invalid = core_patterns._select_classic_engines(
        "unit_test", ensemble=False
    )

    assert engines == ["unit_test"]
    assert invalid == []


def test_apply_config_to_obj_coerces_bool_strings():
    cfg = ClassicDetectorConfig()
    assert cfg.use_robust_fit is True

    _apply_config_to_obj(cfg, {"use_robust_fit": "false", "use_dtw_check": "0"})
    assert cfg.use_robust_fit is False
    assert cfg.use_dtw_check is False

    _apply_config_to_obj(cfg, {"use_robust_fit": "true", "use_dtw_check": "yes"})
    assert cfg.use_robust_fit is True
    assert cfg.use_dtw_check is True


def test_apply_config_to_obj_rejects_invalid_bool_strings():
    cfg = ClassicDetectorConfig()
    original = cfg.use_robust_fit

    invalid = _apply_config_to_obj(cfg, {"use_robust_fit": "maybe"})

    assert cfg.use_robust_fit is original
    assert invalid == ["use_robust_fit"]


def test_apply_config_to_obj_rejects_invalid_type_coercion():
    cfg = ClassicDetectorConfig()
    original = cfg.min_distance

    invalid = _apply_config_to_obj(cfg, {"min_distance": "not-an-int"})

    assert cfg.min_distance == original
    assert invalid == ["min_distance"]


def test_build_pattern_response_include_completed_filter_behavior():
    df = pd.DataFrame({"time": [1, 2, 3], "close": [10.0, 11.0, 12.0]})
    patterns = [
        {"name": "A", "status": "forming", "confidence": 0.5},
        {"name": "B", "status": "completed", "confidence": 0.6},
    ]

    forming_only = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "classic",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
    )
    with_completed = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "classic",
        patterns,
        include_completed=True,
        include_series=False,
        series_time="string",
        df=df,
    )

    assert forming_only["n_patterns"] == 1
    assert forming_only["patterns"][0]["status"] == "forming"
    assert forming_only["completed_patterns_hidden"] == 1
    assert "include_completed=true" in forming_only["note"]
    assert with_completed["n_patterns"] == 2


def test_build_pattern_response_hoists_repeated_regime_context():
    df = pd.DataFrame({"time": [1, 2, 3], "close": [10.0, 11.0, 12.0]})
    shared = {
        "state": "ranging",
        "direction": "bearish",
        "window_bars": 160,
        "trend_strength": 3.266,
        "efficiency_ratio": 0.1035,
        "window_move_pct": -0.706,
    }
    patterns = [
        {
            "name": "A",
            "status": "forming",
            "confidence": 0.5,
            "details": {
                "regime_context": {
                    **shared,
                    "pattern_bias": "bullish",
                    "alignment": "neutral",
                }
            },
        },
        {
            "name": "B",
            "status": "forming",
            "confidence": 0.6,
            "details": {
                "regime_context": {
                    **shared,
                    "pattern_bias": "bearish",
                    "alignment": "neutral",
                }
            },
        },
    ]

    result = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "classic",
        patterns,
        include_completed=True,
        include_series=False,
        series_time="string",
        df=df,
        detail="full",
    )

    assert result["regime_context"] == shared
    assert result["patterns"][0]["details"]["regime_context"] == {
        "pattern_bias": "bullish",
        "alignment": "neutral",
    }
    assert result["patterns"][1]["details"]["regime_context"] == {
        "pattern_bias": "bearish",
        "alignment": "neutral",
    }


def test_build_pattern_response_elliott_hidden_completed_preview_is_truthful():
    df = pd.DataFrame({"time": [1, 2, 3, 4], "close": [10.0, 11.0, 12.0, 13.0]})
    patterns = [
        {
            "wave_type": "Correction",
            "status": "completed",
            "confidence": 0.71,
            "start_index": 0,
            "end_index": 1,
            "start_date": "2026-03-01 00:00",
            "end_date": "2026-03-02 00:00",
            "details": {
                "trend": "bear",
                "pattern_confirmed": True,
                "has_unconfirmed_terminal_pivot": False,
            },
        },
        {
            "wave_type": "Impulse",
            "status": "completed",
            "confidence": 0.82,
            "start_index": 1,
            "end_index": 3,
            "start_date": "2026-03-02 00:00",
            "end_date": "2026-03-04 00:00",
            "details": {
                "trend": "bull",
                "pattern_confirmed": True,
                "has_unconfirmed_terminal_pivot": False,
            },
        },
    ]

    res = _build_pattern_response(
        "EURUSD",
        "H4",
        100,
        "elliott",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
    )

    assert res["n_patterns"] == 0
    assert res["completed_patterns_hidden"] == 2
    assert "No forming Elliott Wave structures detected" in res["diagnostic"]
    assert "No valid Elliott Wave structures detected" not in res["diagnostic"]
    assert res["completed_patterns_preview"][0]["pattern"] == "Impulse"
    assert res["completed_patterns_preview"][0]["timeframe"] == "H4"
    assert "strongest hidden count" in res["note"]
    assert "include_completed=true" in res["note"]


def test_build_pattern_response_elliott_compact_keeps_hidden_completed_preview():
    df = pd.DataFrame({"time": [1, 2, 3, 4], "close": [10.0, 11.0, 12.0, 13.0]})
    patterns = [
        {
            "wave_type": "Impulse",
            "status": "forming",
            "confidence": 0.55,
            "start_index": 1,
            "end_index": 3,
            "start_date": "2026-03-02 00:00",
            "end_date": "2026-03-04 00:00",
            "details": {"trend": "bull"},
        },
        {
            "wave_type": "Correction",
            "status": "completed",
            "confidence": 0.77,
            "start_index": 0,
            "end_index": 1,
            "start_date": "2026-03-01 00:00",
            "end_date": "2026-03-02 00:00",
            "details": {"trend": "bear", "pattern_confirmed": True},
        },
    ]

    compact = _build_pattern_response(
        "EURUSD",
        "H4",
        100,
        "elliott",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
        detail="compact",
    )

    assert compact["n_patterns"] == 1
    assert compact["completed_patterns_hidden"] == 1
    assert compact["completed_patterns_preview"][0]["pattern"] == "Correction"
    assert compact["completed_patterns_preview"][0]["timeframe"] == "H4"


def test_build_pattern_response_compact_detail_returns_summary():
    df = pd.DataFrame({"time": [1, 2, 3], "close": [10.0, 11.0, 12.0]})
    patterns = [
        {"name": "A", "status": "forming", "confidence": 0.5, "end_index": 1},
        {"name": "B", "status": "forming", "confidence": 0.7, "end_index": 2},
    ]

    compact = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "classic",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
        detail="compact",
    )

    assert compact["n_patterns"] == 2
    assert compact["top_patterns"] == [
        {"name": "B", "status": "forming", "confidence": 0.7},
        {"name": "A", "status": "forming", "confidence": 0.5},
    ]
    assert "recent_patterns" not in compact
    assert "summary" not in compact
    assert "patterns" not in compact


def test_build_pattern_response_compact_keeps_actionable_fields():
    df = pd.DataFrame({"time": [1, 2, 3], "close": [10.0, 11.0, 12.0]})
    patterns = [
        {
            "name": "Double Bottom",
            "status": "forming",
            "confidence": 0.85,
            "end_index": 2,
            "end_date": "2026-03-02 00:00",
            "bias": "bullish",
            "reference_price": 12.0,
            "target_price": 13.2,
            "invalidation_price": 11.4,
        }
    ]

    compact = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "classic",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
        detail="compact",
    )

    assert compact["bias"] == "bullish"
    assert compact["suggested_review"] == "long_setup"
    assert compact["pattern_confidence"] == 0.85
    assert compact["review_recommended"] is True
    assert compact["pattern_status"] == "bullish"
    assert compact["top_patterns"] == [
        {
            "name": "Double Bottom",
            "direction": "bullish",
            "status": "forming",
            "confidence": 0.85,
            "time": "2026-03-02 00:00",
            "price": 12.0,
        }
    ]
    assert "recent_patterns" not in compact


def test_build_pattern_response_compact_keeps_elliott_candidate_context():
    df = pd.DataFrame({"time": [1, 2, 3], "close": [10.0, 11.0, 12.0]})
    candidate_note = (
        "Low-confidence fallback candidate; Elliott rules did not validate a "
        "specific impulse or correction."
    )
    patterns = [
        {
            "pattern": "Elliott impulse-like candidate",
            "wave_type": "Candidate",
            "status": "forming",
            "confidence": 0.1,
            "end_index": 2,
            "wave_count": 6,
            "validation_status": "fallback_candidate",
            "candidate_note": candidate_note,
            "details": {"sequence_direction": "bullish"},
        }
    ]

    compact = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "elliott",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
        detail="compact",
    )

    assert compact["pattern_status"] == "uncertain"
    assert compact["review_recommended"] is False
    assert "bias" not in compact
    assert compact["pattern_confidence"] == 0.1
    assert "suggested_review" not in compact
    assert compact["top_patterns"] == [
        {
            "name": "Elliott impulse-like candidate",
            "status": "forming",
            "confidence": 0.1,
            "wave_count": 6,
            "candidate_note": candidate_note,
            "validation_status": "fallback_candidate",
        }
    ]


def test_build_pattern_response_compact_keeps_fractal_breakout_fields():
    df = pd.DataFrame({"time": [1, 2, 3], "close": [10.0, 10.5, 9.8]})
    patterns = [
        {
            "name": "Bullish Fractal",
            "status": "completed",
            "confidence": 0.82,
            "end_index": 2,
            "direction": "bullish",
            "bias": "bearish",
            "price": 9.9,
            "level_price": 9.9,
            "level_state": "broken",
            "confirmation_date": "2026-03-01 00:00",
            "breakout_direction": "bearish",
            "breakout_date": "2026-03-02 00:00",
            "breakout_price": 9.7,
        }
    ]

    compact = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "fractal",
        patterns,
        include_completed=True,
        include_series=False,
        series_time="string",
        df=df,
        detail="compact",
    )

    assert compact["bias"] == "bearish"
    assert compact["suggested_review"] == "short_setup"
    assert compact["top_patterns"][0]["name"] == "Bullish Fractal"
    assert "recent_patterns" not in compact


def test_build_pattern_response_compact_hides_completed_fractal_rows_by_default():
    df = pd.DataFrame({"time": [1, 2, 3], "close": [10.0, 10.5, 9.8]})
    patterns = [
        {
            "name": "Active Fractal",
            "status": "forming",
            "confidence": 0.82,
            "end_index": 1,
            "direction": "bullish",
            "bias": "bullish",
            "price": 9.9,
            "level_price": 9.9,
            "level_state": "active",
        },
        {
            "name": "Broken Fractal",
            "status": "completed",
            "confidence": 0.88,
            "end_index": 2,
            "direction": "bearish",
            "bias": "bullish",
            "price": 10.7,
            "level_price": 10.7,
            "level_state": "broken",
            "breakout_direction": "bullish",
        },
    ]

    compact = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "fractal",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
        detail="compact",
    )

    assert compact["top_patterns"][0]["name"] == "Active Fractal"
    assert compact["completed_patterns_hidden"] == 1


def test_build_pattern_response_compact_counts_omitted_rows_when_truncated():
    df = pd.DataFrame(
        {"time": list(range(12)), "close": [100.0 + i for i in range(12)]}
    )
    patterns = [
        {
            "name": f"Pattern {i}",
            "status": "forming",
            "confidence": 0.9 - (i * 0.01),
            "bias": "bullish" if i < 4 else "bearish",
            "end_index": min(i, len(df) - 1),
        }
        for i in range(9)
    ]

    compact = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "classic",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
        detail="compact",
    )

    assert len(compact["top_patterns"]) == 3
    assert compact["patterns_shown"] == 3
    assert compact["n_patterns"] == 9
    assert "patterns_omitted" not in compact
    assert compact["pattern_status"] == "conflicting"
    assert compact["review_recommended"] is False
    assert "suggested_review" not in compact
    assert "confidence" not in compact
    assert "pattern_distribution" not in compact
    assert "strong_patterns" not in compact
    assert "verdict" not in compact
    assert "hints" not in compact
    assert "show_all_hint" not in compact


def test_build_pattern_response_compact_promotes_data_gap_warning():
    df = pd.DataFrame(
        {"time": list(range(12)), "close": [100.0 + i for i in range(12)]}
    )
    df.attrs["warnings"] = [
        "Data quality warning: detected time gaps larger than 1.5 bar intervals."
    ]
    patterns = [
        {
            "name": "Hammer",
            "status": "forming",
            "confidence": 0.8,
            "bias": "bullish",
            "end_index": 10,
        }
    ]

    compact = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "classic",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
        detail="compact",
    )

    assert compact["data_quality"] == {
        "status": "warning",
        "patterns_reliability": "degraded",
        "issues": ["time_gaps"],
    }
    keys = list(compact)
    assert keys.index("data_quality") < keys.index("pattern_status")
    assert compact["warnings"] == df.attrs["warnings"]


def test_build_pattern_response_compact_suppresses_low_confidence_elliott_bias():
    df = pd.DataFrame(
        {"time": list(range(12)), "close": [100.0 + i for i in range(12)]}
    )

    compact = _build_pattern_response(
        "EURUSD",
        "H1",
        150,
        "elliott",
        [
            {
                "wave_type": "Impulse",
                "status": "forming",
                "confidence": 0.1,
                "bias": "bullish",
                "end_index": 10,
            }
        ],
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
        detail="compact",
    )

    assert compact["pattern_status"] == "uncertain"
    assert compact["review_recommended"] is False
    assert compact["pattern_confidence"] == pytest.approx(0.1)
    assert "suggested_review" not in compact
    assert "bias" not in compact
    assert "dominant_direction" not in compact


def test_patterns_detect_elliott_without_timeframe_scans_default_subset(monkeypatch):
    monkeypatch.setattr(
        core_patterns, "TIMEFRAME_MAP", {"M1": 1, "H1": 2, "H4": 3, "D1": 4}
    )

    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )

    def _fake_fetch(symbol, timeframe, limit, denoise):
        return df.copy(), None

    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", _fake_fetch)

    def _fake_detect(_df, _cfg):
        return [
            SimpleNamespace(
                wave_type="Impulse",
                confidence=0.91,
                start_index=0,
                end_index=5,
                start_time=1.0,
                end_time=6.0,
                details={"score": 0.8},
            )
        ]

    monkeypatch.setattr(core_patterns, "_detect_elliott_waves", _fake_detect)

    res = patterns_detect(
        symbol="EURUSD",
        mode="elliott",
        detail="full",
        timeframe=None,
        include_completed=True,
        __cli_raw=True,
    )

    assert res["success"] is True
    assert res["timeframe"] == "ALL"
    assert res["scanned_timeframes"] == ["H1", "H4", "D1"]
    assert res["n_patterns"] == 3
    assert len(res["findings"]) == 3
    assert {p["timeframe"] for p in res["patterns"]} == {"H1", "H4", "D1"}


def test_patterns_detect_elliott_scan_timeframes_can_override_default(monkeypatch):
    monkeypatch.setattr(
        core_patterns, "TIMEFRAME_MAP", {"M15": 1, "H1": 2, "H4": 3, "D1": 4}
    )

    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )

    def _fake_fetch(symbol, timeframe, limit, denoise):
        return df.copy(), None

    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", _fake_fetch)

    def _fake_detect(_df, _cfg):
        return [
            SimpleNamespace(
                wave_type="Impulse",
                confidence=0.88,
                start_index=0,
                end_index=5,
                start_time=1.0,
                end_time=6.0,
                details={"score": 0.75},
            )
        ]

    monkeypatch.setattr(core_patterns, "_detect_elliott_waves", _fake_detect)

    res = patterns_detect(
        symbol="EURUSD",
        mode="elliott",
        detail="full",
        timeframe=None,
        include_completed=True,
        config={"scan_timeframes": ["M15", "H1"], "max_scan_timeframes": 2},
        __cli_raw=True,
    )

    assert res["success"] is True
    assert res["scanned_timeframes"] == ["M15", "H1"]
    assert res["n_patterns"] == 2


def test_patterns_detect_classic_applies_regime_context(monkeypatch):
    n = 180
    df = pd.DataFrame(
        {
            "time": np.arange(n, dtype=float),
            "close": np.linspace(100.0, 140.0, n),
            "tick_volume": np.full(n, 100.0),
        }
    )

    monkeypatch.setattr(
        core_patterns,
        "_fetch_pattern_data",
        lambda symbol, timeframe, limit, denoise: (df.copy(), None),
    )

    def _fake_run_classic_engine(engine, symbol, source_df, cfg, config):
        _ = (engine, symbol, source_df, cfg, config)
        return (
            [
                {
                    "name": "Ascending Triangle",
                    "status": "forming",
                    "confidence": 0.60,
                    "start_index": 20,
                    "end_index": n - 2,
                    "details": {
                        "bias": "bullish",
                        "support": 132.0,
                        "resistance": 141.0,
                    },
                }
            ],
            None,
        )

    monkeypatch.setattr(core_patterns, "_run_classic_engine", _fake_run_classic_engine)

    res = patterns_detect(
        symbol="EURUSD",
        mode="classic",
        detail="full",
        timeframe="H1",
        include_completed=True,
        __cli_raw=True,
    )

    assert res["success"] is True
    assert res["n_patterns"] == 1
    row = res["patterns"][0]
    regime = row["details"]["regime_context"]
    assert regime["state"] == "trending"
    assert regime["direction"] == "bullish"
    assert regime["status"] == "aligned"
    assert float(row["confidence"]) > 0.60


def test_patterns_detect_elliott_with_explicit_timeframe_uses_single_output(
    monkeypatch,
):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )

    def _fake_fetch(symbol, timeframe, limit, denoise):
        return df.copy(), None

    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", _fake_fetch)

    def _fake_detect(_df, _cfg):
        return [
            SimpleNamespace(
                wave_type="Correction",
                confidence=0.75,
                start_index=1,
                end_index=5,
                start_time=2.0,
                end_time=6.0,
                details={"score": 0.7},
            )
        ]

    monkeypatch.setattr(core_patterns, "_detect_elliott_waves", _fake_detect)

    res = patterns_detect(
        symbol="EURUSD",
        mode="elliott",
        detail="full",
        timeframe="H1",
        include_completed=True,
        __cli_raw=True,
    )

    assert res["success"] is True
    assert res["timeframe"] == "H1"
    assert "findings" not in res
    assert res["n_patterns"] == 1


def test_patterns_detect_elliott_with_explicit_timeframe_hidden_completed_is_truthful(
    monkeypatch,
):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )

    monkeypatch.setattr(
        core_patterns,
        "_fetch_pattern_data",
        lambda symbol, timeframe, limit, denoise: (df.copy(), None),
    )

    def _fake_detect(_df, _cfg):
        return [
            SimpleNamespace(
                wave_type="Impulse",
                confidence=0.81,
                start_index=0,
                end_index=1,
                start_time=1.0,
                end_time=2.0,
                details={
                    "trend": "bull",
                    "pattern_confirmed": True,
                    "has_unconfirmed_terminal_pivot": False,
                },
            )
        ]

    monkeypatch.setattr(core_patterns, "_detect_elliott_waves", _fake_detect)

    res = patterns_detect(
        symbol="EURUSD",
        mode="elliott",
        detail="full",
        timeframe="H1",
        limit=100,
        include_completed=False,
        __cli_raw=True,
    )

    assert res["success"] is True
    assert res["n_patterns"] == 0
    assert res["completed_patterns_hidden"] == 1
    assert "No forming Elliott Wave structures detected" in res["diagnostic"]
    assert "completed_patterns_preview" in res
    assert res["completed_patterns_preview"][0]["pattern"] == "Impulse"


def test_patterns_detect_elliott_scan_hidden_completed_is_truthful(monkeypatch):
    monkeypatch.setattr(
        core_patterns, "TIMEFRAME_MAP", {"M1": 1, "H1": 2, "H4": 3, "D1": 4}
    )

    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )

    monkeypatch.setattr(
        core_patterns,
        "_fetch_pattern_data",
        lambda symbol, timeframe, limit, denoise: (df.copy(), None),
    )

    def _fake_detect(_df, _cfg):
        return [
            SimpleNamespace(
                wave_type="Correction",
                confidence=0.79,
                start_index=0,
                end_index=1,
                start_time=1.0,
                end_time=2.0,
                details={
                    "trend": "bear",
                    "pattern_confirmed": True,
                    "has_unconfirmed_terminal_pivot": False,
                },
            )
        ]

    monkeypatch.setattr(core_patterns, "_detect_elliott_waves", _fake_detect)

    res = patterns_detect(
        symbol="EURUSD",
        mode="elliott",
        detail="full",
        timeframe=None,
        include_completed=False,
        __cli_raw=True,
    )

    assert res["success"] is True
    assert res["n_patterns"] == 0
    assert res["completed_patterns_hidden"] == 3
    assert (
        "No forming Elliott Wave structures were detected across scanned timeframes."
        in res["diagnostic"]
    )
    assert len(res["completed_patterns_preview"]) == 3
    assert {item["timeframe"] for item in res["completed_patterns_preview"]} == {
        "H1",
        "H4",
        "D1",
    }
    assert all(
        "No forming Elliott Wave structures detected" in row["diagnostic"]
        for row in res["findings"]
    )


def test_patterns_detect_classic_ensemble_merges_engine_outputs(monkeypatch, caplog):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )

    monkeypatch.setattr(
        core_patterns,
        "_fetch_pattern_data",
        lambda symbol, timeframe, limit, denoise: (df.copy(), None),
    )

    def _fake_engine(engine, symbol, df_in, cfg, config):
        _ = symbol
        _ = df_in
        _ = cfg
        _ = config
        if engine == "native":
            return [
                {
                    "name": "Double Top",
                    "status": "forming",
                    "confidence": 0.8,
                    "start_index": 10,
                    "end_index": 20,
                    "start_date": "A",
                    "end_date": "B",
                    "details": {"x": 1},
                }
            ], None
        if engine == "stock_pattern":
            return [
                {
                    "name": "Double Top",
                    "status": "forming",
                    "confidence": 0.6,
                    "start_index": 12,
                    "end_index": 22,
                    "start_date": "C",
                    "end_date": "D",
                    "details": {"y": 2},
                }
            ], None
        return [], "unexpected"

    monkeypatch.setattr(core_patterns, "_run_classic_engine", _fake_engine)

    with caplog.at_level(logging.DEBUG, logger=core_patterns.logger.name):
        res = patterns_detect(
            symbol="EURUSD",
            mode="classic",
            detail="full",
            timeframe="H1",
            engine="native,stock_pattern",
            ensemble=True,
            include_completed=True,
            __cli_raw=True,
        )

    assert res["success"] is True
    assert res["engine"] == "ensemble"
    assert res["n_patterns"] == 1
    assert res["patterns"][0]["support_count"] == 2
    assert set(res["patterns"][0]["source_engines"]) == {"native", "stock_pattern"}
    assert "event=finish operation=patterns_detect success=True" in caplog.text


def test_patterns_detect_rejects_hidden_precise_engine(monkeypatch):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3],
            "close": [100.0, 101.0, 102.0],
            "high": [100.5, 101.5, 102.5],
            "low": [99.5, 100.5, 101.5],
        }
    )
    monkeypatch.setattr(
        core_patterns,
        "_fetch_pattern_data",
        lambda *_args, **_kwargs: (df.copy(), None),
    )

    res = patterns_detect(
        symbol="EURUSD",
        mode="classic",
        timeframe="H1",
        engine="precise_patterns",
        __cli_raw=True,
    )

    assert "error" in res
    assert "Invalid classic engine" in str(res["error"])


def test_patterns_detect_engine_findings_report_hidden_completed(monkeypatch):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )
    monkeypatch.setattr(
        core_patterns,
        "_fetch_pattern_data",
        lambda symbol, timeframe, limit, denoise: (df.copy(), None),
    )
    monkeypatch.setattr(
        core_patterns,
        "_run_classic_engine",
        lambda engine, symbol, df_in, cfg, config: (
            [
                {
                    "name": "Triangle",
                    "status": "forming",
                    "confidence": 0.8,
                    "start_index": 1,
                    "end_index": 4,
                },
                {
                    "name": "Triangle",
                    "status": "completed",
                    "confidence": 0.7,
                    "start_index": 0,
                    "end_index": 3,
                },
            ],
            None,
        ),
    )

    res = patterns_detect(
        symbol="EURUSD",
        mode="classic",
        detail="full",
        timeframe="H1",
        include_completed=False,
        __cli_raw=True,
    )

    assert res["success"] is True
    finding = res["engine_findings"][0]
    assert finding["n_patterns"] == 1
    assert finding["n_completed"] == 0
    assert finding["n_completed_hidden"] == 1
    assert finding["n_patterns_total"] == 2


def test_patterns_detect_classic_invalid_engine_returns_error(monkeypatch):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )
    monkeypatch.setattr(
        core_patterns,
        "_fetch_pattern_data",
        lambda symbol, timeframe, limit, denoise: (df.copy(), None),
    )

    res = patterns_detect(
        symbol="EURUSD",
        mode="classic",
        timeframe="H1",
        engine="bad_engine",
        __cli_raw=True,
    )

    assert "error" in res
    assert "Invalid classic engine" in str(res["error"])


def test_patterns_detect_classic_adds_signal_summary_and_levels(monkeypatch):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [100.0, 101.0, 102.0, 101.0, 100.5, 100.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )
    monkeypatch.setattr(
        core_patterns,
        "_fetch_pattern_data",
        lambda symbol, timeframe, limit, denoise: (df.copy(), None),
    )

    def _fake_engine(engine, symbol, df_in, cfg, config):
        _ = engine
        _ = symbol
        _ = df_in
        _ = cfg
        _ = config
        return [
            {
                "name": "Double Top",
                "status": "forming",
                "confidence": 0.9,
                "start_index": 1,
                "end_index": 5,
                "details": {"support": 98.5, "resistance": 102.5},
            },
            {
                "name": "Bull Pennant",
                "status": "forming",
                "confidence": 0.7,
                "start_index": 1,
                "end_index": 5,
                "details": {"support": 98.8, "resistance": 102.2},
            },
        ], None

    monkeypatch.setattr(core_patterns, "_run_classic_engine", _fake_engine)

    res = patterns_detect(
        symbol="EURUSD",
        mode="classic",
        detail="full",
        timeframe="H1",
        include_completed=True,
        __cli_raw=True,
    )

    assert res["success"] is True
    assert "signal_summary" not in res
    assert res["summary"]["signal_bias"]["conflict"] is True
    assert res["summary"]["signal_bias"]["net_bias"] == "mixed"
    rows = res["patterns"]
    assert all("bias" in row for row in rows)
    assert all("reference_price" in row for row in rows)
    assert all(
        "volume_confirmation" in row["details"]
        for row in rows
        if isinstance(row.get("details"), dict)
    )
    assert all("target_price" in row for row in rows)
    assert all("invalidation_price" in row for row in rows)


def test_build_pattern_response_compact_marks_conflicting_signal_wait():
    df = pd.DataFrame({"time": [1, 2, 3], "close": [10.0, 10.5, 10.0]})
    patterns = [
        {
            "name": "Double Top",
            "status": "forming",
            "confidence": 0.9,
            "end_index": 2,
            "bias": "bearish",
        },
        {
            "name": "Bull Pennant",
            "status": "forming",
            "confidence": 0.9,
            "end_index": 2,
            "bias": "bullish",
        },
    ]

    compact = _build_pattern_response(
        "EURUSD",
        "H1",
        100,
        "classic",
        patterns,
        include_completed=False,
        include_series=False,
        series_time="string",
        df=df,
        detail="compact",
    )

    assert compact["pattern_status"] == "conflicting"
    assert compact["review_recommended"] is False
    assert "bias" not in compact
    assert compact["pattern_confidence"] == 0.0
    assert "suggested_review" not in compact
    assert compact["conflict"] == "both_bullish_and_bearish_patterns_present"


def test_run_classic_engine_native_multiscale_merges(monkeypatch):
    cfg = ClassicDetectorConfig(min_distance=6, min_prominence_pct=0.6)
    df = pd.DataFrame(
        {"close": np.linspace(100.0, 120.0, 200), "time": np.arange(200, dtype=float)}
    )

    def _fake_format(_df, cfg_in):
        md = int(cfg_in.min_distance)
        return [
            {
                "name": "Double Top",
                "status": "forming",
                "confidence": 0.4 + 0.03 * md,
                "start_index": 40 + (md % 2),
                "end_index": 90 + (md % 3),
                "start_date": None,
                "end_date": None,
                "details": {"min_distance": md},
            }
        ]

    monkeypatch.setattr(core_patterns, "_format_classic_native_patterns", _fake_format)

    out, err = core_patterns._run_classic_engine_native(
        symbol="EURUSD",
        df=df,
        cfg=cfg,
        config={"native_multiscale": True, "native_scale_factors": [0.8, 1.0, 1.25]},
    )
    assert err is None
    assert out
    assert out[0]["details"].get("native_multiscale") is True
    assert int(out[0].get("support_count", 1)) >= 2


def test_attach_pattern_usage_notice_compact_adds_confidence_basis():
    result = {"success": True, "patterns_shown": 2, "pattern_confidence": 0.3}
    core_patterns._attach_pattern_usage_notice(result)
    assert result["is_signal"] is False
    assert result["usage"] == "information_only"
    assert "heuristic" in result["confidence_basis"]
    # compact shape should not carry the verbose calibration block
    assert "calibration" not in result


def test_attach_pattern_usage_notice_full_adds_calibration():
    result = {"success": True, "patterns": []}
    core_patterns._attach_pattern_usage_notice(result)
    assert result["calibration"]["confidence"].startswith("heuristic")
