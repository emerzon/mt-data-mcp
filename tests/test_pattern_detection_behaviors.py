import numpy as np
import pandas as pd
from types import SimpleNamespace

from src.mtdata.core import patterns as core_patterns
from src.mtdata.core.patterns import _apply_config_to_obj, _build_pattern_response
from src.mtdata.patterns.candlestick import _is_candlestick_allowed
from src.mtdata.patterns.classic import ClassicDetectorConfig, _fit_lines_and_arrays, _count_recent_touches, detect_classic_patterns
import src.mtdata.patterns.classic as classic_mod


def test_fit_lines_and_arrays_uses_cfg_for_robust_fit(monkeypatch):
    calls = []

    def _fake_robust(x, y, cfg):
        calls.append(bool(cfg.use_robust_fit))
        # slope, intercept, r2
        return 1.0, 2.0, 0.9

    monkeypatch.setattr(classic_mod, "_fit_line_robust", _fake_robust)

    ih = np.array([1, 5, 9], dtype=int)
    il = np.array([2, 6, 10], dtype=int)
    c = np.linspace(10.0, 20.0, 12)
    cfg = ClassicDetectorConfig(use_robust_fit=True)

    sh, bh, r2h, sl, bl, r2l, upper, lower = _fit_lines_and_arrays(ih, il, c, len(c), cfg)

    assert calls == [True, True]
    assert (sh, bh, r2h) == (1.0, 2.0, 0.9)
    assert (sl, bl, r2l) == (1.0, 2.0, 0.9)
    assert upper.shape[0] == len(c)
    assert lower.shape[0] == len(c)


def test_candlestick_allowed_respects_whitelist_when_not_robust_only():
    robust_set = {"engulfing", "harami"}
    whitelist = {"engulfing"}

    assert _is_candlestick_allowed(
        "engulfing",
        robust_only=False,
        robust_set=robust_set,
        whitelist_set=whitelist,
    )
    assert not _is_candlestick_allowed(
        "doji",
        robust_only=False,
        robust_set=robust_set,
        whitelist_set=whitelist,
    )


def test_candlestick_allowed_requires_robust_when_enabled():
    robust_set = {"engulfing", "harami"}

    assert _is_candlestick_allowed(
        "engulfing",
        robust_only=True,
        robust_set=robust_set,
        whitelist_set=None,
    )
    assert not _is_candlestick_allowed(
        "doji",
        robust_only=True,
        robust_set=robust_set,
        whitelist_set=None,
    )


def test_apply_config_to_obj_coerces_bool_strings():
    cfg = ClassicDetectorConfig()
    assert cfg.use_robust_fit is True

    _apply_config_to_obj(cfg, {"use_robust_fit": "false", "use_dtw_check": "0"})
    assert cfg.use_robust_fit is False
    assert cfg.use_dtw_check is False

    _apply_config_to_obj(cfg, {"use_robust_fit": "true", "use_dtw_check": "yes"})
    assert cfg.use_robust_fit is True
    assert cfg.use_dtw_check is True


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
    assert with_completed["n_patterns"] == 2


def test_patterns_detect_elliott_without_timeframe_scans_all(monkeypatch):
    monkeypatch.setattr(core_patterns, "TIMEFRAME_MAP", {"M1": 1, "H1": 2})

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

    res = core_patterns.patterns_detect(
        symbol="EURUSD",
        mode="elliott",
        timeframe=None,
        include_completed=True,
        __cli_raw=True,
    )

    assert res["success"] is True
    assert res["timeframe"] == "ALL"
    assert res["scanned_timeframes"] == ["M1", "H1"]
    assert res["n_patterns"] == 2
    assert len(res["findings"]) == 2
    assert {p["timeframe"] for p in res["patterns"]} == {"M1", "H1"}


def test_patterns_detect_elliott_with_explicit_timeframe_uses_single_output(monkeypatch):
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

    res = core_patterns.patterns_detect(
        symbol="EURUSD",
        mode="elliott",
        timeframe="H1",
        include_completed=True,
        __cli_raw=True,
    )

    assert res["success"] is True
    assert res["timeframe"] == "H1"
    assert "findings" not in res
    assert res["n_patterns"] == 1


def test_count_recent_touches_respects_lookback():
    series = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    close = np.array([9.7, 10.3, 9.9, 10.0, 10.1])
    # lookback=3 => compare [9.9, 10.0, 10.1] against 10.0 with tol 0.15 => 3 touches
    assert _count_recent_touches(series, close, tol_abs=0.15, lookback_bars=3) == 3


def test_detect_classic_patterns_disables_aliases_by_default(monkeypatch):
    n = 150
    x = np.linspace(0, 4 * np.pi, n)
    close = 100 + 0.3 * np.arange(n) + 4.0 * np.sin(x)
    df = pd.DataFrame({"time": np.arange(n, dtype=float), "close": close})

    peaks = np.array([20, 45, 70, 95, 120, 145], dtype=int)
    troughs = np.array([10, 35, 60, 85, 110, 135], dtype=int)
    monkeypatch.setattr(classic_mod, "_detect_pivots_close", lambda c, cfg: (peaks, troughs))

    out_default = detect_classic_patterns(df, ClassicDetectorConfig(max_consolidation_bars=5))
    names_default = {p.name for p in out_default}
    assert "Ascending Trend Line" in names_default
    assert "Trend Line" not in names_default
    assert "Trend Channel" not in names_default

    out_alias = detect_classic_patterns(
        df, ClassicDetectorConfig(include_aliases=True, max_consolidation_bars=5)
    )
    names_alias = {p.name for p in out_alias}
    assert "Trend Line" in names_alias or "Trend Channel" in names_alias


def test_patterns_detect_classic_ensemble_merges_engine_outputs(monkeypatch):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )

    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", lambda symbol, timeframe, limit, denoise: (df.copy(), None))

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
        if engine == "precise_patterns":
            return [], "precise-patterns unavailable"
        return [], "unexpected"

    monkeypatch.setattr(core_patterns, "_run_classic_engine", _fake_engine)

    res = core_patterns.patterns_detect(
        symbol="EURUSD",
        mode="classic",
        timeframe="H1",
        engine="native,stock_pattern,precise_patterns",
        ensemble=True,
        include_completed=True,
        __cli_raw=True,
    )

    assert res["success"] is True
    assert res["engine"] == "ensemble"
    assert res["n_patterns"] == 1
    assert res["patterns"][0]["support_count"] == 2
    assert set(res["patterns"][0]["source_engines"]) == {"native", "stock_pattern"}
    assert "engine_errors" in res
    assert "precise_patterns" in res["engine_errors"]


def test_patterns_detect_classic_invalid_engine_returns_error(monkeypatch):
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "tick_volume": [100, 100, 100, 100, 100, 100],
        }
    )
    monkeypatch.setattr(core_patterns, "_fetch_pattern_data", lambda symbol, timeframe, limit, denoise: (df.copy(), None))

    res = core_patterns.patterns_detect(
        symbol="EURUSD",
        mode="classic",
        timeframe="H1",
        engine="bad_engine",
        __cli_raw=True,
    )

    assert "error" in res
    assert "Invalid classic engine" in str(res["error"])


def test_detect_pivots_close_prefers_high_low_when_enabled():
    n = 160
    close = np.full(n, 100.0, dtype=float)
    high = close.copy()
    low = close.copy()
    high[[30, 75, 120]] = [106.0, 107.0, 106.5]
    low[[45, 95, 140]] = [94.0, 93.5, 94.2]

    cfg_hl = ClassicDetectorConfig(
        pivot_use_hl=True,
        pivot_use_atr_adaptive_prominence=False,
        pivot_use_atr_adaptive_distance=False,
        min_prominence_pct=1.0,
        min_distance=5,
    )
    cfg_close = ClassicDetectorConfig(
        pivot_use_hl=False,
        pivot_use_atr_adaptive_prominence=False,
        pivot_use_atr_adaptive_distance=False,
        min_prominence_pct=1.0,
        min_distance=5,
    )

    peaks_hl, troughs_hl = classic_mod._detect_pivots_close(close, cfg_hl, high, low)
    peaks_close, troughs_close = classic_mod._detect_pivots_close(close, cfg_close, high, low)

    assert peaks_hl.size >= 2
    assert troughs_hl.size >= 2
    assert peaks_close.size == 0
    assert troughs_close.size == 0


def test_detect_classic_triangle_marks_completed_on_breakout(monkeypatch):
    n = 170
    close = np.full(n, 100.0, dtype=float)
    top_line = np.linspace(106.0, 100.0, n)
    bot_line = np.linspace(94.0, 99.0, n)
    close[:-3] = (top_line[:-3] + bot_line[:-3]) / 2.0
    close[-3:] = top_line[-3:] + 1.0

    peaks = np.array([30, 60, 90, 120, 150], dtype=int)
    troughs = np.array([20, 50, 80, 110, 140], dtype=int)
    close[peaks] = top_line[peaks]
    close[troughs] = bot_line[troughs]
    close[-3:] = top_line[-3:] + 1.0
    df = pd.DataFrame({"time": np.arange(n, dtype=float), "close": close, "high": close + 0.2, "low": close - 0.2})
    monkeypatch.setattr(classic_mod, "_detect_pivots_close", lambda c, cfg, *args: (peaks, troughs))

    def _fake_fit_lines(ih, il, c, n, cfg):
        return -0.03, 106.0, 0.9, 0.03, 94.0, 0.9, top_line.copy(), bot_line.copy()

    monkeypatch.setattr(classic_mod, "_fit_lines_and_arrays", _fake_fit_lines)

    out = detect_classic_patterns(
        df,
        ClassicDetectorConfig(
            min_channel_touches=2,
            breakout_lookahead=6,
            completion_lookback_bars=6,
            completion_confirm_bars=1,
            max_consolidation_bars=5,
        ),
    )
    tri = [p for p in out if "Triangle" in p.name]
    assert tri
    assert any(p.status == "completed" for p in tri)


def test_detect_classic_confidence_calibration_and_lifecycle(monkeypatch):
    n = 150
    x = np.linspace(0, 4 * np.pi, n)
    close = 100 + 0.25 * np.arange(n) + 3.0 * np.sin(x)
    df = pd.DataFrame({"time": np.arange(n, dtype=float), "close": close, "high": close + 0.3, "low": close - 0.3})

    peaks = np.array([20, 45, 70, 95, 120, 145], dtype=int)
    troughs = np.array([10, 35, 60, 85, 110, 135], dtype=int)
    monkeypatch.setattr(classic_mod, "_detect_pivots_close", lambda c, cfg, *args: (peaks, troughs))

    cfg = ClassicDetectorConfig(
        calibrate_confidence=True,
        confidence_calibration_map={"default": {"0.0": 0.0, "1.0": 0.5}},
        confidence_calibration_blend=1.0,
        max_consolidation_bars=5,
    )
    out = detect_classic_patterns(df, cfg)
    assert out
    sample = out[0]
    assert isinstance(sample.details, dict)
    assert "raw_confidence" in sample.details
    assert "calibrated_confidence" in sample.details
    assert sample.confidence <= float(sample.details["raw_confidence"])
    assert sample.details.get("lifecycle_state") in {"forming", "confirmed"}


def test_run_classic_engine_native_multiscale_merges(monkeypatch):
    cfg = ClassicDetectorConfig(min_distance=6, min_prominence_pct=0.6)
    df = pd.DataFrame({"close": np.linspace(100.0, 120.0, 200), "time": np.arange(200, dtype=float)})

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
