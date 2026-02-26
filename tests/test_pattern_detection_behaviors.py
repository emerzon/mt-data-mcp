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
