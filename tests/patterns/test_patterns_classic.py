"""Classic pattern tests."""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.mtdata.patterns.classic as classic_mod
from src.mtdata.core import patterns as core_patterns
from src.mtdata.core.patterns import _apply_config_to_obj, _build_pattern_response
from src.mtdata.patterns.classic import (
    ClassicDetectorConfig,
    ClassicPatternResult,
    _count_recent_touches,
    _fit_lines_and_arrays,
    detect_classic_patterns,
)


def test_count_recent_touches_respects_lookback():
    series = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    close = np.array([9.7, 10.3, 9.9, 10.0, 10.1])
    # lookback=3 => compare [9.9, 10.0, 10.1] against 10.0 with tol 0.15 => 3 touches
    assert _count_recent_touches(series, close, tol_abs=0.15, lookback_bars=3) == 3


def test_detect_classic_uses_singular_pennant_name(monkeypatch):
    n = 110
    close = np.linspace(100.0, 130.0, n)
    window = 30
    seg = close[-window:].copy()
    seg_peaks = np.array([6, 13, 20, 27], dtype=int)
    seg_troughs = np.array([4, 11, 18, 25], dtype=int)

    top = np.linspace(150.0, 148.0, window)
    bot = np.linspace(144.0, 145.5, window)
    seg[:] = (top + bot) / 2.0
    seg[seg_peaks] = top[seg_peaks]
    seg[seg_troughs] = bot[seg_troughs]
    close[-window:] = seg

    df = pd.DataFrame(
        {
            "time": np.arange(n, dtype=float),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
        }
    )

    def _fake_pivots(c, cfg, *args):
        n_local = len(c)
        if n_local >= 20:
            return np.array([2, 7, 13, n_local - 4], dtype=int), np.array(
                [4, 10, 16, n_local - 2], dtype=int
            )
        return np.array([], dtype=int), np.array([], dtype=int)

    from src.mtdata.patterns.classic_impl import continuation

    monkeypatch.setattr(continuation, "_detect_pivots_close", _fake_pivots)

    def _fake_fit_lines(ih, il, c, n, cfg):
        if n >= 20:
            return (
                -0.07,
                150.0,
                0.9,
                0.05,
                144.0,
                0.9,
                np.linspace(150.0, 148.0, n),
                np.linspace(144.0, 145.5, n),
            )
        x = np.arange(n, dtype=float)
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x.copy(), x.copy()

    monkeypatch.setattr(continuation, "_fit_lines_and_arrays", _fake_fit_lines)

    out = detect_classic_patterns(
        df,
        ClassicDetectorConfig(min_pole_return_pct=1.0, max_consolidation_bars=window),
    )
    names = {p.name for p in out}
    assert "Bull Pennant" in names
    assert "Bull Pennants" not in names


def test_detect_flags_pennants_measure_pole_from_tip_not_last_bar(monkeypatch):
    from src.mtdata.patterns.classic_impl import continuation

    n = 120
    window = 30
    close = np.full(n, 102.0, dtype=float)
    close[75:90] = np.linspace(100.0, 106.0, 15)
    seg = np.linspace(104.0, 103.0, window)
    seg[0] = 106.0
    seg[-1] = 101.8
    close[-window:] = seg

    peaks = np.array([4, 11, 18, 25], dtype=int)
    troughs = np.array([2, 9, 16, 23], dtype=int)
    top = np.linspace(106.0, 104.0, window)
    bot = np.linspace(102.0, 103.0, window)

    monkeypatch.setattr(
        continuation, "_detect_pivots_close", lambda *_args, **_kwargs: (peaks, troughs)
    )
    monkeypatch.setattr(
        continuation,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (
            -0.05,
            106.0,
            0.9,
            0.03,
            102.0,
            0.9,
            top.copy(),
            bot.copy(),
        ),
    )

    out = continuation.detect_flags_pennants(
        close,
        close + 0.2,
        close - 0.2,
        np.arange(n, dtype=float),
        n,
        ClassicDetectorConfig(max_consolidation_bars=window, min_pole_return_pct=4.0),
    )

    assert out
    assert out[0].name == "Bull Pennant"
    assert out[0].details["pole_return_pct"] > 4.0
    assert out[0].details["pole_tip_price"] == pytest.approx(106.0)


def test_detect_flags_pennants_reject_protrend_consolidation(monkeypatch):
    from src.mtdata.patterns.classic_impl import continuation

    n = 120
    window = 30
    close = np.full(n, 102.0, dtype=float)
    close[75:90] = np.linspace(100.0, 106.0, 15)
    seg = np.linspace(103.5, 105.8, window)
    seg[0] = 106.0
    close[-window:] = seg

    peaks = np.array([4, 11, 18, 25], dtype=int)
    troughs = np.array([2, 9, 16, 23], dtype=int)
    top = np.linspace(104.0, 105.6, window)
    bot = np.linspace(102.0, 103.6, window)

    monkeypatch.setattr(
        continuation, "_detect_pivots_close", lambda *_args, **_kwargs: (peaks, troughs)
    )
    monkeypatch.setattr(
        continuation,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (
            0.06,
            104.0,
            0.9,
            0.06,
            102.0,
            0.9,
            top.copy(),
            bot.copy(),
        ),
    )

    out = continuation.detect_flags_pennants(
        close,
        close + 0.2,
        close - 0.2,
        np.arange(n, dtype=float),
        n,
        ClassicDetectorConfig(max_consolidation_bars=window, min_pole_return_pct=4.0),
    )

    assert out == []


def test_detect_classic_channel_parallel_ratio_uses_config(monkeypatch):
    n = 150
    close = np.linspace(100.0, 120.0, n)
    peaks = np.array([20, 45, 70, 95, 120, 145], dtype=int)
    troughs = np.array([10, 35, 60, 85, 110, 135], dtype=int)
    upper = 1.18 * np.arange(n, dtype=float) + 150.0
    lower = 1.00 * np.arange(n, dtype=float) + 120.0
    close[peaks] = upper[peaks]
    close[troughs] = lower[troughs]
    df = pd.DataFrame(
        {
            "time": np.arange(n, dtype=float),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
        }
    )

    monkeypatch.setattr(
        classic_mod, "_detect_pivots_close", lambda c, cfg, *args: (peaks, troughs)
    )
    monkeypatch.setattr(classic_mod, "_is_converging", lambda *args, **kwargs: False)

    def _fake_fit_lines(ih, il, c, n, cfg):
        return 1.18, 150.0, 0.95, 1.00, 120.0, 0.95, upper, lower

    monkeypatch.setattr(classic_mod, "_fit_lines_and_arrays", _fake_fit_lines)

    out_default = detect_classic_patterns(
        df, ClassicDetectorConfig(min_channel_touches=2, max_consolidation_bars=5)
    )
    out_relaxed = detect_classic_patterns(
        df,
        ClassicDetectorConfig(
            min_channel_touches=2,
            max_consolidation_bars=5,
            channel_parallel_slope_ratio=0.2,
        ),
    )

    assert not any("Channel" in p.name for p in out_default)
    assert any("Channel" in p.name for p in out_relaxed)


def test_detect_classic_patterns_historical_scan_finds_older_prefix_pattern(
    monkeypatch,
):
    n = 220
    df = pd.DataFrame(
        {"time": np.arange(n, dtype=float), "close": np.linspace(100.0, 120.0, n)}
    )

    def _fake_rounding(c, t, cfg):
        _ = cfg
        if len(c) != 140:
            return []
        return [
            ClassicPatternResult(
                name="Rounding Bottom",
                status="forming",
                confidence=0.82,
                start_index=100,
                end_index=139,
                start_time=float(t[100]),
                end_time=float(t[139]),
                details={},
            )
        ]

    monkeypatch.setattr(
        classic_mod,
        "_detect_pivots_close",
        lambda c, cfg, *args: (np.array([], dtype=int), np.array([], dtype=int)),
    )
    monkeypatch.setattr(classic_mod, "detect_rounding", _fake_rounding)

    out_default = detect_classic_patterns(
        df, ClassicDetectorConfig(max_consolidation_bars=5)
    )
    out_scan = detect_classic_patterns(
        df,
        ClassicDetectorConfig(
            max_consolidation_bars=5,
            scan_historical=True,
            scan_step_bars=10,
            scan_min_prefix_bars=120,
        ),
    )

    assert not any(p.name == "Rounding Bottom" for p in out_default)
    match = next(p for p in out_scan if p.name == "Rounding Bottom")
    assert match.status == "forming"
    assert match.end_index == 139

    out_scan_completed = detect_classic_patterns(
        df,
        ClassicDetectorConfig(
            max_consolidation_bars=5,
            scan_historical=True,
            scan_step_bars=10,
            scan_min_prefix_bars=120,
            auto_complete_stale_forming=True,
        ),
    )
    assert (
        next(p for p in out_scan_completed if p.name == "Rounding Bottom").status
        == "completed"
    )


def test_detect_classic_patterns_historical_scan_reuses_global_pivots(monkeypatch):
    n = 220
    df = pd.DataFrame(
        {"time": np.arange(n, dtype=float), "close": np.linspace(100.0, 120.0, n)}
    )
    calls = {"count": 0}

    def _fake_pivots(c, cfg, *args):
        _ = c
        _ = cfg
        _ = args
        calls["count"] += 1
        return np.array([], dtype=int), np.array([], dtype=int)

    monkeypatch.setattr(classic_mod, "_detect_pivots_close", _fake_pivots)
    monkeypatch.setattr(classic_mod, "detect_diamonds", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        classic_mod, "detect_flags_pennants", lambda *args, **kwargs: []
    )

    out = detect_classic_patterns(
        df,
        ClassicDetectorConfig(
            max_consolidation_bars=5,
            scan_historical=True,
            scan_step_bars=10,
            scan_min_prefix_bars=120,
        ),
    )

    assert out == []
    assert calls["count"] == 1


def test_detect_classic_patterns_surfaces_confidence_calibration_errors(monkeypatch):
    n = 150
    df = pd.DataFrame(
        {"time": np.arange(n, dtype=float), "close": np.linspace(100.0, 110.0, n)}
    )

    monkeypatch.setattr(
        classic_mod,
        "_detect_pivots_close",
        lambda c, cfg, *args: (np.array([], dtype=int), np.array([], dtype=int)),
    )
    monkeypatch.setattr(
        classic_mod,
        "detect_rounding",
        lambda c, t, cfg: [
            ClassicPatternResult(
                name="Rounding Bottom",
                status="forming",
                confidence=0.7,
                start_index=40,
                end_index=149,
                start_time=float(t[40]),
                end_time=float(t[149]),
                details={},
            )
        ],
    )
    monkeypatch.setattr(
        classic_mod,
        "_calibrate_confidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        detect_classic_patterns(df, ClassicDetectorConfig(max_consolidation_bars=5))


def test_detect_cup_handle_respects_configurable_handle_pullback():
    from src.mtdata.patterns.classic_impl.continuation import detect_cup_handle

    n = 180
    anchors = [
        (0, 100.0),
        (25, 100.0),
        (90, 82.0),
        (135, 100.0),
        (150, 98.0),
        (165, 95.0),
        (179, 101.0),
    ]
    close = np.full(n, 100.0, dtype=float)
    for (a_i, a_v), (b_i, b_v) in zip(anchors, anchors[1:]):
        close[a_i : b_i + 1] = np.linspace(a_v, b_v, b_i - a_i + 1)

    strict_cfg = ClassicDetectorConfig(
        cup_handle_max_handle_pullback_pct=4.0,
        breakout_lookahead=40,
        completion_lookback_bars=40,
    )
    relaxed_cfg = ClassicDetectorConfig(
        cup_handle_max_handle_pullback_pct=6.0,
        breakout_lookahead=40,
        completion_lookback_bars=40,
    )

    out_strict = detect_cup_handle(close, np.arange(n, dtype=float), strict_cfg)
    out_relaxed = detect_cup_handle(close, np.arange(n, dtype=float), relaxed_cfg)

    assert out_strict == []
    assert out_relaxed
    assert out_relaxed[0].status == "completed"
    assert out_relaxed[0].details["handle_pullback_pct"] == pytest.approx(5.0)


def test_detect_cup_handle_scores_rim_mismatch_instead_of_hard_reject():
    from src.mtdata.patterns.classic_impl.continuation import detect_cup_handle

    n = 180
    anchors = [
        (0, 100.0),
        (25, 100.0),
        (90, 82.0),
        (135, 104.0),
        (150, 102.0),
        (165, 99.0),
        (179, 106.0),
    ]
    close = np.full(n, 100.0, dtype=float)
    for (a_i, a_v), (b_i, b_v) in zip(anchors, anchors[1:]):
        close[a_i : b_i + 1] = np.linspace(a_v, b_v, b_i - a_i + 1)

    strict_cfg = ClassicDetectorConfig(
        cup_handle_max_rim_mismatch_pct=2.0,
        cup_handle_max_handle_pullback_pct=6.0,
        breakout_lookahead=40,
        completion_lookback_bars=40,
        same_level_tol_pct=0.5,
    )
    relaxed_cfg = ClassicDetectorConfig(
        cup_handle_max_rim_mismatch_pct=6.0,
        cup_handle_max_handle_pullback_pct=6.0,
        breakout_lookahead=40,
        completion_lookback_bars=40,
        same_level_tol_pct=0.5,
    )

    out_strict = detect_cup_handle(close, np.arange(n, dtype=float), strict_cfg)
    out_relaxed = detect_cup_handle(close, np.arange(n, dtype=float), relaxed_cfg)

    assert out_strict == []
    assert out_relaxed
    assert out_relaxed[0].status == "completed"
    assert out_relaxed[0].details["near_equal_rim"] == "no"
    assert out_relaxed[0].details["rim_mismatch_pct"] == pytest.approx(
        3.8461538461538463
    )
    assert 0.0 < out_relaxed[0].details["rim_symmetry"] < 1.0


def test_detect_inverted_cup_handle_detects_bearish_variant():
    from src.mtdata.patterns.classic_impl.continuation import detect_cup_handle

    n = 180
    anchors = [
        (0, 100.0),
        (25, 100.0),
        (90, 118.0),
        (135, 100.0),
        (150, 102.0),
        (165, 105.0),
        (179, 99.0),
    ]
    close = np.full(n, 100.0, dtype=float)
    for (a_i, a_v), (b_i, b_v) in zip(anchors, anchors[1:]):
        close[a_i : b_i + 1] = np.linspace(a_v, b_v, b_i - a_i + 1)

        out = detect_cup_handle(
            close,
            np.arange(n, dtype=float),
            ClassicDetectorConfig(
                breakout_lookahead=40,
                completion_lookback_bars=40,
                cup_handle_max_depth_pct=100.0,
                cup_handle_max_handle_pullback_pct=30.0,
            ),
        )

    inverted = next(
        pattern for pattern in out if pattern.name == "Inverted Cup and Handle"
    )
    assert inverted.details["breakout_direction"] == "down"
    assert inverted.details["bias"] == "bearish"
    assert inverted.status == "completed"


def test_detect_triangles_skip_same_sign_converging_shapes(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 150
    peaks = np.array([30, 60, 90, 120], dtype=int)
    troughs = np.array([20, 50, 80, 110], dtype=int)
    close = np.linspace(100.0, 130.0, n)
    top = np.linspace(112.0, 118.0, n)
    bot = np.linspace(102.0, 106.0, n)
    close[peaks] = top[peaks]
    close[troughs] = bot[troughs]

    monkeypatch.setattr(
        shapes,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (0.04, 112.0, 0.9, 0.02, 102.0, 0.9, top, bot),
    )
    monkeypatch.setattr(shapes, "_is_converging", lambda *_args, **_kwargs: True)

    tri = shapes.detect_triangles(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2),
    )
    wedge = shapes.detect_wedges(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2),
    )

    assert tri == []
    assert wedge
    assert wedge[0].name == "Rising Wedge"


def test_detect_triangles_allows_near_flat_boundary_with_same_sign_slopes(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 150
    peaks = np.array([30, 60, 90, 120], dtype=int)
    troughs = np.array([20, 50, 80, 110], dtype=int)
    close = np.linspace(100.0, 130.0, n)
    top = np.linspace(112.0, 114.0, n)
    bot = np.linspace(102.0, 112.0, n)
    close[peaks] = top[peaks]
    close[troughs] = bot[troughs]

    monkeypatch.setattr(
        shapes,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (0.01, 112.0, 0.9, 0.08, 102.0, 0.9, top, bot),
    )
    monkeypatch.setattr(shapes, "_is_converging", lambda *_args, **_kwargs: True)

    tri = shapes.detect_triangles(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2, max_flat_slope=0.02),
    )

    assert tri
    assert tri[0].name == "Ascending Triangle"


def test_detect_triangles_reject_crossed_boundaries(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 150
    peaks = np.array([30, 60, 90, 120], dtype=int)
    troughs = np.array([20, 50, 80, 110], dtype=int)
    close = np.linspace(100.0, 130.0, n)
    top = np.linspace(112.0, 98.0, n)
    bot = np.linspace(102.0, 104.0, n)

    monkeypatch.setattr(
        shapes,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (
            -0.09,
            112.0,
            0.9,
            0.013,
            102.0,
            0.9,
            top.copy(),
            bot.copy(),
        ),
    )
    monkeypatch.setattr(shapes, "_is_converging", lambda *_args, **_kwargs: True)

    out = shapes.detect_triangles(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2),
    )

    assert out == []


def test_detect_triangles_reject_both_flat_boundaries(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 150
    peaks = np.array([30, 60, 90, 120], dtype=int)
    troughs = np.array([20, 50, 80, 110], dtype=int)
    close = np.linspace(100.0, 130.0, n)
    top = np.linspace(112.0, 112.8, n)
    bot = np.linspace(102.0, 101.2, n)
    close[peaks] = top[peaks]
    close[troughs] = bot[troughs]

    monkeypatch.setattr(
        shapes,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (0.005, 112.0, 0.9, -0.005, 102.0, 0.9, top, bot),
    )
    monkeypatch.setattr(shapes, "_is_converging", lambda *_args, **_kwargs: True)

    out = shapes.detect_triangles(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2, max_flat_slope=0.02),
    )

    assert out == []


def test_detect_wedges_reject_near_flat_boundaries(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 150
    peaks = np.array([30, 60, 90, 120], dtype=int)
    troughs = np.array([20, 50, 80, 110], dtype=int)
    close = np.linspace(100.0, 130.0, n)
    top = np.linspace(112.0, 113.0, n)
    bot = np.linspace(102.0, 102.9, n)
    close[peaks] = top[peaks]
    close[troughs] = bot[troughs]

    monkeypatch.setattr(
        shapes,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (0.01, 112.0, 0.9, 0.008, 102.0, 0.9, top, bot),
    )
    monkeypatch.setattr(shapes, "_is_converging", lambda *_args, **_kwargs: True)

    out = shapes.detect_wedges(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2, max_flat_slope=0.02),
    )

    assert out == []


def test_detect_flags_prefers_flag_when_convergence_is_only_noise(monkeypatch):
    from src.mtdata.patterns.classic_impl import continuation

    n = 160
    window = 30
    close = np.full(n, 100.0, dtype=float)
    close[-window:] = np.linspace(104.0, 103.0, window)
    high = close + 0.1
    low = close - 0.1
    peaks = np.array([5, 12, 19, 26], dtype=int)
    troughs = np.array([3, 10, 17, 24], dtype=int)
    top = np.linspace(104.0, 103.01, window)
    bot = np.linspace(102.0, 101.02, window)

    monkeypatch.setattr(
        continuation, "_detect_pivots_close", lambda *_args, **_kwargs: (peaks, troughs)
    )
    monkeypatch.setattr(
        continuation,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (
            -0.03,
            104.0,
            0.9,
            -0.029,
            102.0,
            0.9,
            top.copy(),
            bot.copy(),
        ),
    )

    out = continuation.detect_flags_pennants(
        close,
        high,
        low,
        np.arange(n, dtype=float),
        n,
        ClassicDetectorConfig(
            max_consolidation_bars=window,
            min_pole_return_pct=2.0,
            pennant_min_convergence_ratio=0.05,
        ),
    )

    assert out
    assert out[0].name == "Bull Flag"


def test_detect_flags_pennants_excludes_pole_from_consolidation_fit(monkeypatch):
    from src.mtdata.patterns.classic_impl import continuation

    n = 130
    window = 30
    close = np.full(n, 98.0, dtype=float)
    close[-window:] = np.concatenate(
        (
            np.linspace(100.0, 110.0, 10),
            np.linspace(109.5, 106.0, 20),
        )
    )
    high = close + 0.1
    low = close - 0.1
    captured = {}

    def _fake_pivots(seg, cfg, *args):
        _ = cfg
        _ = args
        captured["seg"] = seg.copy()
        return np.array([2, 7, 12, 17], dtype=int), np.array([4, 9, 14, 19], dtype=int)

    monkeypatch.setattr(continuation, "_detect_pivots_close", _fake_pivots)
    monkeypatch.setattr(
        continuation,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (
            -0.08,
            110.0,
            0.9,
            -0.05,
            107.0,
            0.9,
            np.linspace(110.0, 106.5, 21),
            np.linspace(107.5, 104.5, 21),
        ),
    )

    out = continuation.detect_flags_pennants(
        close,
        high,
        low,
        np.arange(n, dtype=float),
        n,
        ClassicDetectorConfig(max_consolidation_bars=window, min_pole_return_pct=5.0),
    )

    assert out
    assert captured["seg"].size == 21
    assert captured["seg"][0] == pytest.approx(110.0)
    assert out[0].details["consolidation_start_index"] == (n - window + 9)


def test_detect_rectangles_mark_completed_on_breakout():
    from src.mtdata.patterns.classic_impl.shapes import detect_rectangles

    n = 120
    close = np.full(n, 100.0, dtype=float)
    peaks = np.array([20, 40, 60], dtype=int)
    troughs = np.array([30, 50, 70], dtype=int)
    close[peaks] = 105.0
    close[troughs] = 95.0
    close[-1] = 106.0

    out = detect_rectangles(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2),
    )

    assert out
    assert out[0].status == "completed"
    assert out[0].details["breakout_direction"] == "up"


def test_detect_rectangles_without_time_values_return_none_timestamps():
    from src.mtdata.patterns.classic_impl.shapes import detect_rectangles

    n = 120
    close = np.full(n, 100.0, dtype=float)
    peaks = np.array([20, 40, 60], dtype=int)
    troughs = np.array([30, 50, 70], dtype=int)
    close[peaks] = 105.0
    close[troughs] = 95.0
    close[-1] = 106.0

    out = detect_rectangles(
        close,
        peaks,
        troughs,
        np.asarray([], dtype=float),
        ClassicDetectorConfig(min_channel_touches=2),
    )

    assert out
    assert out[0].start_time is None
    assert out[0].end_time is None


def test_detect_trend_lines_extend_to_current_bar():
    from src.mtdata.patterns.classic_impl.trend import detect_trend_lines

    n = 140
    close = np.linspace(100.0, 120.0, n)
    peaks = np.array([20, 50, 80, 110], dtype=int)
    troughs = np.array([10, 40, 70, 100], dtype=int)
    close[peaks] = np.linspace(104.0, 118.0, peaks.size)
    close[troughs] = np.linspace(98.0, 112.0, troughs.size)

    out = detect_trend_lines(
        close, peaks, troughs, np.arange(n, dtype=float), ClassicDetectorConfig()
    )

    assert out
    assert all(p.end_index == (n - 1) for p in out)


def test_detect_channels_allow_small_absolute_slope_spread(monkeypatch):
    from src.mtdata.patterns.classic_impl import trend

    n = 160
    close = np.linspace(100.0, 101.0, n)
    peaks = np.array([30, 60, 90, 120, 150], dtype=int)
    troughs = np.array([20, 50, 80, 110, 140], dtype=int)
    upper = 110.0 + (2e-5 * np.arange(n, dtype=float))
    lower = 100.0 + (9e-5 * np.arange(n, dtype=float))

    monkeypatch.setattr(
        trend,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (2e-5, 110.0, 0.95, 9e-5, 100.0, 0.95, upper, lower),
    )
    monkeypatch.setattr(trend, "_is_converging", lambda *_args, **_kwargs: False)

    out = trend.detect_channels(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2, channel_parallel_slope_ratio=0.15),
    )

    assert out
    assert out[0].name == "Horizontal Channel"


def test_detect_channels_reject_widening_parallel_structure(monkeypatch):
    from src.mtdata.patterns.classic_impl import trend

    n = 60
    x = np.arange(n, dtype=float)
    upper = 110.0 + (0.470 * x)
    lower = 100.0 + (0.400 * x)
    close = (upper + lower) / 2.0
    peaks = np.array([10, 20, 30, 40, 50], dtype=int)
    troughs = np.array([5, 15, 25, 35, 45], dtype=int)

    monkeypatch.setattr(
        trend,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (
            0.470,
            110.0,
            0.95,
            0.400,
            100.0,
            0.95,
            upper.copy(),
            lower.copy(),
        ),
    )
    monkeypatch.setattr(trend, "_is_converging", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(trend, "_count_touches", lambda *_args, **_kwargs: 6)

    out = trend.detect_channels(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2),
    )

    assert out == []


def test_detect_channels_reject_crossed_boundaries(monkeypatch):
    from src.mtdata.patterns.classic_impl import trend

    n = 60
    x = np.arange(n, dtype=float)
    upper = 110.0 - (0.08 * x)
    lower = 100.0 + (0.10 * x)
    close = (upper + lower) / 2.0
    peaks = np.array([10, 20, 30, 40, 50], dtype=int)
    troughs = np.array([5, 15, 25, 35, 45], dtype=int)

    monkeypatch.setattr(
        trend,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (
            -0.08,
            110.0,
            0.95,
            0.10,
            100.0,
            0.95,
            upper.copy(),
            lower.copy(),
        ),
    )
    monkeypatch.setattr(trend, "_is_converging", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(trend, "_count_touches", lambda *_args, **_kwargs: 6)

    out = trend.detect_channels(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=2, channel_parallel_slope_ratio=2.0),
    )

    assert out == []


def test_detect_trend_lines_require_breakout_for_completed_status(monkeypatch):
    from src.mtdata.patterns.classic_impl import trend

    n = 40
    peaks = np.array([5, 15, 25], dtype=int)
    troughs = np.array([10, 20, 30], dtype=int)

    def _fake_fit(x, _y):
        xs = tuple(int(v) for v in x.tolist())
        if xs == tuple(peaks.tolist()):
            return 0.0, 110.0, 0.95
        return 0.0, 100.0, 0.95

    monkeypatch.setattr(trend, "_fit_line", _fake_fit)
    monkeypatch.setattr(
        trend, "_last_touch_indexes", lambda _line, idxs, _c, _tol: idxs.tolist()
    )

    cfg = ClassicDetectorConfig(
        use_robust_fit=False,
        same_level_tol_pct=0.1,
        completion_lookback_bars=4,
        completion_confirm_bars=2,
    )

    close_forming = np.full(n, 100.0, dtype=float)
    close_forming[-4:] = np.array([100.02, 99.98, 100.01, 100.0], dtype=float)
    out_forming = trend.detect_trend_lines(
        close_forming,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        cfg,
    )
    support_forming = next(
        pattern for pattern in out_forming if pattern.details["side"] == "low"
    )

    assert support_forming.status == "forming"
    assert support_forming.details["touches_recent"] == 4
    assert support_forming.details["breakout_index"] is None

    close_break = close_forming.copy()
    close_break[-1] = 99.0
    out_break = trend.detect_trend_lines(
        close_break,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        cfg,
    )
    support_break = next(
        pattern for pattern in out_break if pattern.details["side"] == "low"
    )

    assert support_break.status == "completed"
    assert support_break.end_index == n - 1
    assert support_break.details["breakout_direction"] == "down"
    assert support_break.details["breakout_index"] == n - 1


def test_detect_channels_require_breakout_for_completed_status(monkeypatch):
    from src.mtdata.patterns.classic_impl import trend

    n = 60
    peaks = np.array([10, 20, 30, 40, 50], dtype=int)
    troughs = np.array([5, 15, 25, 35, 45], dtype=int)
    upper = np.full(n, 105.0, dtype=float)
    lower = np.full(n, 95.0, dtype=float)

    monkeypatch.setattr(
        trend,
        "_fit_lines_and_arrays",
        lambda *_args, **_kwargs: (
            0.0,
            105.0,
            0.95,
            0.0,
            95.0,
            0.95,
            upper.copy(),
            lower.copy(),
        ),
    )
    monkeypatch.setattr(trend, "_is_converging", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(trend, "_count_touches", lambda *_args, **_kwargs: 6)

    cfg = ClassicDetectorConfig(
        min_channel_touches=2,
        same_level_tol_pct=0.1,
        completion_lookback_bars=4,
        completion_confirm_bars=2,
    )

    close_forming = np.full(n, 100.0, dtype=float)
    close_forming[-4:] = np.array([105.0, 95.0, 100.0, 104.95], dtype=float)
    out_forming = trend.detect_channels(
        close_forming,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        cfg,
    )

    assert out_forming
    assert out_forming[0].status == "forming"
    assert out_forming[0].details["touches_recent"] == 3
    assert out_forming[0].details["breakout_index"] is None

    close_break = close_forming.copy()
    close_break[-1] = 106.0
    out_break = trend.detect_channels(
        close_break,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        cfg,
    )

    assert out_break[0].status == "completed"
    assert out_break[0].end_index == n - 1
    assert out_break[0].details["breakout_direction"] == "up"
    assert out_break[0].details["breakout_index"] == n - 1


def test_detect_diamonds_respects_geometry_threshold(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 200
    close = np.linspace(100.0, 101.0, n)
    close[-1] = 111.0
    peaks = np.array([30, 60, 90, 130, 160], dtype=int)
    troughs = np.array([20, 50, 100, 140, 170], dtype=int)

    def _fake_fit(x, y):
        _ = y
        xs = tuple(int(v) for v in x.tolist())
        if xs == tuple(peaks[peaks < 100].tolist()):
            return 0.05, 102.0, 0.9
        if xs == tuple(troughs[troughs < 100].tolist()):
            return -0.05, 98.0, 0.9
        if xs == tuple(peaks[peaks >= 100].tolist()):
            return -0.05, 115.0, 0.9
        if xs == tuple(troughs[troughs >= 100].tolist()):
            return 0.05, 85.0, 0.9
        return 0.0, 100.0, 0.0

    monkeypatch.setattr(
        shapes, "_detect_pivots_close", lambda seg, cfg, *args: (peaks, troughs)
    )
    monkeypatch.setattr(shapes, "_fit_line", _fake_fit)

    out_strict = shapes.detect_diamonds(
        close,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(
            use_robust_fit=False,
            diamond_min_boundary_r2=0.0,
            diamond_min_width_ratio=1.5,
            breakout_lookahead=40,
            completion_lookback_bars=40,
        ),
    )
    out_relaxed = shapes.detect_diamonds(
        close,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(
            use_robust_fit=False,
            diamond_min_boundary_r2=0.0,
            diamond_min_width_ratio=1.2,
            breakout_lookahead=40,
            completion_lookback_bars=40,
        ),
    )

    assert out_strict == []
    assert out_relaxed
    assert out_relaxed[0].status == "completed"
    assert out_relaxed[0].details["diamond_split_index"] == 100
    assert out_relaxed[0].details["prior_pole_span_bars"] > 0
    assert 0.0 < out_relaxed[0].details["geometry_score"] < 1.0


def test_detect_diamonds_forward_high_low_arrays_to_pivot_detection(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 150
    close = np.linspace(100.0, 105.0, n)
    high = close + 1.0
    low = close - 1.0
    captured = {}

    def _fake_pivots(seg, cfg, seg_h, seg_l):
        _ = seg
        _ = cfg
        captured["high"] = seg_h.copy()
        captured["low"] = seg_l.copy()
        return np.array([], dtype=int), np.array([], dtype=int)

    monkeypatch.setattr(shapes, "_detect_pivots_close", _fake_pivots)

    out = shapes.detect_diamonds(
        close, np.arange(n, dtype=float), ClassicDetectorConfig(), high, low
    )

    assert out == []
    assert np.array_equal(captured["high"], high[-n:])
    assert np.array_equal(captured["low"], low[-n:])


def test_detect_diamonds_accepts_asymmetric_split_with_stricter_default_r2(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 200
    close = np.linspace(100.0, 101.0, n)
    close[-1] = 111.0
    peaks = np.array([10, 30, 50, 160, 180], dtype=int)
    troughs = np.array([5, 25, 49, 170, 190], dtype=int)

    def _fake_fit(x, y):
        _ = y
        xs = tuple(int(v) for v in x.tolist())
        if xs == tuple(peaks[peaks < 50].tolist()):
            return 0.05, 102.0, 0.75
        if xs == tuple(troughs[troughs < 50].tolist()):
            return -0.05, 98.0, 0.75
        if xs == tuple(peaks[peaks >= 50].tolist()):
            return -0.02, 105.5, 0.75
        if xs == tuple(troughs[troughs >= 50].tolist()):
            return 0.02, 94.5, 0.75
        return 0.0, 100.0, 0.0

    monkeypatch.setattr(
        shapes, "_detect_pivots_close", lambda seg, cfg, *args: (peaks, troughs)
    )
    monkeypatch.setattr(shapes, "_fit_line", _fake_fit)

    out = shapes.detect_diamonds(
        close,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(
            use_robust_fit=False,
            breakout_lookahead=40,
            completion_lookback_bars=40,
        ),
    )

    assert out
    assert out[0].details["diamond_split_index"] in {49, 50}


def test_detect_diamonds_reject_disjoint_split_boundaries(monkeypatch):
    from src.mtdata.patterns.classic_impl import shapes

    n = 200
    close = np.linspace(100.0, 101.0, n)
    peaks = np.array([30, 60, 90, 130, 160], dtype=int)
    troughs = np.array([20, 50, 100, 140, 170], dtype=int)

    def _fake_fit(x, y):
        _ = y
        xs = tuple(int(v) for v in x.tolist())
        if xs == tuple(peaks[peaks < 100].tolist()):
            return 0.05, 102.0, 0.9
        if xs == tuple(troughs[troughs < 100].tolist()):
            return -0.05, 98.0, 0.9
        if xs == tuple(peaks[peaks >= 100].tolist()):
            return -0.05, 130.0, 0.9
        if xs == tuple(troughs[troughs >= 100].tolist()):
            return 0.05, 70.0, 0.9
        return 0.0, 100.0, 0.0

    monkeypatch.setattr(
        shapes, "_detect_pivots_close", lambda seg, cfg, *args: (peaks, troughs)
    )
    monkeypatch.setattr(shapes, "_fit_line", _fake_fit)

    out = shapes.detect_diamonds(
        close,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(
            use_robust_fit=False,
            diamond_min_boundary_r2=0.0,
            breakout_lookahead=40,
            completion_lookback_bars=40,
        ),
    )

    assert out == []


def test_detect_tops_bottoms_merges_connected_same_level_cluster():
    from src.mtdata.patterns.classic_impl.reversal import detect_tops_bottoms

    close = np.array(
        [98.0, 100.1, 95.0, 100.0, 94.8, 99.9, 95.2, 100.2, 96.0],
        dtype=float,
    )
    peaks = np.array([1, 3, 5, 7], dtype=int)
    troughs = np.array([2, 4, 6], dtype=int)

    out = detect_tops_bottoms(
        close,
        peaks,
        troughs,
        np.arange(close.size, dtype=float),
        ClassicDetectorConfig(same_level_tol_pct=0.5),
    )

    assert out
    triple_top = next(pattern for pattern in out if pattern.name == "Triple Top")
    assert triple_top.details["touches"] == 4
    assert triple_top.confidence == pytest.approx(0.7)


def test_level_components_preserves_transitive_clusters_in_input_order():
    from src.mtdata.patterns.classic_impl.reversal import _level_components

    vals = np.array([100.0, 112.0, 103.0, 115.0, 106.0], dtype=float)

    assert _level_components(vals, 5.0) == [[0, 2, 4], [1, 3]]


def test_detect_head_shoulders_fits_neckline_with_reaction_troughs(monkeypatch):
    from src.mtdata.patterns.classic_impl import reversal

    captured = {}

    def _fake_fit(x, y):
        captured["x"] = x.tolist()
        captured["y"] = y.tolist()
        return 0.0, 95.0, 0.8

    monkeypatch.setattr(reversal, "_fit_line", _fake_fit)

    close = np.array(
        [96.0, 100.0, 95.0, 103.0, 110.0, 96.0, 95.5, 99.0, 100.5, 97.0], dtype=float
    )
    peaks = np.array([1, 4, 8], dtype=int)
    troughs = np.array([2, 5, 6], dtype=int)

    out = reversal.detect_head_shoulders(
        close,
        peaks,
        troughs,
        np.arange(close.size, dtype=float),
        ClassicDetectorConfig(
            same_level_tol_pct=1.0, use_dtw_check=False, use_robust_fit=False
        ),
    )

    assert out
    assert captured["x"] == [2.0, 5.0]
    assert out[0].details["neckline_source"] == "troughs"
    assert out[0].details["neck_points"] == 2
    assert out[0].details["neck_validation_points"] == 1


def test_detect_inverse_head_shoulders_uses_reaction_peaks_for_neckline(monkeypatch):
    from src.mtdata.patterns.classic_impl import reversal

    captured = {}

    def _fake_fit(x, y):
        captured["x"] = x.tolist()
        captured["y"] = y.tolist()
        return 0.0, 105.0, 0.85

    monkeypatch.setattr(reversal, "_fit_line", _fake_fit)

    close = np.array([104.0, 95.0, 103.0, 90.0, 102.0, 96.0, 108.0], dtype=float)
    peaks = np.array([0, 2, 4, 6], dtype=int)
    troughs = np.array([1, 3, 5], dtype=int)

    out = reversal.detect_head_shoulders(
        close,
        peaks,
        troughs,
        np.arange(close.size, dtype=float),
        ClassicDetectorConfig(
            same_level_tol_pct=2.0, use_dtw_check=False, use_robust_fit=False
        ),
    )

    assert out
    inverse = next(
        pattern for pattern in out if pattern.name == "Inverse Head and Shoulders"
    )
    assert captured["x"] == [2.0, 4.0]
    assert inverse.details["neckline_source"] == "peaks"
    assert inverse.details["neck_points"] == 2


def test_detect_head_shoulders_rejects_neckline_above_shoulders():
    from src.mtdata.patterns.classic_impl import reversal

    close = np.array(
        [96.0, 100.0, 101.0, 103.0, 110.0, 101.5, 101.2, 99.0, 100.5, 97.0], dtype=float
    )
    peaks = np.array([1, 4, 8], dtype=int)
    troughs = np.array([2, 5, 6], dtype=int)

    out = reversal.detect_head_shoulders(
        close,
        peaks,
        troughs,
        np.arange(close.size, dtype=float),
        ClassicDetectorConfig(
            same_level_tol_pct=1.0, use_dtw_check=False, use_robust_fit=False
        ),
    )

    assert out == []


def test_detect_inverse_head_shoulders_rejects_neckline_below_shoulders():
    from src.mtdata.patterns.classic_impl import reversal

    close = np.array([104.0, 95.0, 94.0, 90.0, 94.5, 96.0, 108.0], dtype=float)
    peaks = np.array([0, 2, 4, 6], dtype=int)
    troughs = np.array([1, 3, 5], dtype=int)

    out = reversal.detect_head_shoulders(
        close,
        peaks,
        troughs,
        np.arange(close.size, dtype=float),
        ClassicDetectorConfig(
            same_level_tol_pct=2.0, use_dtw_check=False, use_robust_fit=False
        ),
    )

    assert out == []


def test_head_shoulders_two_point_neckline_does_not_get_free_r2_boost():
    from src.mtdata.patterns.classic_impl import reversal

    cfg = ClassicDetectorConfig(max_flat_slope=1e-4)
    quality = reversal._neckline_quality_score(
        slope=cfg.max_flat_slope * 2.5,
        r2=1.0,
        point_count=2,
        cfg=cfg,
    )

    assert quality == pytest.approx(0.5)


def test_detect_rounding_tries_multiple_windows(monkeypatch):
    from src.mtdata.patterns.classic_impl import reversal

    n = 260
    close = np.linspace(100.0, 110.0, n)
    called = []

    def _fake_polyfit(x, y, deg):
        _ = y
        _ = deg
        called.append(len(x))
        if len(x) == 100:
            return np.array([0.6, 0.0, 95.0], dtype=float)
        raise np.linalg.LinAlgError("skip other windows")

    monkeypatch.setattr(reversal.np, "polyfit", _fake_polyfit)
    monkeypatch.setattr(reversal, "_level_close", lambda *_args, **_kwargs: True)

    out = reversal.detect_rounding(
        close,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(
            rounding_window_bars=220, rounding_window_sizes=[100, 220]
        ),
    )

    assert called == [100, 220] or called == [220, 100]
    assert out
    assert out[0].details["window_bars"] == 100


def test_detect_rounding_returns_multiple_non_overlapping_windows(monkeypatch):
    from src.mtdata.patterns.classic_impl import reversal

    n = 320
    close = np.linspace(100.0, 110.0, n)
    fits = {
        100: np.array([0.5, 0.0, 95.0], dtype=float),
        220: np.array([0.55, 0.0, 95.0], dtype=float),
    }

    def _fake_polyfit(x, y, deg):
        _ = (y, deg)
        value = fits.get(len(x))
        if value is None:
            raise np.linalg.LinAlgError("skip")
        return value

    monkeypatch.setattr(reversal.np, "polyfit", _fake_polyfit)
    monkeypatch.setattr(reversal, "_level_close", lambda *_args, **_kwargs: True)

    out = reversal.detect_rounding(
        close,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(
            rounding_window_bars=220, rounding_window_sizes=[100, 220]
        ),
    )

    assert len(out) == 2
    assert {pattern.details["window_bars"] for pattern in out} == {100, 220}


def test_dedupe_overlapping_head_shoulders_results():
    from src.mtdata.patterns.classic_impl.reversal import _dedupe_overlapping_patterns

    patterns = [
        ClassicPatternResult(
            name="Head and Shoulders",
            status="forming",
            confidence=0.82,
            start_index=10,
            end_index=40,
            start_time=None,
            end_time=None,
            details={},
        ),
        ClassicPatternResult(
            name="Head and Shoulders",
            status="forming",
            confidence=0.75,
            start_index=14,
            end_index=38,
            start_time=None,
            end_time=None,
            details={},
        ),
        ClassicPatternResult(
            name="Head and Shoulders",
            status="forming",
            confidence=0.9,
            start_index=70,
            end_index=100,
            start_time=None,
            end_time=None,
            details={},
        ),
    ]

    out = _dedupe_overlapping_patterns(patterns, overlap_threshold=0.6)

    assert len(out) == 2
    assert any(p.start_index == 10 and p.end_index == 40 for p in out)
    assert any(p.start_index == 70 and p.end_index == 100 for p in out)


def test_detect_classic_patterns_disables_aliases_by_default(monkeypatch):
    n = 150
    x = np.linspace(0, 4 * np.pi, n)
    close = 100 + 0.3 * np.arange(n) + 4.0 * np.sin(x)
    df = pd.DataFrame({"time": np.arange(n, dtype=float), "close": close})

    peaks = np.array([20, 45, 70, 95, 120, 145], dtype=int)
    troughs = np.array([10, 35, 60, 85, 110, 135], dtype=int)
    monkeypatch.setattr(
        classic_mod, "_detect_pivots_close", lambda c, cfg: (peaks, troughs)
    )

    out_default = detect_classic_patterns(
        df, ClassicDetectorConfig(max_consolidation_bars=5)
    )
    names_default = {p.name for p in out_default}
    assert "Ascending Trend Line" in names_default
    assert "Trend Line" not in names_default
    assert "Trend Channel" not in names_default

    out_alias = detect_classic_patterns(
        df, ClassicDetectorConfig(include_aliases=True, max_consolidation_bars=5)
    )
    names_alias = {p.name for p in out_alias}
    assert "Trend Line" in names_alias or "Trend Channel" in names_alias


def test_detect_classic_patterns_reraises_internal_pivot_type_errors(monkeypatch):
    n = 150
    x = np.linspace(0, 4 * np.pi, n)
    close = 100 + 0.3 * np.arange(n) + 4.0 * np.sin(x)
    df = pd.DataFrame({"time": np.arange(n, dtype=float), "close": close})

    def _bad_pivots(c, cfg, *args):
        raise TypeError("internal pivot failure")

    monkeypatch.setattr(classic_mod, "_detect_pivots_close", _bad_pivots)

    with pytest.raises(TypeError, match="internal pivot failure"):
        detect_classic_patterns(df, ClassicDetectorConfig(max_consolidation_bars=5))


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
    df = pd.DataFrame(
        {
            "time": np.arange(n, dtype=float),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
        }
    )
    monkeypatch.setattr(
        classic_mod, "_detect_pivots_close", lambda c, cfg, *args: (peaks, troughs)
    )

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


def test_detect_classic_converging_parallel_shape_excludes_channel(monkeypatch):
    n = 170
    x = np.arange(n, dtype=float)
    top_line = 110.0 - 0.020 * x
    bot_line = 100.0 - 0.019 * x
    close = (top_line + bot_line) / 2.0

    peaks = np.array([30, 60, 90, 120, 150], dtype=int)
    troughs = np.array([20, 50, 80, 110, 140], dtype=int)
    close[peaks] = top_line[peaks]
    close[troughs] = bot_line[troughs]

    df = pd.DataFrame(
        {
            "time": np.arange(n, dtype=float),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
        }
    )
    monkeypatch.setattr(
        classic_mod, "_detect_pivots_close", lambda c, cfg, *args: (peaks, troughs)
    )

    def _fake_fit_lines(ih, il, c, n, cfg):
        return (
            -0.020,
            110.0,
            0.95,
            -0.019,
            100.0,
            0.95,
            top_line.copy(),
            bot_line.copy(),
        )

    monkeypatch.setattr(classic_mod, "_fit_lines_and_arrays", _fake_fit_lines)

    out = detect_classic_patterns(
        df,
        ClassicDetectorConfig(
            min_channel_touches=2,
            max_consolidation_bars=5,
        ),
    )
    names = {p.name for p in out}
    assert "Falling Wedge" in names
    assert "Symmetrical Triangle" not in names
    assert not any("Channel" in name for name in names)


def test_detect_classic_confidence_calibration_and_lifecycle(monkeypatch):
    n = 150
    x = np.linspace(0, 4 * np.pi, n)
    close = 100 + 0.25 * np.arange(n) + 3.0 * np.sin(x)
    df = pd.DataFrame(
        {
            "time": np.arange(n, dtype=float),
            "close": close,
            "high": close + 0.3,
            "low": close - 0.3,
        }
    )

    peaks = np.array([20, 45, 70, 95, 120, 145], dtype=int)
    troughs = np.array([10, 35, 60, 85, 110, 135], dtype=int)
    monkeypatch.setattr(
        classic_mod, "_detect_pivots_close", lambda c, cfg, *args: (peaks, troughs)
    )

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


def test_detect_pivots_close_prefers_high_low_when_enabled():
    n = 160
    close = np.full(n, 100.0, dtype=float)
    high = close.copy()
    low = close.copy()
    high[[30, 75, 120]] = [106.0, 107.0, 106.5]
    low[[45, 95, 140]] = [94.0, 93.5, 94.2]

    cfg_hl = ClassicDetectorConfig(
        pivot_use_hl=True,
        pivot_enable_fallback=False,
        pivot_use_atr_adaptive_prominence=False,
        pivot_use_atr_adaptive_distance=False,
        min_prominence_pct=1.0,
        min_distance=5,
    )
    cfg_close = ClassicDetectorConfig(
        pivot_use_hl=False,
        pivot_enable_fallback=False,
        pivot_use_atr_adaptive_prominence=False,
        pivot_use_atr_adaptive_distance=False,
        min_prominence_pct=1.0,
        min_distance=5,
    )

    peaks_hl, troughs_hl = classic_mod._detect_pivots_close(close, cfg_hl, high, low)
    peaks_close, troughs_close = classic_mod._detect_pivots_close(
        close, cfg_close, high, low
    )

    assert peaks_hl.size >= 2
    assert troughs_hl.size >= 2
    assert peaks_close.size == 0
    assert troughs_close.size == 0
