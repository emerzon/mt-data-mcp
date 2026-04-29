from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.core import regime as regime_mod
from mtdata.core.regime import _consolidate_payload
from mtdata.core.regime import api as regime_api
from mtdata.core.regime import regime_detect
from mtdata.core.regime.api import _build_all_method_comparison
from mtdata.core.regime.payload import _build_regime_descriptions


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


@pytest.fixture(autouse=True)
def _skip_mt5_connection(monkeypatch):
    monkeypatch.setattr(regime_mod, "ensure_mt5_connection_or_raise", lambda: None)


def _downtrend_df(n: int = 120) -> pd.DataFrame:
    t = np.arange(float(n))
    close = np.linspace(100.0, 70.0, n)
    return pd.DataFrame({"time": t, "close": close})


def _choppy_bearish_df(n: int = 120) -> pd.DataFrame:
    t = np.arange(float(n))
    alternating = np.where(np.arange(n) % 2 == 0, 2.0, -2.0)
    close = 100.0 + alternating + np.linspace(0.0, -10.0, n)
    return pd.DataFrame({"time": t, "close": close})


def test_build_all_method_comparison_uses_semantic_signals() -> None:
    comparison = _build_all_method_comparison(
        {
            "bocpd": {
                "current_regime": {
                    "status": "no_recent_change_detected",
                    "started_at": "T1",
                    "bars_since_change": 500,
                    "transition_risk": "low",
                },
                "transition_summary": {
                    "recent_change_points_count": 0,
                    "recent_transition_activity": "none",
                    "calibration_status": "calibrated",
                },
                "regime_context": {
                    "bias": "bullish",
                    "return_pct": 4.2,
                    "volatility_pct": 1.1,
                },
            },
            "hmm": {
                "current_regime": {
                    "regime_id": 1,
                    "label": "positive_mod_vol",
                    "confidence": 0.857,
                },
                "regime_info": {
                    1: {
                        "label": "positive_mod_vol",
                        "mean_return": 0.000122,
                        "mean_return_pct": 0.0122,
                        "volatility": 0.00279,
                        "volatility_pct": 0.279,
                    }
                },
            },
            "clustering": {
                "current_regime": {
                    "regime_id": 1,
                    "label": "regime_bullish",
                    "confidence": 1.0,
                }
            },
            "garch": {
                "current_regime": {
                    "regime_id": 0,
                    "label": "low_vol",
                    "confidence": 1.0,
                },
                "regime_info": {
                    0: {
                        "label": "low_vol",
                        "mean_return": 0.000105,
                        "mean_return_pct": 0.0105,
                        "volatility": 0.00367,
                        "volatility_pct": 0.367,
                    }
                },
            },
            "rule_based": {
                "current_regime": {
                    "state": "ranging",
                    "direction": "bullish",
                    "trend_strength": 0.1497,
                    "efficiency_ratio": 0.0009,
                    "window_bars": 160,
                    "window_move_pct": 1.25,
                    "signal_source": "price",
                }
            },
            "ensemble": {
                "current_regime": {
                    "regime_id": 1,
                    "label": "negative_mod_vol",
                    "confidence": 0.3524,
                },
                "regime_info": {
                    1: {
                        "label": "negative_mod_vol",
                        "mean_return": -0.000415,
                        "mean_return_pct": -0.0415,
                        "volatility": 0.00386,
                        "volatility_pct": 0.386,
                    }
                },
            },
        }
    )

    assert comparison["current_regimes"]["bocpd"]["status"] == "no_recent_change_detected"
    assert comparison["current_regimes"]["bocpd"]["recent_transition_activity"] == "none"
    assert comparison["current_regimes"]["bocpd"]["bias"] == "bullish"
    assert comparison["current_regimes"]["hmm"]["regime_confidence"] == 0.857
    assert comparison["current_regimes"]["garch"]["volatility"] == "low_vol"
    assert comparison["current_regimes"]["rule_based"]["signal_source"] == "price"
    assert "majority_state" not in comparison["agreement"]
    assert comparison["agreement"]["direction"] == {
        "majority": "bullish",
        "agreement_pct": 75.0,
        "methods_considered": ["hmm", "clustering", "rule_based", "ensemble"],
    }
    assert comparison["agreement"]["volatility"] == {
        "majority": "moderate_vol",
        "agreement_pct": 66.67,
        "methods_considered": ["hmm", "garch", "ensemble"],
    }


def test_regime_detect_all_respects_full_and_summary_detail(monkeypatch) -> None:
    real = _unwrap(regime_api.regime_detect)
    monkeypatch.setattr(regime_api, "_fetch_history", lambda *args, **kwargs: _downtrend_df(120))
    monkeypatch.setattr(regime_api, "_regime_connection_error", lambda: None)
    monkeypatch.setattr(
        regime_api,
        "_build_all_method_comparison",
        lambda results: {"methods": sorted(results.keys())},
    )

    subcall_details: list[tuple[str, str, bool]] = []

    def fake_recursive(*args, **kwargs):
        method_name = str(kwargs.get("method") or "")
        detail_name = str(kwargs.get("detail") or "")
        include_series = bool(kwargs.get("include_series"))
        subcall_details.append((method_name, detail_name, include_series))
        result = {
            "success": True,
            "symbol": kwargs.get("symbol"),
            "timeframe": kwargs.get("timeframe"),
            "method": method_name,
            "target": kwargs.get("target"),
        }
        if detail_name == "full":
            result["detail_marker"] = f"{method_name}-full"
            if include_series:
                result["series"] = {"state": [0, 1, 1]}
        return result

    monkeypatch.setattr(regime_api, "regime_detect", fake_recursive)

    full = real("EURUSD", method="all", detail="full", include_series=True)
    assert full["detail"] == "full"
    assert "results" in full
    assert full["results"]["bocpd"]["detail_marker"] == "bocpd-full"
    assert full["results"]["hmm"]["series"] == {"state": [0, 1, 1]}
    assert subcall_details
    assert all(detail == "full" for _, detail, _ in subcall_details)
    assert all(include_series is True for _, _, include_series in subcall_details)

    subcall_details.clear()
    summary = real("EURUSD", method="all", detail="summary", include_series=True)
    assert summary["detail"] == "summary"
    assert "results" not in summary
    # Summary mode must not embed per-method regimes or params_used.
    assert "params_used" not in summary
    comparison = summary.get("comparison", {})
    assert "current_regimes" not in comparison
    assert "agreement" in comparison
    assert subcall_details
    assert all(detail == "compact" for _, detail, _ in subcall_details)
    assert all(include_series is False for _, _, include_series in subcall_details)


def test_regime_detect_all_reports_runtime_diagnostics_for_partial_results(monkeypatch) -> None:
    real = _unwrap(regime_api.regime_detect)
    monkeypatch.setattr(regime_api, "_fetch_history", lambda *args, **kwargs: _downtrend_df(120))
    monkeypatch.setattr(regime_api, "_regime_connection_error", lambda: None)
    monkeypatch.setattr(
        regime_api,
        "_build_all_method_comparison",
        lambda results: {"methods_run": sorted(results.keys())},
    )

    def fake_recursive(*args, **kwargs):
        method_name = str(kwargs.get("method") or "")
        if method_name == "ms_ar":
            return {"error": "simulated slow fit timeout"}
        return {
            "success": True,
            "symbol": kwargs.get("symbol"),
            "timeframe": kwargs.get("timeframe"),
            "method": method_name,
            "target": kwargs.get("target"),
            "current_regime": {"label": "neutral"},
        }

    monkeypatch.setattr(regime_api, "regime_detect", fake_recursive)

    result = real("EURUSD", method="all", detail="compact")

    assert result["success"] is True
    assert result["runtime"]["partial_results"] is True
    assert "ms_ar" in result["runtime"]["failed_methods"]
    assert result["runtime"]["method_errors"]["ms_ar"] == "simulated slow fit timeout"
    assert "ms_ar" in result["runtime"]["method_durations_ms"]
    assert result["runtime"]["method_guidance"]["all"]["speed_tier"] == "slow"
    assert result["runtime"]["method_guidance"]["rule_based"]["speed_tier"] == "fast"
    assert result["runtime"]["suggested_faster_methods"]


def test_ensemble_labels_follow_mean_return_sign() -> None:
    descriptions = _build_regime_descriptions(
        {
            "mean_return": [
                -0.000725,
                -0.000415,
                0.000079,
                0.000179,
                0.000607,
                0.00119,
            ],
            "volatility": [0.00435, 0.00386, 0.00466, 0.00614, 0.00469, 0.00593],
        },
        "ensemble",
    )

    assert descriptions[0]["label"].startswith("negative_")
    assert descriptions[1]["label"].startswith("negative_")
    assert descriptions[2]["label"].startswith("neutral_")
    assert descriptions[3]["label"].startswith("positive_")
    assert "bearish" not in descriptions[2]["label"]


def test_wavelet_labels_include_frequency_character() -> None:
    descriptions = _build_regime_descriptions(
        {
            "mean_return": [-0.00025, -0.00016, 0.00055],
            "volatility": [0.0043, 0.00398, 0.00503],
            "energy_profiles": {
                "0": {
                    "band_0_energy": 0.084377,
                    "band_1_energy": 0.212702,
                    "band_2_energy": 0.271006,
                    "band_3_energy": 0.431915,
                },
                "1": {
                    "band_0_energy": 0.081442,
                    "band_1_energy": 0.114511,
                    "band_2_energy": 0.196177,
                    "band_3_energy": 0.60787,
                },
                "2": {
                    "band_0_energy": 0.037996,
                    "band_1_energy": 0.081256,
                    "band_2_energy": 0.419504,
                    "band_3_energy": 0.461244,
                },
            },
        },
        "wavelet",
    )

    assert descriptions[0]["label"] == "negative_mixed_freq_high_vol"
    assert descriptions[1]["label"] == "negative_trend_dominant_high_vol"
    assert descriptions[2]["label"] == "positive_trend_dominant_high_vol"


def test_bocpd_consolidation_uses_segment_language() -> None:
    out = _consolidate_payload(
        {
            "symbol": "X",
            "timeframe": "H1",
            "method": "bocpd",
            "target": "return",
            "times": ["T1", "T2", "T3", "T4"],
            "cp_prob": [0.05, 0.9, 0.1, 0.08],
            "change_points": [{"idx": 1, "time": "T2", "prob": 0.9}],
            "_series_values": [0.01, 0.02, -0.01, -0.02],
            "threshold": 0.5,
            "reliability": {
                "expected_false_alarm_rate": 0.02,
                "recent_cp_density": 0.0,
                "threshold_calibrated": True,
            },
            "summary": {
                "lookback": 4,
                "last_cp_prob": 0.08,
                "change_points_count": 1,
            },
        },
        "bocpd",
        "compact",
    )

    assert "current_regime" in out
    assert out["current_regime"]["status"] == "recent_change_detected"
    assert out["transition_summary"]["recent_change_points_count"] == 1
    assert out["transition_summary"]["calibration_status"] == "calibrated"
    assert out["regime_context"]["source"] == "derived_from_return_series"
    assert out["regimes"][1]["transition_prob_at_start"] == 0.9
    assert "current_segment" not in out
    assert "segments" not in out


def test_consolidate_payload_uses_regime_confidence_name() -> None:
    out = _consolidate_payload(
        {
            "symbol": "X",
            "timeframe": "H1",
            "method": "hmm",
            "times": ["T1", "T2", "T3", "T4"],
            "state": [0, 0, 1, 1],
            "state_probabilities": [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.15, 0.85],
                [0.2, 0.8],
            ],
        },
        "hmm",
        "compact",
    )

    current_regime = out["current_regime"]
    assert current_regime["regime_confidence"] == 0.825
    assert "confidence" not in current_regime


def test_rule_based_uses_price_window_metrics_for_return_target() -> None:
    raw = _unwrap(regime_detect)
    history = _downtrend_df()
    expected_segment = history["close"].to_numpy()[-60:]
    expected_move_pct = round(
        (expected_segment[-1] - expected_segment[0]) / expected_segment[0] * 100.0,
        4,
    )

    with (
        patch("mtdata.core.regime._fetch_history", return_value=history),
        patch("mtdata.core.regime._resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime._format_time_minimal", side_effect=lambda x: f"T{x}"),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            limit=len(history),
            method="rule_based",
            target="return",
            params={"window_bars": 60},
            detail="full",
        )

    regime = out["current_regime"]
    assert regime["direction"] == "bearish"
    assert regime["signal_source"] == "price"
    assert regime["window_move_pct"] == expected_move_pct
    assert out["params_used"]["signal_source"] == "price"


def test_rule_based_explains_ranging_direction_bias() -> None:
    raw = _unwrap(regime_detect)

    with (
        patch("mtdata.core.regime._fetch_history", return_value=_choppy_bearish_df()),
        patch("mtdata.core.regime._resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime._format_time_minimal", side_effect=lambda x: f"T{x}"),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            limit=120,
            method="rule_based",
            params={"window_bars": 60},
            detail="full",
        )

    regime = out["current_regime"]
    assert regime["state"] == "ranging"
    assert regime["direction"] == "bearish"
    assert regime["direction_basis"] == "net_window_move"
    assert "window bias, not a trend classification" in regime["interpretation"]
    assert regime["trend_strength"] >= out["params_used"]["trend_strength_threshold"]
    assert "efficiency_ratio indicates a choppy path" in regime["note"]


def test_rule_based_compact_omits_explanatory_fields() -> None:
    raw = _unwrap(regime_detect)

    with (
        patch("mtdata.core.regime._fetch_history", return_value=_choppy_bearish_df()),
        patch("mtdata.core.regime._resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime._format_time_minimal", side_effect=lambda x: f"T{x}"),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            limit=120,
            method="rule_based",
            params={"window_bars": 60},
        )

    regime = out["current_regime"]
    assert {"state", "direction", "efficiency_ratio"}.issubset(regime)
    assert regime["state"] == "ranging"
    assert regime["direction"] == "bearish"
    assert out["current_regime"]["state"] == "ranging"
    assert out["current_regime"]["direction"] == "bearish"
    assert out["current_regime"]["bars"] == 60
    assert "interpretation" not in regime
    assert "note" not in regime
    assert "signal_source" not in regime
    assert "params_used" not in out
    assert "regime" not in out
