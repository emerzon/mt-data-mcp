from __future__ import annotations

import inspect
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.core import regime as regime_mod
from mtdata.core.regime import _auto_calibrate_bocpd_params, regime_detect
from mtdata.core.regime.methods.bocpd.core import (
    _walkforward_quantile_threshold_calibration,
)
from mtdata.utils.mt5 import MT5ConnectionError


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


@pytest.fixture(autouse=True)
def _skip_mt5_connection(monkeypatch):
    monkeypatch.setattr(regime_mod, "ensure_mt5_connection_or_raise", lambda: None)


def _sample_df(n: int = 80) -> pd.DataFrame:
    t = np.arange(float(n))
    close = 100.0 + np.linspace(0.0, 1.0, n)
    return pd.DataFrame({"time": t, "close": close})


def test_regime_detect_defaults_to_compact_output() -> None:
    raw = _unwrap(regime_detect)
    cp = np.zeros(79, dtype=float)
    cp[-2] = 0.9

    with (
        patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)),
        patch(
            "mtdata.core.regime._resolve_denoise_base_col",
            return_value="close",
        ),
        patch(
            "mtdata.core.regime._format_time_minimal",
            side_effect=lambda x: f"T{x}",
        ),
        patch(
            "mtdata.utils.regime.bocpd_gaussian",
            return_value={"cp_prob": cp},
        ),
    ):
        out = raw(
            symbol="EURUSD",
            timeframe="H1",
            limit=80,
            method="bocpd",
            threshold=0.5,
            lookback=20,
        )

    assert out.get("success") is True
    assert "summary" not in out
    assert "reliability" in out
    assert "regimes" in out
    assert "current_regime" in out
    assert "current_segment" not in out


def test_regime_detect_accepts_standard_detail_as_compact() -> None:
    raw = _unwrap(regime_detect)
    cp = np.zeros(79, dtype=float)

    with (
        patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)),
        patch("mtdata.core.regime._resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch("mtdata.utils.regime.bocpd_gaussian", return_value={"cp_prob": cp}),
    ):
        out = raw(
            symbol="EURUSD",
            timeframe="H1",
            limit=80,
            method="bocpd",
            threshold=0.5,
            detail="standard",
            lookback=20,
        )

    assert out.get("success") is True
    assert "regimes" in out
    assert "series" not in out


def test_regime_detect_returns_connection_error_payload(monkeypatch) -> None:
    raw = _unwrap(regime_detect)

    def fail_connection() -> None:
        raise MT5ConnectionError(
            "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
        )

    monkeypatch.setattr(regime_mod, "ensure_mt5_connection_or_raise", fail_connection)

    out = raw(symbol="EURUSD", timeframe="H1", limit=80, method="bocpd")

    assert out["error"] == "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
    assert out["success"] is False
    assert out["error_code"] == "mt5_connection_error"
    assert out["operation"] == "mt5_ensure_connection"
    assert isinstance(out.get("request_id"), str)


def test_bocpd_uses_crypto_sensitive_auto_hazard_default() -> None:
    raw = _unwrap(regime_detect)
    capture = {}

    def _fake_bocpd(x, hazard_lambda=0, max_run_length=0, **kwargs):
        capture["hazard_lambda"] = int(hazard_lambda)
        capture["max_run_length"] = int(max_run_length)
        return {"cp_prob": np.zeros_like(x, dtype=float)}

    with (
        patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)),
        patch(
            "mtdata.core.regime._resolve_denoise_base_col",
            return_value="close",
        ),
        patch(
            "mtdata.core.regime._format_time_minimal",
            side_effect=lambda x: f"T{x}",
        ),
        patch(
            "mtdata.utils.regime.bocpd_gaussian",
            side_effect=_fake_bocpd,
        ),
    ):
        out = raw(
            symbol="BTCUSD",
            timeframe="H1",
            limit=80,
            method="bocpd",
            threshold=0.5,
            lookback=20,
            detail="full",  # params_used only in full mode
        )

    params_used = out.get("params_used", {})
    auto_diag = params_used.get("auto_calibration", {})

    assert capture["hazard_lambda"] == int(params_used["hazard_lambda"])
    assert capture["hazard_lambda"] != 72
    assert params_used["hazard_lambda_source"] == "auto_calibrated"
    assert params_used["cp_threshold_source"] == "auto_calibrated"
    assert 0.15 <= float(params_used["cp_threshold"]) <= 0.75
    assert isinstance(auto_diag, dict)
    assert auto_diag.get("calibrated") is True
    assert int(auto_diag.get("base_hazard_lambda", 0)) == 72
    assert abs(float(auto_diag.get("base_cp_threshold", 0.0)) - 0.35) < 1e-12


def test_bocpd_hazard_lambda_param_override_is_preserved() -> None:
    raw = _unwrap(regime_detect)
    capture = {}

    def _fake_bocpd(x, hazard_lambda=0, max_run_length=0, **kwargs):
        capture["hazard_lambda"] = int(hazard_lambda)
        capture["max_run_length"] = int(max_run_length)
        return {"cp_prob": np.zeros_like(x, dtype=float)}

    with (
        patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)),
        patch(
            "mtdata.core.regime._resolve_denoise_base_col",
            return_value="close",
        ),
        patch(
            "mtdata.core.regime._format_time_minimal",
            side_effect=lambda x: f"T{x}",
        ),
        patch(
            "mtdata.utils.regime.bocpd_gaussian",
            side_effect=_fake_bocpd,
        ),
    ):
        out = raw(
            symbol="BTCUSD",
            timeframe="H1",
            limit=80,
            method="bocpd",
            params={"hazard_lambda": 140},
            threshold=0.5,
            lookback=20,
            detail="full",  # params_used only in full mode
        )

    assert capture["hazard_lambda"] == 140
    assert out["params_used"]["hazard_lambda"] == 140
    assert out["params_used"]["hazard_lambda_source"] == "params"
    assert out["params_used"]["cp_threshold_source"] == "auto_calibrated"


def test_bocpd_cp_threshold_param_override_is_preserved() -> None:
    raw = _unwrap(regime_detect)
    cp = np.zeros(79, dtype=float)

    with (
        patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)),
        patch(
            "mtdata.core.regime._resolve_denoise_base_col",
            return_value="close",
        ),
        patch(
            "mtdata.core.regime._format_time_minimal",
            side_effect=lambda x: f"T{x}",
        ),
        patch(
            "mtdata.utils.regime.bocpd_gaussian",
            return_value={"cp_prob": cp},
        ),
    ):
        out = raw(
            symbol="BTCUSD",
            timeframe="H1",
            limit=80,
            method="bocpd",
            params={"cp_threshold": 0.2},
            threshold=0.5,
            lookback=20,
            detail="full",
        )

    assert abs(float(out["params_used"]["cp_threshold"]) - 0.2) < 1e-12
    assert out["params_used"]["cp_threshold_source"] == "params.cp_threshold"


def test_bocpd_hazard_mode_auto_calibrated_sets_sources_and_diagnostics() -> None:
    raw = _unwrap(regime_detect)
    cp = np.zeros(79, dtype=float)

    with (
        patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)),
        patch(
            "mtdata.core.regime._resolve_denoise_base_col",
            return_value="close",
        ),
        patch(
            "mtdata.core.regime._format_time_minimal",
            side_effect=lambda x: f"T{x}",
        ),
        patch(
            "mtdata.utils.regime.bocpd_gaussian",
            return_value={"cp_prob": cp},
        ),
    ):
        out = raw(
            symbol="BTCUSD",
            timeframe="H1",
            limit=80,
            method="bocpd",
            params={"hazard_mode": "auto_calibrated"},
            threshold=0.5,
            lookback=20,
            detail="full",  # params_used only in full mode
        )

    params_used = out.get("params_used", {})
    assert params_used.get("hazard_mode") == "auto_calibrated"
    assert params_used.get("hazard_lambda_source") == "auto_calibrated"
    assert params_used.get("cp_threshold_source") == "auto_calibrated"
    auto_diag = params_used.get("auto_calibration")
    assert isinstance(auto_diag, dict)
    assert auto_diag.get("calibrated") is True


def test_bocpd_hazard_lambda_override_beats_auto_calibrated_mode() -> None:
    raw = _unwrap(regime_detect)
    cp = np.zeros(79, dtype=float)

    with (
        patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)),
        patch(
            "mtdata.core.regime._resolve_denoise_base_col",
            return_value="close",
        ),
        patch(
            "mtdata.core.regime._format_time_minimal",
            side_effect=lambda x: f"T{x}",
        ),
        patch(
            "mtdata.utils.regime.bocpd_gaussian",
            return_value={"cp_prob": cp},
        ),
    ):
        out = raw(
            symbol="BTCUSD",
            timeframe="H1",
            limit=80,
            method="bocpd",
            params={"hazard_mode": "auto_calibrated", "hazard_lambda": 111},
            threshold=0.5,
            lookback=20,
            detail="full",  # params_used only in full mode
        )

    params_used = out.get("params_used", {})
    assert params_used.get("hazard_lambda") == 111
    assert params_used.get("hazard_lambda_source") == "params"
    assert params_used.get("cp_threshold_source") == "auto_calibrated"


def test_auto_calibrate_bocpd_params_significant_move_lowers_hazard_and_threshold() -> (
    None
):
    rng = np.random.default_rng(7)
    returns = rng.normal(loc=-1.5e-4, scale=6.0e-4, size=240)
    hazard_lambda, cp_threshold, diagnostics = _auto_calibrate_bocpd_params(
        returns=returns,
        symbol="EURUSD",
        timeframe="H1",
    )

    assert diagnostics.get("calibrated") is True
    assert diagnostics.get("asset_class_hint") == "other"
    assert float(diagnostics.get("move_zscore", 0.0)) > 0.0
    assert float(diagnostics.get("move_sig_norm", 0.0)) > 0.0
    assert int(hazard_lambda) < 250
    assert float(cp_threshold) < 0.50


def test_regime_detect_rejects_invalid_min_regime_bars() -> None:
    raw = _unwrap(regime_detect)
    with (
        patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)),
        patch(
            "mtdata.core.regime._resolve_denoise_base_col",
            return_value="close",
        ),
    ):
        out = raw(
            symbol="EURUSD", timeframe="H1", limit=80, method="hmm", min_regime_bars=0
        )
    assert "error" in out
    assert "min_regime_bars" in str(out["error"])


def test_regime_detect_default_min_regime_bars_is_dynamic() -> None:
    """min_regime_bars defaults to -1 which triggers timeframe-based defaults."""
    raw = _unwrap(regime_detect)
    default_val = inspect.signature(raw).parameters["min_regime_bars"].default
    assert int(default_val) == -1  # -1 means "use timeframe-based default"

    # Verify that effective defaults are applied based on timeframe
    with (
        patch("mtdata.core.regime._fetch_history", return_value=_sample_df(800)),
        patch("mtdata.core.regime._resolve_denoise_base_col", return_value="close"),
    ):
        # M1 should get higher defaults
        out_m1 = raw(symbol="TEST", timeframe="M1", method="hmm", detail="full")
        params_m1 = out_m1.get("params_used", {})

        # D1 should get lower defaults
        out_d1 = raw(symbol="TEST", timeframe="D1", method="hmm", detail="full")
        params_d1 = out_d1.get("params_used", {})

        # M1 should have higher min_regime_bars than D1
        assert params_m1.get("min_regime_bars", 0) >= params_d1.get(
            "min_regime_bars", 0
        )


def test_bocpd_zero_change_points_includes_tuning_hint() -> None:
    raw = _unwrap(regime_detect)
    cp = np.zeros(79, dtype=float)

    with (
        patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)),
        patch(
            "mtdata.core.regime._resolve_denoise_base_col",
            return_value="close",
        ),
        patch(
            "mtdata.core.regime._format_time_minimal",
            side_effect=lambda x: f"T{x}",
        ),
        patch(
            "mtdata.utils.regime.bocpd_gaussian",
            return_value={"cp_prob": cp},
        ),
    ):
        out = raw(
            symbol="BTCUSD",
            timeframe="H1",
            limit=80,
            method="bocpd",
            threshold=0.6,
            detail="summary",
        )

    summary = out.get("summary", {})
    assert summary.get("change_points_count") == 0
    hint = str(summary.get("tuning_hint", ""))
    assert "hazard_lambda" in hint
    assert "threshold" in hint


def test_bocpd_filters_last_bar_spike_with_strict_confirmation() -> None:
    raw = _unwrap(regime_detect)

    def _fake_bocpd(x, hazard_lambda=0, max_run_length=0, **kwargs):
        cp = np.zeros(len(x), dtype=float)
        if cp.size:
            cp[-1] = 0.9
        return {"cp_prob": cp}

    with (
        patch("mtdata.core.regime._fetch_history", return_value=_sample_df(220)),
        patch(
            "mtdata.core.regime._resolve_denoise_base_col",
            return_value="close",
        ),
        patch(
            "mtdata.core.regime._format_time_minimal",
            side_effect=lambda x: f"T{x}",
        ),
        patch(
            "mtdata.utils.regime.bocpd_gaussian",
            side_effect=_fake_bocpd,
        ),
    ):
        out = raw(
            symbol="EURUSD",
            timeframe="H1",
            limit=220,
            method="bocpd",
            detail="summary",
            params={"cp_confirm_bars": 2},
        )

    summary = out.get("summary", {})
    assert int(summary.get("raw_change_points_count", 0)) >= 1
    assert int(summary.get("change_points_count", 0)) == 0
    assert int(summary.get("filtered_change_points_count", 0)) >= 1
    params_used = out.get("params_used", {})
    cp_filter = params_used.get("cp_filter", {})
    assert int(cp_filter.get("confirm_bars", 0)) == 2
    assert int(cp_filter.get("filtered_count", 0)) >= 1


def test_bocpd_walkforward_threshold_calibration_metadata_is_exposed() -> None:
    raw = _unwrap(regime_detect)

    def _fake_bocpd(x, hazard_lambda=0, max_run_length=0, **kwargs):
        cp = np.full(len(x), 0.01, dtype=float)
        return {"cp_prob": cp}

    with (
        patch("mtdata.core.regime._fetch_history", return_value=_sample_df(220)),
        patch(
            "mtdata.core.regime._resolve_denoise_base_col",
            return_value="close",
        ),
        patch(
            "mtdata.core.regime._format_time_minimal",
            side_effect=lambda x: f"T{x}",
        ),
        patch(
            "mtdata.utils.regime.bocpd_gaussian",
            side_effect=_fake_bocpd,
        ),
    ):
        out = raw(
            symbol="BTCUSD", timeframe="H1", limit=220, method="bocpd", detail="summary"
        )

    cal = out.get("params_used", {}).get("cp_threshold_calibration", {})
    assert cal.get("mode") == "walkforward_quantile"
    assert cal.get("calibrated") is True
    assert int(cal.get("points", 0)) >= 200


def test_bocpd_summary_contains_reliability_fields() -> None:
    raw = _unwrap(regime_detect)
    cp = np.zeros(219, dtype=float)
    cp[150] = 0.9

    with (
        patch("mtdata.core.regime._fetch_history", return_value=_sample_df(220)),
        patch(
            "mtdata.core.regime._resolve_denoise_base_col",
            return_value="close",
        ),
        patch(
            "mtdata.core.regime._format_time_minimal",
            side_effect=lambda x: f"T{x}",
        ),
        patch(
            "mtdata.utils.regime.bocpd_gaussian",
            return_value={"cp_prob": cp},
        ),
    ):
        out = raw(
            symbol="EURUSD", timeframe="H1", limit=220, method="bocpd", detail="summary"
        )

    summary = out.get("summary", {})
    assert "confidence" in summary
    assert "expected_false_alarm_rate" in summary
    assert "calibration_age_bars" in summary
    rel = out.get("reliability", {})
    assert "confidence" in rel
    assert "expected_false_alarm_rate" in rel
    assert "calibration_age_bars" in rel


def test_bocpd_calibrated_threshold_does_not_overreject_at_edge_by_default() -> None:
    raw = _unwrap(regime_detect)

    def _fake_bocpd(x, hazard_lambda=0, max_run_length=0, **kwargs):
        cp = np.zeros(len(x), dtype=float)
        if cp.size >= 2:
            cp[-2] = 0.44
            cp[-1] = 0.44
        return {"cp_prob": cp}

    fake_auto = (
        168,
        0.43,
        {
            "calibrated": True,
            "points": 219,
            "base_hazard_lambda": 250,
            "base_cp_threshold": 0.5,
        },
    )
    fake_thr_cal = (
        0.43,
        {
            "mode": "walkforward_quantile",
            "calibrated": True,
            "points": 219,
            "target_false_alarm_rate": 0.02,
        },
    )

    with (
        patch("mtdata.core.regime._fetch_history", return_value=_sample_df(220)),
        patch(
            "mtdata.core.regime._resolve_denoise_base_col",
            return_value="close",
        ),
        patch(
            "mtdata.core.regime._format_time_minimal",
            side_effect=lambda x: f"T{x}",
        ),
        patch(
            "mtdata.core.regime._auto_calibrate_bocpd_params",
            return_value=fake_auto,
        ),
        patch(
            "mtdata.core.regime._walkforward_quantile_threshold_calibration",
            return_value=fake_thr_cal,
        ),
        patch(
            "mtdata.utils.regime.bocpd_gaussian",
            side_effect=_fake_bocpd,
        ),
    ):
        out = raw(
            symbol="EURUSD", timeframe="D1", limit=220, method="bocpd", detail="summary"
        )

    params_used = out.get("params_used", {})
    cp_filter = params_used.get("cp_filter", {})
    assert abs(float(cp_filter.get("edge_multiplier", 0.0)) - 1.0) < 1e-12
    assert int(cp_filter.get("raw_candidates_count", 0)) >= 2
    assert int(cp_filter.get("accepted_count", 0)) >= 1
    summary = out.get("summary", {})
    assert int(summary.get("change_points_count", 0)) >= 1


def test_bocpd_default_cp_confirm_bars_is_live_mode_one() -> None:
    raw = _unwrap(regime_detect)

    def _fake_bocpd(x, hazard_lambda=0, max_run_length=0, **kwargs):
        cp = np.zeros(len(x), dtype=float)
        if cp.size >= 1:
            cp[-1] = 0.44
        return {"cp_prob": cp}

    fake_auto = (
        168,
        0.43,
        {
            "calibrated": True,
            "points": 219,
            "base_hazard_lambda": 250,
            "base_cp_threshold": 0.5,
        },
    )
    fake_thr_cal = (
        0.43,
        {
            "mode": "walkforward_quantile",
            "calibrated": True,
            "points": 219,
            "target_false_alarm_rate": 0.02,
        },
    )

    with (
        patch("mtdata.core.regime._fetch_history", return_value=_sample_df(220)),
        patch(
            "mtdata.core.regime._resolve_denoise_base_col",
            return_value="close",
        ),
        patch(
            "mtdata.core.regime._format_time_minimal",
            side_effect=lambda x: f"T{x}",
        ),
        patch(
            "mtdata.core.regime._auto_calibrate_bocpd_params",
            return_value=fake_auto,
        ),
        patch(
            "mtdata.core.regime._walkforward_quantile_threshold_calibration",
            return_value=fake_thr_cal,
        ),
        patch(
            "mtdata.utils.regime.bocpd_gaussian",
            side_effect=_fake_bocpd,
        ),
    ):
        out = raw(
            symbol="EURUSD", timeframe="D1", limit=220, method="bocpd", detail="summary"
        )

    cp_filter = out.get("params_used", {}).get("cp_filter", {})
    assert int(cp_filter.get("confirm_bars", 0)) == 1
    assert int(cp_filter.get("raw_candidates_count", 0)) >= 1
    assert int(cp_filter.get("accepted_count", 0)) >= 1


def test_bocpd_walkforward_calibration_defaults_collect_larger_null_sample() -> None:
    sig = inspect.signature(_walkforward_quantile_threshold_calibration)
    assert int(sig.parameters["max_windows"].default) == 10
    assert int(sig.parameters["bootstrap_runs"].default) == 20
