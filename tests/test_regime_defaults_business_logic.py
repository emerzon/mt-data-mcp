from __future__ import annotations

import inspect
from unittest.mock import patch

import numpy as np
import pandas as pd

from mtdata.core.regime import regime_detect


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _sample_df(n: int = 80) -> pd.DataFrame:
    t = np.arange(float(n))
    close = 100.0 + np.linspace(0.0, 1.0, n)
    return pd.DataFrame({"time": t, "close": close})


def test_regime_detect_defaults_to_compact_output() -> None:
    raw = _unwrap(regime_detect)
    cp = np.zeros(79, dtype=float)
    cp[-2] = 0.9

    with patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)), patch(
        "mtdata.core.regime._resolve_denoise_base_col",
        return_value="close",
    ), patch(
        "mtdata.core.regime._format_time_minimal",
        side_effect=lambda x: f"T{x}",
    ), patch(
        "mtdata.utils.regime.bocpd_gaussian",
        return_value={"cp_prob": cp},
    ):
        out = raw(symbol="EURUSD", timeframe="H1", limit=80, method="bocpd", threshold=0.5, lookback=20)

    assert out.get("success") is True
    assert "summary" in out
    assert out["summary"].get("lookback") == 20
    assert "regimes" in out


def test_bocpd_uses_crypto_sensitive_auto_hazard_default() -> None:
    raw = _unwrap(regime_detect)
    capture = {}

    def _fake_bocpd(x, hazard_lambda=0, max_run_length=0):
        capture["hazard_lambda"] = int(hazard_lambda)
        capture["max_run_length"] = int(max_run_length)
        return {"cp_prob": np.zeros_like(x, dtype=float)}

    with patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)), patch(
        "mtdata.core.regime._resolve_denoise_base_col",
        return_value="close",
    ), patch(
        "mtdata.core.regime._format_time_minimal",
        side_effect=lambda x: f"T{x}",
    ), patch(
        "mtdata.utils.regime.bocpd_gaussian",
        side_effect=_fake_bocpd,
    ):
        out = raw(symbol="BTCUSD", timeframe="H1", limit=80, method="bocpd", threshold=0.5, lookback=20)

    assert capture["hazard_lambda"] == 72
    assert out["params_used"]["hazard_lambda"] == 72
    assert out["params_used"]["hazard_lambda_source"] == "auto_default"
    assert abs(float(out["params_used"]["cp_threshold"]) - 0.35) < 1e-12
    assert out["params_used"]["cp_threshold_source"] == "auto_default"


def test_bocpd_hazard_lambda_param_override_is_preserved() -> None:
    raw = _unwrap(regime_detect)
    capture = {}

    def _fake_bocpd(x, hazard_lambda=0, max_run_length=0):
        capture["hazard_lambda"] = int(hazard_lambda)
        capture["max_run_length"] = int(max_run_length)
        return {"cp_prob": np.zeros_like(x, dtype=float)}

    with patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)), patch(
        "mtdata.core.regime._resolve_denoise_base_col",
        return_value="close",
    ), patch(
        "mtdata.core.regime._format_time_minimal",
        side_effect=lambda x: f"T{x}",
    ), patch(
        "mtdata.utils.regime.bocpd_gaussian",
        side_effect=_fake_bocpd,
    ):
        out = raw(
            symbol="BTCUSD",
            timeframe="H1",
            limit=80,
            method="bocpd",
            params={"hazard_lambda": 140},
            threshold=0.5,
            lookback=20,
        )

    assert capture["hazard_lambda"] == 140
    assert out["params_used"]["hazard_lambda"] == 140
    assert out["params_used"]["hazard_lambda_source"] == "params"
    assert out["params_used"]["cp_threshold_source"] == "auto_default"


def test_bocpd_cp_threshold_param_override_is_preserved() -> None:
    raw = _unwrap(regime_detect)
    cp = np.zeros(79, dtype=float)

    with patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)), patch(
        "mtdata.core.regime._resolve_denoise_base_col",
        return_value="close",
    ), patch(
        "mtdata.core.regime._format_time_minimal",
        side_effect=lambda x: f"T{x}",
    ), patch(
        "mtdata.utils.regime.bocpd_gaussian",
        return_value={"cp_prob": cp},
    ):
        out = raw(
            symbol="BTCUSD",
            timeframe="H1",
            limit=80,
            method="bocpd",
            params={"cp_threshold": 0.2},
            threshold=0.5,
            lookback=20,
            output="full",
        )

    assert abs(float(out["params_used"]["cp_threshold"]) - 0.2) < 1e-12
    assert out["params_used"]["cp_threshold_source"] == "params.cp_threshold"


def test_regime_detect_rejects_invalid_min_regime_bars() -> None:
    raw = _unwrap(regime_detect)
    with patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)), patch(
        "mtdata.core.regime._resolve_denoise_base_col",
        return_value="close",
    ):
        out = raw(symbol="EURUSD", timeframe="H1", limit=80, method="hmm", min_regime_bars=0)
    assert "error" in out
    assert "min_regime_bars" in str(out["error"])


def test_regime_detect_default_min_regime_bars_is_5() -> None:
    raw = _unwrap(regime_detect)
    default_val = inspect.signature(raw).parameters["min_regime_bars"].default
    assert int(default_val) == 5


def test_bocpd_zero_change_points_includes_tuning_hint() -> None:
    raw = _unwrap(regime_detect)
    cp = np.zeros(79, dtype=float)

    with patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)), patch(
        "mtdata.core.regime._resolve_denoise_base_col",
        return_value="close",
    ), patch(
        "mtdata.core.regime._format_time_minimal",
        side_effect=lambda x: f"T{x}",
    ), patch(
        "mtdata.utils.regime.bocpd_gaussian",
        return_value={"cp_prob": cp},
    ):
        out = raw(symbol="BTCUSD", timeframe="H1", limit=80, method="bocpd", threshold=0.6, output="summary")

    summary = out.get("summary", {})
    assert summary.get("change_points_count") == 0
    hint = str(summary.get("tuning_hint", ""))
    assert "hazard_lambda" in hint
    assert "threshold" in hint
