from __future__ import annotations

import numpy as np
import pandas as pd

import mtdata.core.causal as causal


def _raw(tool):
    return getattr(tool, "__wrapped__", tool)


def test_cross_correlation_identifies_first_symbol_lead(monkeypatch):
    rng = np.random.default_rng(4)
    left = rng.normal(size=300)
    right = np.concatenate([np.zeros(3), left[:-3]])
    index = pd.date_range("2025-01-01", periods=300, freq="h")
    series = {
        "LEFT": pd.Series(left, index=index),
        "RIGHT": pd.Series(right, index=index),
    }
    monkeypatch.setattr(causal, "_causal_connection_error", lambda: None)
    monkeypatch.setattr(
        causal,
        "_fetch_series_for_window",
        lambda symbol, *args, **kwargs: (series[symbol], None),
    )

    result = _raw(causal.cross_correlation)(
        symbols="LEFT,RIGHT",
        transform="level",
        max_lag=8,
        min_overlap=50,
        bootstrap_samples=50,
    )

    assert result["success"] is True
    assert result["best"]["lag"] == 3
    assert result["best"]["leader"] == "LEFT"
    assert result["context"]["lag_tests"] == 17
    assert result["context"]["significance_correction"] == "bonferroni_across_lags"
    assert result["context"]["ci_per_lag_confidence"] > 0.95


def test_cross_correlation_adjusts_selected_lag_interval(monkeypatch):
    index = pd.date_range("2025-01-01", periods=80, freq="h")
    series = {
        "LEFT": pd.Series(np.arange(80, dtype=float), index=index),
        "RIGHT": pd.Series(np.arange(80, dtype=float), index=index),
    }
    observed: dict[str, float] = {}

    monkeypatch.setattr(causal, "_causal_connection_error", lambda: None)
    monkeypatch.setattr(
        causal,
        "_fetch_series_for_window",
        lambda symbol, *args, **kwargs: (series[symbol], None),
    )

    def _ci(*args, confidence, **kwargs):
        observed["confidence"] = confidence
        return -0.01, 0.01

    monkeypatch.setattr(causal, "_block_bootstrap_correlation_ci", _ci)
    result = _raw(causal.cross_correlation)(
        symbols="LEFT,RIGHT",
        transform="level",
        max_lag=2,
        min_overlap=20,
    )

    assert result["success"] is True
    assert observed["confidence"] == 0.99
    assert result["best"]["significant"] is False


def test_cointegration_johansen_reports_positive_rank(monkeypatch):
    rng = np.random.default_rng(8)
    base = np.cumsum(rng.normal(size=500)) + 100.0
    linked = 1.5 * base + rng.normal(scale=0.2, size=500)
    index = pd.date_range("2024-01-01", periods=500, freq="h")
    series = {
        "AAA": pd.Series(base, index=index),
        "BBB": pd.Series(linked, index=index),
    }
    monkeypatch.setattr(causal, "_causal_connection_error", lambda: None)
    monkeypatch.setattr(
        causal,
        "_fetch_series_for_window",
        lambda symbol, *args, **kwargs: (series[symbol], None),
    )

    result = _raw(causal.cointegration_test)(
        symbols="AAA,BBB",
        method="johansen",
        transform="level",
        min_overlap=100,
        window_bars=400,
    )

    assert result["success"] is True
    assert result["method"] == "johansen"
    assert result["cointegration_rank"] >= 1
    assert result["cointegrating_vectors"]


def test_cointegration_corrects_significance_across_tested_pairs(monkeypatch):
    index = pd.date_range("2025-01-01", periods=120, freq="h")
    series = {
        symbol: pd.Series(np.linspace(100.0 + offset, 120.0 + offset, 120), index=index)
        for offset, symbol in enumerate(("AAA", "BBB", "CCC"))
    }
    p_values = iter((0.02, 0.03, 0.9))

    monkeypatch.setattr(causal, "_causal_connection_error", lambda: None)
    monkeypatch.setattr(
        causal,
        "_fetch_series_for_window",
        lambda symbol, *args, **kwargs: (series[symbol], None),
    )
    monkeypatch.setattr(
        "statsmodels.tsa.stattools.coint",
        lambda *args, **kwargs: (-4.0, next(p_values), [-3.9, -3.3, -3.0]),
    )

    result = _raw(causal.cointegration_test)(
        symbols="AAA,BBB,CCC",
        transform="level",
        min_overlap=80,
        significance=0.05,
    )

    assert result["success"] is True
    assert result["summary"]["counts"]["cointegrated"] == 0
    assert [item["p_value_raw"] for item in result["items"]] == [0.02, 0.03, 0.9]
    assert [item["p_value"] for item in result["items"]] == [0.06, 0.06, 0.9]
    assert all(item["p_value_correction"] == "holm_across_pairs" for item in result["items"])
