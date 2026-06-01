from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.core import regime as regime_mod
from mtdata.core.regime import regime_detect
from mtdata.core.regime.methods.hmm import fit_temporal_gaussian_hmm_1d


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


@pytest.fixture(autouse=True)
def _skip_mt5_connection(monkeypatch):
    monkeypatch.setattr(regime_mod, "ensure_mt5_connection_or_raise", lambda: None)


def _history_from_returns(returns: np.ndarray) -> pd.DataFrame:
    prices = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], returns])))
    n = prices.size
    return pd.DataFrame(
        {
            "time": np.arange(n, dtype=float) * 3600 + 1_700_000_000,
            "open": prices,
            "high": prices + 0.001,
            "low": prices - 0.001,
            "close": prices,
            "tick_volume": np.ones(n),
        }
    )


def test_fit_temporal_gaussian_hmm_1d_uses_true_hmmlearn_backend() -> None:
    rng = np.random.default_rng(42)
    x = np.concatenate(
        [
            rng.normal(-0.004, 0.0004, 40),
            rng.normal(0.004, 0.0004, 40),
        ]
    )

    weights, mu, sigma, gamma, meta = fit_temporal_gaussian_hmm_1d(
        x, n_states=2, seed=42
    )

    assert meta["backend"] == "gaussian_hmm"
    assert gamma.shape == (x.size, int(meta["fitted_n_states"]))
    assert meta["transition_matrix"].shape == (
        int(meta["fitted_n_states"]),
        int(meta["fitted_n_states"]),
    )
    assert np.all(np.diag(meta["transition_matrix"]) > 0.5)
    assert np.all(np.diff(mu) >= 0.0)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(sigma > 0.0)


def test_regime_detect_hmm_prefers_temporal_backend_payload() -> None:
    raw = _unwrap(regime_detect)
    history = _history_from_returns(
        np.concatenate([np.full(30, -0.002), np.full(30, 0.002)])
    )
    gamma = np.vstack(
        [
            np.tile(np.array([[0.98, 0.02]]), (30, 1)),
            np.tile(np.array([[0.03, 0.97]]), (30, 1)),
        ]
    )
    weights = np.array([0.5, 0.5])
    mu = np.array([-0.002, 0.002])
    sigma = np.array([0.001, 0.0015])

    with (
        patch("mtdata.core.regime._fetch_history", return_value=history),
        patch("mtdata.core.regime._resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch(
            "mtdata.core.regime.fit_temporal_gaussian_hmm_1d",
            return_value=(weights, mu, sigma, gamma, {"backend": "gaussian_hmm"}),
        ) as mock_hmm,
        patch(
            "mtdata.core.regime.api.fit_gaussian_mixture_1d",
            side_effect=AssertionError("fallback should not run"),
            create=True,
        ),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            limit=len(history),
            method="hmm",
            params={"n_states": 2},
            detail="full",
            include_series=True,
        )

    mock_hmm.assert_called_once()
    assert out["success"] is True
    assert out["params_used"]["backend"] == "gaussian_hmm"
    assert out["params_used"]["temporal_model"] is True
    assert out["regime_params"]["mu"] == pytest.approx([-0.002, 0.002])
    assert len(out["series"]["state_probabilities"]) == len(history) - 1


def test_regime_detect_hmm_fallback_is_explicit() -> None:
    raw = _unwrap(regime_detect)
    history = _history_from_returns(np.full(40, 0.001))
    gamma = np.tile(np.array([[1.0, 0.0]]), (40, 1))
    weights = np.array([1.0])
    mu = np.array([0.001])
    sigma = np.array([0.0005])

    with (
        patch("mtdata.core.regime._fetch_history", return_value=history),
        patch("mtdata.core.regime._resolve_denoise_base_col", return_value="close"),
        patch("mtdata.core.regime._format_time_minimal", side_effect=lambda x: f"T{x}"),
        patch(
            "mtdata.core.regime.fit_temporal_gaussian_hmm_1d",
            side_effect=RuntimeError("boom"),
        ),
        patch(
            "mtdata.forecast.monte_carlo.fit_gaussian_mixture_1d",
            return_value=(weights, mu, sigma, gamma, 12.5),
            create=True,
        ),
    ):
        out = raw(
            symbol="TEST",
            timeframe="H1",
            limit=len(history),
            method="hmm",
            detail="full",
        )

    assert out["success"] is True
    assert out["params_used"]["backend"] == "gaussian_mixture_fallback"
    assert out["params_used"]["temporal_model"] is False
    assert out["params_used"]["fallback_reason"] == "boom"
    assert "fell back" in " ".join(out.get("warnings", [])).lower()
