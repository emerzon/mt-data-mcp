"""Tests for ensemble (consensus voting) regime detection.

Runs `regime_detect` with `method="ensemble"` while patching the MT5 history
fetch to use synthetic data with two distinct volatility regimes.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.core.regime import regime_detect


def _mock_fetch_history(
    symbol: str, timeframe: str, limit: int, as_of=None
) -> pd.DataFrame:
    rng = np.random.default_rng(12345)
    n = int(limit)
    t = pd.date_range("2024-01-01", periods=n, freq="h")

    split = min(400, n)
    y = np.empty(n, dtype=float)
    y[:split] = 100.0 + np.cumsum(rng.normal(0.0, 0.5, split))
    if split < n:
        y[split:] = y[split - 1] + np.cumsum(rng.normal(0.0, 2.0, n - split))

    t_seconds = (t.astype("int64") // 10**9).astype(np.int64)
    return pd.DataFrame(
        {
            "time": t_seconds,
            "close": y,
            "open": y,
            "high": y,
            "low": y,
            "tick_volume": 100,
            "spread": 1,
            "real_volume": 100,
        }
    )


@patch("mtdata.core.regime._fetch_history", side_effect=_mock_fetch_history)
class TestEnsembleRegime:
    """Tests for the ensemble (consensus voting) regime detection method."""

    def test_ensemble_default_methods(self, _mock):
        """Default sub-methods (bocpd, hmm, clustering) should all succeed."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="ensemble",
            params={"n_states": 2},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error"), f"Error: {res.get('error')}"
        regimes = res.get("regimes", [])
        assert len(regimes) >= 1

        params_used = res.get("params_used", {})
        assert params_used.get("n_methods_succeeded") == 4
        assert set(params_used.get("methods", [])) == {
            "bocpd",
            "hmm",
            "clustering",
            "wavelet",
        }

    def test_ensemble_soft_voting(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="ensemble",
            params={"n_states": 2, "methods": ["hmm", "clustering"], "voting": "soft"},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error")
        assert res.get("params_used", {}).get("voting") == "soft"

    def test_ensemble_hard_voting(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="ensemble",
            params={"n_states": 2, "methods": ["hmm", "clustering"], "voting": "hard"},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error")
        assert res.get("params_used", {}).get("voting") == "hard"

    def test_ensemble_agreement_score(self, _mock):
        """Ensemble should report mean_agreement in regime_params."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="ensemble",
            params={"n_states": 2, "methods": ["hmm", "clustering"]},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        rp = res.get("regime_params", {})
        assert "mean_agreement" in rp
        assert 0.0 <= rp["mean_agreement"] <= 1.0

    def test_ensemble_compact_output(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="ensemble",
            params={"n_states": 2},
            output="compact",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert "regimes" in res or "summary" in res

    def test_ensemble_summary_output(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="ensemble",
            params={"n_states": 2},
            output="summary",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        summary = res.get("summary", {})
        assert "last_state" in summary
        assert "mean_agreement" in summary

    def test_ensemble_excludes_self(self, _mock):
        """If 'ensemble' is passed as a sub-method, it's silently excluded."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="ensemble",
            params={"n_states": 2, "methods": ["hmm", "ensemble", "clustering"]},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error")
        methods_used = res.get("params_used", {}).get("methods", [])
        assert "ensemble" not in methods_used

    def test_ensemble_with_wavelet(self, _mock):
        """Ensemble with wavelet sub-method."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="ensemble",
            params={"n_states": 2, "methods": ["hmm", "wavelet"]},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error")

    def test_ensemble_single_method_fallback(self, _mock):
        """Ensemble with only one valid sub-method should still work."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="ensemble",
            params={"n_states": 2, "methods": ["hmm"]},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error")

    def test_ensemble_no_valid_methods(self, _mock):
        """Ensemble with no valid sub-methods should error cleanly."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="ensemble",
            params={"n_states": 2, "methods": ["ensemble", "rule_based"]},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert "error" in res
