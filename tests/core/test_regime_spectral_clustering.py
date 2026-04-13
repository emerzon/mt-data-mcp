"""Smoke test for spectral-clustering-based regime detection on synthetic data.

Runs `regime_detect` with `method="clustering", params={"algorithm": "spectral"}`
while patching the MT5 history fetch.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.core.regime import regime_detect


def _mock_fetch_history(symbol: str, timeframe: str, limit: int, as_of=None) -> pd.DataFrame:
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
class TestSpectralClustering:
    """Tests for clustering method with algorithm='spectral'."""

    def test_spectral_basic(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="clustering",
            params={"algorithm": "spectral", "k_regimes": 2, "window_size": 20},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict), f"Expected dict, got {type(res)}"
        assert res.get("success") or not res.get("error"), f"Error: {res.get('error')}"
        regimes = res.get("regimes", [])
        assert len(regimes) >= 1, "Expected at least 1 regime segment"
        unique_ids = {r["regime"] for r in regimes}
        assert len(unique_ids) >= 2, f"Expected >=2 regime IDs, got {unique_ids}"

    def test_spectral_params_used(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="clustering",
            params={"algorithm": "spectral", "k_regimes": 3},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        params_used = res.get("params_used", {})
        assert params_used.get("algorithm") == "spectral"
        assert params_used.get("k_regimes") == 3

    def test_spectral_compact_output(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="clustering",
            params={"algorithm": "spectral", "k_regimes": 2},
            output="compact",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert "summary" in res or "regimes" in res

    def test_spectral_rbf_affinity(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="clustering",
            params={"algorithm": "spectral", "affinity": "rbf", "k_regimes": 2},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error"), f"Error: {res.get('error')}"
        assert len(res.get("regimes", [])) >= 1

    def test_kmeans_default_unchanged(self, _mock):
        """Verify that omitting algorithm still uses kmeans (backward compat)."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="clustering",
            params={"k_regimes": 2, "window_size": 20},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error")
        params_used = res.get("params_used", {})
        assert params_used.get("algorithm") == "kmeans"
