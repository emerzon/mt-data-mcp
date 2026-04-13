"""Tests for wavelet-based regime detection.

Runs `regime_detect` with `method="wavelet"` while patching the MT5 history
fetch to use synthetic data with two distinct volatility regimes.
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
class TestWaveletRegime:
    """Tests for the wavelet regime detection method."""

    def test_wavelet_basic(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="wavelet",
            params={"n_states": 3},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error"), f"Error: {res.get('error')}"
        regimes = res.get("regimes", [])
        assert len(regimes) >= 1, "Expected at least 1 regime segment"

    def test_wavelet_two_states(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="wavelet",
            params={"n_states": 2},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error")
        regimes = res.get("regimes", [])
        unique_ids = {r["regime"] for r in regimes}
        assert len(unique_ids) >= 2, f"Expected >=2 regime IDs, got {unique_ids}"

    def test_wavelet_params_used(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="wavelet",
            params={"wavelet": "db4", "n_states": 3, "energy_window": 40},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        params_used = res.get("params_used", {})
        assert params_used.get("wavelet") == "db4"
        assert params_used.get("n_states") == 3
        assert params_used.get("energy_window") == 40

    def test_wavelet_compact_output(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="wavelet",
            params={"n_states": 2},
            output="compact",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert "regimes" in res or "summary" in res

    def test_wavelet_summary_output(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="wavelet",
            params={"n_states": 2},
            output="summary",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert "summary" in res
        summary = res["summary"]
        assert "last_state" in summary
        assert "state_shares" in summary

    def test_wavelet_custom_wavelet(self, _mock):
        """Test with a different wavelet family."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="wavelet",
            params={"wavelet": "haar", "n_states": 2},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error"), f"Error: {res.get('error')}"
        assert res.get("params_used", {}).get("wavelet") == "haar"

    def test_wavelet_regime_params(self, _mock):
        """Verify energy profiles are returned in regime_params."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="wavelet",
            params={"n_states": 2},
            output="full",
            include_series=True,
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        rp = res.get("regime_params", {})
        assert "energy_profiles" in rp
        assert "n_bands" in rp
        assert rp["n_bands"] >= 1

    def test_wavelet_current_regime_has_descriptive_label(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="wavelet",
            params={"n_states": 3},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        current = res.get("current_regime", {})
        label = str(current.get("label", ""))
        assert label
        assert not label.startswith("regime_")
        assert any(
            token in label
            for token in ("trend_dominant", "noise_dominant", "mixed_freq")
        )
        info = res.get("regime_info", {})
        regime_id = current.get("regime_id")
        assert regime_id in info
        assert info[regime_id]["label"] == label
        assert "volatility" in info[regime_id]
        labels = {entry["label"] for entry in info.values() if isinstance(entry, dict)}
        assert len(labels) >= 2

    def test_wavelet_insufficient_data(self, _mock):
        """Small limit should produce an error, not crash."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=30,
            method="wavelet",
            params={"n_states": 2, "energy_window": 30},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        # Should either succeed or return a clean error
        if not res.get("success"):
            assert "error" in res

    def test_wavelet_invalid_wavelet_name(self, _mock):
        """Unknown wavelet name should return a clean error."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="wavelet",
            params={"wavelet": "totally_invalid_wavelet_xyz"},
            output="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert "error" in res
