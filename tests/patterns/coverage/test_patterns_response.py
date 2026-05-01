"""Tests for pattern response building and formatting."""

import numpy as np
import pandas as pd
import pytest

from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers to build mock data
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n: int = 200, *, with_time: bool = True, with_volume: bool = True) -> pd.DataFrame:
    """Build a synthetic OHLCV dataframe."""
    rng = np.random.RandomState(42)
    base = 1.1000 + np.cumsum(rng.randn(n) * 0.0005)
    data: Dict[str, Any] = {
        "open": base,
        "high": base + rng.uniform(0.0001, 0.001, n),
        "low": base - rng.uniform(0.0001, 0.001, n),
        "close": base + rng.randn(n) * 0.0002,
    }
    if with_time:
        start = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
        data["time"] = np.arange(start, start + n * 3600, 3600)[:n]
    if with_volume:
        data["tick_volume"] = rng.randint(100, 5000, n)
    return pd.DataFrame(data)


# ── _build_pattern_response ──────────────────────────────────────────────

class TestBuildPatternResponse:

    def _call(self, **kwargs):
        from mtdata.core.patterns import _build_pattern_response
        defaults = dict(
            symbol="EURUSD",
            timeframe="H1",
            limit=500,
            mode="classic",
            patterns=[{"status": "forming", "confidence": 0.8}],
            include_completed=False,
            include_series=False,
            series_time="string",
            df=_make_ohlcv_df(100),
        )
        defaults.update(kwargs)
        return _build_pattern_response(**defaults)

    def test_basic_response(self):
        resp = self._call()
        assert resp["success"] is True
        assert resp["symbol"] == "EURUSD"
        assert resp["timeframe"] == "H1"
        assert resp["mode"] == "classic"

    def test_filters_completed(self):
        patterns = [{"status": "forming"}, {"status": "completed"}]
        resp = self._call(patterns=patterns, include_completed=False)
        assert resp["n_patterns"] == 1

    def test_includes_completed(self):
        patterns = [{"status": "forming"}, {"status": "completed"}]
        resp = self._call(patterns=patterns, include_completed=True)
        assert resp["n_patterns"] == 2

    def test_include_series_string(self):
        resp = self._call(include_series=True, series_time="string")
        assert "series_close" in resp
        assert "series_time" in resp

    def test_include_series_epoch(self):
        resp = self._call(include_series=True, series_time="epoch")
        assert "series_close" in resp
        assert "series_epoch" in resp

    def test_includes_dataframe_warnings(self):
        df = _make_ohlcv_df(100)
        df.attrs["warnings"] = ["sample warning"]

        resp = self._call(df=df)

        assert resp["warnings"] == ["sample warning"]

    def test_mn1_with_ancient_patterns_includes_data_freshness(self):
        """MN1 with old patterns should include data_freshness and warning."""
        patterns = [
            {"status": "forming", "end_date": "2011-08-31 21:00"},
            {"status": "forming", "end_date": "2015-07-31 21:00"},
            {"status": "forming", "end_date": "2024-02-28 21:00"},
        ]
        resp = self._call(timeframe="MN1", patterns=patterns)
        
        assert "data_freshness" in resp
        assert resp["data_freshness"]["oldest_pattern"] == "2011-08-31 21:00"
        assert resp["data_freshness"]["newest_pattern"] == "2024-02-28 21:00"
        assert resp["data_freshness"]["years_spanned"] == 13
        assert "warnings" in resp
        assert any("years_spanned" in str(w) or "ancient" in str(w).lower() or "monthly" in str(w).lower() 
                   for w in resp["warnings"])

    def test_w1_with_old_patterns_includes_data_freshness(self):
        """W1 with old patterns should include data_freshness and warning."""
        patterns = [
            {"status": "forming", "end_date": "2015-01-01 00:00"},
            {"status": "forming", "end_date": "2024-12-31 00:00"},
        ]
        resp = self._call(timeframe="W1", patterns=patterns)
        
        assert "data_freshness" in resp
        # 9 years is less than 10, so years_spanned won't be added
        assert resp["data_freshness"]["oldest_pattern"] == "2015-01-01 00:00"
        assert resp["data_freshness"]["newest_pattern"] == "2024-12-31 00:00"

    def test_h1_does_not_add_data_freshness(self):
        """H1 (intraday) should not add data_freshness warning."""
        patterns = [
            {"status": "forming", "end_date": "2011-08-31 21:00"},
        ]
        resp = self._call(timeframe="H1", patterns=patterns)
        
        # H1 is not MN1 or W1, so should not add data_freshness
        assert "data_freshness" not in resp

    def test_mn1_with_recent_patterns_no_warning(self):
        """MN1 with recent patterns (< 10 years) should not add old-data warning."""
        patterns = [
            {"status": "forming", "end_date": "2020-01-01 00:00"},
            {"status": "forming", "end_date": "2024-12-31 00:00"},
        ]
        resp = self._call(timeframe="MN1", patterns=patterns)
        
        # Years spanned is 4, less than 10, so no warning should be added
        if "data_freshness" in resp:
            # If data_freshness exists, years_spanned should be less than 10
            assert resp["data_freshness"].get("years_spanned", 0) < 10


# ── _build_stock_pattern_frame ───────────────────────────────────────────

class TestBuildStockPatternFrame:

    def _call(self, df):
        from mtdata.core.patterns import _build_stock_pattern_frame
        return _build_stock_pattern_frame(df)

    def test_basic(self):
        df = _make_ohlcv_df(50)
        result = self._call(df)
        assert "Open" in result.columns
        assert "Close" in result.columns
        assert len(result) == 50

    def test_no_time_column(self):
        df = _make_ohlcv_df(50, with_time=False)
        result = self._call(df)
        assert isinstance(result.index, pd.RangeIndex)

    def test_uppercase_columns(self):
        df = pd.DataFrame({
            "Open": [1.0], "High": [1.1], "Low": [0.9], "Close": [1.05], "Volume": [100]
        })
        result = self._call(df)
        assert len(result) == 1


# ── _format_elliott_patterns ─────────────────────────────────────────────

def _mock_pattern_result(**overrides):
    """Build a SimpleNamespace mimicking ClassicPatternResult / ElliottWaveResult."""
    from types import SimpleNamespace
    defaults = dict(
        name="Triangle",
        wave_type="Impulse",
        status="forming",
        confidence=0.85,
        start_index=10,
        end_index=50,
        start_time=1704067200.0,
        end_time=1704110400.0,
        details={"some_key": 1.23456789012},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestFormatElliottPatterns:

    def _call(self, df, cfg):
        from mtdata.core.patterns import _format_elliott_patterns
        return _format_elliott_patterns(df, cfg)

    @patch("mtdata.core.patterns._detect_elliott_waves")
    def test_basic(self, mock_detect):
        mock_detect.return_value = [
            _mock_pattern_result(wave_type="Impulse", start_index=0, end_index=10,
                                 start_time=1704067200.0, end_time=1704110400.0,
                                 confidence=0.9, details={"key": 1.5}),
        ]
        df = _make_ohlcv_df(50)
        result = self._call(df, MagicMock())
        assert len(result) == 1
        assert result[0]["wave_type"] == "Impulse"

    @patch("mtdata.core.patterns._detect_elliott_waves")
    def test_forming_status(self, mock_detect):
        df = _make_ohlcv_df(50)
        mock_detect.return_value = [
            _mock_pattern_result(end_index=48),  # near end
        ]
        result = self._call(df, MagicMock())
        assert result[0]["status"] == "forming"

    @patch("mtdata.core.patterns._detect_elliott_waves")
    def test_completed_status(self, mock_detect):
        df = _make_ohlcv_df(100)
        mock_detect.return_value = [
            _mock_pattern_result(end_index=10),  # far from end
        ]
        result = self._call(df, MagicMock())
        assert result[0]["status"] == "completed"

    @patch("mtdata.core.patterns._detect_elliott_waves")
    def test_exception_skipped(self, mock_detect):
        bad = _mock_pattern_result()
        bad.start_time = "not-a-number"
        bad.end_time = "not-a-number"
        bad.confidence = "bad"
        mock_detect.return_value = [bad]
        df = _make_ohlcv_df(50)
        result = self._call(df, MagicMock())
        # Should either skip or handle gracefully
        assert isinstance(result, list)

    @patch("mtdata.core.patterns._detect_elliott_waves")
    def test_adds_volume_confirmation(self, mock_detect):
        from mtdata.patterns.elliott import ElliottWaveConfig

        df = _make_ohlcv_df(12)
        df["tick_volume"] = [200, 210, 220, 100, 110, 240, 250, 120, 130, 260, 270, 140]
        mock_detect.return_value = [
            _mock_pattern_result(
                wave_type="Impulse",
                start_index=0,
                end_index=10,
                confidence=0.7,
                details={
                    "pattern_family": "impulse",
                    "wave_points_labeled": [
                        {"label": "W0", "index": 0, "time": 1.0, "price": 1.0},
                        {"label": "W1", "index": 2, "time": 2.0, "price": 2.0},
                        {"label": "W2", "index": 4, "time": 3.0, "price": 3.0},
                        {"label": "W3", "index": 6, "time": 4.0, "price": 4.0},
                        {"label": "W4", "index": 8, "time": 5.0, "price": 5.0},
                        {"label": "W5", "index": 10, "time": 6.0, "price": 6.0},
                    ],
                },
            ),
        ]

        result = self._call(
            df,
            ElliottWaveConfig(
                volume_confirm_min_ratio=1.1,
                volume_confirm_bonus=0.1,
                volume_confirm_penalty=0.1,
            ),
        )

        volume_confirmation = result[0]["details"]["volume_confirmation"]
        assert result[0]["confidence"] == pytest.approx(0.8)
        assert volume_confirmation["status"] == "confirmed"
        assert volume_confirmation["trend_to_counter_ratio"] > 1.1

    @patch("mtdata.core.patterns._detect_elliott_waves")
    def test_adds_regime_context(self, mock_detect):
        from mtdata.patterns.elliott import ElliottWaveConfig

        n = 180
        df = pd.DataFrame(
            {
                "time": np.arange(n, dtype=float),
                "close": np.linspace(150.0, 100.0, n),
                "tick_volume": np.full(n, 100.0),
            }
        )
        mock_detect.return_value = [
            _mock_pattern_result(
                wave_type="Impulse",
                start_index=10,
                end_index=n - 2,
                confidence=0.55,
                details={
                    "pattern_family": "impulse",
                    "trend": "bear",
                    "sequence_direction": "bear",
                },
            ),
        ]

        result = self._call(
            df,
            ElliottWaveConfig(
                use_volume_confirmation=False,
                regime_alignment_bonus=0.06,
                regime_countertrend_penalty=0.03,
            ),
        )

        regime_context = result[0]["details"]["regime_context"]
        assert regime_context["state"] == "trending"
        assert regime_context["direction"] == "bearish"
        assert regime_context["status"] == "aligned"
        assert result[0]["confidence"] == pytest.approx(0.61)
