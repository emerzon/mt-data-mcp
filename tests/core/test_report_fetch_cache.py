"""Tests for report request-scoped fetch cache.

Verifies that context_for_tf and attach_multi_timeframes reuse cached
results instead of issuing redundant data fetches for the same
(symbol, timeframe) pair within a single report request.
"""

from unittest.mock import patch

import pytest

from mtdata.core.report.utils import attach_multi_timeframes, context_for_tf


_ROWS = [
    {
        "close": 1.1050,
        "EMA_20": 1.1040,
        "EMA_50": 1.1020,
        "EMA_200": 1.1000,
        "RSI_14": 55.0,
        "MACD_12_26_9": 0.0005,
        "MACDs_12_26_9": 0.0003,
    }
]


def _fake_fetch(**kwargs):
    return {"data": list(_ROWS)}


class TestContextForTfCache:
    """context_for_tf should use _fetch_cache to avoid redundant fetches."""

    def test_cache_hit_skips_fetch(self, monkeypatch):
        """Second call for same (symbol, tf) should not invoke the tool."""
        call_count = 0

        def _counting_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"data": list(_ROWS)}

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _counting_fetch)
        monkeypatch.setattr(
            "mtdata.core.report_templates.basic._compute_compact_trend",
            lambda _rows: None,
        )

        cache = {}
        r1 = context_for_tf("EURUSD", "H1", None, limit=200, tail=1, _fetch_cache=cache)
        r2 = context_for_tf("EURUSD", "H1", None, limit=200, tail=1, _fetch_cache=cache)

        assert call_count == 1, f"Expected 1 fetch call, got {call_count}"
        assert r1 is not None
        assert r2 is not None
        assert r1["close"] == r2["close"]

    def test_cache_is_case_insensitive(self, monkeypatch):
        """'h1' and 'H1' should share the same cache entry."""
        call_count = 0

        def _counting_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"data": list(_ROWS)}

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _counting_fetch)
        monkeypatch.setattr(
            "mtdata.core.report_templates.basic._compute_compact_trend",
            lambda _rows: None,
        )

        cache = {}
        context_for_tf("EURUSD", "h1", None, _fetch_cache=cache)
        context_for_tf("EURUSD", "H1", None, _fetch_cache=cache)

        assert call_count == 1

    def test_different_timeframes_cached_separately(self, monkeypatch):
        call_count = 0

        def _counting_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"data": list(_ROWS)}

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _counting_fetch)
        monkeypatch.setattr(
            "mtdata.core.report_templates.basic._compute_compact_trend",
            lambda _rows: None,
        )

        cache = {}
        context_for_tf("EURUSD", "M15", None, _fetch_cache=cache)
        context_for_tf("EURUSD", "H4", None, _fetch_cache=cache)

        assert call_count == 2
        assert len(cache) == 2

    def test_failed_fetch_cached_as_none(self, monkeypatch):
        """Error result should be cached so we don't retry."""
        call_count = 0

        def _failing_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"error": "no data"}

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _failing_fetch)

        cache = {}
        r1 = context_for_tf("EURUSD", "H1", None, _fetch_cache=cache)
        r2 = context_for_tf("EURUSD", "H1", None, _fetch_cache=cache)

        assert call_count == 1
        assert r1 is None
        assert r2 is None
        assert ("EURUSD", "H1") in cache
        assert cache[("EURUSD", "H1")] is None

    def test_no_cache_allows_repeated_fetches(self, monkeypatch):
        """Without _fetch_cache, each call fetches independently."""
        call_count = 0

        def _counting_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"data": list(_ROWS)}

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _counting_fetch)
        monkeypatch.setattr(
            "mtdata.core.report_templates.basic._compute_compact_trend",
            lambda _rows: None,
        )

        context_for_tf("EURUSD", "H1", None)
        context_for_tf("EURUSD", "H1", None)

        assert call_count == 2


class TestAttachMultiTimeframesCacheThreading:
    """attach_multi_timeframes should thread _fetch_cache to context_for_tf."""

    def test_shared_cache_prevents_duplicate_fetches(self, monkeypatch):
        """Calling attach_multi_timeframes + fallback with same cache => no dupes."""
        fetched_tfs = []

        def _tracking_fetch(**kwargs):
            fetched_tfs.append(kwargs.get("timeframe"))
            return {"data": list(_ROWS)}

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _tracking_fetch)
        monkeypatch.setattr(
            "mtdata.core.report_templates.basic._compute_compact_trend",
            lambda _rows: None,
        )

        cache = {}
        report = {"sections": {"context": {"timeframe": "H1"}}, "meta": {"timeframe": "H1"}}

        # First pass: attach_multi_timeframes fetches M15, H4, D1 (H1 skipped as base)
        attach_multi_timeframes(
            report, "EURUSD", None,
            extra_timeframes=["M15", "H1", "H4", "D1"],
            _fetch_cache=cache,
        )
        first_pass_count = len(fetched_tfs)

        # Second pass: simulate fallback calling context_for_tf for same TFs
        for tf in ["M15", "H4", "D1"]:
            context_for_tf("EURUSD", tf, None, limit=200, tail=30, _fetch_cache=cache)

        # No additional fetches should have occurred
        assert len(fetched_tfs) == first_pass_count, (
            f"Expected {first_pass_count} fetches total, got {len(fetched_tfs)}: {fetched_tfs}"
        )
