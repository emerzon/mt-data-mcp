"""Regression test for market-status summaries with no open exchanges."""

from datetime import datetime, timezone

import mtdata.core.market_status as market_status_mod


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_market_status_summary_includes_all_statuses(monkeypatch):
    statuses = {
        "ASX": "pre_market",
        "EURONEXT": "pre_market",
        "HKEX": "pre_market",
        "SSE": "pre_market",
        "TSE": "pre_market",
        "XETRA": "closed",
        "LSE": "closed",
        "NASDAQ": "closed",
        "NYSE": "closed",
    }
    fixed_now = datetime(2024, 4, 22, 19, 0, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    def mock_check(market_id, _now_local):
        return {"symbol": market_id, "status": statuses[market_id]}

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "_get_local_time", lambda _tz: fixed_now)
    monkeypatch.setattr(market_status_mod, "_check_market_status", mock_check)
    monkeypatch.setattr(market_status_mod, "_get_upcoming_holidays", lambda _markets: [])

    result = _unwrap(market_status_mod.market_status)(detail="full")

    assert result["success"] is True
    assert result["summary"] == "5 pre-market: ASX, EURONEXT, HKEX, SSE, TSE; 4 closed"
    assert result["markets_open"] == 0
    assert result["markets_pre_market"] == 5
    assert result["markets_closed"] == 4
