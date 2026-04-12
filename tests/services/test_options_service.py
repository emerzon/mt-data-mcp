from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import requests

from mtdata.services import options_service as osvc


def test_to_numeric_logs_non_empty_conversion_failures(caplog):
    with caplog.at_level("WARNING"):
        out = osvc._to_numeric("bad-data", float, float("nan"), field_name="strike")

    assert out != out
    assert "Failed to coerce Yahoo options 'strike' value 'bad-data' to float" in caplog.text


def test_get_options_expirations_parses_payload(monkeypatch):
    expiry_a = osvc._ymd_to_epoch("2026-04-17")
    expiry_b = osvc._ymd_to_epoch("2026-05-15")

    monkeypatch.setattr(
        osvc,
        "_fetch_yahoo_options_payload",
        lambda symbol, expiry_epoch=None: {
            "expirationDates": [expiry_b, expiry_a],
            "quote": {"regularMarketPrice": 212.34, "currency": "USD"},
        },
    )

    out = osvc.get_options_expirations("aapl")
    assert out["success"] is True
    assert out["symbol"] == "AAPL"
    assert out["underlying_price"] == 212.34
    assert out["expirations"] == ["2026-04-17", "2026-05-15"]
    assert out["expiration_count"] == 2


def test_get_options_chain_filters_and_selects_expiration(monkeypatch):
    expiry_a = osvc._ymd_to_epoch("2026-04-17")
    expiry_b = osvc._ymd_to_epoch("2026-05-15")

    def fake_fetch(symbol, expiry_epoch=None):
        if expiry_epoch is None:
            return {
                "expirationDates": [expiry_a, expiry_b],
                "quote": {"regularMarketPrice": 100.5, "currency": "USD"},
            }
        return {
            "expirationDates": [expiry_a, expiry_b],
            "quote": {"regularMarketPrice": 100.5, "currency": "USD"},
            "options": [
                {
                    "calls": [
                        {
                            "contractSymbol": "AAPL260417C00100000",
                            "strike": 100.0,
                            "lastPrice": 2.1,
                            "bid": 2.0,
                            "ask": 2.2,
                            "change": 0.1,
                            "percentChange": 5.0,
                            "volume": 15,
                            "openInterest": 20,
                            "impliedVolatility": 0.25,
                            "inTheMoney": True,
                            "lastTradeDate": 0,
                            "currency": "USD",
                        },
                        {
                            "contractSymbol": "AAPL260417C00110000",
                            "strike": 110.0,
                            "lastPrice": 1.0,
                            "bid": 0.9,
                            "ask": 1.1,
                            "change": 0.0,
                            "percentChange": 0.0,
                            "volume": 1,
                            "openInterest": 1,
                            "impliedVolatility": 0.3,
                            "inTheMoney": False,
                            "lastTradeDate": 0,
                            "currency": "USD",
                        },
                    ],
                    "puts": [
                        {
                            "contractSymbol": "AAPL260417P00100000",
                            "strike": 100.0,
                            "lastPrice": 1.8,
                            "bid": 1.7,
                            "ask": 1.9,
                            "change": -0.1,
                            "percentChange": -5.0,
                            "volume": 12,
                            "openInterest": 18,
                            "impliedVolatility": 0.22,
                            "inTheMoney": False,
                            "lastTradeDate": 0,
                            "currency": "USD",
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(osvc, "_fetch_yahoo_options_payload", fake_fetch)

    out = osvc.get_options_chain(
        symbol="aapl",
        expiration="2026-04-17",
        option_type="call",
        min_open_interest=10,
        min_volume=10,
        limit=10,
    )
    assert out["success"] is True
    assert out["symbol"] == "AAPL"
    assert out["expiration"] == "2026-04-17"
    assert out["option_type"] == "call"
    assert out["count"] == 1
    assert out["calls_count"] == 1
    assert out["puts_count"] == 0
    assert out["options"][0]["contract"] == "AAPL260417C00100000"


def test_get_options_chain_rejects_unavailable_expiration(monkeypatch):
    expiry = osvc._ymd_to_epoch("2026-04-17")
    monkeypatch.setattr(
        osvc,
        "_fetch_yahoo_options_payload",
        lambda symbol, expiry_epoch=None: {
            "expirationDates": [expiry],
            "quote": {"regularMarketPrice": 100.5, "currency": "USD"},
        },
    )

    out = osvc.get_options_chain(symbol="AAPL", expiration="2026-05-15")
    assert "error" in out
    assert "not available" in out["error"]
    assert out["expirations"] == ["2026-04-17"]


def test_get_yahoo_session_reuses_single_session(monkeypatch):
    sessions = []

    def fake_session():
        session = MagicMock()
        sessions.append(session)
        return session

    monkeypatch.setattr(osvc.requests, "Session", fake_session)
    monkeypatch.setattr(osvc, "_YAHOO_SESSION", None)

    first = osvc._get_yahoo_session()
    second = osvc._get_yahoo_session()

    assert first is second
    assert len(sessions) == 1


def test_fetch_yahoo_options_payload_retries_rate_limited_response(monkeypatch):
    retry_response = SimpleNamespace(status_code=429, headers={"Retry-After": "1"})
    ok_response = MagicMock(status_code=200, headers={})
    ok_response.raise_for_status.return_value = None
    ok_response.json.return_value = {
        "optionChain": {
            "result": [
                {
                    "quote": {"regularMarketPrice": 100.5, "currency": "USD"},
                    "expirationDates": [],
                }
            ]
        }
    }
    session = MagicMock()
    session.get.side_effect = [retry_response, ok_response]
    sleep_calls = []

    monkeypatch.setattr(osvc, "_get_yahoo_session", lambda: session)
    monkeypatch.setattr(osvc, "_throttle_yahoo_request", lambda: None)
    monkeypatch.setattr(osvc._time, "sleep", lambda seconds: sleep_calls.append(seconds))

    out = osvc._fetch_yahoo_options_payload("AAPL")

    assert out["quote"]["regularMarketPrice"] == 100.5
    assert session.get.call_count == 2
    assert sleep_calls == [1.0]


# ---------------------------------------------------------------------------
# Yahoo session lifecycle tests
# ---------------------------------------------------------------------------


def test_build_yahoo_session_returns_fresh_session():
    session = osvc._build_yahoo_session()
    assert isinstance(session, requests.Session)
    session.close()


def test_reset_yahoo_session_clears_singleton(monkeypatch):
    sentinel = osvc._build_yahoo_session()
    monkeypatch.setattr(osvc, "_YAHOO_SESSION", sentinel)

    osvc._reset_yahoo_session()
    assert osvc._YAHOO_SESSION is None


def test_reset_yahoo_session_tolerates_already_none():
    original = osvc._YAHOO_SESSION
    try:
        osvc._YAHOO_SESSION = None
        osvc._reset_yahoo_session()
        assert osvc._YAHOO_SESSION is None
    finally:
        osvc._YAHOO_SESSION = original


def test_get_yahoo_session_delegates_to_builder(monkeypatch):
    calls = {"built": 0}

    class FakeSession:
        pass

    def fake_build():
        calls["built"] += 1
        return FakeSession()

    monkeypatch.setattr(osvc, "_YAHOO_SESSION", None)
    monkeypatch.setattr(osvc, "_build_yahoo_session", fake_build)

    first = osvc._get_yahoo_session()
    second = osvc._get_yahoo_session()

    assert first is second
    assert calls["built"] == 1
    assert isinstance(first, FakeSession)
