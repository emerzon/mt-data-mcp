from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mtdata.core.radar import MarketRadarRequest, parse_radar_symbols, run_market_radar
from mtdata.core.web_api import app
from mtdata.core.web_api_radar import compact_session_strip

_client = TestClient(app)


def test_parse_radar_symbols_caps_and_dedupes() -> None:
    symbols = parse_radar_symbols("eurusd, GBPUSD, EURUSD, , XAUUSD", limit=2)
    assert symbols == ["EURUSD", "GBPUSD"]


def test_market_radar_rejects_explicit_empty_watchlist() -> None:
    with pytest.raises(ValidationError, match="contains no symbols"):
        MarketRadarRequest(symbols=" , ; ")

    assert MarketRadarRequest().symbols is None


def _scan_rows(*symbols: str) -> Dict[str, Any]:
    return {
        "success": True,
        "data": [
            {
                "symbol": symbol,
                "bid": 1.1,
                "ask": 1.2,
                "quote_as_of": "2026-08-19T13:48:00Z",
                "quote_age_seconds": 21_560.0 if symbol == "USDJPY" else 1.0,
                "quote_usable_for_live_trading": symbol != "USDJPY",
                "quote_stale": symbol == "USDJPY",
                "price_change_pct": 0.2 if symbol == "EURUSD" else -0.1,
            }
            for symbol in symbols
        ],
    }


def test_market_radar_keeps_watchlist_order() -> None:
    def caller(name: str, kwargs: Dict[str, Any]) -> Any:
        assert name == "scan"
        return _scan_rows("GBPUSD", "EURUSD")

    result = run_market_radar(
        MarketRadarRequest(symbols="EURUSD,GBPUSD", rank_by="watchlist"),
        call_section=caller,
    )
    assert [row["symbol"] for row in result["rows"]] == ["EURUSD", "GBPUSD"]
    assert result["rows"][1]["quote_not_live_ready"] is False


def test_market_radar_marks_unusable_quotes() -> None:
    result = run_market_radar(
        MarketRadarRequest(symbols="USDJPY"),
        call_section=lambda name, kwargs: _scan_rows("USDJPY"),
    )
    assert result["rows"][0]["quote_not_live_ready"] is True
    assert result["rows"][0]["quote_as_of"] == "2026-08-19T13:48:00Z"
    assert result["rows"][0]["quote_age_seconds"] == 21_560.0
    assert result["rows"][0]["quote_stale"] is True


def test_market_radar_fails_closed_when_quote_readiness_is_missing() -> None:
    result = run_market_radar(
        MarketRadarRequest(symbols="EURUSD"),
        call_section=lambda name, kwargs: {
            "success": True,
            "data": [{"symbol": "EURUSD", "bid": 1.1, "ask": 1.2}],
        },
    )

    assert result["rows"][0]["quote_not_live_ready"] is True


def test_market_radar_full_detail_requests_full_scan_rows() -> None:
    observed: Dict[str, Any] = {}

    def caller(name: str, kwargs: Dict[str, Any]) -> Any:
        observed.update(kwargs)
        return _scan_rows("EURUSD")

    run_market_radar(
        MarketRadarRequest(symbols="EURUSD", detail="full"),
        call_section=caller,
    )

    assert observed["detail"] == "full"


def test_market_radar_reports_missing_names() -> None:
    result = run_market_radar(
        MarketRadarRequest(symbols="EURUSD,NOPE"),
        call_section=lambda name, kwargs: _scan_rows("EURUSD"),
    )
    assert result["partial_failure"] is True
    assert result["missing"] == ["NOPE"]


def test_market_radar_seeds_from_top_markets_when_majors_missing() -> None:
    calls: list[str] = []

    def caller(name: str, kwargs: Dict[str, Any]) -> Any:
        calls.append(name)
        if name == "scan" and "EURUSD" in str(kwargs.get("symbols")):
            return {"success": True, "data": []}
        if name == "top_markets":
            return {"data": [{"symbol": "US500"}, {"symbol": "DE40"}]}
        return _scan_rows("US500", "DE40")

    result = run_market_radar(MarketRadarRequest(), call_section=caller)
    assert "top_markets" in calls
    assert [row["symbol"] for row in result["rows"]] == ["US500", "DE40"]
    assert result["seeded"] is True


def test_get_radar_route_returns_compact_rows() -> None:
    payload = {
        "success": True,
        "timeframe": "H1",
        "rank_by": "watchlist",
        "rows": [{"symbol": "EURUSD", "mid": 1.1, "quote_not_live_ready": False}],
        "count": 1,
    }
    with __import__("unittest.mock").mock.patch(
        "mtdata.core.web_api_radar.run_market_radar",
        return_value=payload,
    ):
        response = _client.get("/api/v1/radar", params={"symbols": "EURUSD"})
    assert response.status_code == 200
    assert response.json()["rows"][0]["symbol"] == "EURUSD"


def test_compact_session_strip_survives_partial_failures() -> None:
    payload = compact_session_strip(
        account={"login": 1, "equity": 10000, "currency": "USD", "server": "Demo"},
        news={"error": "news down"},
        exposure={"count": 2},
        market_status={"status": "open", "is_tradable": True},
    )
    assert payload["account"]["equity"] == 10000
    assert payload["exposure_count"] == 2
    assert payload["partial_failure"] is True
    assert "news" in payload["failed_sections"]
    assert "news" not in payload
