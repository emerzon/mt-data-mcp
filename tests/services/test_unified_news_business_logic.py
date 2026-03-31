from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mtdata.services import unified_news as svc


def _reset_aggregator(monkeypatch) -> None:
    monkeypatch.setattr(svc, "_news_aggregator", None)


def test_fetch_unified_news_without_symbol_returns_only_general_bucket(monkeypatch) -> None:
    def fake_general_news(news_type: str = "news", limit: int = 20, page: int = 1):
        return {
            "success": True,
            "items": [
                {
                    "Title": "Fed holds rates steady ahead of CPI",
                    "Source": "Reuters",
                    "Date": "2026-03-29T08:00:00Z",
                    "Link": "https://example.com/fed",
                }
            ],
        }

    def fake_mt5_news(**_kwargs):
        return {
            "success": True,
            "news": [
                {
                    "subject": "US inflation release due later today",
                    "source": "FXStreet",
                    "category": "Economic Calendar",
                    "priority": 1,
                    "relative_time": "10 minutes ago",
                }
            ],
        }

    monkeypatch.setattr(svc, "get_general_news", fake_general_news)
    monkeypatch.setattr(svc, "get_mt5_news", fake_mt5_news)
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news()

    assert result["success"] is True
    assert result["related_news"] == []
    assert result["general_count"] == 2
    assert set(result["sources_used"]) == {"finviz", "mt5"}


def test_fetch_unified_news_uses_symbol_metadata_for_crypto_classification(monkeypatch) -> None:
    class FakeInfo:
        path = "Crypto\\Majors"
        description = "Bitcoin vs US Dollar"
        currency_base = "BTC"
        currency_profit = "USD"
        currency_margin = "USD"

    def fake_general_news(news_type: str = "news", limit: int = 20, page: int = 1):
        return {
            "success": True,
            "items": [
                {
                    "Title": "Crypto traders await Fed decision",
                    "Source": "Reuters",
                    "Date": "2026-03-29T07:00:00Z",
                    "Link": "https://example.com/crypto-fed",
                }
            ],
        }

    def fake_mt5_news(**_kwargs):
        return {
            "success": True,
            "news": [
                {
                    "subject": "Bitcoin steady ahead of CPI release",
                    "source": "FXStreet",
                    "category": "Economic Calendar",
                    "priority": 1,
                    "relative_time": "5 minutes ago",
                }
            ],
        }

    def fake_crypto_performance():
        return {
            "success": True,
            "coins": [
                {"Ticker": "BTCUSD", "Price": "70000.00", "Change": "2.5%"},
                {"Ticker": "ETHUSD", "Price": "3500.00", "Change": "1.0%"},
            ],
        }

    def fake_economic_calendar(limit: int = 100, page: int = 1, impact=None, date_from=None, date_to=None):
        return {
            "success": True,
            "items": [
                {
                    "Datetime": "2026-03-29T12:30:00Z",
                    "Release": "US CPI",
                    "For": "USD",
                    "Impact": "high",
                    "Category": "Inflation",
                    "Reference": "BLS",
                }
            ],
        }

    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: FakeInfo())
    monkeypatch.setattr(svc, "get_general_news", fake_general_news)
    monkeypatch.setattr(svc, "get_mt5_news", fake_mt5_news)
    monkeypatch.setattr(svc, "get_crypto_performance", fake_crypto_performance)
    monkeypatch.setattr(svc, "get_economic_calendar", fake_economic_calendar)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("BTCUSD")
    titles = [item["title"] for item in result["related_news"]]

    assert result["success"] is True
    assert result["instrument"]["asset_class"] == "crypto"
    assert result["instrument"]["metadata_hints"]["currency_base"] == "BTC"
    assert any("market snapshot" in title.lower() for title in titles)
    assert any("US CPI" in title for title in titles)
    assert any("Bitcoin steady ahead of CPI release" == title for title in titles)


def test_fetch_unified_news_includes_direct_equity_news(monkeypatch) -> None:
    class FakeInfo:
        path = "Stocks\\US"
        description = "Apple Inc"
        currency_profit = "USD"

    def fake_general_news(news_type: str = "news", limit: int = 20, page: int = 1):
        return {"success": True, "items": []}

    def fake_stock_news(symbol: str, limit: int = 20, page: int = 1):
        return {
            "success": True,
            "news": [
                {
                    "Title": "Apple unveils new AI features",
                    "Source": "Reuters",
                    "Date": "2026-03-29T09:00:00Z",
                    "Link": "https://example.com/aapl-ai",
                }
            ],
        }

    def fake_mt5_news(**_kwargs):
        return {"success": True, "news": []}

    def fake_economic_calendar(limit: int = 100, page: int = 1, impact=None, date_from=None, date_to=None):
        return {"success": True, "items": []}

    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: FakeInfo())
    monkeypatch.setattr(svc, "get_general_news", fake_general_news)
    monkeypatch.setattr(svc, "get_stock_news", fake_stock_news)
    monkeypatch.setattr(svc, "get_mt5_news", fake_mt5_news)
    monkeypatch.setattr(svc, "get_economic_calendar", fake_economic_calendar)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("AAPL")

    assert result["success"] is True
    assert result["instrument"]["asset_class"] == "equity"
    assert result["related_news"][0]["kind"] == "direct_symbol"
    assert result["related_news"][0]["title"] == "Apple unveils new AI features"


def test_fetch_unified_news_treats_whitespace_symbol_as_general_news(monkeypatch) -> None:
    monkeypatch.setattr(
        svc,
        "get_general_news",
        lambda news_type="news", limit=20, page=1: {
            "success": True,
            "items": [{"Title": "Fed preview", "Source": "Reuters", "Date": "2026-03-29T08:00:00Z"}],
        },
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("   ")

    assert result["success"] is True
    assert result["symbol"] is None
    assert result["instrument"] is None
    assert result["related_news"] == []
    assert result["general_count"] == 1


def test_classify_instrument_handles_four_character_crypto_bases(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)

    context = svc._classify_instrument("DOGEUSD")

    assert context.asset_class == "crypto"
    assert context.base_asset == "DOGE"
    assert context.quote_asset == "USD"
    assert "DOGE/USD" in context.aliases


def test_classify_instrument_handles_short_and_alias_commodities(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)

    natgas = svc._classify_instrument("NGAS")
    brent = svc._classify_instrument("XBRUSD")

    assert natgas.asset_class == "commodity"
    assert natgas.base_asset == "NG"
    assert brent.asset_class == "commodity"
    assert brent.base_asset == "BRENT"
    assert brent.quote_asset == "USD"
    assert "XBR" in brent.aliases


def test_parse_relative_time_handles_yesterday() -> None:
    parsed = svc._parse_relative_time("yesterday")

    assert parsed is not None
    assert datetime.now(timezone.utc) - timedelta(days=1, minutes=1) <= parsed <= datetime.now(timezone.utc)


def test_unknown_equity_without_specific_evidence_returns_no_related_items(monkeypatch) -> None:
    monkeypatch.setattr(
        svc,
        "get_general_news",
        lambda news_type="news", limit=20, page=1: {
            "success": True,
            "items": [
                {
                    "Title": "Fed officials signal patience on rates",
                    "Source": "Reuters",
                    "Date": "2026-03-29T08:00:00Z",
                }
            ],
        },
    )
    monkeypatch.setattr(svc, "get_stock_news", lambda symbol, limit=20, page=1: {"success": True, "news": []})
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {
            "success": True,
            "items": [
                {
                    "Datetime": "2026-03-29T12:30:00Z",
                    "Release": "House Price Index",
                    "For": "USD",
                    "Impact": "medium",
                    "Category": "Housing",
                    "Reference": "FHFA",
                }
            ],
        },
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("ZZZZNOTREAL")

    assert result["success"] is True
    assert result["instrument"]["asset_class"] == "equity"
    assert result["related_news"] == []
    assert result["source_details"]["finviz"]["selected_related"] == 0


def test_index_aliases_match_futures_snapshots(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(
        svc,
        "get_futures_performance",
        lambda: {
            "success": True,
            "futures": [{"Ticker": "NQ", "Price": "20100.00", "Change": "0.8%"}],
        },
    )
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {"success": True, "items": []},
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("NAS100")

    snapshots = [item for item in result["related_news"] if item["kind"] == "market_snapshot"]
    assert snapshots
    assert snapshots[0]["published_at"] is not None
    assert snapshots[0]["metadata"]["snapshot_time_inferred"] is True
    assert result["source_details"]["finviz"]["selected_related"] >= 1


def test_us_index_filters_non_macro_calendar_noise(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(svc, "get_futures_performance", lambda: {"success": True, "futures": []})
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {
            "success": True,
            "items": [
                {
                    "Datetime": "2026-03-29T12:30:00Z",
                    "Release": "House Price Index",
                    "For": "USD",
                    "Impact": "medium",
                    "Category": "Housing",
                    "Reference": "FHFA",
                }
            ],
        },
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("NAS100")

    assert result["related_news"] == []


def test_european_index_matches_regional_macro_events(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(svc, "get_futures_performance", lambda: {"success": True, "futures": []})
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {
            "success": True,
            "items": [
                {
                    "Datetime": "2026-03-29T09:00:00Z",
                    "Release": "German CPI",
                    "For": "EUR",
                    "Impact": "high",
                    "Category": "Inflation",
                    "Reference": "Destatis",
                }
            ],
        },
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("GER40")

    assert any("German CPI" in item["title"] for item in result["related_news"])
