from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mtdata.services import unified_news as svc


def _reset_aggregator(monkeypatch) -> None:
    monkeypatch.setattr(svc, "_news_aggregator", None)


def _disable_ycnbc(monkeypatch) -> None:
    monkeypatch.setattr(svc, "_import_ycnbc", lambda: (_ for _ in ()).throw(ImportError("ycnbc unavailable")))


def _disable_embeddings(monkeypatch) -> None:
    class FakeEmbeddingService:
        enabled = True
        top_n = 0
        weight = 1.0

        def is_available(self) -> bool:
            return False

        def score_documents(self, context, items):
            return {}

        def status(self):
            return {"enabled": True, "available": False, "model": "Qwen/Qwen3-Embedding-0.6B"}

    monkeypatch.setattr(svc, "get_news_embedding_service", lambda: FakeEmbeddingService())


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
    _disable_ycnbc(monkeypatch)
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
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
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


def test_classify_instrument_preserves_usdt_quotes(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)

    context = svc._classify_instrument("BNBUSDT")

    assert context.asset_class == "crypto"
    assert context.base_asset == "BNB"
    assert context.quote_asset == "USDT"


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


def test_classify_index_hints_override_mt5_currency_metadata(monkeypatch) -> None:
    class FakeInfo:
        path = "Indices\\Cash"
        description = "US Tech 100"
        currency_base = "USD"
        currency_profit = "USD"
        currency_margin = "USD"

    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: FakeInfo())

    context = svc._classify_instrument("USTEC")

    assert context.asset_class == "index"
    assert context.base_asset == "NAS"
    assert context.quote_asset is None
    assert "NQ" in context.aliases


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
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
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
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("NAS100")

    snapshots = [item for item in result["related_news"] if item["kind"] == "market_snapshot"]
    assert snapshots
    assert snapshots[0]["published_at"] is not None
    assert snapshots[0]["metadata"]["snapshot_time_inferred"] is True
    assert result["source_details"]["finviz"]["selected_related"] >= 1


def test_market_snapshots_handle_lowercase_and_pair_keys(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(
        svc,
        "get_forex_performance",
        lambda: {
            "success": True,
            "pairs": [{"Pair": "EUR/USD", "Change": "0.5%"}],
        },
    )
    monkeypatch.setattr(
        svc,
        "get_futures_performance",
        lambda: {
            "success": True,
            "futures": [{"ticker": "NQ", "label": "Nasdaq 100 E-mini", "group": "Indices", "perf": "0.8%"}],
        },
    )
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {"success": True, "items": []},
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
    _reset_aggregator(monkeypatch)

    forex_result = svc.fetch_unified_news("EURUSD")
    futures_result = svc.fetch_unified_news("NAS100")

    assert forex_result["related_news"][0]["title"] == "EUR/USD market snapshot"
    assert futures_result["related_news"][0]["title"] == "NQ market snapshot"
    assert "Label: Nasdaq 100 E-mini" in (futures_result["related_news"][0]["summary"] or "")
    assert "Perf: 0.8%" in (futures_result["related_news"][0]["summary"] or "")


def test_index_snapshot_candidate_pool_keeps_more_than_three_rows(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(
        svc,
        "get_futures_performance",
        lambda: {
            "success": True,
            "futures": [
                {"Ticker": "NQ", "Price": "20100.00", "Change": "0.8%"},
                {"Ticker": "NAS100", "Price": "20105.00", "Change": "0.7%"},
                {"Ticker": "USTEC", "Price": "20110.00", "Change": "0.6%"},
                {"Ticker": "NDX", "Price": "20115.00", "Change": "0.5%"},
                {"Ticker": "ES", "Price": "5300.00", "Change": "0.2%"},
            ],
        },
    )
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {"success": True, "items": []},
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("NAS100")

    snapshots = [item for item in result["related_news"] if item["kind"] == "market_snapshot"]
    assert len(snapshots) >= 4


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
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("NAS100")

    assert result["related_news"] == []


def test_systemic_impact_news_surfaces_major_war_headline(monkeypatch) -> None:
    monkeypatch.setattr(
        svc,
        "get_general_news",
        lambda news_type="news", limit=20, page=1: {
            "success": True,
            "items": [
                {
                    "Title": "Oil surges as war fears deepen after overnight missile strikes",
                    "Source": "Reuters",
                    "Date": "2026-03-29T08:00:00Z",
                    "Link": "https://example.com/war-oil",
                }
            ],
        },
    )
    monkeypatch.setattr(svc, "get_futures_performance", lambda: {"success": True, "futures": []})
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {"success": True, "items": []},
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("NAS100")

    assert result["impact_count"] >= 1
    assert any("war fears deepen" in item["title"].lower() for item in result["impact_news"])
    assert result["impact_news"][0]["metadata"]["systemic_impact_score"] >= 2.4


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


def test_fetch_unified_news_can_rerank_with_embeddings(monkeypatch) -> None:
    class FakeEmbeddingService:
        enabled = True
        top_n = 5
        weight = 1.0

        def is_available(self) -> bool:
            return True

        def score_documents(self, context, items):
            return {
                items[0].dedupe_key(): 0.05,
                items[1].dedupe_key(): 0.95,
            }

        def status(self):
            return {"enabled": True, "available": True, "model": "Qwen/Qwen3-Embedding-0.6B"}

    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(
        svc,
        "get_futures_performance",
        lambda: {
            "success": True,
            "futures": [
                {"Ticker": "NQ", "Price": "20100.00", "Change": "0.4%"},
                {"Ticker": "NAS100", "Price": "20110.00", "Change": "0.5%"},
            ],
        },
    )
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {"success": True, "items": []},
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    monkeypatch.setattr(svc, "get_news_embedding_service", lambda: FakeEmbeddingService())
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("NAS100")

    assert result["related_news"][0]["title"] == "NAS100 market snapshot"
    assert result["related_news"][0]["metadata"]["embedding_used"] is True
    assert result["matching"]["embeddings"]["enabled"] is True


def test_fetch_unified_news_includes_ycnbc_general_candidates_when_enabled(monkeypatch) -> None:
    class FakeNews:
        def latest(self):
            return [
                {
                    "headline": "Oil prices reverse course as traders assess Iran conflict",
                    "time": "23 min ago",
                    "link": "https://example.com/cnbc-oil",
                }
            ]

        def __getattr__(self, _name):
            return lambda: []

    class FakeStocksUtil:
        def news(self, symbol: str):
            return []

    monkeypatch.setattr(svc, "_import_ycnbc", lambda: (FakeNews, FakeStocksUtil))
    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": False, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news()

    assert result["success"] is True
    assert "ycnbc" in result["sources_used"]
    assert any(item["provider"] == "ycnbc" for item in result["general_news"])
    assert result["source_details"]["ycnbc"]["selected_general"] >= 1


def test_fetch_unified_news_maps_indices_to_ycnbc_quote_symbols(monkeypatch) -> None:
    class FakeNews:
        def latest(self):
            return []

        def __getattr__(self, _name):
            return lambda: []

    class FakeStocksUtil:
        def news(self, symbol: str):
            assert symbol == ".NDX"
            return [
                {
                    "headline": "Nasdaq 100 rebounds as chip stocks lead gains",
                    "posttime": "4 Hours Ago",
                    "link": "https://example.com/cnbc-ndx",
                }
            ]

    monkeypatch.setattr(svc, "_import_ycnbc", lambda: (FakeNews, FakeStocksUtil))
    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(svc, "get_futures_performance", lambda: {"success": True, "futures": []})
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {"success": True, "items": []},
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": False, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("NAS100")

    assert result["success"] is True
    assert any("Nasdaq 100 rebounds" in item["title"] for item in result["related_news"])
    assert result["source_details"]["ycnbc"]["related_candidates"] >= 1
    ycnbc_items = [item for item in result["related_news"] if item["provider"] == "ycnbc"]
    assert ycnbc_items[0]["metadata"]["cnbc_symbol"] == ".NDX"


def test_fetch_unified_news_returns_failure_when_all_sources_error(monkeypatch) -> None:
    class BrokenSource:
        name = "broken"

        def is_available(self) -> bool:
            return True

        def fetch_general_candidates(self, limit: int):
            raise RuntimeError("boom")

        def fetch_related_candidates(self, context, limit: int):
            raise RuntimeError("boom")

    aggregator = svc.NewsAggregator()
    aggregator._sources = {"broken": BrokenSource()}

    result = aggregator.fetch_news("AAPL")

    assert result["success"] is False
    assert result["error"] == "All news sources failed"
    assert result["source_details"]["broken"]["success"] is False
