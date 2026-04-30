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


def test_fetch_unified_news_defaults_to_twenty_general_items(monkeypatch) -> None:
    def fake_general_news(news_type: str = "news", limit: int = 20, page: int = 1):
        return {
            "success": True,
            "items": [
                {
                    "Title": f"Headline {idx}",
                    "Source": "Reuters",
                    "Date": f"2026-03-{idx + 1:02d}T08:00:00Z",
                    "Link": f"https://example.com/headline-{idx}",
                }
                for idx in range(30)
            ],
        }

    monkeypatch.setattr(svc, "get_general_news", fake_general_news)
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news()

    assert result["success"] is True
    assert result["general_count"] == 20
    assert len(result["general_news"]) == 20
    assert result["general_news"][0]["title"] == "Headline 29"


def test_fetch_unified_news_prioritizes_more_recent_general_headlines(monkeypatch) -> None:
    recent_time = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
    stale_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()

    monkeypatch.setattr(
        svc,
        "get_general_news",
        lambda news_type="news", limit=20, page=1: {
            "success": True,
            "items": [
                {
                    "Title": "Inflation update older",
                    "Source": "Reuters",
                    "Date": stale_time,
                    "Link": "https://example.com/older",
                },
                {
                    "Title": "Inflation update recent",
                    "Source": "Reuters",
                    "Date": recent_time,
                    "Link": "https://example.com/recent",
                },
            ],
        },
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news()

    assert [item["title"] for item in result["general_news"][:2]] == [
        "Inflation update recent",
        "Inflation update older",
    ]


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
                    "date": "2026-03-29T12:30:00Z",
                    "event": "US CPI",
                    "ticker": "USD",
                    "importance": 3,
                    "category": "Inflation",
                    "reference": "BLS",
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
    assert list(result).index("related_news") < list(result).index("general_news")
    assert result["instrument"]["asset_class"] == "equity"
    assert result["related_news"][0]["kind"] == "direct_symbol"
    assert result["related_news"][0]["title"] == "Apple unveils new AI features"


def test_fetch_unified_news_uses_root_ticker_for_equity_cfd_symbols(monkeypatch) -> None:
    class FakeInfo:
        path = "Stock CFD's\\Nasdaq\\24HR NAS"
        description = "Apple Inc"
        currency_profit = "USD"

    seen_symbols: list[str] = []

    def fake_stock_news(symbol: str, limit: int = 20, page: int = 1):
        seen_symbols.append(symbol)
        if symbol != "AAPL":
            return {"success": False, "error": "unsupported"}
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

    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: FakeInfo())
    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(svc, "get_stock_news", fake_stock_news)
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {"success": True, "items": []},
    )
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("AAPL.NAS-24")

    assert seen_symbols == ["AAPL"]
    assert result["success"] is True
    assert result["instrument"]["symbol"] == "AAPL.NAS-24"
    assert result["related_news"][0]["title"] == "Apple unveils new AI features"
    assert result["related_news"][0]["metadata"]["direct_symbol"] == "AAPL.NAS-24"
    assert result["related_news"][0]["metadata"]["source_symbol"] == "AAPL"


def test_equity_stock_page_news_requires_company_evidence(monkeypatch) -> None:
    class FakeInfo:
        path = "Stocks\\US"
        description = "Apple Inc designs consumer electronics and software"
        currency_base = ""
        currency_profit = ""
        currency_margin = ""

    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: FakeInfo())
    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(
        svc,
        "get_stock_news",
        lambda symbol, limit=20, page=1: {
            "success": True,
            "news": [
                {"Title": "Apple expands AI tooling for developers", "Source": "Reuters", "Date": "2026-03-29T09:00:00Z"},
                {"Title": "Netflix initiated, Instacart upgraded: Wall Street's top analyst calls", "Source": "The Fly", "Date": "2026-03-29T10:00:00Z"},
            ],
        },
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {"success": True, "items": []},
    )
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("AAPL")

    titles = [item["title"] for item in result["related_news"]]
    assert "Apple expands AI tooling for developers" in titles
    assert "Netflix initiated, Instacart upgraded: Wall Street's top analyst calls" not in titles


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
    assert list(result).index("general_news") < list(result).index("related_news")
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


def test_classify_instrument_handles_broker_suffixes(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)

    nas = svc._classify_instrument("NAS100.cash")
    spx = svc._classify_instrument("US500.cash")
    gold = svc._classify_instrument("XAUUSD.a")

    assert nas.asset_class == "index"
    assert nas.base_asset == "NAS"
    assert spx.asset_class == "index"
    assert spx.base_asset == "SPX"
    assert gold.asset_class == "commodity"
    assert gold.base_asset == "XAU"
    assert gold.quote_asset == "USD"


def test_classify_instrument_handles_generic_crypto_quote_splits(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)

    aave = svc._classify_instrument("AAVEUSDT")
    link = svc._classify_instrument("LINKUSD")

    assert aave.asset_class == "crypto"
    assert aave.base_asset == "AAVE"
    assert aave.quote_asset == "USDT"
    assert link.asset_class == "crypto"
    assert link.base_asset == "LINK"
    assert link.quote_asset == "USD"


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


def test_classify_instrument_handles_common_broker_commodity_aliases(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)

    usoil = svc._classify_instrument("USOIL")
    gold = svc._classify_instrument("GOLD")

    assert usoil.asset_class == "commodity"
    assert usoil.base_asset == "WTI"
    assert gold.asset_class == "commodity"
    assert gold.base_asset == "XAU"


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


def test_classify_instrument_supports_common_index_aliases(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)

    ftse = svc._classify_instrument("FTSE100")
    nikkei = svc._classify_instrument("NIKKEI225")
    jpn = svc._classify_instrument("JPN225")

    assert ftse.asset_class == "index"
    assert ftse.base_asset == "FTSE"
    assert nikkei.asset_class == "index"
    assert nikkei.base_asset == "NIKKEI"
    assert jpn.asset_class == "index"
    assert jpn.base_asset == "NIKKEI"


def test_parse_relative_time_handles_yesterday() -> None:
    parsed = svc._parse_relative_time("yesterday")

    assert parsed is not None
    assert datetime.now(timezone.utc) - timedelta(days=1, minutes=1) <= parsed <= datetime.now(timezone.utc)


def test_parse_relative_time_rejects_negative_values() -> None:
    assert svc._parse_relative_time("-5 minutes ago") is None


def test_parse_relative_time_rejects_overflowing_values() -> None:
    assert svc._parse_relative_time("999999 days ago") is None


def test_maybe_parse_datetime_rejects_numeric_values() -> None:
    assert svc._maybe_parse_datetime(42) is None
    assert svc._maybe_parse_datetime("42") is None


def test_unknown_equity_classification_does_not_infer_description(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)

    context = svc._classify_instrument("ZZZZNOTREAL")

    assert context.asset_class == "equity"
    assert context.description is None


def test_equity_symbol_hints_help_without_mt5_metadata(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(
        svc,
        "get_stock_news",
        lambda symbol, limit=20, page=1: {
            "success": True,
            "news": [
                {"Title": "Apple expands AI tooling for developers", "Source": "Reuters", "Date": "2026-03-29T09:00:00Z"},
            ],
        },
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {"success": True, "items": []},
    )
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("AAPL")

    assert any("Apple expands AI tooling" in item["title"] for item in result["related_news"])


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
                    "date": "2026-03-29T12:30:00Z",
                    "event": "House Price Index",
                    "ticker": "USD",
                    "importance": 2,
                    "category": "Housing",
                    "reference": "FHFA",
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
            "pairs": [
                {
                    "Pair": "EUR/USD",
                    "Price": 1.1717,
                    "Perf Day": 0.0029,
                    "Perf Week": -0.0039000000000000003,
                }
            ],
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
    assert "Perf Week: -0.0039" in (forex_result["related_news"][0]["summary"] or "")
    assert "-0.0039000000000000003" not in (forex_result["related_news"][0]["summary"] or "")
    assert futures_result["related_news"][0]["title"] == "Nasdaq 100 E-mini market snapshot"
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


def test_market_snapshots_require_meaningful_relevance(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(
        svc,
        "get_futures_performance",
        lambda: {
            "success": True,
            "futures": [
                {"Ticker": "NQ", "Label": "Nasdaq 100 E-mini", "Perf": "0.8%"},
                {"Ticker": "DY", "Label": "US Dollar Index", "Perf": "0.2%"},
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

    titles = [item["title"] for item in result["related_news"]]
    assert any(title == "Nasdaq 100 E-mini market snapshot" for title in titles)
    assert all(title != "DY market snapshot" for title in titles)


def test_crypto_snapshots_require_exact_symbol_family_match(monkeypatch) -> None:
    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(
        svc,
        "get_crypto_performance",
        lambda: {
            "success": True,
            "coins": [
                {"Ticker": "BTC", "Label": "Bitcoin", "Perf": "1.2%"},
                {"Ticker": "BCH", "Label": "Bitcoin Cash", "Perf": "0.9%"},
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

    result = svc.fetch_unified_news("BTCUSD")

    titles = [item["title"] for item in result["related_news"]]
    assert any(title == "BTC market snapshot" for title in titles)
    assert all(title != "BCH market snapshot" for title in titles)


def test_short_symbol_aliases_require_real_token_matches(monkeypatch) -> None:
    monkeypatch.setattr(
        svc,
        "get_general_news",
        lambda news_type="news", limit=20, page=1: {
            "success": True,
            "items": [
                {
                    "Title": "Pete Hegseth's broker attempted to make defense investments before Iran war",
                    "Source": "CNBC",
                    "Date": "2026-03-29T08:00:00Z",
                },
                {
                    "Title": "ETH jumps after ETF rumor",
                    "Source": "Reuters",
                    "Date": "2026-03-29T09:00:00Z",
                },
                {
                    "Title": "EUR/USD Price Forecast: Consolidates above one-week low",
                    "Source": "FXStreet",
                    "Date": "2026-03-29T10:00:00Z",
                },
                {
                    "Title": "SOL rallies on network activity",
                    "Source": "CoinDesk",
                    "Date": "2026-03-29T11:00:00Z",
                },
            ],
        },
    )
    monkeypatch.setattr(svc, "get_crypto_performance", lambda: {"success": True, "coins": []})
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

    eth_result = svc.fetch_unified_news("ETHUSDT")
    sol_result = svc.fetch_unified_news("SOLUSDT")

    eth_titles = [item["title"] for item in eth_result["related_news"]]
    sol_titles = [item["title"] for item in sol_result["related_news"]]
    assert "ETH jumps after ETF rumor" in eth_titles
    assert all("Hegseth" not in title for title in eth_titles)
    assert "SOL rallies on network activity" in sol_titles
    assert all("Consolidates above one-week low" not in title for title in sol_titles)


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
                    "date": "2026-03-29T12:30:00Z",
                    "event": "House Price Index",
                    "ticker": "USD",
                    "importance": 2,
                    "category": "Housing",
                    "reference": "FHFA",
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


def test_index_related_bucket_ignores_generic_general_headlines(monkeypatch) -> None:
    monkeypatch.setattr(
        svc,
        "get_general_news",
        lambda news_type="news", limit=20, page=1: {
            "success": True,
            "items": [
                {
                    "Title": "AI leaders rally after upbeat semiconductor outlook",
                    "Source": "Reuters",
                    "Date": "2026-03-29T08:00:00Z",
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
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("NAS100")

    assert result["related_news"] == []


def test_crypto_related_bucket_rejects_generic_usd_headlines(monkeypatch) -> None:
    monkeypatch.setattr(
        svc,
        "get_general_news",
        lambda news_type="news", limit=20, page=1: {
            "success": True,
            "items": [
                {
                    "Title": "US Dollar Index trades broadly firm above 100.00 amid war fears",
                    "Source": "Reuters",
                    "Date": "2026-03-29T08:00:00Z",
                }
            ],
        },
    )
    monkeypatch.setattr(svc, "get_crypto_performance", lambda: {"success": True, "coins": []})
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

    result = svc.fetch_unified_news("BTCUSD")

    assert result["related_news"] == []


def test_crypto_related_bucket_can_promote_crypto_market_headlines(monkeypatch) -> None:
    monkeypatch.setattr(
        svc,
        "get_general_news",
        lambda news_type="news", limit=20, page=1: {
            "success": True,
            "items": [
                {
                    "Title": "Crypto market cap climbs as digital asset momentum returns",
                    "Source": "Reuters",
                    "Date": "2026-03-29T08:00:00Z",
                },
                {
                    "Title": "US stocks rally on improving consumer confidence",
                    "Source": "Reuters",
                    "Date": "2026-03-29T09:00:00Z",
                },
            ],
        },
    )
    monkeypatch.setattr(svc, "get_crypto_performance", lambda: {"success": True, "coins": []})
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

    result = svc.fetch_unified_news("BTCUSD")

    titles = [item["title"] for item in result["related_news"]]
    assert "Crypto market cap climbs as digital asset momentum returns" in titles
    assert "US stocks rally on improving consumer confidence" not in titles


def test_forex_related_bucket_accepts_currency_side_headlines(monkeypatch) -> None:
    monkeypatch.setattr(
        svc,
        "get_general_news",
        lambda news_type="news", limit=20, page=1: {
            "success": True,
            "items": [
                {
                    "Title": "Japanese Yen outperforms as BoJ's Ueda signals readiness to intervene",
                    "Source": "Reuters",
                    "Date": "2026-03-29T08:00:00Z",
                },
                {
                    "Title": "Fed officials stress patience on interest rates",
                    "Source": "Reuters",
                    "Date": "2026-03-29T09:00:00Z",
                },
            ],
        },
    )
    monkeypatch.setattr(svc, "get_forex_performance", lambda: {"success": True, "pairs": []})
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

    result = svc.fetch_unified_news("USDJPY")

    titles = [item["title"] for item in result["related_news"]]
    assert "Japanese Yen outperforms as BoJ's Ueda signals readiness to intervene" in titles
    assert "Fed officials stress patience on interest rates" not in titles


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
    future_time = (datetime.now(timezone.utc) + timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(svc, "get_futures_performance", lambda: {"success": True, "futures": []})
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {
            "success": True,
            "items": [
                {
                    "date": future_time,
                    "event": "German CPI",
                    "ticker": "EUR",
                    "importance": 3,
                    "category": "Inflation",
                    "reference": "Destatis",
                }
            ],
        },
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("GER40")

    assert any("German CPI" in item["title"] for item in result["upcoming_events"])


def test_fetch_unified_news_surfaces_future_currency_events_in_dedicated_bucket(monkeypatch) -> None:
    future_time = (datetime.now(timezone.utc) + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")

    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(svc, "get_forex_performance", lambda: {"success": True, "pairs": []})
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {
            "success": True,
            "items": [
                {
                    "date": future_time,
                    "event": "ADP Employment Change Weekly",
                    "ticker": "USAAECW",
                    "importance": 2,
                    "category": "Labor",
                    "reference": "Mar",
                },
                {
                    "date": future_time,
                    "event": "Machine Orders",
                    "ticker": "JAPANMACHINEORDERS",
                    "importance": 2,
                    "category": "Manufacturing",
                    "reference": "Feb",
                },
                {
                    "date": future_time,
                    "event": "German Factory Orders",
                    "ticker": "GERMANYFACTORYORDERS",
                    "importance": 3,
                    "category": "Manufacturing",
                    "reference": "Feb",
                },
            ],
        },
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("USDJPY")

    upcoming_titles = [item["title"] for item in result["upcoming_events"]]
    assert set(upcoming_titles) == {
        "ADP Employment Change Weekly (USD)",
        "Machine Orders (JPY)",
    }
    assert result["upcoming_count"] == 2
    assert result["related_news"] == []
    assert all(item["kind"] == "economic_event" for item in result["upcoming_events"])
    assert {item["metadata"]["event_for"] for item in result["upcoming_events"]} == {"USD", "JPY"}


def test_fetch_unified_news_keeps_future_macro_events_out_of_related_bucket(monkeypatch) -> None:
    future_time = (datetime.now(timezone.utc) + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")

    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(svc, "get_forex_performance", lambda: {"success": True, "pairs": []})
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {
            "success": True,
            "items": [
                {
                    "date": future_time,
                    "event": "CPI",
                    "ticker": "USD",
                    "importance": 3,
                    "category": "Consumer Price Index CPI",
                    "reference": "Mar",
                }
            ],
        },
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("USDJPY")

    assert [item["title"] for item in result["upcoming_events"]] == ["CPI (USD)"]
    assert result["related_news"] == []


def test_fetch_unified_news_caps_upcoming_events_bucket_at_twenty(monkeypatch) -> None:
    future_base = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(svc, "get_forex_performance", lambda: {"success": True, "pairs": []})
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {
            "success": True,
            "items": [
                {
                    "date": (future_base + timedelta(hours=idx)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "event": f"CPI {idx}",
                    "ticker": "USD",
                    "importance": 3,
                    "category": "Consumer Price Index CPI",
                    "reference": "Mar",
                }
                for idx in range(25)
            ],
        },
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("USDJPY")

    assert result["upcoming_count"] == 20
    assert len(result["upcoming_events"]) == 20
    assert [item["title"] for item in result["upcoming_events"][:2]] == ["CPI 0 (USD)", "CPI 1 (USD)"]
    assert result["related_news"] == []


def test_fetch_unified_news_surfaces_recent_events_bucket_with_latest_five(monkeypatch) -> None:
    recent_base = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)

    monkeypatch.setattr(svc, "get_general_news", lambda news_type="news", limit=20, page=1: {"success": True, "items": []})
    monkeypatch.setattr(svc, "get_forex_performance", lambda: {"success": True, "pairs": []})
    monkeypatch.setattr(
        svc,
        "get_economic_calendar",
        lambda limit=100, page=1, impact=None, date_from=None, date_to=None: {
            "success": True,
            "items": [
                {
                    "date": (recent_base - timedelta(hours=idx)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "event": f"CPI {idx}",
                    "ticker": "USD",
                    "importance": 3,
                    "actual": f"{idx}.0%",
                    "forecast": f"{idx}.1%",
                    "previous": f"{idx}.2%",
                    "category": "Consumer Price Index CPI",
                    "reference": "Mar",
                }
                for idx in range(7)
            ],
        },
    )
    monkeypatch.setattr(svc, "get_mt5_news", lambda **_kwargs: {"success": True, "news": []})
    monkeypatch.setattr(svc, "get_symbol_info_cached", lambda symbol: None)
    _disable_ycnbc(monkeypatch)
    _disable_embeddings(monkeypatch)
    _reset_aggregator(monkeypatch)

    result = svc.fetch_unified_news("USDJPY")

    assert result["recent_count"] == 5
    assert [item["title"] for item in result["recent_events"]] == [
        "CPI 0 (USD)",
        "CPI 1 (USD)",
        "CPI 2 (USD)",
        "CPI 3 (USD)",
        "CPI 4 (USD)",
    ]
    assert all("Actual:" in (item["summary"] or "") for item in result["recent_events"])
    assert result["related_news"] == []


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


def test_ycnbc_general_candidates_are_cached(monkeypatch) -> None:
    call_counts = {"latest": 0}

    class FakeNews:
        def latest(self):
            call_counts["latest"] += 1
            return [{"headline": "CNBC latest", "time": "5 min ago", "link": "https://example.com/cnbc"}]

        def __getattr__(self, _name):
            return lambda: []

    class FakeStocksUtil:
        def news(self, symbol: str):
            return []

    monkeypatch.setattr(svc, "_import_ycnbc", lambda: (FakeNews, FakeStocksUtil))

    source = svc.YCNBCNewsSource()
    first = source.fetch_general_candidates(5)
    second = source.fetch_general_candidates(5)

    assert len(first) == 1
    assert len(second) == 1
    assert call_counts["latest"] == 1


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
