from __future__ import annotations

from datetime import datetime, timedelta, timezone
from inspect import signature

from mtdata.core.news import news, normalize_news_output
from mtdata.core.output_contract import apply_output_verbosity


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _prepare_news_output(payload, *, detail: str):
    return apply_output_verbosity(
        normalize_news_output(payload, detail=detail),
        detail=detail,
        tool_name="news",
    )


def test_news_tool_has_only_optional_symbol_parameter() -> None:
    raw = _unwrap(news)
    params = list(signature(raw).parameters.values())

    assert [param.name for param in params] == ["symbol", "detail", "limit"]
    assert params[0].default is None
    assert params[1].default == "compact"
    assert params[2].default is None


def test_news_tool_forwards_symbol(monkeypatch) -> None:
    raw = _unwrap(news)

    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda symbol=None: {
            "success": True,
            "symbol": symbol,
            "general_news": [],
            "related_news": [],
        },
    )

    result = raw(symbol="EURUSD")

    assert result["success"] is True
    assert result["symbol"] == "EURUSD"


def test_news_tool_limits_each_bucket_without_changing_default(monkeypatch) -> None:
    raw = _unwrap(news)

    payload = {
        "success": True,
        "general_news": [{"title": "g1"}, {"title": "g2"}],
        "related_news": [{"title": "r1"}, {"title": "r2"}],
        "impact_news": [{"title": "i1"}, {"title": "i2"}],
        "upcoming_events": [{"title": "u1"}, {"title": "u2"}],
        "recent_events": [{"title": "e1"}, {"title": "e2"}],
    }
    monkeypatch.setattr("mtdata.core.news.fetch_unified_news", lambda symbol=None: payload)

    unlimited = raw()
    limited = raw(limit=1)

    assert len(unlimited["general_news"]) == 2
    assert limited["general_news"] == [{"title": "g1"}]
    assert limited["related_news"] == [{"title": "r1"}]
    assert limited["impact_news"] == [{"title": "i1"}]
    assert limited["upcoming_events"] == [{"title": "u1"}]
    assert limited["recent_events"] == [{"title": "e1"}]


def test_news_tool_rejects_invalid_limit() -> None:
    raw = _unwrap(news)

    assert raw(limit=0)["error"] == "limit must be a positive integer."


def test_news_tool_compact_and_full_detail_contract(monkeypatch) -> None:
    raw = _unwrap(news)

    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda symbol=None: {
            "success": True,
            "symbol": symbol,
            "instrument": {"symbol": symbol},
            "matching": {"embeddings": {"enabled": True}},
            "general_news": [
                {
                    "title": "Fed preview",
                    "provider": "finviz",
                    "published_at": "2026-03-29T08:00:00Z",
                    "metadata": {"relative_time": "9 days ago"},
                }
            ],
            "related_news": [],
            "impact_news": [],
        },
    )

    compact = raw(symbol="EURUSD", detail="compact")
    full = raw(symbol="EURUSD", detail="full")

    assert "instrument" not in compact
    assert "matching" not in compact
    assert "tool_scope" not in compact
    assert "timezone" not in compact
    assert compact["general_news"] == [
        {
            "title": "Fed preview",
            "relative_time": "9 days ago",
        }
    ]

    assert full["instrument"] == {"symbol": "EURUSD"}
    assert full["matching"] == {"embeddings": {"enabled": True}}
    assert full["tool_scope"] == "unified_trading_news"
    assert full["timezone"] == "UTC"
    assert full["general_news"][0]["provider"] == "finviz"


def test_news_output_hides_debug_fields_when_not_verbose() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "instrument": {"symbol": "EURUSD", "aliases": ["EURUSD", "EUR/USD"]},
        "sources_used": ["finviz", "mt5"],
        "source_details": {"finviz": {"selected_total": 1}},
        "matching": {"embeddings": {"enabled": True}},
        "general_count": 1,
        "related_count": 1,
        "impact_count": 0,
        "general_news": [
            {
                "title": "Fed preview",
                "provider": "finviz",
                "source": "Reuters",
                "kind": "headline",
                "published_at": "2026-03-29T08:00:00Z",
                "url": "https://example.com/fed-preview",
                "summary": None,
                "category": "market_news",
                "priority": "MEDIUM",
                "relevance_score": 0.4,
                "importance_score": 5.2,
                "metadata": {"source_rank": 0, "relative_time": "9 days ago"},
            }
        ],
        "related_news": [
            {
                "title": "EUR/USD market snapshot",
                "provider": "finviz",
                "source": "Finviz Forex",
                "kind": "market_snapshot",
                "published_at": "2026-03-29T08:05:00Z",
                "url": None,
                "summary": "Price: 1.1541",
                "category": "forex",
                "priority": "HIGH",
                "relevance_score": 8.3,
                "importance_score": 4.8,
                "metadata": {"ticker": "EUR/USD", "relative_time": "9 days ago"},
            }
        ],
        "impact_news": [],
    }

    result = _prepare_news_output(payload, detail="compact")

    assert "instrument" not in result
    assert "sources_used" not in result
    assert "source_details" not in result
    assert "matching" not in result
    assert "general_count" not in result
    assert "related_count" not in result
    assert "impact_count" not in result
    assert result["general_news"] == [
        {
            "title": "Fed preview",
            "source": "Reuters",
            "kind": "headline",
            "relative_time": "9 days ago",
        }
    ]
    assert "url" not in result["general_news"][0]
    assert result["related_news"] == [
        {
            "title": "EUR/USD market snapshot",
            "source": "Finviz Forex",
            "kind": "market_snapshot",
            "relative_time": "9 days ago",
            "summary": "Price: 1.1541",
        }
    ]
    assert "url" not in result["related_news"][0]
    assert "category" not in result["general_news"][0]


def test_news_output_keeps_debug_fields_when_verbose() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "instrument": {"symbol": "EURUSD"},
        "matching": {"embeddings": {"enabled": True}},
        "general_news": [
            {
                "title": "Fed preview",
                "provider": "finviz",
                "source": "Reuters",
                "kind": "headline",
                "published_at": "2026-03-29T08:00:00Z",
                "category": "market_news",
                "priority": "MEDIUM",
                "relevance_score": 0.4,
                "importance_score": 5.2,
                "metadata": {"source_rank": 0},
            }
        ],
        "related_news": [],
        "impact_news": [],
    }

    result = _prepare_news_output(payload, detail="full")

    assert "instrument" in result
    assert "matching" in result
    assert result["general_news"][0]["provider"] == "finviz"
    assert result["general_news"][0]["metadata"]["source_rank"] == 0


def test_news_output_derives_relative_time_from_published_at_when_needed() -> None:
    published_at = (datetime.now(timezone.utc) - timedelta(hours=2, minutes=10)).isoformat()
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "general_news": [
            {
                "title": "Fed preview",
                "provider": "finviz",
                "source": "Reuters",
                "kind": "headline",
                "published_at": published_at,
                "category": "market_news",
                "priority": "MEDIUM",
                "relevance_score": 0.4,
                "importance_score": 5.2,
                "metadata": {"source_rank": 0},
            }
        ],
        "related_news": [],
        "impact_news": [],
    }

    result = _prepare_news_output(payload, detail="compact")

    item = result["general_news"][0]
    assert item["title"] == "Fed preview"
    assert item["source"] == "Reuters"
    assert item["kind"] == "headline"
    assert item["relative_time"].endswith("ago")
    assert "published_at" not in item


def test_news_output_uses_relative_time_for_future_events() -> None:
    published_at = datetime.now(timezone.utc).replace(second=0, microsecond=0) + timedelta(hours=3, minutes=15)
    payload = {
        "success": True,
        "symbol": "USDJPY",
        "general_news": [],
        "related_news": [
            {
                "title": "US CPI (USD)",
                "provider": "finviz",
                "source": "Finviz Economic Calendar",
                "kind": "economic_event",
                "published_at": published_at.isoformat(),
                "summary": "Expected: 3.2% | Prior: 3.1%",
                "category": "economic_calendar",
                "priority": "HIGH",
                "relevance_score": 9.1,
                "importance_score": 6.3,
                "metadata": {"impact": "high"},
            }
        ],
        "impact_news": [],
    }

    result = _prepare_news_output(payload, detail="compact")

    item = result["related_news"][0]
    assert item["title"] == "US CPI (USD)"
    assert item["relative_time"] == "in 3 hours"
    assert "time_utc" not in item
    assert "published_at" not in item
    assert item["source"] == "Finviz Economic Calendar"
    assert item["kind"] == "economic_event"


def test_news_output_compaction_is_idempotent() -> None:
    payload = {
        "success": True,
        "symbol": "USDJPY",
        "general_news": [{"title": "Fed preview", "relative_time": "2 hours ago"}],
        "related_news": [
            {
                "title": "US CPI (USD)",
                "time_utc": "2026-04-07 12:30 UTC",
                "kind": "economic_event",
                "summary": "Expected: 3.2% | Prior: 3.1%",
            }
        ],
        "impact_news": [{"title": "Oil jumps on war fears", "relative_time": "6 hours ago"}],
    }

    result = _prepare_news_output(payload, detail="compact")

    assert result == payload


def test_news_output_compacts_upcoming_events_bucket() -> None:
    published_at = datetime.now(timezone.utc).replace(second=0, microsecond=0) + timedelta(hours=2)
    payload = {
        "success": True,
        "symbol": "USDJPY",
        "general_news": [],
        "related_news": [],
        "impact_news": [],
        "upcoming_events": [
            {
                "title": "US CPI (USD)",
                "provider": "finviz",
                "source": "Finviz Economic Calendar",
                "kind": "economic_event",
                "published_at": published_at.isoformat(),
                "summary": "Expected: 3.2% | Prior: 3.1%",
                "category": "economic_calendar",
                "priority": "HIGH",
                "relevance_score": 9.1,
                "importance_score": 6.3,
                "metadata": {"event_for": "USD", "impact": "high"},
            }
        ],
        "upcoming_count": 1,
    }

    result = _prepare_news_output(payload, detail="compact")

    assert "upcoming_count" not in result
    item = result["upcoming_events"][0]
    assert item["title"] == "US CPI (USD)"
    assert item["source"] == "Finviz Economic Calendar"
    assert item["kind"] == "economic_event"
    assert "published_at" not in item
    assert item["relative_time"].startswith("in ")
    assert "time_utc" not in item
    assert item["summary"] == "Expected: 3.2% | Prior: 3.1%"


def test_news_output_compacts_recent_events_bucket() -> None:
    published_at = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(hours=2)
    payload = {
        "success": True,
        "symbol": "USDJPY",
        "general_news": [],
        "related_news": [],
        "impact_news": [],
        "upcoming_events": [],
        "recent_events": [
            {
                "title": "US CPI (USD)",
                "provider": "finviz",
                "source": "Finviz Economic Calendar",
                "kind": "economic_event",
                "published_at": published_at.isoformat(),
                "summary": "Actual: 3.2% | Expected: 3.1% | Prior: 3.0%",
                "category": "economic_calendar",
                "priority": "HIGH",
                "relevance_score": 9.1,
                "importance_score": 6.3,
                "metadata": {"event_for": "USD", "impact": "high"},
            }
        ],
        "recent_count": 1,
    }

    result = _prepare_news_output(payload, detail="compact")

    assert "recent_count" not in result
    assert result["recent_events"] == [
        {
            "title": "US CPI (USD)",
            "source": "Finviz Economic Calendar",
            "kind": "economic_event",
            "relative_time": "2 hours ago",
            "summary": "Actual: 3.2% | Expected: 3.1% | Prior: 3.0%",
        }
    ]


def test_generic_output_contract_no_longer_special_cases_news() -> None:
    payload = {
        "success": True,
        "source_details": {"finviz": {"selected_total": 1}},
        "general_news": [
            {
                "title": "Fed preview",
                "provider": "finviz",
                "published_at": "2026-03-29T08:00:00Z",
            }
        ],
    }

    result = apply_output_verbosity(payload, detail="compact", tool_name="news")

    assert "source_details" in result
    assert result["general_news"][0]["provider"] == "finviz"
