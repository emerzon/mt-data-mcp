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

    assert [param.name for param in params] == ["symbol", "detail", "limit", "offset", "limit_per_bucket"]
    assert params[0].default is None
    assert params[1].default == "compact"
    assert params[2].default is None
    assert params[3].default == 0
    assert params[4].default is None


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
    assert result["data_fetched_at"].endswith("Z")
    assert "T" in result["data_fetched_at"]


def test_news_tool_preserves_symbol_validation_error(monkeypatch) -> None:
    raw = _unwrap(news)
    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda symbol=None: {
            "success": False,
            "error": f"Symbol '{symbol}' was not found by the equity news provider.",
            "error_code": "news_symbol_unavailable",
            "symbol": symbol,
            "remediation": "Check the ticker spelling.",
        },
    )

    result = raw(symbol="ZZZZZ", limit=2)

    assert result["success"] is False
    assert result["error_code"] == "news_symbol_unavailable"
    assert result["symbol"] == "ZZZZZ"
    assert "limit_scope" not in result


def test_news_tool_limits_globally_without_changing_default(monkeypatch) -> None:
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
    limited = raw(limit=3)

    assert len(unlimited["general_news"]) == 2
    assert limited["related_news"] == [{"title": "r1"}, {"title": "r2"}]
    assert limited["general_news"] == [{"title": "g1"}]
    assert limited["row_keys"] == ["related_news", "general_news"]
    assert "impact_news" not in limited
    assert "upcoming_events" not in limited
    assert "recent_events" not in limited
    assert limited["total_candidates"] == 10
    assert limited["returned"] == 3
    assert limited["limit_scope"] == "global"
    assert limited["truncated"] is True
    assert limited["has_more"] is True


def test_news_tool_symbol_limit_caps_related_bucket_only(monkeypatch) -> None:
    raw = _unwrap(news)

    payload = {
        "success": True,
        "symbol": "EURUSD",
        "general_news": [{"title": "g1"}, {"title": "g2"}],
        "related_news": [{"title": "r1"}, {"title": "r2"}, {"title": "r3"}],
        "impact_news": [{"title": "i1"}],
        "upcoming_events": [{"title": "u1"}],
        "recent_events": [{"title": "e1"}],
    }
    monkeypatch.setattr("mtdata.core.news.fetch_unified_news", lambda symbol=None: payload)

    limited = raw(symbol="EURUSD", limit=2)

    assert limited["related_news"] == [{"title": "r1"}, {"title": "r2"}]
    assert limited["row_keys"] == ["related_news"]
    assert "general_news" not in limited
    assert "impact_news" not in limited
    assert "upcoming_events" not in limited
    assert "recent_events" not in limited
    assert limited["total_candidates"] == 3
    assert limited["returned"] == 2
    assert limited["limit_scope"] == "symbol"
    assert limited["bucket_truncation"] == {"related_news": True}
    assert limited["truncated"] is True
    assert limited["has_more"] is True


def test_compact_symbol_news_caps_each_bucket_by_default(monkeypatch) -> None:
    raw = _unwrap(news)
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "general_news": [{"title": f"g{i}"} for i in range(6)],
        "related_news": [],
        "impact_news": [{"title": f"i{i}"} for i in range(6)],
        "upcoming_events": [{"title": f"u{i}"} for i in range(6)],
        "recent_events": [{"title": f"e{i}"} for i in range(6)],
        "symbol_news_note": "No EURUSD-specific related news passed relevance gates.",
    }
    monkeypatch.setattr("mtdata.core.news.fetch_unified_news", lambda symbol=None: payload)

    compact = raw(symbol="EURUSD")
    full = raw(symbol="EURUSD", detail="full")

    for key in ("general_news", "impact_news", "upcoming_events", "recent_events"):
        assert len(compact[key]) == 5
        assert len(full[key]) == 6
    assert compact["compact_bucket_limit"] == 5
    assert compact["truncated"] is True
    assert compact["symbol_news_note"] == payload["symbol_news_note"]
    assert "compact_bucket_limit" not in full


def test_news_tool_fx_symbol_limit_does_not_fill_from_general_buckets(monkeypatch) -> None:
    raw = _unwrap(news)

    payload = {
        "success": True,
        "symbol": "EURUSD",
        "instrument": {"symbol": "EURUSD", "asset_class": "forex"},
        "general_news": [{"title": "g1"}, {"title": "g2"}],
        "related_news": [{"title": "r1"}],
        "impact_news": [{"title": "i1"}],
        "upcoming_events": [{"title": "u1"}],
        "recent_events": [{"title": "e1"}],
        "market_context": [{"title": "m1"}],
    }
    monkeypatch.setattr("mtdata.core.news.fetch_unified_news", lambda symbol=None: payload)

    limited = raw(symbol="EURUSD", limit=3)

    assert limited["related_news"] == [{"title": "r1"}]
    assert limited["row_keys"] == ["related_news"]
    assert "general_news" not in limited
    assert "impact_news" not in limited
    assert "upcoming_events" not in limited
    assert "recent_events" not in limited
    assert "market_context" not in limited
    assert limited["total_candidates"] == 1
    assert limited["returned"] == 1
    assert limited["limit_scope"] == "symbol"
    assert "macro_fallback" not in limited
    assert limited["has_more"] is False


def test_news_tool_supports_global_offset(monkeypatch) -> None:
    raw = _unwrap(news)

    payload = {
        "success": True,
        "general_news": [{"title": "g1"}, {"title": "g2"}],
        "related_news": [{"title": "r1"}, {"title": "r2"}],
        "impact_news": [{"title": "i1"}, {"title": "i2"}],
    }
    monkeypatch.setattr("mtdata.core.news.fetch_unified_news", lambda symbol=None: payload)

    page = raw(limit=2, offset=2)

    assert page["general_news"] == [{"title": "g1"}, {"title": "g2"}]
    assert page["row_keys"] == ["general_news"]
    assert "related_news" not in page
    assert "impact_news" not in page
    assert page["total_candidates"] == 6
    assert page["returned"] == 2
    assert page["offset"] == 2
    assert page["has_more"] is True
    assert page["limit_scope"] == "global"


def test_news_tool_keeps_per_bucket_limit_mode(monkeypatch) -> None:
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

    limited = raw(limit_per_bucket=1)

    assert limited["general_news"] == [{"title": "g1"}]
    assert limited["related_news"] == [{"title": "r1"}]
    assert limited["impact_news"] == [{"title": "i1"}]
    assert limited["upcoming_events"] == [{"title": "u1"}]
    assert limited["recent_events"] == [{"title": "e1"}]
    assert limited["total_candidates"] == 10
    assert limited["returned"] == 5
    assert limited["limit_scope"] == "per_bucket"
    assert limited["truncated"] is True


def test_news_tool_rejects_invalid_limit() -> None:
    raw = _unwrap(news)

    assert raw(limit=0)["error"] == "limit must be a positive integer."
    assert raw(offset=-1)["error"] == "offset must be >= 0."
    assert raw(limit_per_bucket=0)["error"] == "limit_per_bucket must be a positive integer."


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
            "published_at": "2026-03-29T08:00:00Z",
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
        "related_count": 0,
        "market_context_count": 1,
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
        "related_news": [],
        "market_context": [
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
    assert "market_context_count" not in result
    assert "impact_count" not in result
    assert result["general_news"] == [
        {
            "title": "Fed preview",
            "source": "Reuters",
            "kind": "headline",
            "published_at": "2026-03-29T08:00:00Z",
            "relative_time": "9 days ago",
        }
    ]
    assert "url" not in result["general_news"][0]
    assert "related_news" not in result
    assert "market_context" not in result
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
    assert item["published_at"] == published_at.replace("+00:00", "Z")


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
                "published_at": published_at.isoformat().replace("+00:00", "Z"),
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
    assert item["published_at"] == published_at.isoformat().replace("+00:00", "Z")
    assert item["source"] == "Finviz Economic Calendar"
    assert item["kind"] == "economic_event"
    assert item["relevance_score"] == 9.1
    assert item["match_reason"] == {"basis": "symbol_relevance_gate"}


def test_news_compact_related_items_explain_term_matches() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "related_news": [
            {
                "title": "ECB outlook shifts",
                "source": "Reuters",
                "kind": "headline",
                "relevance_score": 2.4,
                "metadata": {"matched_terms": ["EUR", "ECB"]},
            }
        ],
    }

    result = _prepare_news_output(payload, detail="compact")

    assert result["related_news"][0]["relevance_score"] == 2.4
    assert result["related_news"][0]["match_reason"] == {
        "basis": "matched_terms",
        "terms": ["EUR", "ECB"],
    }


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
    assert item["published_at"] == published_at.isoformat().replace("+00:00", "Z")
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
            "published_at": published_at.isoformat().replace("+00:00", "Z"),
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
