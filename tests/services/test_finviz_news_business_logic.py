from __future__ import annotations

import logging
from unittest.mock import patch

from mtdata.core import finviz as core_finviz
from mtdata.core.finviz import finviz_market_news, finviz_news
from mtdata.services.finviz import get_stock_news


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_finviz_news_rejects_pair_style_symbol_before_service_call() -> None:
    raw = _unwrap(finviz_news)
    with patch("mtdata.core.finviz.get_stock_news") as mock_news:
        out = raw(symbol="BTCUSD", limit=5, page=1)

    assert "error" in out
    assert "not a Finviz-supported equity ticker" in out["error"]
    mock_news.assert_not_called()


def test_get_stock_news_returns_clean_message_for_404_like_errors() -> None:
    class Boom:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("404 Client Error: Not Found for url: https://finviz.com/quote.ashx?t=BTCUSD")

    with patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch", lambda: None), patch.dict(
        "sys.modules",
        {"finvizfinance.quote": type("Q", (), {"finvizfinance": Boom})},
    ):
        out = get_stock_news("BTCUSD", limit=5, page=1)

    assert out["error"] == "BTCUSD is not a Finviz-supported symbol. finviz_news only covers US equities."


def test_finviz_news_logs_finish_event_for_success(caplog) -> None:
    raw = _unwrap(finviz_news)

    with patch("mtdata.core.finviz.get_stock_news", return_value={"success": True, "items": []}), caplog.at_level(logging.DEBUG,
        logger=core_finviz.logger.name,
    ):
        out = raw(symbol="AAPL", limit=5, page=1)

    assert out["success"] is True
    assert any(
        "event=finish" in record.message and "operation=finviz_news" in record.message
        for record in caplog.records
    )


def test_finviz_news_normalizes_stock_results_to_single_items_array() -> None:
    raw = _unwrap(finviz_news)

    service_result = {
        "success": True,
        "symbol": "AAPL",
        "count": 1,
        "total": 1,
        "page": 1,
        "pages": 1,
        "news": [
            {
                "Title": "  Apple launches new chips  ",
                "Source": " Reuters ",
                "Date": " 2026-04-18 ",
                "Link": " https://example.test/apple ",
            }
        ],
    }

    with patch("mtdata.core.finviz.get_stock_news", return_value=service_result):
        out = raw(symbol="AAPL", limit=5, page=1)

    assert out["items"] == [
        {
            "title": "Apple launches new chips",
            "source": "Reuters",
            "published_at": "2026-04-18T00:00:00+00:00",
            "url": "https://example.test/apple",
        }
    ]
    assert out["tool_scope"] == "raw_finviz_provider"
    assert out["preferred_tool"] == "news"
    assert out["output_shape"] == "flat_paginated_items"
    assert "news" not in out


def test_finviz_news_without_symbol_normalizes_general_items() -> None:
    raw = _unwrap(finviz_news)
    service_result = {
        "success": True,
        "type": "news",
        "count": 1,
        "items": [
            {
                "Title": " Market wrap ",
                "Source": "Finviz",
                "Date": "01:30PM",
                "Link": "https://example.test/news",
            }
        ],
    }

    with patch("mtdata.core.finviz.get_general_news", return_value=service_result):
        out = raw(symbol=None, limit=5, page=1)

    assert out["items"][0]["title"] == "Market wrap"
    assert out["items"][0]["source"] == "Finviz"
    assert "T13:30:00+00:00" in out["items"][0]["published_at"]
    assert out["items"][0]["url"] == "https://example.test/news"
    assert out["preferred_tool"] == "news"


def test_finviz_market_news_normalizes_items() -> None:
    raw = _unwrap(finviz_market_news)
    service_result = {
        "success": True,
        "type": "news",
        "items": [{"Title": "Stocks rise", "Source": "AP", "Date": "02:00PM"}],
    }

    with patch("mtdata.core.finviz.get_general_news", return_value=service_result):
        out = raw(news_type="news", limit=5, page=1)

    assert out["items"][0]["title"] == "Stocks rise"
    assert out["items"][0]["source"] == "AP"
    assert "T14:00:00+00:00" in out["items"][0]["published_at"]
    assert out["tool_scope"] == "raw_finviz_provider"
    assert out["preferred_tool"] == "news"


def test_finviz_news_helpers_are_registered_tools() -> None:
    assert hasattr(finviz_news, "__wrapped__")
    assert hasattr(finviz_market_news, "__wrapped__")


def test_news_tool_docstrings_describe_tool_boundaries() -> None:
    from mtdata.core.news import news

    assert "preferred trader-facing news tool" in (news.__doc__ or "")
    assert "Raw Finviz news provider endpoint" in (finviz_news.__doc__ or "")
    assert "Raw Finviz general market news/blog provider endpoint" in (
        finviz_market_news.__doc__ or ""
    )
