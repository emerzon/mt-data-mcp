from unittest.mock import patch

from mtdata.core.finviz import (
    finviz_calendar,
    finviz_earnings,
    finviz_insider,
    finviz_insider_activity,
    finviz_peers,
    finviz_ratings,
)


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


class TestFinvizEarningsOutputContract:
    def _unwrapped(self):
        return _unwrap(finviz_earnings)

    @patch("mtdata.core.finviz.get_earnings_calendar")
    def test_success_returns_flat_normalized_items(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "period": "This Week",
            "count": 2,
            "total": 6,
            "page": 2,
            "pages": 3,
            "truncated": False,
            "earnings": [
                {"Ticker": "AAPL", "Market Cap": "3T", "Date": "2026-01-10"},
                {"Ticker": "MSFT", "Market Cap": "2T", "Date": "2026-01-11"},
            ],
        }

        result = self._unwrapped()(period="This Week", limit=2, page=2)

        assert result["success"] is True
        assert result["items"][0] == {
            "symbol": "AAPL",
            "market_cap": "3T",
        }
        assert result["count"] == 2
        assert result["page"] == 2
        assert result["total"] == 6
        assert result["pages"] == 3
        assert "data" not in result
        assert "summary" not in result
        assert "meta" not in result
        assert "earnings" not in result

    @patch("mtdata.core.finviz.get_earnings_calendar")
    def test_full_includes_metadata(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "period": "This Week",
            "count": 2,
            "total": 6,
            "page": 2,
            "pages": 3,
            "truncated": False,
            "earnings": [{"Ticker": "AAPL", "Date": "2026-01-10"}],
        }

        result = self._unwrapped()(period="This Week", limit=2, page=2, detail="full")

        assert result["success"] is True
        assert result["detail"] == "full"
        assert result["meta"]["tool"] == "finviz_earnings"
        assert result["meta"]["request"] == {
            "period": "This Week",
            "limit": 2,
            "page": 2,
            "detail": "full",
        }
        assert result["meta"]["pagination"] == {
            "page": 2,
            "total": 6,
            "pages": 3,
        }
        assert result["meta"]["stats"]["truncated"] is False

    @patch("mtdata.core.finviz.get_earnings_calendar")
    def test_invalid_period_returns_error_envelope(self, mock_get):
        mock_get.return_value = {
            "error": "Invalid period 'Bad'. Available period: ['This Week']"
        }

        result = self._unwrapped()(period="Bad", limit=50, page=1)

        assert result["success"] is False
        assert result["error_code"] == "finviz_earnings_invalid_period"
        assert result["meta"]["tool"] == "finviz_earnings"
        assert result["meta"]["request"]["period"] == "Bad"
        assert result["meta"]["request"]["limit"] == 50
        assert result["meta"]["request"]["page"] == 1
        assert result["meta"]["request"]["detail"] == "compact"
        assert "operation" not in result


class TestFinvizCalendarOutputContract:
    @patch("mtdata.core.finviz.get_economic_calendar")
    def test_calendar_normalizes_top_level_and_item_keys(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "dateFrom": "2026-01-05",
            "dateTo": "2026-01-12",
            "items": [
                {
                    "Datetime": "2026-01-06T13:30:00",
                    "Release": "CPI",
                    "Impact": "high",
                    "For": "USD",
                    "Country": "United States",
                    "ReferenceDate": "2025-12",
                }
            ],
        }

        result = _unwrap(finviz_calendar)(start="2026-01-05", end="2026-01-12")

        assert result["date_from"] == "2026-01-05"
        assert result["date_to"] == "2026-01-12"
        assert result["items"] == [
            {
                "datetime": "2026-01-06T13:30:00",
                "release": "CPI",
                "impact": "high",
                "for_currency": "USD",
                "country": "United States",
                "reference_date": "2025-12",
            }
        ]

    @patch("mtdata.core.finviz.get_earnings_calendar_api")
    def test_calendar_earnings_normalizes_api_keys(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "items": [
                {
                    "earningsdate": "2026-04-29T08:30:00",
                    "isearningdateestimate": False,
                    "symbol": "ABBV",
                    "marketcap": 357812,
                    "epsestimate": 2.59,
                    "epsactual": 2.65,
                    "epssurprise": 2.23,
                    "salesestimate": 12900,
                    "salesactual": 13100,
                }
            ],
        }

        result = _unwrap(finviz_calendar)(
            calendar="earnings",
            start="2026-04-29",
            end="2026-04-30",
        )

        assert result["items"] == [
            {
                "earnings_date": "2026-04-29T08:30:00",
                "is_earning_date_estimate": False,
                "symbol": "ABBV",
                "market_cap": 357812,
                "eps_estimate": 2.59,
                "eps_actual": 2.65,
                "eps_surprise": 2.23,
                "sales_estimate": 12900,
                "sales_actual": 13100,
            }
        ]


class TestFinvizInsiderActivityOutputContract:
    @patch("mtdata.core.finviz.get_insider_activity")
    def test_compact_normalizes_items_and_summarizes_without_urls(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "option": "latest",
            "count": 6,
            "total": 6,
            "page": 1,
            "pages": 1,
            "insider_trades": [
                {
                    "Ticker": "AAPL",
                    "SEC Form 4": "Apr 27 06:30 PM",
                    "SEC Form 4 Link": "https://sec.example/a",
                    "Insider_id": "123",
                    "#Shares Total": "200",
                    "Transaction": "Sale",
                    "#Shares": "10",
                    "Value ($)": "1000",
                },
                {
                    "Ticker": "AAPL",
                    "SEC Form 4 Link": "https://sec.example/b",
                    "Transaction": "Buy",
                    "#Shares": "5",
                    "Value ($)": "600",
                },
                {"Ticker": "MSFT", "Transaction": "Sale", "#Shares": "2", "Value ($)": "200"},
                {"Ticker": "NVDA", "Transaction": "Option Exercise"},
                {"Ticker": "TSLA", "Transaction": "Sale"},
                {"Ticker": "META", "Transaction": "Buy"},
            ],
        }

        result = _unwrap(finviz_insider_activity)(detail="compact")

        assert result["detail"] == "compact"
        assert "insider_trades" not in result
        assert len(result["items"]) == 5
        assert result["items"][0]["symbol"] == "AAPL"
        assert result["items"][0] == {
            "symbol": "AAPL",
            "transaction": "Sale",
            "shares": "10",
            "value_usd": "1000",
        }
        assert "sec_form_4" not in result["items"][0]
        assert "sec_form_4_link" not in result["items"][0]
        assert "insider_id" not in result["items"][0]
        assert "shares_total" not in result["items"][0]
        assert result["summary"]["counts"] == {
            "returned": 5,
            "available": 6,
            "total": 6,
            "buy_transactions": 2,
            "sell_transactions": 3,
        }
        assert result["summary"]["top_symbols"][0] == {
            "symbol": "AAPL",
            "transactions": 2,
            "shares": 15.0,
            "value_usd": 1600.0,
        }
        assert result["omitted_item_count"] == 1

    @patch("mtdata.core.finviz.get_insider_activity")
    def test_full_keeps_all_normalized_rows_including_urls(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "insider_trades": [
                {"Ticker": "AAPL", "SEC Form 4 Link": "https://sec.example/a"}
            ],
        }

        result = _unwrap(finviz_insider_activity)(detail="full")

        assert result["detail"] == "full"
        assert result["items"] == [
            {"symbol": "AAPL", "sec_form_4_link": "https://sec.example/a"}
        ]
        assert "insider_trades" not in result


class TestFinvizInsiderOutputContract:
    @patch("mtdata.core.finviz.get_stock_insider_trades")
    def test_compact_normalizes_items(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "symbol": "AAPL",
            "total": 4,
            "insider_trades": [
                {
                    "Insider Trading": "Parekh Kevan",
                    "Relationship": "CFO",
                    "Transaction": "Sale",
                    "#Shares": "1534",
                    "Value ($)": "421850",
                    "SEC Form 4 Link": "https://sec.example/a",
                    "Insider_id": "123",
                },
                {"Insider Trading": "Cook Tim", "Transaction": "Buy"},
                {"Insider Trading": "Maestri Luca", "Transaction": "Sale"},
                {"Insider Trading": "Williams Jeff", "Transaction": "Sale"},
            ],
        }

        result = _unwrap(finviz_insider)("AAPL", detail="compact")

        assert result["detail"] == "compact"
        assert "insider_trades" not in result
        assert result["items"][0] == {
            "owner": "Parekh Kevan",
            "transaction": "Sale",
            "shares": "1534",
            "value_usd": "421850",
        }
        assert result["summary"]["counts"]["returned"] == 3
        assert result["summary"]["counts"]["sell_transactions"] == 3
        assert result["omitted_item_count"] == 1

    @patch("mtdata.core.finviz.get_stock_insider_trades")
    def test_full_normalizes_items(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "symbol": "AAPL",
            "insider_trades": [
                {"Insider Trading": "Parekh Kevan", "SEC Form 4": "Apr 27 06:30 PM"}
            ],
        }

        result = _unwrap(finviz_insider)("AAPL", detail="full")

        assert result["detail"] == "full"
        assert result["items"] == [
            {"owner": "Parekh Kevan", "sec_form_4": "Apr 27 06:30 PM"}
        ]
        assert "insider_trades" not in result


class TestFinvizProgressiveDisclosure:
    @patch("mtdata.core.finviz.get_stock_insider_trades")
    def test_insider_compact_truncates_rows_and_adds_counts(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "symbol": "AAPL",
            "total": 4,
            "insider_trades": [
                {"Transaction": "Buy", "Owner": "A"},
                {"Transaction": "Sale", "Owner": "B"},
                {"Transaction": "Option Exercise", "Owner": "C"},
                {"Transaction": "Buy", "Owner": "D"},
            ],
        }

        result = _unwrap(finviz_insider)("AAPL", detail="compact")

        assert result["detail"] == "compact"
        assert len(result["items"]) == 3
        assert "insider_trades" not in result
        assert result["summary"]["counts"]["available"] == 4
        assert result["summary"]["counts"]["buy_transactions"] == 2
        assert result["summary"]["counts"]["sell_transactions"] == 1
        assert result["omitted_item_count"] == 1

    @patch("mtdata.core.finviz.get_stock_ratings")
    def test_ratings_compact_returns_latest_rows_and_summary(self, mock_get):
        rows = [
            {"Date": f"2026-01-0{i}", "Outer": "UBS", "Rating": "Buy"}
            for i in range(1, 6)
        ]
        mock_get.return_value = {"success": True, "symbol": "AAPL", "ratings": rows}

        result = _unwrap(finviz_ratings)("AAPL", detail="compact")

        expected_rows = [
            {"date": f"2026-01-0{i}", "firm": "UBS", "rating": "Buy"}
            for i in range(1, 4)
        ]
        assert result["detail"] == "compact"
        assert result["ratings"] == expected_rows
        assert result["count"] == 3
        assert result["available_count"] == 5
        assert result["truncated"] is True
        assert result["summary"]["latest"] == expected_rows[0]
        assert result["summary"]["counts"]["available"] == 5
        assert result["show_all_hint"] == "Set detail='full' or limit=5 to view all ratings."

    @patch("mtdata.core.finviz.get_stock_ratings")
    def test_ratings_limit_controls_returned_rows(self, mock_get):
        rows = [{"Date": f"2026-01-0{i}", "Rating": "Buy"} for i in range(1, 6)]
        mock_get.return_value = {"success": True, "symbol": "AAPL", "ratings": rows}

        result = _unwrap(finviz_ratings)("AAPL", limit=2)

        assert result["detail"] == "compact"
        assert len(result["ratings"]) == 2
        assert result["count"] == 2
        assert result["available_count"] == 5
        assert result["truncated"] is True
        assert result["summary"]["counts"] == {"returned": 2, "available": 5}
        assert result["omitted_item_count"] == 3

    @patch("mtdata.core.finviz.get_stock_ratings")
    def test_ratings_normalizes_mixed_date_formats(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "symbol": "AAPL",
            "ratings": [
                {"Date": "2026-04-28", "Rating": "Neutral"},
                {"Date": "2026-04-17 00:00:00", "Rating": "Outperform"},
            ],
        }

        result = _unwrap(finviz_ratings)("AAPL", detail="compact", limit=2)

        assert [row["date"] for row in result["ratings"]] == [
            "2026-04-28",
            "2026-04-17",
        ]

    @patch("mtdata.core.finviz.get_stock_peers")
    def test_peers_compact_returns_top_five_and_counts(self, mock_get):
        peers = ["MSFT", "GOOGL", "META", "AMZN", "NVDA", "ORCL"]
        mock_get.return_value = {"success": True, "symbol": "AAPL", "peers": peers}

        result = _unwrap(finviz_peers)("AAPL", detail="compact")

        assert result["detail"] == "compact"
        assert result["peers"] == peers[:5]
        assert result["summary"]["counts"]["available"] == 6
        assert result["omitted_item_count"] == 1

    @patch("mtdata.core.finviz.get_stock_peers")
    def test_peers_limit_controls_returned_rows(self, mock_get):
        peers = ["MSFT", "GOOGL", "META"]
        mock_get.return_value = {"success": True, "symbol": "AAPL", "peers": peers}

        result = _unwrap(finviz_peers)("AAPL", limit=2)

        assert result["peers"] == ["MSFT", "GOOGL"]
        assert result["summary"]["counts"] == {"returned": 2, "available": 3}
        assert result["omitted_item_count"] == 1

    @patch("mtdata.core.finviz.get_stock_ratings")
    def test_finviz_detail_accepts_standard_alias_as_compact(self, mock_get):
        mock_get.return_value = {"success": True, "symbol": "AAPL", "ratings": []}

        result = _unwrap(finviz_ratings)("AAPL", detail="standard")  # type: ignore[arg-type]

        assert result["success"] is True
        assert result["detail"] == "compact"
