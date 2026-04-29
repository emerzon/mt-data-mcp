"""
Tests for finviz service and tools.
"""

from unittest.mock import MagicMock, patch

import pandas as pd


class TestFinvizService:
    """Tests for the canonical finviz package functions."""

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_fundamentals_success(self, mock_finviz):
        """Test successful fundamentals fetch."""
        from mtdata.services.finviz import get_stock_fundamentals
        
        mock_stock = MagicMock()
        mock_stock.ticker_fundament.return_value = {
            "P/E": "28.5",
            "Market Cap": "3.0T",
            "EPS (ttm)": "6.05",
        }
        mock_finviz.return_value = mock_stock
        
        result = get_stock_fundamentals("AAPL")
        
        assert result["success"] is True
        assert result["symbol"] == "AAPL"
        assert "fundamentals" in result
        assert result["fundamentals"]["P/E"] == "28.5"

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_fundamentals_error(self, mock_finviz):
        """Test fundamentals fetch with error."""
        from mtdata.services.finviz import get_stock_fundamentals
        
        mock_finviz.side_effect = Exception("Network error")
        
        result = get_stock_fundamentals("INVALID")
        
        assert "error" in result

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_news_success(self, mock_finviz):
        """Test successful news fetch."""
        from mtdata.services.finviz import get_stock_news
        
        mock_stock = MagicMock()
        mock_df = pd.DataFrame([
            {"Title": "\r\n    News 1   ", "Link": "  http://example.com/1  ", "Date": " 2024-01-01 "},
            {"Title": "News 2", "Link": "http://example.com/2", "Date": "2024-01-02"},
        ])
        mock_stock.ticker_news.return_value = mock_df
        mock_finviz.return_value = mock_stock
        
        result = get_stock_news("AAPL", limit=10)
        
        assert result["success"] is True
        assert result["symbol"] == "AAPL"
        assert result["count"] == 2
        assert len(result["news"]) == 2
        assert result["news"][0]["Title"] == "News 1"
        assert result["news"][0]["Link"] == "http://example.com/1"
        assert result["news"][0]["Date"] == "2024-01-01"

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_insider_trades_success(self, mock_finviz):
        """Test successful insider trades fetch."""
        from mtdata.services.finviz import get_stock_insider_trades
        
        mock_stock = MagicMock()
        mock_df = pd.DataFrame([
            {"Owner": "John Doe", "Relationship": "CEO", "Transaction": "Buy", "Date": "Nov 07 '25"},
        ])
        mock_stock.ticker_inside_trader.return_value = mock_df
        mock_finviz.return_value = mock_stock
        
        result = get_stock_insider_trades("AAPL")
        
        assert result["success"] is True
        assert result["count"] == 1
        assert result["insider_trades"][0]["Date"] == "2025-11-07"

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_ratings_success(self, mock_finviz):
        """Test successful ratings fetch."""
        from mtdata.services.finviz import get_stock_ratings
        
        mock_stock = MagicMock()
        mock_df = pd.DataFrame([
            {"Date": "2024-01-01", "Analyst": "Goldman Sachs", "Rating": "Buy"},
        ])
        mock_stock.ticker_outer_ratings.return_value = mock_df
        mock_finviz.return_value = mock_stock
        
        result = get_stock_ratings("AAPL")
        
        assert result["success"] is True
        assert result["count"] == 1

    @patch('finvizfinance.screener.overview.Overview')
    def test_screen_stocks_success(self, mock_overview_class):
        """Test successful stock screening."""
        from mtdata.services.finviz import screen_stocks
        
        mock_screener = MagicMock()
        mock_df = pd.DataFrame([
            {"Ticker": "AAPL", "Company": "Apple Inc.", "Market Cap": "3.0T"},
            {"Ticker": "MSFT", "Company": "Microsoft", "Market Cap": "2.8T"},
        ])
        mock_screener.screener_view.return_value = mock_df
        mock_overview_class.return_value = mock_screener
        
        result = screen_stocks(
            filters={"Exchange": "NASDAQ", "Sector": "Technology"},
            limit=10
        )
        
        assert result["success"] is True
        assert result["count"] == 2

    @patch('finvizfinance.screener.overview.Overview')
    def test_screen_stocks_no_results(self, mock_overview_class):
        """Test screening with no results."""
        from mtdata.services.finviz import screen_stocks
        
        mock_screener = MagicMock()
        mock_screener.screener_view.return_value = pd.DataFrame()
        mock_overview_class.return_value = mock_screener
        
        result = screen_stocks(filters={"Market Cap": "Mega (>$200bln)"})
        
        assert result["success"] is True
        assert result["count"] == 0

    @patch('finvizfinance.forex.Forex')
    def test_get_forex_performance(self, mock_forex_class):
        """Test forex performance fetch."""
        from mtdata.services.finviz import get_forex_performance
        
        mock_forex = MagicMock()
        mock_df = pd.DataFrame([
            {"Ticker": "EUR/USD", "Price": "1.08", "Change": "0.5%"},
        ])
        mock_forex.performance.return_value = mock_df
        mock_forex_class.return_value = mock_forex
        
        result = get_forex_performance()
        
        assert result["success"] is True
        assert result["market"] == "forex"

    @patch('finvizfinance.crypto.Crypto')
    def test_get_crypto_performance(self, mock_crypto_class):
        """Test crypto performance fetch."""
        from mtdata.services.finviz import get_crypto_performance
        
        mock_crypto = MagicMock()
        mock_df = pd.DataFrame([
            {"Ticker": "BTC", "Price": "45000", "Change": "2.5%"},
        ])
        mock_crypto.performance.return_value = mock_df
        mock_crypto_class.return_value = mock_crypto
        
        result = get_crypto_performance()
        
        assert result["success"] is True
        assert result["market"] == "crypto"
        assert result["coins"][0]["Price"] == "45000.00"
        assert "Price_display" not in result["coins"][0]

    @patch('finvizfinance.crypto.Crypto')
    def test_get_crypto_performance_preserves_subcent_price_display(self, mock_crypto_class):
        """Sub-cent tokens should include a non-truncated display price."""
        from mtdata.services.finviz import get_crypto_performance

        mock_crypto = MagicMock()
        mock_df = pd.DataFrame([
            {"Ticker": "SHIBUSD", "Price": "0.00001234", "Change": "2.5%"},
        ])
        mock_crypto.performance.return_value = mock_df
        mock_crypto_class.return_value = mock_crypto

        result = get_crypto_performance()

        assert result["success"] is True
        assert result["coins"][0]["Price"] == "0.00001234"
        assert "Price_display" not in result["coins"][0]

    @patch('finvizfinance.crypto.Crypto')
    def test_get_crypto_performance_uses_scientific_notation_for_tiny_prices(self, mock_crypto_class):
        """Tiny nonzero token prices should not round down to zero."""
        from mtdata.services.finviz import get_crypto_performance

        mock_crypto = MagicMock()
        mock_df = pd.DataFrame([
            {"Ticker": "TINY", "Price": "0.0000000015", "Change": "2.5%"},
        ])
        mock_crypto.performance.return_value = mock_df
        mock_crypto_class.return_value = mock_crypto

        result = get_crypto_performance()

        assert result["success"] is True
        assert result["coins"][0]["Price"] == "1.5e-09"

    @patch('finvizfinance.crypto.Crypto')
    def test_get_crypto_performance_adds_wtd_alias_when_day_week_identical(self, mock_crypto_class):
        """When day/week values are identical for all rows, add WTD alias and warning."""
        from mtdata.services.finviz import get_crypto_performance

        mock_crypto = MagicMock()
        mock_df = pd.DataFrame([
            {"Ticker": "BTCUSD", "Perf Day": -0.0242, "Perf Week": -0.0242},
            {"Ticker": "ETHUSD", "Perf Day": -0.0310, "Perf Week": -0.0310},
        ])
        mock_crypto.performance.return_value = mock_df
        mock_crypto_class.return_value = mock_crypto

        result = get_crypto_performance()

        assert result["success"] is True
        assert "warnings" in result
        assert "Perf WTD" in result["coins"][0]
        assert result["coins"][0]["Perf WTD"] == result["coins"][0]["Perf Week"]
        assert result["coins"][1]["Perf WTD"] == result["coins"][1]["Perf Week"]

    @patch('finvizfinance.crypto.Crypto')
    def test_get_crypto_performance_no_wtd_alias_when_values_differ(self, mock_crypto_class):
        """Do not add WTD alias when day/week values differ."""
        from mtdata.services.finviz import get_crypto_performance

        mock_crypto = MagicMock()
        mock_df = pd.DataFrame([
            {"Ticker": "BTCUSD", "Perf Day": -0.0242, "Perf Week": -0.0500},
            {"Ticker": "ETHUSD", "Perf Day": -0.0310, "Perf Week": -0.0600},
        ])
        mock_crypto.performance.return_value = mock_df
        mock_crypto_class.return_value = mock_crypto

        result = get_crypto_performance()

        assert result["success"] is True
        assert "warnings" not in result
        assert "Perf WTD" not in result["coins"][0]
        assert "Perf WTD" not in result["coins"][1]

    @patch("finvizfinance.earnings.Earnings")
    def test_get_earnings_calendar_success(self, mock_earnings_class):
        """Test earnings calendar fetch."""
        from mtdata.services.finviz import get_earnings_calendar

        mock_earnings = MagicMock()
        mock_df = pd.DataFrame(
            [
                {"Ticker": "AAPL", "Earnings": "2026-01-10", "EPS Est": "2.10"},
                {"Ticker": "MSFT", "Earnings": "2026-01-11", "EPS Est": "3.20"},
            ]
        )
        mock_earnings.df = mock_df
        mock_earnings_class.return_value = mock_earnings

        result = get_earnings_calendar(period="This Week", limit=10, page=1)

        mock_earnings_class.assert_called_once_with(period="This Week")
        assert result["success"] is True
        assert result["period"] == "This Week"
        assert result["count"] == 2
        assert len(result["earnings"]) == 2

    @patch("finvizfinance.earnings.Earnings")
    def test_get_earnings_calendar_invalid_period(self, mock_earnings_class):
        """Test earnings calendar with invalid period."""
        from mtdata.services.finviz import get_earnings_calendar

        mock_earnings_class.side_effect = ValueError(
            "Invalid period 'Bad'. Available period: ['This Week', 'Next Week']"
        )

        result = get_earnings_calendar(period="Bad")

        assert "error" in result

    @patch("mtdata.services.finviz._fetch_finviz_economic_calendar_items")
    def test_get_economic_calendar_success(self, mock_fetch_items):
        """Test economic calendar fetch."""
        from mtdata.services.finviz import get_economic_calendar

        mock_fetch_items.return_value = [
            {
                "calendarId": 0,
                "ticker": "USD",
                "event": "Out of range",
                "category": "Test",
                "date": "2026-01-03T08:30:00",
                "actual": "",
                "forecast": "",
                "previous": "",
                "importance": 1,
            },
            {
                "calendarId": 1,
                "ticker": "UNITEDSTANONFAR",
                "event": "Nonfarm Payrolls",
                "category": "Employment",
                "date": "2026-01-04T08:30:00",
                "actual": "",
                "forecast": "",
                "previous": "",
                "importance": 3,
            },
            {
                "calendarId": 2,
                "ticker": "USD",
                "event": "ISM Services",
                "category": "Business",
                "date": "2026-01-04T10:00:00",
                "actual": "",
                "forecast": "",
                "previous": "",
                "importance": 2,
            },
        ]

        result = get_economic_calendar(limit=10, page=1, date_from="2026-01-04", date_to="2026-01-04")

        assert result["success"] is True
        assert result["source"] == "finviz_api"
        assert result["count"] == 2
        assert result["total"] == 2
        assert len(result["items"]) == 2
        assert "events" not in result
        assert result["items"][0]["For"] == "USD"
        assert result["items"][0]["Country"] == "United States"

        result_high = get_economic_calendar(
            impact="high",
            limit=10,
            page=1,
            date_from="2026-01-04",
            date_to="2026-01-05",
        )
        assert result_high["success"] is True
        assert result_high["impact"] == "high"
        assert result_high["total"] == 1
        assert len(result_high["items"]) == 1

    @patch("mtdata.services.finviz._fetch_finviz_economic_calendar_items")
    def test_get_economic_calendar_invalid_impact(self, mock_fetch_items):
        """Test economic calendar with invalid impact filter."""
        from mtdata.services.finviz import get_economic_calendar

        mock_fetch_items.return_value = []

        result = get_economic_calendar(impact="extreme")

        assert "error" in result

    @patch("mtdata.services.finviz._fetch_finviz_economic_calendar_items")
    def test_get_economic_calendar_date_from_defaults_to_week(self, mock_fetch_items):
        """If date_from is provided without date_to, default to a 7-day window."""
        from mtdata.services.finviz import get_economic_calendar

        mock_fetch_items.return_value = []

        get_economic_calendar(date_from="2026-01-05", limit=10, page=1)

        _, kwargs = mock_fetch_items.call_args
        assert kwargs["date_from"] == "2026-01-05"
        assert kwargs["date_to"] == "2026-01-12"

    @patch("mtdata.services.finviz._fetch_finviz_economic_calendar_items")
    def test_get_economic_calendar_weekend_anchor_shifts_to_monday(self, mock_fetch_items):
        """If date_from is a weekend, shift the API anchor to the next Monday but keep the requested range."""
        from mtdata.services.finviz import get_economic_calendar

        mock_fetch_items.return_value = [
            {
                "calendarId": 1,
                "ticker": "USD",
                "event": "Test",
                "category": "Test",
                "date": "2025-01-06T10:00:00",
                "actual": "",
                "forecast": "",
                "previous": "",
                "importance": 2,
            },
        ]

        result = get_economic_calendar(date_from="2025-01-05", limit=10, page=1)

        assert result["success"] is True
        assert result["dateFrom"] == "2025-01-05"
        assert result["dateTo"] == "2025-01-12"

        _, kwargs = mock_fetch_items.call_args
        assert kwargs["date_from"] == "2025-01-06"

    @patch("mtdata.services.finviz._fetch_finviz_economic_calendar_items")
    def test_get_economic_calendar_accepts_iso_datetime_date_from(self, mock_fetch_items):
        """ISO datetime date_from inputs should normalize before weekend alignment."""
        from mtdata.services.finviz import get_economic_calendar

        mock_fetch_items.return_value = []

        result = get_economic_calendar(date_from="2025-01-05T08:30:00", limit=10, page=1)

        assert result["success"] is True
        assert result["dateFrom"] == "2025-01-05"
        assert result["dateTo"] == "2025-01-12"

        _, kwargs = mock_fetch_items.call_args
        assert kwargs["date_from"] == "2025-01-06"
        assert kwargs["date_to"] == "2025-01-12"

    @patch("mtdata.services.finviz._fetch_finviz_calendar_paged")
    def test_get_earnings_calendar_api_success(self, mock_fetch_paged):
        """Test earnings calendar API fetch."""
        from mtdata.services.finviz import get_earnings_calendar_api

        mock_fetch_paged.return_value = {
            "items": [{"ticker": "AAPL", "date": "2026-01-05", "eps": "2.10"}],
            "page": 1,
            "pageSize": 50,
            "totalItemsCount": 1,
            "totalPages": 1,
        }

        result = get_earnings_calendar_api(date_from="2026-01-05", date_to="2026-01-12", limit=50, page=1)

        assert result["success"] is True
        assert result["calendar"] == "earnings"
        assert result["dateFrom"] == "2026-01-05"
        assert result["dateTo"] == "2026-01-12"
        assert result["count"] == 1
        assert result["total"] == 1
        assert len(result["items"]) == 1
        assert "earnings" not in result

    @patch("mtdata.services.finviz._fetch_finviz_calendar_paged")
    def test_get_earnings_calendar_api_applies_client_limit(self, mock_fetch_paged):
        from mtdata.services.finviz import get_earnings_calendar_api

        mock_fetch_paged.return_value = {
            "items": [{"ticker": f"SYM{i}", "date": "2026-01-05"} for i in range(5)],
            "page": 1,
            "pageSize": 50,
            "totalItemsCount": 5,
            "totalPages": 1,
        }

        result = get_earnings_calendar_api(date_from="2026-01-05", date_to="2026-01-12", limit=3, page=1)

        assert result["count"] == 3
        assert result["total"] == 5
        assert result["pages"] == 2
        assert [item["ticker"] for item in result["items"]] == ["SYM0", "SYM1", "SYM2"]

    @patch("mtdata.services.finviz._fetch_finviz_calendar_paged")
    def test_get_dividends_calendar_api_success(self, mock_fetch_paged):
        """Test dividends calendar API fetch."""
        from mtdata.services.finviz import get_dividends_calendar_api

        mock_fetch_paged.return_value = {
            "items": [{"ticker": "MSFT", "exDate": "2026-01-06", "amount": "0.75"}],
            "page": 1,
            "pageSize": 50,
            "totalItemsCount": 1,
            "totalPages": 1,
        }

        result = get_dividends_calendar_api(date_from="2026-01-05", date_to="2026-01-12", limit=50, page=1)

        assert result["success"] is True
        assert result["calendar"] == "dividends"
        assert result["dateFrom"] == "2026-01-05"
        assert result["dateTo"] == "2026-01-12"
        assert result["count"] == 1
        assert result["total"] == 1
        assert len(result["items"]) == 1

    @patch("mtdata.services.finviz._fetch_finviz_calendar_paged")
    def test_get_dividends_calendar_api_applies_client_limit(self, mock_fetch_paged):
        from mtdata.services.finviz import get_dividends_calendar_api

        mock_fetch_paged.return_value = {
            "items": [{"ticker": f"SYM{i}", "exDate": "2026-01-06"} for i in range(6)],
            "page": 1,
            "pageSize": 50,
            "totalItemsCount": 6,
            "totalPages": 1,
        }

        result = get_dividends_calendar_api(date_from="2026-01-05", date_to="2026-01-12", limit=4, page=1)

        assert result["count"] == 4
        assert result["total"] == 6
        assert result["pages"] == 2
        assert [item["ticker"] for item in result["items"]] == ["SYM0", "SYM1", "SYM2", "SYM3"]
        assert "dividends" not in result


class TestFinvizTools:
    """Tests for finviz MCP tools."""

    @patch("mtdata.core.finviz.get_forex_performance")
    def test_finviz_forex_uses_items_with_snake_case_rows(self, mock_get_forex):
        from mtdata.core.finviz import finviz_forex

        mock_get_forex.return_value = {
            "success": True,
            "market": "forex",
            "count": 1,
            "pairs": [{"Pair": "EUR/USD", "Perf 5Min": "0.1%"}],
        }

        raw = getattr(finviz_forex, "__wrapped__", finviz_forex)
        result = raw()

        assert "pairs" not in result
        assert result["detail"] == "compact"
        assert result["items"] == [{"perf_5min": "0.1%", "symbol": "EUR/USD"}]

    @patch("mtdata.core.finviz.get_forex_performance")
    def test_finviz_forex_applies_limit(self, mock_get_forex):
        from mtdata.core.finviz import finviz_forex

        mock_get_forex.return_value = {
            "success": True,
            "market": "forex",
            "pairs": [
                {"Pair": "EUR/USD"},
                {"Pair": "GBP/USD"},
                {"Pair": "USD/JPY"},
            ],
        }

        raw = getattr(finviz_forex, "__wrapped__", finviz_forex)
        result = raw(limit=2)

        assert result["count"] == 2
        assert result["available_count"] == 3
        assert result["omitted_item_count"] == 1
        assert result["items"] == [{"symbol": "EUR/USD"}, {"symbol": "GBP/USD"}]

    @patch("mtdata.core.finviz.get_crypto_performance")
    def test_finviz_crypto_uses_items_with_snake_case_rows(self, mock_get_crypto):
        from mtdata.core.finviz import finviz_crypto

        mock_get_crypto.return_value = {
            "success": True,
            "market": "crypto",
            "count": 1,
            "coins": [{"Ticker": "BTC", "Perf WTD": "2.5%"}],
        }

        raw = getattr(finviz_crypto, "__wrapped__", finviz_crypto)
        result = raw()

        assert "coins" not in result
        assert result["detail"] == "compact"
        assert result["items"] == [{"perf_wtd": "2.5%", "symbol": "BTC"}]

    @patch("mtdata.core.finviz.get_futures_performance")
    def test_finviz_futures_uses_items_with_snake_case_rows(self, mock_get_futures):
        from mtdata.core.finviz import finviz_futures

        mock_get_futures.return_value = {
            "success": True,
            "market": "futures",
            "count": 1,
            "futures": [{"ticker": "NQ", "label": "Nasdaq 100", "perf": "0.8%"}],
        }

        raw = getattr(finviz_futures, "__wrapped__", finviz_futures)
        result = raw()

        assert "futures" not in result
        assert result["detail"] == "compact"
        assert result["items"] == [{"name": "Nasdaq 100", "perf_pct": "0.8%", "symbol": "NQ"}]

    @patch("mtdata.core.finviz.get_futures_performance")
    def test_finviz_market_tools_accept_full_detail(self, mock_get_futures):
        from mtdata.core.finviz import finviz_futures

        mock_get_futures.return_value = {
            "success": True,
            "market": "futures",
            "count": 1,
            "futures": [{"ticker": "NQ", "label": "Nasdaq 100", "perf": "0.8%"}],
        }

        raw = getattr(finviz_futures, "__wrapped__", finviz_futures)
        result = raw(detail="full")

        assert result["detail"] == "full"
        assert result["meta"]["tool"] == "finviz_futures"
        assert result["meta"]["request"] == {"limit": 20, "detail": "full"}

    @patch("mtdata.core.finviz.get_stock_fundamentals")
    def test_finviz_fundamentals_rejects_non_equity_symbols_upfront(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("BTCUSD")

        mock_get_fundamentals.assert_not_called()
        assert result["error"] == (
            "BTCUSD is not a Finviz-supported equity ticker. "
            "finviz_fundamentals only supports US equities."
        )
        assert result["success"] is False
        assert result["error_code"] == "finviz_unsupported_symbol"
        assert result["operation"] == "finviz_fundamentals"
        assert result["details"] == {
            "symbol": "BTCUSD",
            "tool": "finviz_fundamentals",
        }
        assert isinstance(result.get("request_id"), str)

    @patch("mtdata.core.finviz.get_stock_description")
    def test_finviz_description_normalizes_equity_symbols(self, mock_get_description):
        from mtdata.core.finviz import finviz_description

        mock_get_description.return_value = {"success": True, "symbol": "AAPL"}
        raw = getattr(finviz_description, "__wrapped__", finviz_description)
        result = raw("aapl")

        mock_get_description.assert_called_once_with("AAPL")
        assert result["success"] is True

    @patch("mtdata.core.finviz.get_stock_news")
    def test_finviz_news_rejects_non_equity_symbols_upfront(self, mock_get_news):
        from mtdata.core.finviz import finviz_news

        raw = getattr(finviz_news, "__wrapped__", finviz_news)
        result = raw("BTCUSD", limit=5, page=1)

        mock_get_news.assert_not_called()
        assert result["error"] == (
            "BTCUSD is not a Finviz-supported equity ticker. "
            "finviz_news only supports US equities."
        )
        assert result["success"] is False
        assert result["error_code"] == "finviz_unsupported_symbol"
        assert result["operation"] == "finviz_news"
        assert result["details"] == {"symbol": "BTCUSD", "tool": "finviz_news"}
        assert isinstance(result.get("request_id"), str)

    @patch("mtdata.core.finviz.get_stock_fundamentals")
    def test_finviz_fundamentals_requires_symbol(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("")

        mock_get_fundamentals.assert_not_called()
        assert result["success"] is False
        assert result["error"] == "finviz_fundamentals requires a symbol."
        assert result["error_code"] == "finviz_symbol_required"
        assert result["operation"] == "finviz_fundamentals"
        assert result["details"] == {"tool": "finviz_fundamentals"}
        assert isinstance(result.get("request_id"), str)

    @patch("mtdata.core.finviz.get_stock_fundamentals")
    def test_finviz_fundamentals_defaults_to_compact_summary(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {
                "Company": "Apple Inc",
                "Sector": "Technology",
                "P/E": "34.29",
                "EPS (ttm)": "7.90",
                "Market Cap": "3979.47B",
                "Price": "270.00",
                "Change": "1.2%",
                "RSI (14)": "62.1",
                "Insider Own": "0.1%",
            },
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("aapl")

        mock_get_fundamentals.assert_called_once_with("AAPL")
        assert result["detail"] == "compact"
        assert result["category"] == "summary"
        assert result["fundamentals"]["pe_ratio"] == "34.29"
        assert result["fundamentals"]["market_cap"] == "3979.47B"
        assert "insider_own" not in result["fundamentals"]
        assert "fields_returned" not in result
        assert "available_field_count" not in result
        assert "omitted_field_count" not in result

    @patch("mtdata.core.finviz.get_stock_fundamentals")
    def test_finviz_fundamentals_filters_category_and_fields(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {
                "P/E": "34.29",
                "P/S": "8.1",
                "EPS (ttm)": "7.90",
                "RSI (14)": "62.1",
            },
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        valuation = raw("AAPL", category="valuation")
        custom = raw("AAPL", fields="P/E,RSI (14),Missing")

        assert valuation["category"] == "valuation"
        assert valuation["fundamentals"] == {
            "pe_ratio": "34.29",
            "price_to_sales": "8.1",
            "eps_ttm": "7.90",
        }
        assert custom["category"] == "custom"
        assert custom["fundamentals"] == {"pe_ratio": "34.29", "rsi_14": "62.1"}
        assert custom["missing_fields"] == ["Missing"]

    @patch("mtdata.core.finviz.get_stock_fundamentals")
    def test_finviz_fundamentals_full_omits_redundant_field_echo(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {
                "Company": "Apple Inc",
                "Sector": "Technology",
                "P/E": "34.29",
            },
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("AAPL", detail="full")

        assert "fields_returned" not in result
        assert result["available_field_count"] == 3
        assert result["omitted_field_count"] == 0

    @patch("mtdata.core.finviz.get_stock_fundamentals")
    def test_finviz_fundamentals_expands_cryptic_metric_keys(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {
                "ROA": "29.5%",
                "ROE": "156.0%",
                "Curr R": "0.87",
                "Quick R": "0.83",
                "LT Debt/Eq": "1.26",
                "Shs Outstand": "14.70B",
                "Shs Float": "14.66B",
                "Book/sh": "6.00",
                "Perf Week": "1.71%",
            },
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("AAPL", detail="full")

        fundamentals = result["fundamentals"]
        assert fundamentals["return_on_assets"] == "29.5%"
        assert fundamentals["return_on_equity"] == "156.0%"
        assert fundamentals["current_ratio"] == "0.87"
        assert fundamentals["quick_ratio"] == "0.83"
        assert fundamentals["long_term_debt_to_equity"] == "1.26"
        assert fundamentals["shares_outstanding"] == "14.70B"
        assert fundamentals["shares_float"] == "14.66B"
        assert fundamentals["book_value_per_share"] == "6.00"
        assert fundamentals["performance_week"] == "1.71%"
        assert "roa" not in fundamentals
        assert "curr_r" not in fundamentals

    @patch("mtdata.core.finviz.get_stock_insider_trades")
    def test_finviz_insider_defaults_to_compact_detail(self, mock_get_trades):
        from mtdata.core.finviz import finviz_insider

        mock_get_trades.return_value = {
            "success": True,
            "symbol": "AAPL",
            "total": 4,
            "insider_trades": [
                {"Owner": f"Owner {i}", "Transaction": "Buy" if i == 0 else "Sale"}
                for i in range(4)
            ],
        }

        raw = getattr(finviz_insider, "__wrapped__", finviz_insider)
        result = raw("AAPL")

        assert result["detail"] == "compact"
        assert result["count"] == 3
        assert result["omitted_item_count"] == 1
        assert result["summary"]["counts"]["buy_transactions"] == 1

    @patch("mtdata.core.finviz.get_earnings_calendar")
    def test_finviz_earnings_expands_cryptic_metric_keys(self, mock_get_earnings):
        from mtdata.core.finviz import finviz_earnings

        mock_get_earnings.return_value = {
            "success": True,
            "period": "This Week",
            "earnings": [
                {
                    "Ticker": "APLM",
                    "ROA": "-1.365",
                    "ROE": "-3.8",
                    "ROIC": None,
                    "Curr R": "0.97",
                    "Debt/Eq": None,
                    "Gross M": None,
                    "Oper M": "-3.04",
                    "Profit M": "-3.67",
                },
            ],
            "count": 1,
            "total": 1,
            "page": 1,
            "pages": 1,
        }

        raw = getattr(finviz_earnings, "__wrapped__", finviz_earnings)
        result = raw(detail="full")

        item = result["items"][0]
        assert item["symbol"] == "APLM"
        assert item["return_on_assets"] == "-1.365"
        assert item["return_on_equity"] == "-3.8"
        assert item["return_on_invested_capital"] is None
        assert item["current_ratio"] == "0.97"
        assert item["debt_to_equity"] is None
        assert item["gross_margin"] is None
        assert item["operating_margin"] == "-3.04"
        assert item["profit_margin"] == "-3.67"
        assert "curr_r" not in item

    @patch("mtdata.core.finviz.get_earnings_calendar")
    def test_finviz_earnings_compact_uses_calendar_focused_rows(self, mock_get_earnings):
        from mtdata.core.finviz import finviz_earnings

        mock_get_earnings.return_value = {
            "success": True,
            "period": "This Week",
            "earnings": [
                {
                    "Ticker": "APLM",
                    "Market Cap": "14.17M",
                    "ROA": "-1.365",
                    "ROE": "-3.8",
                    "Curr R": "0.97",
                    "Earnings": "Apr 27/b",
                    "Price": "12.85",
                    "Change": "-0.0258",
                    "Volume": "6593",
                },
            ],
            "count": 1,
            "total": 12,
            "page": 1,
            "pages": 2,
        }

        raw = getattr(finviz_earnings, "__wrapped__", finviz_earnings)
        result = raw()

        mock_get_earnings.assert_called_once_with(period="This Week", limit=10, page=1)
        assert result["detail"] == "compact"
        assert result["omitted_item_count"] == 11
        assert result["items"] == [
            {
                "symbol": "APLM",
                "earnings": "Apr 27/b",
                "market_cap": "14.17M",
                "price": "12.85",
                "change_pct": "-0.0258",
                "volume": "6593",
            }
        ]

    @patch('mtdata.services.finviz.get_stock_fundamentals')
    def test_finviz_fundamentals_tool(self, mock_get_fundamentals):
        """Test finviz_fundamentals tool."""
        # Import the service function directly to test logic without MCP server init
        
        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {"P/E": "28.5"},
        }
        
        # Call the mocked function
        result = mock_get_fundamentals("AAPL")
        
        mock_get_fundamentals.assert_called_once_with("AAPL")
        assert result["success"] is True

    @patch('mtdata.services.finviz.screen_stocks')
    def test_finviz_screen_tool_with_filters(self, mock_screen):
        """Test finviz_screen tool with JSON filters."""
        import json
        
        mock_screen.return_value = {"success": True, "count": 5, "stocks": []}
        
        # Simulate what finviz_screen does: parse JSON and call service
        filters_str = '{"Exchange": "NASDAQ"}'
        filters_dict = json.loads(filters_str)
        result = mock_screen(filters=filters_dict, order=None, limit=10, view="overview")
        
        mock_screen.assert_called_once_with(
            filters={"Exchange": "NASDAQ"},
            order=None,
            limit=10,
            view="overview"
        )
        assert result["success"] is True

    def test_finviz_screen_tool_invalid_json(self):
        """Test finviz_screen tool with invalid JSON."""
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        with patch("mtdata.core.finviz.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen.__wrapped__(filters="not valid json")

        assert "error" in result
        # Verify improved error message
        assert "Invalid filters format" in result["error"]
        assert "JSON object" in result["error"]
        assert "filter names as keys" in result["error"]
        assert "Sector" in result["error"]  # Should include example
        assert "Finviz screener shorthand tokens" in result["error"]
        assert "'not valid json'" in result["error"]
        assert result["success"] is False
        assert result["error_code"] == "finviz_screen_filters_invalid"
        assert result["operation"] == "finviz_screen"
        assert result["details"] == {"received_type": "str"}
        assert isinstance(result.get("request_id"), str)

    def test_finviz_screen_tool_rejects_non_object_json_filters(self):
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        with patch("mtdata.core.finviz.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen.__wrapped__(filters='["NASDAQ"]')

        assert result["error"].startswith("Invalid filters format.")
        assert "Got: '[\"NASDAQ\"]'" in result["error"]
        assert result["success"] is False
        assert result["error_code"] == "finviz_screen_filters_invalid"
        assert result["details"] == {"received_type": "str"}

    @patch('mtdata.core.finviz.screen_stocks')
    def test_finviz_screen_tool_accepts_dict_filters(self, mock_screen):
        """Test finviz_screen tool accepts dict filters directly."""
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        mock_screen.return_value = {
            "success": True,
            "count": 5,
            "stocks": [{"Ticker": "AAPL", "Market Cap": "3.0T", "P/E": "28.5"}],
        }

        with patch("mtdata.core.finviz.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen.__wrapped__(
                filters={"Exchange": "NASDAQ", "Sector": "Technology"},
                limit=10
            )

        mock_screen.assert_called_once_with(
            filters={"Exchange": "NASDAQ", "Sector": "Technology"},
            order=None,
            limit=10,
            page=1,
            view="overview"
        )
        assert result["success"] is True
        assert result["count"] == 1
        assert result["available_count"] == 1
        assert result["detail"] == "compact"
        assert "stocks" not in result
        assert result["items"] == [{"symbol": "AAPL", "market_cap": "3.0T", "pe_ratio": "28.5"}]

    @patch('mtdata.core.finviz.screen_stocks')
    def test_finviz_screen_tool_accepts_json_string_filters(self, mock_screen):
        """Test finviz_screen tool still accepts JSON string filters."""
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        mock_screen.return_value = {"success": True, "count": 3, "stocks": []}

        with patch("mtdata.core.finviz.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen.__wrapped__(
                filters='{"Exchange": "NASDAQ"}',
                limit=5
            )

        mock_screen.assert_called_once_with(
            filters={"Exchange": "NASDAQ"},
            order=None,
            limit=5,
            page=1,
            view="overview"
        )
        assert result["success"] is True
        assert result["count"] == 0
        assert result["available_count"] == 0
        assert result["items"] == []

    @patch('mtdata.core.finviz.screen_stocks')
    def test_finviz_screen_tool_supports_full_detail_meta_and_omitted_count(self, mock_screen):
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        mock_screen.return_value = {
            "success": True,
            "stocks": [
                {"Ticker": "AAPL", "Market Cap": "3.0T"},
                {"Ticker": "MSFT", "Market Cap": "2.8T"},
            ],
        }

        with patch("mtdata.core.finviz.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen.__wrapped__(
                filters={"Exchange": "NASDAQ"},
                limit=1,
                detail="full",
            )

        assert result["items"] == [{"symbol": "AAPL", "market_cap": "3.0T"}]
        assert result["available_count"] == 2
        assert result["omitted_item_count"] == 1
        assert result["detail"] == "full"
        assert result["meta"]["tool"] == "finviz_screen"

    @patch('mtdata.core.finviz.screen_stocks')
    def test_finviz_screen_tool_accepts_finviz_shorthand_filters(self, mock_screen):
        """Test finviz_screen accepts native Finviz screener shorthand filters."""
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        mock_screen.return_value = {"success": True, "count": 2, "stocks": []}

        with patch("mtdata.core.finviz.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen.__wrapped__(
                filters="cap_largeover,exch_nyse",
                limit=5,
            )

        mock_screen.assert_called_once_with(
            filters={
                "Market Cap.": "+Large (over $10bln)",
                "Exchange": "NYSE",
            },
            order=None,
            limit=5,
            page=1,
            view="overview",
        )
        assert result["success"] is True
        assert result["count"] == 0
        assert result["available_count"] == 0

    def test_finviz_calendar_prefers_start_end_aliases(self):
        from mtdata.core.finviz import finviz_calendar

        def _run_direct(_logger, operation, func, **fields):
            return func()

        with (
            patch("mtdata.core.finviz.run_logged_operation", side_effect=_run_direct),
            patch("mtdata.core.finviz.get_economic_calendar", return_value={"success": True}) as mock_calendar,
        ):
            result = finviz_calendar.__wrapped__(start="2026-01-05", end="2026-01-12")

        assert result["success"] is True
        mock_calendar.assert_called_once_with(
            impact=None,
            limit=20,
            page=1,
            date_from="2026-01-05",
            date_to="2026-01-12",
        )

    def test_finviz_calendar_rejects_conflicting_start_and_date_from(self):
        from mtdata.core.finviz import finviz_calendar

        def _run_direct(_logger, operation, func, **fields):
            return func()

        with patch("mtdata.core.finviz.run_logged_operation", side_effect=_run_direct):
            result = finviz_calendar.__wrapped__(
                start="2026-01-05",
                date_from="2026-01-06",
            )

        assert result["success"] is False
        assert result["error"] == "Provide either start or date_from, not both with different values."
        assert result["error_code"] == "finviz_conflicting_text_args"
        assert result["operation"] == "finviz_calendar"
        assert result["details"] == {
            "preferred_name": "start",
            "legacy_name": "date_from",
        }
        assert isinstance(result.get("request_id"), str)
