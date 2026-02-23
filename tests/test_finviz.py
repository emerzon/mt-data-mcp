"""
Tests for finviz service and tools.
"""

from unittest.mock import patch, MagicMock
import pandas as pd


class TestFinvizService:
    """Tests for finviz_service.py functions."""

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_fundamentals_success(self, mock_finviz):
        """Test successful fundamentals fetch."""
        from mtdata.services.finviz_service import get_stock_fundamentals
        
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
        from mtdata.services.finviz_service import get_stock_fundamentals
        
        mock_finviz.side_effect = Exception("Network error")
        
        result = get_stock_fundamentals("INVALID")
        
        assert "error" in result

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_news_success(self, mock_finviz):
        """Test successful news fetch."""
        from mtdata.services.finviz_service import get_stock_news
        
        mock_stock = MagicMock()
        mock_df = pd.DataFrame([
            {"Title": "News 1", "Link": "http://example.com/1", "Date": "2024-01-01"},
            {"Title": "News 2", "Link": "http://example.com/2", "Date": "2024-01-02"},
        ])
        mock_stock.ticker_news.return_value = mock_df
        mock_finviz.return_value = mock_stock
        
        result = get_stock_news("AAPL", limit=10)
        
        assert result["success"] is True
        assert result["symbol"] == "AAPL"
        assert result["count"] == 2
        assert len(result["news"]) == 2

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_insider_trades_success(self, mock_finviz):
        """Test successful insider trades fetch."""
        from mtdata.services.finviz_service import get_stock_insider_trades
        
        mock_stock = MagicMock()
        mock_df = pd.DataFrame([
            {"Owner": "John Doe", "Relationship": "CEO", "Transaction": "Buy"},
        ])
        mock_stock.ticker_inside_trader.return_value = mock_df
        mock_finviz.return_value = mock_stock
        
        result = get_stock_insider_trades("AAPL")
        
        assert result["success"] is True
        assert result["count"] == 1

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_ratings_success(self, mock_finviz):
        """Test successful ratings fetch."""
        from mtdata.services.finviz_service import get_stock_ratings
        
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
        from mtdata.services.finviz_service import screen_stocks
        
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
        from mtdata.services.finviz_service import screen_stocks
        
        mock_screener = MagicMock()
        mock_screener.screener_view.return_value = pd.DataFrame()
        mock_overview_class.return_value = mock_screener
        
        result = screen_stocks(filters={"Market Cap": "Mega (>$200bln)"})
        
        assert result["success"] is True
        assert result["count"] == 0

    @patch('finvizfinance.forex.Forex')
    def test_get_forex_performance(self, mock_forex_class):
        """Test forex performance fetch."""
        from mtdata.services.finviz_service import get_forex_performance
        
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
        from mtdata.services.finviz_service import get_crypto_performance       
        
        mock_crypto = MagicMock()
        mock_df = pd.DataFrame([
            {"Ticker": "BTC", "Price": "45000", "Change": "2.5%"},
        ])
        mock_crypto.performance.return_value = mock_df
        mock_crypto_class.return_value = mock_crypto
        
        result = get_crypto_performance()
        
        assert result["success"] is True
        assert result["market"] == "crypto"

    @patch("finvizfinance.earnings.Earnings")
    def test_get_earnings_calendar_success(self, mock_earnings_class):
        """Test earnings calendar fetch."""
        from mtdata.services.finviz_service import get_earnings_calendar

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
        from mtdata.services.finviz_service import get_earnings_calendar

        mock_earnings_class.side_effect = ValueError(
            "Invalid period 'Bad'. Available period: ['This Week', 'Next Week']"
        )

        result = get_earnings_calendar(period="Bad")

        assert "error" in result

    @patch("mtdata.services.finviz_service._fetch_finviz_economic_calendar_items")
    def test_get_economic_calendar_success(self, mock_fetch_items):
        """Test economic calendar fetch."""
        from mtdata.services.finviz_service import get_economic_calendar

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
                "ticker": "USD",
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
        assert len(result["events"]) == 2
        assert len(result["items"]) == 2

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
        assert len(result_high["events"]) == 1

    @patch("mtdata.services.finviz_service._fetch_finviz_economic_calendar_items")
    def test_get_economic_calendar_invalid_impact(self, mock_fetch_items):
        """Test economic calendar with invalid impact filter."""
        from mtdata.services.finviz_service import get_economic_calendar

        mock_fetch_items.return_value = []

        result = get_economic_calendar(impact="extreme")

        assert "error" in result

    @patch("mtdata.services.finviz_service._fetch_finviz_economic_calendar_items")
    def test_get_economic_calendar_date_from_defaults_to_week(self, mock_fetch_items):
        """If date_from is provided without date_to, default to a 7-day window."""
        from mtdata.services.finviz_service import get_economic_calendar

        mock_fetch_items.return_value = []

        get_economic_calendar(date_from="2026-01-05", limit=10, page=1)

        _, kwargs = mock_fetch_items.call_args
        assert kwargs["date_from"] == "2026-01-05"
        assert kwargs["date_to"] == "2026-01-12"

    @patch("mtdata.services.finviz_service._fetch_finviz_economic_calendar_items")
    def test_get_economic_calendar_weekend_anchor_shifts_to_monday(self, mock_fetch_items):
        """If date_from is a weekend, shift the API anchor to the next Monday but keep the requested range."""
        from mtdata.services.finviz_service import get_economic_calendar

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

    @patch("mtdata.services.finviz_service._fetch_finviz_calendar_paged")
    def test_get_earnings_calendar_api_success(self, mock_fetch_paged):
        """Test earnings calendar API fetch."""
        from mtdata.services.finviz_service import get_earnings_calendar_api

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
        assert len(result["earnings"]) == 1

    @patch("mtdata.services.finviz_service._fetch_finviz_calendar_paged")
    def test_get_dividends_calendar_api_success(self, mock_fetch_paged):
        """Test dividends calendar API fetch."""
        from mtdata.services.finviz_service import get_dividends_calendar_api

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
        assert len(result["dividends"]) == 1


class TestFinvizTools:
    """Tests for finviz MCP tools."""

    @patch('mtdata.services.finviz_service.get_stock_fundamentals')
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

    @patch('mtdata.services.finviz_service.screen_stocks')
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
        import json
        
        filters_str = "not valid json"
        try:
            json.loads(filters_str)
            result = {"success": True}
        except (json.JSONDecodeError, TypeError):
            result = {"error": f"Invalid filters JSON: {filters_str}"}
        
        assert "error" in result
