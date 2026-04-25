from unittest.mock import patch

from mtdata.core.finviz import finviz_earnings, finviz_insider, finviz_peers, finviz_ratings


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


class TestFinvizEarningsOutputContract:
    def _unwrapped(self):
        return _unwrap(finviz_earnings)

    @patch("mtdata.core.finviz.get_earnings_calendar")
    def test_success_moves_items_and_pagination_into_contract(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "period": "This Week",
            "count": 2,
            "total": 6,
            "page": 2,
            "pages": 3,
            "truncated": False,
            "earnings": [
                {"ticker": "AAPL", "date": "2026-01-10"},
                {"ticker": "MSFT", "date": "2026-01-11"},
            ],
        }

        result = self._unwrapped()(period="This Week", limit=2, page=2)

        assert result["success"] is True
        assert result["data"]["items"][0]["ticker"] == "AAPL"
        assert result["summary"]["counts"]["items"] == 2
        assert result["meta"]["tool"] == "finviz_earnings"
        assert result["meta"]["request"] == {
            "period": "This Week",
            "limit": 2,
            "page": 2,
        }
        assert result["meta"]["pagination"] == {
            "page": 2,
            "total": 6,
            "pages": 3,
        }
        assert result["meta"]["stats"]["truncated"] is False
        assert "earnings" not in result
        assert "page" not in result
        assert "total" not in result
        assert "pages" not in result

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
        assert "operation" not in result


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
        assert len(result["insider_trades"]) == 3
        assert result["summary"]["counts"]["available"] == 4
        assert result["summary"]["counts"]["buy_transactions"] == 2
        assert result["summary"]["counts"]["sell_transactions"] == 1
        assert result["omitted_item_count"] == 1

    @patch("mtdata.core.finviz.get_stock_ratings")
    def test_ratings_compact_returns_latest_rows_and_summary(self, mock_get):
        rows = [{"Date": f"2026-01-0{i}", "Rating": "Buy"} for i in range(1, 6)]
        mock_get.return_value = {"success": True, "symbol": "AAPL", "ratings": rows}

        result = _unwrap(finviz_ratings)("AAPL", detail="compact")

        assert result["detail"] == "compact"
        assert result["ratings"] == rows[:3]
        assert result["summary"]["latest"] == rows[0]
        assert result["summary"]["counts"]["available"] == 5

    @patch("mtdata.core.finviz.get_stock_peers")
    def test_peers_compact_returns_top_five_and_counts(self, mock_get):
        peers = ["MSFT", "GOOGL", "META", "AMZN", "NVDA", "ORCL"]
        mock_get.return_value = {"success": True, "symbol": "AAPL", "peers": peers}

        result = _unwrap(finviz_peers)("AAPL", detail="compact")

        assert result["detail"] == "compact"
        assert result["peers"] == peers[:5]
        assert result["summary"]["counts"]["available"] == 6
        assert result["omitted_item_count"] == 1

    @patch("mtdata.core.finviz.get_stock_ratings")
    def test_finviz_detail_rejects_unknown_values(self, mock_get):
        mock_get.return_value = {"success": True, "symbol": "AAPL", "ratings": []}

        result = _unwrap(finviz_ratings)("AAPL", detail="standard")  # type: ignore[arg-type]

        assert result["success"] is False
        assert result["error_code"] == "finviz_ratings_invalid_detail"
