from unittest.mock import patch

from mtdata.core.finviz import finviz_earnings


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
