"""Tests for mtdata.services.finviz with mocked HTTP."""

import datetime
from unittest.mock import patch, MagicMock, PropertyMock

import pandas as pd
import pytest

from mtdata.services import finviz as svc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mock_finviz_stock(**overrides):
    stock = MagicMock()
    stock.ticker_fundament.return_value = overrides.get("fundament", {"P/E": "20"})
    stock.ticker_description.return_value = overrides.get("description", "A company.")
    stock.ticker_news.return_value = overrides.get("news", pd.DataFrame({"Title": ["t1"], "Link": ["l1"]}))
    stock.ticker_inside_trader.return_value = overrides.get("insider", pd.DataFrame({"Owner": ["CEO"]}))
    stock.ticker_outer_ratings.return_value = overrides.get("ratings", pd.DataFrame({"Date": ["2024-01-01"]}))
    stock.ticker_peer.return_value = overrides.get("peers", ["MSFT", "GOOG"])
    return stock


# ---------------------------------------------------------------------------
# _sanitize_pagination / _compute_screener_fetch_limit
# ---------------------------------------------------------------------------


class TestSanitizePagination:
    def test_normal_values(self):
        assert svc._sanitize_pagination(50, 1) == (50, 1)

    def test_clamps_limit_to_max(self):
        lim, pg = svc._sanitize_pagination(99999, 1)
        assert lim == svc._FINVIZ_PAGE_LIMIT_MAX

    def test_clamps_limit_min(self):
        lim, pg = svc._sanitize_pagination(-5, 1)
        assert lim == 1

    def test_clamps_page_min(self):
        lim, pg = svc._sanitize_pagination(50, -1)
        assert pg == 1

    def test_non_int_limit(self):
        lim, pg = svc._sanitize_pagination("bad", 1)  # type: ignore
        assert lim == 50

    def test_non_int_page(self):
        lim, pg = svc._sanitize_pagination(10, "bad")  # type: ignore
        assert pg == 1


class TestComputeScreenerFetchLimit:
    def test_normal(self):
        r = svc._compute_screener_fetch_limit(50, 2, 5000)
        assert r == 100

    def test_caps_at_max(self):
        r = svc._compute_screener_fetch_limit(5000, 100, 5000)
        assert r == 5000

    def test_floor_at_one(self):
        r = svc._compute_screener_fetch_limit(0, 0, 5000)
        assert r >= 1


# ---------------------------------------------------------------------------
# get_stock_description (lines 125-139)
# ---------------------------------------------------------------------------


class TestGetStockDescription:
    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    @patch("mtdata.services.finviz.finvizfinance", create=True)
    def test_success(self, mock_cls, mock_patch):
        mock_cls.return_value = _mock_finviz_stock(description="Foo Corp builds things.")
        with patch.dict("sys.modules", {"finvizfinance.quote": MagicMock(finvizfinance=mock_cls)}):
            result = svc.get_stock_description("FOO")
        assert result["success"] is True
        assert result["symbol"] == "FOO"
        assert result["description"] == "Foo Corp builds things."

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_empty_description(self, mock_patch):
        mock_stock = _mock_finviz_stock(description="")
        with patch.dict("sys.modules", {"finvizfinance.quote": MagicMock(finvizfinance=MagicMock(return_value=mock_stock))}):
            result = svc.get_stock_description("FOO")
        assert "error" in result

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_exception(self, mock_patch):
        mock_fv = MagicMock(side_effect=RuntimeError("boom"))
        with patch.dict("sys.modules", {"finvizfinance.quote": MagicMock(finvizfinance=mock_fv)}):
            result = svc.get_stock_description("BAD")
        assert "error" in result


# ---------------------------------------------------------------------------
# get_stock_peers (lines 235-249)
# ---------------------------------------------------------------------------


class TestGetStockPeers:
    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_success(self, mock_patch):
        stock = _mock_finviz_stock(peers=["MSFT", "GOOG"])
        with patch.dict("sys.modules", {"finvizfinance.quote": MagicMock(finvizfinance=MagicMock(return_value=stock))}):
            result = svc.get_stock_peers("AAPL")
        assert result["success"] is True
        assert "MSFT" in result["peers"]

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_no_peers(self, mock_patch):
        stock = _mock_finviz_stock(peers=[])
        with patch.dict("sys.modules", {"finvizfinance.quote": MagicMock(finvizfinance=MagicMock(return_value=stock))}):
            result = svc.get_stock_peers("LONE")
        assert "error" in result

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_single_peer_wrapped(self, mock_patch):
        stock = _mock_finviz_stock(peers="ONLY")
        with patch.dict("sys.modules", {"finvizfinance.quote": MagicMock(finvizfinance=MagicMock(return_value=stock))}):
            result = svc.get_stock_peers("X")
        assert result["success"] is True
        assert result["peers"] == ["ONLY"]

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_exception(self, mock_patch):
        with patch.dict("sys.modules", {"finvizfinance.quote": MagicMock(finvizfinance=MagicMock(side_effect=Exception("fail")))}):
            result = svc.get_stock_peers("ERR")
        assert "error" in result


# ---------------------------------------------------------------------------
# screen_stocks (lines 302-319, 359-361) — screener view dispatch
# ---------------------------------------------------------------------------


class TestScreenStocks:
    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    @patch("mtdata.services.finviz._run_screener_view")
    def test_overview_empty(self, mock_run, mock_patch):
        mock_run.return_value = (None, 50)
        with patch.dict("sys.modules", {"finvizfinance.screener.overview": MagicMock()}):
            result = svc.screen_stocks(view="overview")
        assert result["error"] == "Failed to fetch screener results from Finviz."

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    @patch("mtdata.services.finviz._run_screener_view")
    def test_valuation_view(self, mock_run, mock_patch):
        df = pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "P/E": [25, 30]})
        mock_run.return_value = (df, 50)
        mod = MagicMock()
        with patch.dict("sys.modules", {"finvizfinance.screener.valuation": mod}):
            result = svc.screen_stocks(view="valuation")
        assert result["success"] is True

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    @patch("mtdata.services.finviz._run_screener_view")
    def test_financial_view(self, mock_run, mock_patch):
        mock_run.return_value = (pd.DataFrame({"Ticker": ["A"]}), 50)
        with patch.dict("sys.modules", {"finvizfinance.screener.financial": MagicMock()}):
            result = svc.screen_stocks(view="financial")
        assert result["success"] is True

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    @patch("mtdata.services.finviz._run_screener_view")
    def test_ownership_view(self, mock_run, mock_patch):
        mock_run.return_value = (pd.DataFrame({"Ticker": ["B"]}), 50)
        with patch.dict("sys.modules", {"finvizfinance.screener.ownership": MagicMock()}):
            result = svc.screen_stocks(view="ownership")
        assert result["success"] is True

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    @patch("mtdata.services.finviz._run_screener_view")
    def test_performance_view(self, mock_run, mock_patch):
        mock_run.return_value = (pd.DataFrame({"Ticker": ["C"]}), 50)
        with patch.dict("sys.modules", {"finvizfinance.screener.performance": MagicMock()}):
            result = svc.screen_stocks(view="performance")
        assert result["success"] is True

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    @patch("mtdata.services.finviz._run_screener_view")
    def test_technical_view(self, mock_run, mock_patch):
        mock_run.return_value = (pd.DataFrame({"Ticker": ["D"]}), 50)
        with patch.dict("sys.modules", {"finvizfinance.screener.technical": MagicMock()}):
            result = svc.screen_stocks(view="technical")
        assert result["success"] is True

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    @patch("mtdata.services.finviz._run_screener_view")
    def test_unknown_view_falls_back(self, mock_run, mock_patch):
        mock_run.return_value = (pd.DataFrame({"Ticker": ["E"]}), 50)
        with patch.dict("sys.modules", {"finvizfinance.screener.overview": MagicMock()}):
            result = svc.screen_stocks(view="random_junk")
        assert result["success"] is True

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_exception_returns_error(self, mock_patch):
        with patch.dict("sys.modules", {"finvizfinance.screener.overview": MagicMock(Overview=MagicMock(side_effect=Exception("x")))}):
            result = svc.screen_stocks()
        assert "error" in result

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_exception_logs_warning_without_traceback(self, mock_patch):
        mock_logger = MagicMock()
        with patch.object(svc, "logger", mock_logger):
            with patch.dict(
                "sys.modules",
                {"finvizfinance.screener.overview": MagicMock(Overview=MagicMock(side_effect=Exception("429 Too Many Requests")))},
            ):
                result = svc.screen_stocks()
        assert "error" in result
        mock_logger.warning.assert_called_once()
        mock_logger.exception.assert_not_called()

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    @patch("mtdata.services.finviz._run_screener_view")
    def test_with_filters_and_order(self, mock_run, mock_patch):
        mock_run.return_value = (pd.DataFrame({"Ticker": ["Z"]}), 50)
        with patch.dict("sys.modules", {"finvizfinance.screener.overview": MagicMock()}):
            result = svc.screen_stocks(filters={"Sector": "Technology"}, order="-marketcap")
        assert result["success"] is True


# ---------------------------------------------------------------------------
# get_general_news (lines 359-421)
# ---------------------------------------------------------------------------


class TestGetGeneralNews:
    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_news_success(self, mock_patch):
        news_df = pd.DataFrame({"Title": [f"n{i}" for i in range(5)]})
        mock_news = MagicMock()
        mock_news.return_value.get_news.return_value = {"news": news_df, "blogs": pd.DataFrame()}
        with patch.dict("sys.modules", {"finvizfinance.news": MagicMock(News=mock_news)}):
            result = svc.get_general_news("news", limit=3, page=1)
        assert result["success"] is True
        assert result["count"] == 3

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_blogs_success(self, mock_patch):
        blog_df = pd.DataFrame({"Title": ["b1", "b2"]})
        mock_news = MagicMock()
        mock_news.return_value.get_news.return_value = {"news": pd.DataFrame(), "blogs": blog_df}
        with patch.dict("sys.modules", {"finvizfinance.news": MagicMock(News=mock_news)}):
            result = svc.get_general_news("blogs", limit=10, page=1)
        assert result["success"] is True
        assert result["count"] == 2

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_empty_news(self, mock_patch):
        mock_news = MagicMock()
        mock_news.return_value.get_news.return_value = {"news": pd.DataFrame()}
        with patch.dict("sys.modules", {"finvizfinance.news": MagicMock(News=mock_news)}):
            result = svc.get_general_news("news")
        assert "error" in result

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_items_as_list(self, mock_patch):
        mock_news = MagicMock()
        mock_news.return_value.get_news.return_value = {"news": [{"title": "a"}, {"title": "b"}]}
        with patch.dict("sys.modules", {"finvizfinance.news": MagicMock(News=mock_news)}):
            result = svc.get_general_news("news", limit=10)
        assert result["success"] is True
        assert result["count"] == 2

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_items_as_empty_list(self, mock_patch):
        mock_news = MagicMock()
        mock_news.return_value.get_news.return_value = {"news": []}
        with patch.dict("sys.modules", {"finvizfinance.news": MagicMock(News=mock_news)}):
            result = svc.get_general_news("news")
        assert "error" in result

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_exception(self, mock_patch):
        mock_news = MagicMock(side_effect=Exception("boom"))
        with patch.dict("sys.modules", {"finvizfinance.news": MagicMock(News=mock_news)}):
            result = svc.get_general_news("news")
        assert "error" in result


# ---------------------------------------------------------------------------
# get_insider_activity (lines 437-464)
# ---------------------------------------------------------------------------


class TestGetInsiderActivity:
    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_success(self, mock_patch):
        df = pd.DataFrame({"Ticker": ["AAPL"], "Owner": ["CEO"], "Date": ["Nov 07 '25"]})
        mock_ins = MagicMock()
        mock_ins.return_value.get_insider.return_value = df
        with patch.dict("sys.modules", {"finvizfinance.insider": MagicMock(Insider=mock_ins)}):
            result = svc.get_insider_activity("latest", limit=10)
        assert result["success"] is True
        assert result["count"] == 1
        assert result["insider_trades"][0]["Date"] == "2025-11-07"

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_empty(self, mock_patch):
        mock_ins = MagicMock()
        mock_ins.return_value.get_insider.return_value = pd.DataFrame()
        with patch.dict("sys.modules", {"finvizfinance.insider": MagicMock(Insider=mock_ins)}):
            result = svc.get_insider_activity("latest")
        assert "error" in result

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_exception(self, mock_patch):
        mock_ins = MagicMock(side_effect=Exception("no"))
        with patch.dict("sys.modules", {"finvizfinance.insider": MagicMock(Insider=mock_ins)}):
            result = svc.get_insider_activity("latest")
        assert "error" in result


# ---------------------------------------------------------------------------
# get_futures_performance (lines 515-533)
# ---------------------------------------------------------------------------


class TestGetFuturesPerformance:
    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_success(self, mock_patch):
        df = pd.DataFrame({"Name": ["Gold"], "Change": ["+1.5%"]})
        mock_f = MagicMock()
        mock_f.return_value.performance.return_value = df
        with patch.dict("sys.modules", {"finvizfinance.future": MagicMock(Future=mock_f)}):
            result = svc.get_futures_performance()
        assert result["success"] is True
        assert result["count"] == 1

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_empty(self, mock_patch):
        mock_f = MagicMock()
        mock_f.return_value.performance.return_value = pd.DataFrame()
        with patch.dict("sys.modules", {"finvizfinance.future": MagicMock(Future=mock_f)}):
            result = svc.get_futures_performance()
        assert "error" in result

    @patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch")
    def test_exception(self, mock_patch):
        mock_f = MagicMock(side_effect=Exception("err"))
        with patch.dict("sys.modules", {"finvizfinance.future": MagicMock(Future=mock_f)}):
            result = svc.get_futures_performance()
        assert "error" in result


# ---------------------------------------------------------------------------
# _resolve_date_range / _align_to_next_monday_if_weekend
# ---------------------------------------------------------------------------


class TestResolveDateRange:
    def test_defaults(self):
        d_from, d_to = svc._resolve_date_range(date_from=None, date_to=None, default_days=7)
        assert d_from == datetime.date.today().isoformat()

    def test_explicit_range(self):
        d_from, d_to = svc._resolve_date_range(date_from="2024-06-01", date_to="2024-06-15", default_days=7)
        assert d_from == "2024-06-01"
        assert d_to == "2024-06-15"

    def test_to_without_from_raises(self):
        with pytest.raises(ValueError, match="date_from is required"):
            svc._resolve_date_range(date_from=None, date_to="2024-06-15", default_days=7)

    def test_bad_from_raises(self):
        with pytest.raises(ValueError, match="Invalid date_from"):
            svc._resolve_date_range(date_from="not-a-date", date_to=None, default_days=7)

    def test_bad_to_raises(self):
        with pytest.raises(ValueError, match="Invalid date_to"):
            svc._resolve_date_range(date_from="2024-06-01", date_to="bad", default_days=7)

    def test_to_before_from_raises(self):
        with pytest.raises(ValueError, match="date_to must be >= date_from"):
            svc._resolve_date_range(date_from="2024-06-15", date_to="2024-06-01", default_days=7)


class TestAlignToMondayIfWeekend:
    def test_saturday(self):
        assert svc._align_to_next_monday_if_weekend("2024-06-08") == "2024-06-10"

    def test_sunday(self):
        assert svc._align_to_next_monday_if_weekend("2024-06-09") == "2024-06-10"

    def test_weekday_unchanged(self):
        assert svc._align_to_next_monday_if_weekend("2024-06-10") == "2024-06-10"


# ---------------------------------------------------------------------------
# _filter_calendar_events_by_date (lines 811-819)
# ---------------------------------------------------------------------------


class TestFilterCalendarEventsByDate:
    def test_filters_by_date_range(self):
        events = [
            {"Datetime": "2024-06-01T10:00:00Z", "Release": "A"},
            {"Datetime": "2024-06-10T08:00:00", "Release": "B"},
            {"Datetime": "2024-07-01T09:00:00", "Release": "C"},
        ]
        filtered = svc._filter_calendar_events_by_date(events, date_from="2024-06-01", date_to="2024-06-15")
        assert len(filtered) == 2

    def test_handles_date_only(self):
        events = [{"Datetime": "2024-06-05", "Release": "X"}]
        filtered = svc._filter_calendar_events_by_date(events, date_from="2024-06-01", date_to="2024-06-10")
        assert len(filtered) == 1

    def test_handles_datetime_object(self):
        events = [{"Datetime": datetime.datetime(2024, 6, 5, 12, 0)}]
        filtered = svc._filter_calendar_events_by_date(events, date_from="2024-06-01", date_to="2024-06-10")
        assert len(filtered) == 1

    def test_handles_date_object(self):
        events = [{"Datetime": datetime.date(2024, 6, 5)}]
        filtered = svc._filter_calendar_events_by_date(events, date_from="2024-06-01", date_to="2024-06-10")
        assert len(filtered) == 1

    def test_skips_non_string_non_date(self):
        events = [{"Datetime": 12345}]
        filtered = svc._filter_calendar_events_by_date(events, date_from="2024-06-01", date_to="2024-06-10")
        assert len(filtered) == 0

    def test_skips_missing_datetime(self):
        events = [{"Release": "no-date"}]
        filtered = svc._filter_calendar_events_by_date(events, date_from="2024-06-01", date_to="2024-06-10")
        assert len(filtered) == 0

    def test_skips_bad_string(self):
        events = [{"Datetime": "not-a-date"}]
        filtered = svc._filter_calendar_events_by_date(events, date_from="2024-06-01", date_to="2024-06-10")
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# _fetch_finviz_economic_calendar_items (lines 829-847)
# ---------------------------------------------------------------------------


class TestFetchFinvizEconomicCalendarItems:
    @patch("mtdata.services.finviz._finviz_http_get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"event": "GDP", "importance": 3}]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        items = svc._fetch_finviz_economic_calendar_items("2024-06-01", "2024-06-07")
        assert len(items) == 1
        assert items[0]["event"] == "GDP"

    @patch("mtdata.services.finviz._finviz_http_get")
    def test_non_list_raises(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "oops"}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        with pytest.raises(TypeError, match="Unexpected response"):
            svc._fetch_finviz_economic_calendar_items("2024-06-01", "2024-06-07")

    @patch("mtdata.services.finviz._finviz_http_get")
    def test_skips_non_dict_items(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"event": "CPI"}, "bad_item", 123]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        items = svc._fetch_finviz_economic_calendar_items("2024-06-01", "2024-06-07")
        assert len(items) == 1


# ---------------------------------------------------------------------------
# _fetch_finviz_calendar_paged (lines 850-880)
# ---------------------------------------------------------------------------


class TestFetchFinvizCalendarPaged:
    @patch("mtdata.services.finviz._finviz_http_get")
    def test_earnings_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"items": [{"ticker": "AAPL"}], "totalItemsCount": 1, "totalPages": 1, "page": 1}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        result = svc._fetch_finviz_calendar_paged(kind="earnings", date_from="2024-06-01", date_to="2024-06-07", page=1, page_size=50)
        assert result["items"][0]["ticker"] == "AAPL"

    @patch("mtdata.services.finviz._finviz_http_get")
    def test_non_dict_raises(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [1, 2, 3]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        with pytest.raises(TypeError, match="Unexpected response"):
            svc._fetch_finviz_calendar_paged(kind="dividends", date_from="2024-06-01", date_to="2024-06-07", page=1, page_size=50)

    @patch("mtdata.services.finviz._finviz_http_get")
    def test_missing_items_raises(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"noitems": True}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        with pytest.raises(TypeError, match="missing items"):
            svc._fetch_finviz_calendar_paged(kind="earnings", date_from="2024-06-01", date_to="2024-06-07", page=1, page_size=50)


# ---------------------------------------------------------------------------
# _normalize_finviz_economic_calendar_items
# ---------------------------------------------------------------------------


class TestNormalizeFinvizEconomicCalendarItems:
    def test_maps_importance(self):
        items = [{"importance": 3, "event": "GDP", "date": "2024-06-01T10:00"}]
        result = svc._normalize_finviz_economic_calendar_items(items)
        assert result[0]["Impact"] == "high"
        assert result[0]["Release"] == "GDP"

    def test_importance_1_is_low(self):
        result = svc._normalize_finviz_economic_calendar_items([{"importance": 1, "event": "X"}])
        assert result[0]["Impact"] == "low"

    def test_importance_2_is_medium(self):
        result = svc._normalize_finviz_economic_calendar_items([{"importance": 2, "event": "Y"}])
        assert result[0]["Impact"] == "medium"

    def test_non_int_importance(self):
        result = svc._normalize_finviz_economic_calendar_items([{"importance": "high", "event": "Z"}])
        assert result[0]["Impact"] == ""

    def test_missing_fields_default_empty(self):
        result = svc._normalize_finviz_economic_calendar_items([{}])
        assert result[0]["Release"] == ""
        assert result[0]["Datetime"] == ""

    def test_internal_calendar_fields_are_omitted(self):
        result = svc._normalize_finviz_economic_calendar_items(
            [
                {
                    "calendarId": 123,
                    "allDay": True,
                    "alert": False,
                    "hasNoDetail": True,
                    "event": "GDP",
                    "date": "2024-06-01T10:00",
                }
            ]
        )
        row = result[0]
        assert "CalendarId" not in row
        assert "AllDay" not in row
        assert "Alert" not in row
        assert "HasNoDetail" not in row


# ---------------------------------------------------------------------------
# get_economic_calendar (lines 659-663)
# ---------------------------------------------------------------------------


class TestGetEconomicCalendar:
    @patch("mtdata.services.finviz._fetch_finviz_economic_calendar_items", return_value=[])
    @patch("mtdata.services.finviz._normalize_finviz_economic_calendar_items", return_value=[])
    @patch("mtdata.services.finviz._filter_calendar_events_by_date", return_value=[])
    def test_success_empty(self, mock_filter, mock_norm, mock_fetch):
        result = svc.get_economic_calendar(limit=50, page=1)
        assert result["success"] is True
        assert result["count"] == 0

    def test_bad_impact(self):
        result = svc.get_economic_calendar(impact="critical")
        assert "error" in result

    @patch("mtdata.services.finviz._fetch_finviz_economic_calendar_items", side_effect=ValueError("bad"))
    def test_value_error(self, mock_fetch):
        result = svc.get_economic_calendar(date_from="2024-06-01", date_to="2024-06-07")
        assert "error" in result

    @patch("mtdata.services.finviz._fetch_finviz_economic_calendar_items", side_effect=RuntimeError("boom"))
    def test_generic_exception(self, mock_fetch):
        result = svc.get_economic_calendar(date_from="2024-06-01", date_to="2024-06-07")
        assert "error" in result


# ---------------------------------------------------------------------------
# get_earnings_calendar_api / get_dividends_calendar_api (700-745)
# ---------------------------------------------------------------------------


class TestGetEarningsCalendarApi:
    @patch("mtdata.services.finviz._fetch_finviz_calendar_paged")
    def test_success(self, mock_fetch):
        mock_fetch.return_value = {"items": [{"ticker": "AAPL"}], "totalItemsCount": 1, "totalPages": 1, "page": 1}
        result = svc.get_earnings_calendar_api(limit=10, page=1, date_from="2024-06-01", date_to="2024-06-07")
        assert result["success"] is True
        assert result["calendar"] == "earnings"

    @patch("mtdata.services.finviz._fetch_finviz_calendar_paged", side_effect=ValueError("bad"))
    def test_value_error(self, mock_fetch):
        result = svc.get_earnings_calendar_api(date_from="2024-06-01", date_to="2024-06-07")
        assert "error" in result

    @patch("mtdata.services.finviz._fetch_finviz_calendar_paged", side_effect=Exception("boom"))
    def test_exception(self, mock_fetch):
        result = svc.get_earnings_calendar_api(date_from="2024-06-01", date_to="2024-06-07")
        assert "error" in result


class TestGetDividendsCalendarApi:
    @patch("mtdata.services.finviz._fetch_finviz_calendar_paged")
    def test_success(self, mock_fetch):
        mock_fetch.return_value = {"items": [{"ticker": "T"}], "totalItemsCount": 1, "totalPages": 1, "page": 1}
        result = svc.get_dividends_calendar_api(limit=10, page=1, date_from="2024-06-01", date_to="2024-06-07")
        assert result["success"] is True
        assert result["calendar"] == "dividends"

    @patch("mtdata.services.finviz._fetch_finviz_calendar_paged", side_effect=ValueError("bad"))
    def test_value_error(self, mock_fetch):
        result = svc.get_dividends_calendar_api(date_from="2024-06-01", date_to="2024-06-07")
        assert "error" in result

    @patch("mtdata.services.finviz._fetch_finviz_calendar_paged", side_effect=Exception("boom"))
    def test_exception(self, mock_fetch):
        result = svc.get_dividends_calendar_api(date_from="2024-06-01", date_to="2024-06-07")
        assert "error" in result


