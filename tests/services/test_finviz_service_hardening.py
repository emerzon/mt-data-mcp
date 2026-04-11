import sys
import types

import pandas as pd

from mtdata.services import finviz as svc
from mtdata.services.finviz import client as finviz_client
from mtdata.services.finviz import pagination as finviz_pagination


def test_compute_screener_fetch_limit_is_bounded(monkeypatch):
    monkeypatch.setattr(svc, "_FINVIZ_PAGE_LIMIT_MAX", 500)

    assert svc._compute_screener_fetch_limit(limit=50, page=1, max_rows=5000) == 50
    assert svc._compute_screener_fetch_limit(limit=50, page=3, max_rows=120) == 120
    assert svc._compute_screener_fetch_limit(limit=-1, page=0, max_rows=120) == 1
    assert svc._compute_screener_fetch_limit(limit=9999, page=1, max_rows=120) == 120


def test_finviz_http_get_applies_default_timeout(monkeypatch):
    calls = {}

    class DummyResp:
        pass

    def _fake_get(url, headers=None, params=None, timeout=None):
        calls["url"] = url
        calls["headers"] = headers
        calls["params"] = params
        calls["timeout"] = timeout
        return DummyResp()

    import requests

    monkeypatch.setattr(svc, "_FINVIZ_HTTP_TIMEOUT", 7.5)
    monkeypatch.setattr(requests, "get", _fake_get)
    _ = svc._finviz_http_get("https://example.test", headers={"A": "B"}, params={"x": 1})

    assert calls["url"] == "https://example.test"
    assert calls["timeout"] == 7.5


def test_finviz_http_get_accepts_requests_timeout_tuple(monkeypatch):
    calls = {}

    class DummyResp:
        pass

    def _fake_get(url, headers=None, params=None, timeout=None):
        calls["timeout"] = timeout
        return DummyResp()

    import requests

    monkeypatch.setattr(requests, "get", _fake_get)
    _ = finviz_client.finviz_http_get(
        "https://example.test",
        headers={"A": "B"},
        params={"x": 1},
        timeout=(1.5, 3.0),
    )

    assert calls["timeout"] == (1.5, 3.0)


def test_finviz_http_get_invalid_timeout_override_falls_back(monkeypatch):
    calls = {}

    class DummyResp:
        pass

    def _fake_get(url, headers=None, params=None, timeout=None):
        calls["timeout"] = timeout
        return DummyResp()

    import requests

    monkeypatch.setattr(requests, "get", _fake_get)
    monkeypatch.setattr(finviz_client, "_FINVIZ_HTTP_TIMEOUT", 7.5)
    _ = finviz_client.finviz_http_get(
        "https://example.test",
        headers={"A": "B"},
        params={"x": 1},
        timeout="bad-timeout",
    )

    assert calls["timeout"] == 7.5


def test_pagination_helpers_coerce_invalid_override_values(monkeypatch):
    monkeypatch.setattr(finviz_pagination, "get_finviz_page_limit_max", lambda: 500)
    monkeypatch.setattr(finviz_pagination, "get_finviz_screener_max_rows", lambda: 120)

    assert finviz_pagination.sanitize_pagination("bad", 0, page_limit_max="bad") == (50, 1)
    assert (
        finviz_pagination.compute_screener_fetch_limit(
            limit=50,
            page=3,
            max_rows="bad",
            page_limit_max="bad",
        )
        == 120
    )


def test_screen_stocks_uses_bounded_screener_view(monkeypatch):
    class FakeOverview:
        last_kwargs = None

        def __init__(self):
            self.filter_args = None

        def set_filter(self, filters_dict):
            self.filter_args = filters_dict

        def screener_view(self, **kwargs):
            FakeOverview.last_kwargs = kwargs
            # 120 rows simulates capped fetch.
            return pd.DataFrame({"Ticker": [f"T{i}" for i in range(120)]})

    overview_mod = types.ModuleType("finvizfinance.screener.overview")
    overview_mod.Overview = FakeOverview
    monkeypatch.setitem(sys.modules, "finvizfinance.screener.overview", overview_mod)
    monkeypatch.setattr(svc, "_apply_finvizfinance_timeout_patch", lambda: None)
    monkeypatch.setattr(svc, "_FINVIZ_SCREENER_MAX_ROWS", 120)
    monkeypatch.setattr(svc, "_FINVIZ_PAGE_LIMIT_MAX", 500)

    result = svc.screen_stocks(filters={"Sector": "Technology"}, limit=50, page=3, view="overview")

    assert result.get("success") is True
    assert result.get("count") == 20
    assert result.get("page") == 3
    assert result.get("truncated") is True
    assert FakeOverview.last_kwargs is not None
    assert int(FakeOverview.last_kwargs.get("limit")) == 120
    assert int(FakeOverview.last_kwargs.get("sleep_sec")) == 0
    assert int(FakeOverview.last_kwargs.get("verbose")) == 0


def test_get_earnings_calendar_uses_earnings_contract_with_pagination(monkeypatch):
    class FakeEarnings:
        last_period = None

        def __init__(self, period):
            FakeEarnings.last_period = period
            self.df = pd.DataFrame(
                {
                    "Ticker": [f"E{i}" for i in range(120)],
                    "Earnings": ["Mon"] * 120,
                }
            )

    earnings_mod = types.ModuleType("finvizfinance.earnings")
    earnings_mod.Earnings = FakeEarnings
    monkeypatch.setitem(sys.modules, "finvizfinance.earnings", earnings_mod)
    monkeypatch.setattr(svc, "_apply_finvizfinance_timeout_patch", lambda: None)
    monkeypatch.setattr(svc, "_FINVIZ_PAGE_LIMIT_MAX", 500)

    result = svc.get_earnings_calendar(period="This Week", limit=50, page=3)

    assert result.get("success") is True
    assert result.get("count") == 20
    assert result.get("page") == 3
    assert result.get("pages") == 3
    assert result.get("truncated") is False
    assert FakeEarnings.last_period == "This Week"


# ---------------------------------------------------------------------------
# Finviz HTTP session lifecycle tests
# ---------------------------------------------------------------------------


def test_build_finviz_session_sets_user_agent():
    session = finviz_client._build_finviz_session()
    assert session.headers.get("User-Agent") == "Mozilla/5.0"
    session.close()


def test_reset_finviz_session_clears_singleton(monkeypatch):
    """After reset, the next _build will create a fresh session."""
    sentinel = finviz_client._build_finviz_session()
    monkeypatch.setattr(finviz_client, "_FINVIZ_HTTP_SESSION", sentinel)

    finviz_client._reset_finviz_session()
    assert finviz_client._FINVIZ_HTTP_SESSION is None


def test_reset_finviz_session_tolerates_already_none():
    """Reset is safe to call even when no session has been created."""
    original = finviz_client._FINVIZ_HTTP_SESSION
    try:
        finviz_client._FINVIZ_HTTP_SESSION = None
        finviz_client._reset_finviz_session()
        assert finviz_client._FINVIZ_HTTP_SESSION is None
    finally:
        finviz_client._FINVIZ_HTTP_SESSION = original


def test_finviz_http_get_uses_build_session(monkeypatch):
    """When the session is None and requests.get is not patched, _build_finviz_session is called."""
    calls = {"built": 0, "get": []}

    class FakeSession:
        headers = {}

        def update(self, _):
            pass

        def get(self, url, **kwargs):
            calls["get"].append(url)
            return "ok"

    fake = FakeSession()
    fake.headers = {"User-Agent": "Mozilla/5.0"}

    def fake_build():
        calls["built"] += 1
        return fake

    monkeypatch.setattr(finviz_client, "_FINVIZ_HTTP_SESSION", None)
    monkeypatch.setattr(finviz_client, "_build_finviz_session", fake_build)
    # Ensure the monkeypatch hook is NOT triggered:
    import requests as _req
    monkeypatch.setattr(_req, "get", _req.api.get)

    result = finviz_client.finviz_http_get(
        "https://example.test", headers={}, params={}, timeout=5.0
    )

    assert calls["built"] == 1
    assert calls["get"] == ["https://example.test"]
    assert result == "ok"
