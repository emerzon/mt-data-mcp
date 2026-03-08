from __future__ import annotations

from unittest.mock import patch
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from mtdata.core.web_api import app


client = TestClient(app)


def test_timeframes_available_on_legacy_and_versioned_routes() -> None:
    legacy = client.get("/api/timeframes")
    versioned = client.get("/api/v1/timeframes")

    assert legacy.status_code == 200
    assert versioned.status_code == 200
    assert legacy.json() == versioned.json()


def test_health_available_on_versioned_route() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"service": "mtdata-webui", "status": "ok"}


def test_history_available_on_versioned_route() -> None:
    payload = {
        "success": True,
        "data": [
            {"time": 1735689600.0, "close": 1.1},
            {"time": 1735693200.0, "close": 1.2},
        ],
        "last_candle_open": False,
    }

    with patch("mtdata.core.web_api.mt5_connection._ensure_connection", return_value=True), patch(
        "mtdata.core.web_api._fetch_candles_impl", return_value=payload
    ), patch("mtdata.core.web_api.mt5_config") as mock_cfg:
        mock_cfg.server_tz_name = "Europe/Nicosia"
        mock_cfg.client_tz_name = None
        mock_cfg.get_server_tz.return_value = ZoneInfo("Europe/Nicosia")
        mock_cfg.get_client_tz.return_value = None
        mock_cfg.get_time_offset_seconds.return_value = 7200
        response = client.get("/api/v1/history", params={"symbol": "EURUSD", "timeframe": "H1", "limit": 2})

    assert response.status_code == 200
    assert response.json() == {
        "bars": payload["data"],
        "meta": {
            "runtime": {
                "timezone": {
                    "output": {"tz": {"hint": "UTC"}},
                    "server": {
                        "source": "MT5_SERVER_TZ",
                        "tz": {
                            "configured": "Europe/Nicosia",
                            "resolved": "Europe/Nicosia",
                            "offset_seconds": 7200,
                        },
                    },
                },
            },
        },
    }
