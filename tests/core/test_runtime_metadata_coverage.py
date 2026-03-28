from __future__ import annotations

from types import SimpleNamespace
from zoneinfo import ZoneInfo

from mtdata.core.runtime_metadata import build_runtime_timezone_meta


def _make_config(
    *,
    server_tz_name=None,
    client_tz_name=None,
    server_tz=None,
    client_tz=None,
    offset_seconds=0,
):
    return SimpleNamespace(
        server_tz_name=server_tz_name,
        client_tz_name=client_tz_name,
        get_server_tz=lambda: server_tz,
        get_client_tz=lambda: client_tz,
        get_time_offset_seconds=lambda: offset_seconds,
    )


def test_omits_unknown_server_and_client_now_fields() -> None:
    cfg = _make_config()

    result = build_runtime_timezone_meta({}, mt5_config=cfg)

    assert isinstance(result["utc"]["now"], str)
    assert result["utc"]["tz"] == "UTC"
    assert "now" not in result.get("server", {})
    assert "now" not in result.get("client", {})
    assert "offset_seconds" not in result.get("server", {})


def test_api_shape_uses_shared_schema_without_local_or_now() -> None:
    cfg = _make_config(
        server_tz_name="Europe/Nicosia",
        client_tz_name="US/Central",
        server_tz=ZoneInfo("Europe/Nicosia"),
        client_tz=ZoneInfo("US/Central"),
        offset_seconds=7200,
    )

    result = build_runtime_timezone_meta(
        {"timezone": "UTC"},
        mt5_config=cfg,
        include_local=False,
        include_now=False,
    )

    assert result == {
        "utc": {
            "tz": "UTC",
        },
        "server": {
            "source": "MT5_SERVER_TZ",
            "tz": "Europe/Nicosia",
            "offset_seconds": 7200,
        },
        "client": {
            "tz": "US/Central",
        },
    }


def test_uses_iana_name_for_client_timezone_when_available() -> None:
    cfg = _make_config(client_tz=ZoneInfo("America/Chicago"))

    result = build_runtime_timezone_meta({}, mt5_config=cfg)

    assert result["client"]["tz"] == "America/Chicago"
    assert result["utc"]["tz"] == "UTC"
