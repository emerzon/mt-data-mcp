from datetime import timedelta, tzinfo
from mtdata.core import config as cfg


class _FixedOffsetTZ(tzinfo):
    def __init__(self, hours):
        self._delta = timedelta(hours=hours)

    def utcoffset(self, dt):
        return self._delta

    def dst(self, dt):
        return timedelta(0)


class _FakePytz:
    def timezone(self, name):
        if name == "bad/tz":
            raise ValueError("invalid tz")
        if name == "client/three":
            return _FixedOffsetTZ(3)
        return _FixedOffsetTZ(2)


def test_mt5_config_credentials_and_login_parsing(monkeypatch):
    monkeypatch.setenv("MT5_LOGIN", "123456")
    monkeypatch.setenv("MT5_PASSWORD", "secret")
    monkeypatch.setenv("MT5_SERVER", "Demo-Server")
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False)

    conf = cfg.MT5Config()

    assert conf.get_login() == 123456
    assert conf.get_password() == "secret"
    assert conf.get_server() == "Demo-Server"
    assert conf.has_credentials() is True


def test_mt5_config_warns_once_when_timezone_info_missing(monkeypatch, caplog):
    monkeypatch.delenv("MT5_SERVER_TZ", raising=False)
    monkeypatch.delenv("MT5_TIME_OFFSET_MINUTES", raising=False)
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False)

    with caplog.at_level("WARNING", logger="mtdata.core.config"):
        cfg.MT5Config()
        cfg.MT5Config()

    warnings = [r.message for r in caplog.records if "MT5_SERVER_TZ or MT5_TIME_OFFSET_MINUTES not set" in r.message]
    assert len(warnings) == 1


def test_get_time_offset_seconds_prefers_explicit_minutes(monkeypatch):
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "90")
    monkeypatch.setenv("MT5_SERVER_TZ", "any/tz")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False)

    conf = cfg.MT5Config()

    assert conf.get_time_offset_seconds() == 5400


def test_get_time_offset_seconds_uses_server_timezone_when_offset_zero(monkeypatch):
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setenv("MT5_SERVER_TZ", "server/two")
    monkeypatch.setattr(cfg, "pytz", _FakePytz())
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False)

    conf = cfg.MT5Config()

    assert conf.get_time_offset_seconds() == 7200
    assert conf.get_server_tz() is not None


def test_mt5_config_handles_invalid_offset_and_bad_timezone(monkeypatch):
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "not-a-number")
    monkeypatch.setenv("MT5_SERVER_TZ", "bad/tz")
    monkeypatch.setenv("CLIENT_TZ", "bad/tz")
    monkeypatch.setattr(cfg, "pytz", _FakePytz())
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False)

    conf = cfg.MT5Config()

    assert conf.time_offset_minutes == 0
    assert conf.get_time_offset_seconds() == 0
    assert conf.get_server_tz() is None
    assert conf.get_client_tz() is None
