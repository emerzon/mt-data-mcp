from datetime import timedelta, tzinfo
from types import SimpleNamespace

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


def test_mt5_config_ignores_invalid_login_and_warns(monkeypatch, caplog):
    monkeypatch.setenv("MT5_LOGIN", "not-a-number")
    monkeypatch.setenv("MT5_PASSWORD", "secret")
    monkeypatch.setenv("MT5_SERVER", "Demo-Server")
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False)

    with caplog.at_level("WARNING"):
        conf = cfg.MT5Config()

    assert conf.get_login() is None
    assert conf.has_credentials() is False
    assert any("Invalid MT5_LOGIN" in record.message for record in caplog.records)


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


def test_mt5_config_autodetects_client_timezone_when_env_unset(monkeypatch):
    sentinel_tz = object()
    monkeypatch.delenv("CLIENT_TZ", raising=False)
    monkeypatch.delenv("MT5_CLIENT_TZ", raising=False)
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False)
    monkeypatch.setattr(cfg, "_detect_local_client_tz", lambda: sentinel_tz)

    conf = cfg.MT5Config()

    assert conf.client_tz_name is None
    assert conf.get_client_tz() is sentinel_tz


def test_detect_local_client_tz_prefers_tzlocal(monkeypatch):
    sentinel_tz = object()
    monkeypatch.setattr(cfg, "tzlocal", SimpleNamespace(get_localzone=lambda: sentinel_tz))
    monkeypatch.setattr(cfg, "dateutil_tz", None)

    assert cfg._detect_local_client_tz() is sentinel_tz


def test_mt5_config_handles_invalid_timeout_with_warning(monkeypatch, caplog):
    monkeypatch.setenv("MT5_TIMEOUT", "not-a-number")
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False)

    with caplog.at_level("WARNING"):
        conf = cfg.MT5Config()

    assert conf.timeout == 30
    assert any("Invalid MT5_TIMEOUT" in record.message for record in caplog.records)


def test_mt5_config_reads_broker_time_check_settings(monkeypatch):
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setenv("MTDATA_BROKER_TIME_CHECK", "true")
    monkeypatch.setenv("MTDATA_BROKER_TIME_CHECK_TTL_SECONDS", "300")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False)

    conf = cfg.MT5Config()

    assert conf.broker_time_check_enabled is True
    assert conf.broker_time_check_ttl_seconds == 300


def test_mt5_config_handles_invalid_broker_time_check_ttl(monkeypatch):
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setenv("MTDATA_BROKER_TIME_CHECK", "off")
    monkeypatch.setenv("MTDATA_BROKER_TIME_CHECK_TTL_SECONDS", "bad")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False)

    conf = cfg.MT5Config()

    assert conf.broker_time_check_enabled is False
    assert conf.broker_time_check_ttl_seconds == 60


def test_load_environment_logs_reload_failures(monkeypatch, caplog):
    monkeypatch.setattr(cfg, "_ENV_LOADED", False)
    monkeypatch.setattr(
        cfg,
        "mt5_config",
        SimpleNamespace(reload_from_env=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("reload exploded"))),
    )

    with caplog.at_level("WARNING"):
        cfg.load_environment(force=True)

    assert any(
        "Failed to reload MT5 configuration from environment: reload exploded" in record.message
        for record in caplog.records
    )
