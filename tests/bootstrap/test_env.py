from mtdata.bootstrap.env import get_bool_env


def test_get_bool_env_uses_default_when_unset_or_invalid(monkeypatch, caplog) -> None:
    monkeypatch.delenv("MTDATA_TEST_BOOL", raising=False)
    assert get_bool_env("MTDATA_TEST_BOOL", default=True) is True

    monkeypatch.setenv("MTDATA_TEST_BOOL", "invalid")
    assert get_bool_env("MTDATA_TEST_BOOL", default=True) is True
    assert "Invalid boolean MTDATA_TEST_BOOL='invalid'" in caplog.text


def test_get_bool_env_accepts_project_truthy_values(monkeypatch) -> None:
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("MTDATA_TEST_BOOL", value)
        assert get_bool_env("MTDATA_TEST_BOOL") is True


def test_get_bool_env_accepts_project_false_values(monkeypatch) -> None:
    for value in ("0", "false", "NO", "off"):
        monkeypatch.setenv("MTDATA_TEST_BOOL", value)
        assert get_bool_env("MTDATA_TEST_BOOL", default=True) is False
