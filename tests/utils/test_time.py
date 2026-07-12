from datetime import timedelta, timezone

import mtdata.utils.time as time_utils
from mtdata.utils.time import format_epoch_utc


def test_format_epoch_utc_uses_second_resolution() -> None:
    assert format_epoch_utc(1000.75) == "1970-01-01T00:16:40Z"


def test_format_epoch_utc_rejects_invalid_values() -> None:
    assert format_epoch_utc(None) is None
    assert format_epoch_utc("not-an-epoch") is None


def test_client_local_formatters_share_resolved_timezone(monkeypatch) -> None:
    client_tz = timezone(timedelta(hours=5, minutes=30))
    monkeypatch.setattr(time_utils, "_resolve_client_tz", lambda: client_tz)

    assert time_utils._use_client_tz() is True
    assert time_utils._format_time_minimal_local(0) == "1970-01-01T05:30+05:30"
    assert time_utils._format_time_explicit_local(0) == "1970-01-01T05:30+05:30"
