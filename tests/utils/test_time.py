from mtdata.utils.time import format_epoch_utc


def test_format_epoch_utc_uses_second_resolution() -> None:
    assert format_epoch_utc(1000.75) == "1970-01-01T00:16:40Z"


def test_format_epoch_utc_rejects_invalid_values() -> None:
    assert format_epoch_utc(None) is None
    assert format_epoch_utc("not-an-epoch") is None
