from __future__ import annotations

from mtdata.core.temporal import _parse_weekday


def test_parse_weekday_numeric_modes_and_aliases() -> None:
    assert _parse_weekday("0") == 0
    assert _parse_weekday("6") == 6
    assert _parse_weekday("7") == 6
    assert _parse_weekday("1") == 1
    assert _parse_weekday("Mon") == 0
