from __future__ import annotations

from mtdata.core.temporal import (
    _compact_temporal_payload,
    _compact_temporal_stats,
    _parse_weekday,
)


def test_parse_weekday_numeric_modes_and_aliases() -> None:
    assert _parse_weekday("0") == 0
    assert _parse_weekday("6") == 6
    assert _parse_weekday("7") == 6
    assert _parse_weekday("1") == 1
    assert _parse_weekday("Mon") == 0


def test_compact_temporal_stats_keep_group_key() -> None:
    result = _compact_temporal_stats(
        {
            "group": 8,
            "group_label": "08:00",
            "bars": 24,
            "avg_return": 0.12,
            "win_rate_pct": 55.0,
            "volatility": 0.03,
        }
    )

    assert result["group"] == 8
    assert result["group_label"] == "08:00"


def test_compact_temporal_payload_best_keeps_group_key() -> None:
    result = _compact_temporal_payload(
        {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "group_by": "hour",
            "return_mode": "pct",
            "units": {"returns": "percentage_points (1.0 = 1%)"},
            "timezone": "UTC",
            "lookback": 100,
            "lookback_source": "request",
            "bars": 48,
            "start": "2024-01-01 00:00",
            "end": "2024-01-03 00:00",
            "groups_analyzed": 2,
            "groups_excluded": 0,
            "groups": [
                {"group": 7, "group_label": "07:00", "bars": 24, "avg_return": 0.1},
                {"group": 8, "group_label": "08:00", "bars": 24, "avg_return": 0.2},
            ],
        }
    )

    assert result["groups"][0]["group"] == 7
    assert result["best"]["group"] == 8
