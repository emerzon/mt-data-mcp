from mtdata.core.report.extras import (
    attach_optional_report_sections,
    compact_report_pattern_row,
    extract_report_pattern_rows,
    summarize_confluence_payload,
    summarize_news_payload,
    summarize_volume_profile_payload,
)


def test_extract_report_pattern_rows_prefers_top_patterns() -> None:
    rows = extract_report_pattern_rows(
        {
            "top_patterns": [{"name": "hammer", "match_score": 0.81}],
            "recent_patterns": [{"pattern": "doji"}],
        }
    )

    assert rows == [{"name": "hammer", "match_score": 0.81}]


def test_extract_report_pattern_rows_falls_back_to_recent_patterns() -> None:
    rows = extract_report_pattern_rows({"recent_patterns": [{"pattern": "doji"}]})
    assert rows == [{"pattern": "doji"}]


def test_compact_report_pattern_row_maps_live_compact_fields() -> None:
    item = compact_report_pattern_row(
        {"name": "engulfing", "direction": "bearish", "match_score": 0.7}
    )
    assert item["name"] == "engulfing"
    assert item["direction"] == "bearish"
    assert item["match_score"] == 0.7


def test_summarize_confluence_payload_keeps_nearest_levels() -> None:
    out = summarize_confluence_payload(
        {
            "reference_price": 1.1,
            "levels": [
                {
                    "price": 1.102,
                    "role": "above",
                    "score": 14.2,
                    "distance_pct": 0.18,
                    "source_families": ["pivot_formula", "touch_derived"],
                    "sources": [{"unused": True}],
                }
            ],
        }
    )
    assert out["reference_price"] == 1.1
    assert out["levels"] == [
        {
            "price": 1.102,
            "role": "above",
            "score": 14.2,
            "distance_pct": 0.18,
            "source_families": ["pivot_formula", "touch_derived"],
        }
    ]


def test_summarize_volume_profile_and_news_payloads() -> None:
    profile = summarize_volume_profile_payload(
        {"poc": 1.101, "vah": 1.105, "val": 1.098, "unused": True}
    )
    news = summarize_news_payload(
        {
            "upcoming_events": [{"title": "CPI", "when": "tomorrow"}],
            "related_news": [{"headline": "Fed speak"}],
        }
    )
    assert profile == {"poc": 1.101, "vah": 1.105, "val": 1.098}
    assert news["upcoming_events"] == [{"title": "CPI", "when": "tomorrow"}]
    assert news["related_news"] == [{"headline": "Fed speak"}]


def test_bounded_report_volume_profile_uses_only_explicit_window() -> None:
    calls = []

    def _call(_tool, **kwargs):
        calls.append(kwargs)
        return {"poc": 1.101, "vah": 1.105, "val": 1.098}

    report = {"sections": {}}
    attach_optional_report_sections(
        report,
        call=_call,
        symbol="EURUSD",
        timeframe="H4",
        params={"_report_execution_sections": ["volume_profile"]},
        start="2026-07-01",
        end="2026-08-01",
    )

    assert calls == [
        {
            "symbol": "EURUSD",
            "start": "2026-07-01",
            "end": "2026-08-01",
            "detail": "compact",
        }
    ]
    assert report["sections"]["volume_profile"]["poc"] == 1.101


def test_unbounded_report_volume_profile_uses_bar_window() -> None:
    calls = []

    def _call(_tool, **kwargs):
        calls.append(kwargs)
        return {"poc": 1.101, "vah": 1.105, "val": 1.098}

    attach_optional_report_sections(
        {"sections": {}},
        call=_call,
        symbol="EURUSD",
        timeframe="H4",
        params={
            "_report_execution_sections": ["volume_profile"],
            "volume_profile_lookback": 24,
        },
        start=None,
        end=None,
    )

    assert calls == [
        {
            "symbol": "EURUSD",
            "end": None,
            "timeframe": "H4",
            "lookback": 24,
            "detail": "compact",
        }
    ]
