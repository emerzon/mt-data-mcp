from __future__ import annotations

from mtdata.utils.level_confluence import build_level_confluence_payload


def test_confluence_clusters_pivot_sr_and_fibonacci_levels():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="auto",
        reference_price=1.08,
        tolerance_pct=0.001,
        pivot_methods=[
            {
                "method": "classic",
                "levels": {"R2": 1.0850},
                "pivot": 1.08,
            }
        ],
        support_resistance_payload={
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "auto",
            "levels": [
                {
                    "type": "resistance",
                    "value": 1.0853,
                    "score": 5.0,
                    "touches": 3,
                }
            ],
            "fibonacci": {
                "levels": [
                    {
                        "label": "61.8%",
                        "ratio": 0.618,
                        "kind": "retracement",
                        "value": 1.0848,
                    }
                ]
            },
        },
        max_distance_pct=1.0,
        detail="standard",
    )

    assert payload["success"] is True
    assert payload["levels"]
    cluster = payload["levels"][0]
    assert cluster["source_families"] == [
        "pivot_formula",
        "swing_fibonacci",
        "touch_derived",
    ]
    assert cluster["role"] == "above"
    assert {source["source"] for source in cluster["sources"]} == {
        "pivot",
        "support_resistance",
        "fibonacci",
    }
    assert cluster["reasons"]


def test_confluence_compact_omits_verbose_source_narration():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.08,
        tolerance_pct=0.001,
        pivot_methods=[
            {
                "method": "classic",
                "levels": {"R1": 1.0802, "R2": 1.0803},
                "pivot": 1.08,
            }
        ],
        support_resistance_payload={
            "levels": [{"type": "support", "value": 1.0801, "score": 4, "touches": 2}]
        },
        detail="compact",
    )

    cluster = payload["levels"][0]
    assert "reasons" not in cluster
    assert "source_labels" not in cluster
    assert "sources" not in cluster
    assert "score_components" not in cluster
    assert cluster["source_count"] == 3
    assert payload["tolerance"]["fraction"] == 0.001
    assert "input_pct" not in payload["tolerance"]
    assert "detail" not in payload
    assert "units" not in payload
    assert "max_distance_pct" not in payload
    assert "min_source_families" not in payload


def test_confluence_standard_keeps_units_and_filter_context():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.08,
        tolerance_pct=0.001,
        pivot_methods=[{"method": "classic", "levels": {"PP": 1.0801}}],
        support_resistance_payload={"levels": []},
        max_distance_pct=2.0,
        min_source_families=1,
        detail="standard",
    )

    assert payload["detail"] == "standard"
    assert payload["units"]["tolerance.pct_points"] == "percentage_points (1.0 = 1%)"
    assert payload["max_distance_pct"] == 2.0
    assert payload["min_source_families"] == 1


def test_pivot_original_resistance_below_reference_is_role_below():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.10,
        tolerance_pct=0.001,
        pivot_methods=[
            {
                "method": "classic",
                "levels": {"R1": 1.09},
                "pivot": 1.085,
            }
        ],
        support_resistance_payload={"levels": []},
        max_distance_pct=2.0,
        detail="full",
    )

    candidate = payload["candidates"][0]
    assert candidate["family"] == "pivot_formula"
    assert candidate["role"] == "below"
    assert "Classic R1" in candidate["label"]


def test_single_family_clusters_are_returned_but_score_lower_than_multi_family():
    single = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.08,
        tolerance_pct=0.001,
        pivot_methods=[{"method": "classic", "levels": {"PP": 1.0801}}],
        support_resistance_payload={"levels": []},
        detail="standard",
    )
    multi = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.08,
        tolerance_pct=0.001,
        pivot_methods=[{"method": "classic", "levels": {"PP": 1.0801}}],
        support_resistance_payload={
            "levels": [{"type": "support", "value": 1.0802, "score": 4, "touches": 2}]
        },
        detail="standard",
    )

    assert single["levels"]
    assert multi["levels"]
    assert single["levels"][0]["source_families"] == ["pivot_formula"]
    assert multi["levels"][0]["score"] > single["levels"][0]["score"]
