from mtdata.utils.level_confluence import build_level_confluence_payload


def test_build_level_confluence_payload_includes_volume_profile_levels():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        pivot_methods=[
            {
                "method": "classic",
                "levels": {"PP": 1.1000},
            }
        ],
        support_resistance_payload={"levels": []},
        reference_price=1.1000,
        tolerance_points=10,
        price_increment=0.0001,
        max_distance_pct=1.0,
        min_source_families=1,
        detail="full",
        volume_profile_payload={
            "success": True,
            "source": "ticks",
            "volume_kind": "tick_count",
            "levels": [
                {
                    "level": "POC",
                    "type": "volume_poc",
                    "price": 1.1005,
                    "volume": 12,
                    "volume_share": 0.5,
                    "bucket_index": 5,
                }
            ],
            "diagnostics": {"bucket_count": 8},
        },
    )

    candidates = payload["candidates"]
    assert any(candidate["family"] == "volume_profile" for candidate in candidates)
    assert payload["volume_profile_diagnostics"]["bucket_count"] == 8


def test_build_level_confluence_payload_compact_keeps_contributing_source_quality():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        pivot_methods=[{"method": "classic", "levels": {"PP": 1.1000}}],
        support_resistance_payload={"levels": []},
        reference_price=1.1000,
        tolerance_points=10,
        price_increment=0.0001,
        max_distance_pct=1.0,
        detail="compact",
        volume_profile_payload={
            "success": True,
            "source": "ticks",
            "profile_source": "ticks",
            "volume_kind": "tick_count",
            "volume_profile_accuracy": "tick_precise",
            "is_synthetic": False,
            "levels": [
                {
                    "level": "POC",
                    "type": "volume_poc",
                    "price": 1.1005,
                    "volume": 12,
                    "volume_share": 0.5,
                    "bucket_index": 5,
                }
            ],
            "diagnostics": {"bucket_count": 8, "tick_rows": 50_000},
            "warnings": ["volume profile diagnostic warning"],
        },
    )

    assert payload["count"] == len(payload["levels"])
    assert "level_counts" not in payload
    assert "volume_profile_diagnostics" not in payload
    assert "volume_profile_warnings" not in payload
    assert payload["source_quality"] == {
        "volume_profile": {
            "source": "ticks",
            "volume_kind": "tick_count",
            "is_synthetic": False,
            "accuracy": "tick_precise",
            "warnings": ["volume profile diagnostic warning"],
        }
    }


def test_compact_confluence_discloses_synthetic_volume_profile_fallback():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        pivot_methods=[{"method": "classic", "levels": {"PP": 1.1000}}],
        support_resistance_payload={"levels": []},
        reference_price=1.1000,
        tolerance_points=10,
        price_increment=0.0001,
        detail="compact",
        volume_profile_payload={
            "success": True,
            "source": "m1_bars",
            "profile_source": "m1_bars",
            "volume_kind": "tick_volume",
            "volume_profile_accuracy": "approximated_from_m1_bars",
            "volume_source_quality": "estimated_m1_bar_proxy",
            "is_synthetic": True,
            "levels": [
                {"level": "POC", "type": "volume_poc", "price": 1.1005}
            ],
            "diagnostics": {
                "auto_fallback_reason": "requested window exceeds bounded tick window"
            },
            "warnings": [
                "Volume profile used M1-bar approximation instead of raw ticks."
            ],
        },
    )

    assert "volume_profile" in payload["levels"][0]["source_families"]
    assert payload["source_quality"]["volume_profile"] == {
        "source": "m1_bars",
        "volume_kind": "tick_volume",
        "is_synthetic": True,
        "accuracy": "approximated_from_m1_bars",
        "volume_source_quality": "estimated_m1_bar_proxy",
        "fallback_reason": "requested window exceeds bounded tick window",
        "warnings": [
            "Volume profile used M1-bar approximation instead of raw ticks."
        ],
    }
