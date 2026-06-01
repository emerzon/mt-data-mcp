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
