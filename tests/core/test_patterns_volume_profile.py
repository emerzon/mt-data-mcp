from mtdata.core.patterns_requests import PatternsDetectRequest
from mtdata.core.patterns_use_cases import PatternsDetectDeps, run_patterns_detect


def test_fractal_mode_opt_in_volume_profile_enriches_rows():
    deps = PatternsDetectDeps(
        compact_patterns_payload=lambda *args, **kwargs: {},
        fetch_pattern_data=lambda *args, **kwargs: (_FakeFrame(), None),
        classic_cfg_cls=object,
        elliott_cfg_cls=object,
        fractal_cfg_cls=_FakeFractalCfg,
        harmonic_cfg_cls=object,
        apply_config_to_obj=lambda cfg, config: [
            key
            for key in (config or {})
            if key not in {"volume_profile", "volume_profile_tolerance_points"}
        ],
        select_classic_engines=lambda *args, **kwargs: [],
        available_classic_engines=lambda: [],
        run_classic_engine=lambda *args, **kwargs: [],
        resolve_engine_weights=lambda *args, **kwargs: {},
        merge_classic_ensemble=lambda *args, **kwargs: [],
        enrich_classic_patterns=lambda *args, **kwargs: [],
        summarize_engine_findings=lambda *args, **kwargs: {},
        summarize_pattern_bias=lambda rows: {},
        build_pattern_response=lambda symbol, timeframe, limit, mode, patterns, *args, **kwargs: {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "lookback": limit,
            "mode": mode,
            "patterns": list(patterns),
            "n_patterns": len(patterns),
        },
        format_elliott_patterns=lambda *args, **kwargs: [],
        format_fractal_patterns=lambda *args, **kwargs: [
            {
                "name": "Bullish Fractal",
                "status": "forming",
                "level_price": 1.0850,
            }
        ],
        format_harmonic_patterns=lambda *args, **kwargs: [],
        detect_candlestick_patterns=lambda *args, **kwargs: {},
        elliott_timeframe_suggestion=lambda *args, **kwargs: "",
        resolve_elliott_scan_timeframes=lambda *args, **kwargs: [],
        validate_classic_config_errors=lambda cfg: [],
        validate_fractal_config=lambda cfg: [],
        validate_harmonic_config=lambda cfg: [],
        summarize_fractal_context=lambda rows: {},
        compute_volume_profile_payload=lambda **kwargs: {
            "success": True,
            "price_point": 0.0001,
            "levels": [
                {
                    "level": "POC",
                    "type": "volume_poc",
                    "price": 1.0855,
                    "volume": 10,
                }
            ],
        },
        annotate_level_confluence=_annotate,
        format_time_minimal=lambda value: str(value),
        to_float_np=lambda value: value,
    )

    result = run_patterns_detect(
        PatternsDetectRequest(
            symbol="EURUSD",
            timeframe="H1",
            mode="fractal",
            config={"volume_profile": True, "volume_profile_tolerance_points": 6},
        ),
        deps,
    )

    assert result["volume_profile"]["success"] is True
    assert result["patterns"][0]["volume_profile_confluence"]["level"] == "POC"
    assert result["n_volume_profile_confluences"] == 1


class _FakeFractalCfg:
    pass


class _FakeFrame:
    pass


def _annotate(rows, levels, **kwargs):
    out = []
    for row in rows:
        enriched = dict(row)
        enriched["volume_profile_confluence"] = {
            "level": levels[0]["level"],
            "within_tolerance": True,
        }
        out.append(enriched)
    return out
