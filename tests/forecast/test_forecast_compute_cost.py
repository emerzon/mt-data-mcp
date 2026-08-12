from mtdata.core.forecast import _forecast_compute_cost


def test_optimize_hints_compute_cost_uses_request_defaults() -> None:
    assert _forecast_compute_cost("forecast_optimize_hints", {}) == {
        "unit": "rolling_backtests",
        "estimated": 190,
        "drivers": (
            "(population+generations*(population-2))*steps "
            "(method/timeframe sampled once per candidate)"
        ),
    }
