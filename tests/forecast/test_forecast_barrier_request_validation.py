import pytest
from pydantic import ValidationError

from mtdata.forecast.requests import (
    ForecastBarrierOptimizeRequest,
    ForecastBarrierProbRequest,
)


def _tp_sl_barrier(unit: str = "pct", take_profit: float = 0.5, stop_loss: float = 0.25):
    return {
        "kind": "tp_sl",
        "unit": unit,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
    }


def test_forecast_barrier_prob_request_rejects_removed_flat_barriers():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            barrier=_tp_sl_barrier(),
            tp_pct=0.5,
        )


def test_forecast_barrier_prob_request_allows_single_shared_unit_family():
    request = ForecastBarrierProbRequest(symbol="EURUSD", barrier=_tp_sl_barrier())

    assert request.tp_pct == 0.5
    assert request.sl_pct == 0.25


def test_forecast_barrier_prob_request_defaults_to_touch_aware_method():
    request = ForecastBarrierProbRequest(symbol="EURUSD", barrier=_tp_sl_barrier())

    assert request.method == "mc_gbm_bb"


def test_forecast_barrier_prob_request_uses_tick_fields_as_canonical_names():
    request = ForecastBarrierProbRequest(
        symbol="EURUSD",
        barrier=_tp_sl_barrier("ticks", 12.0, 9.0),
    )

    assert request.tp_ticks == 12.0
    assert request.sl_ticks == 9.0
    assert request.model_dump()["barrier"]["unit"] == "ticks"


def test_forecast_barrier_prob_request_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            barrier=_tp_sl_barrier(),
            tp_percent=0.5,
        )


def test_forecast_barrier_optimize_request_keeps_ticks_mode_canonical():
    request = ForecastBarrierOptimizeRequest(symbol="EURUSD", mode="ticks")

    assert request.mode == "ticks"
