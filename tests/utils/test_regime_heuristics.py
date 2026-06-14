import numpy as np

from mtdata.utils.regime_heuristics import infer_market_regime


def test_infer_market_regime_detects_direct_trend() -> None:
    result = infer_market_regime(np.linspace(100.0, 120.0, 80))

    assert result is not None
    assert result["state"] == "trending"
    assert result["direction"] == "bullish"
    assert result["window_bars"] == 80


def test_infer_market_regime_rejects_short_series() -> None:
    assert infer_market_regime([1.0] * 19) is None
