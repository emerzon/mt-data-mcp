from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from mtdata.core.regime import regime_detect


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _sample_df(n: int = 80) -> pd.DataFrame:
    t = np.arange(float(n))
    close = 100.0 + np.linspace(0.0, 1.0, n)
    return pd.DataFrame({"time": t, "close": close})


def test_regime_detect_defaults_to_compact_output() -> None:
    raw = _unwrap(regime_detect)
    cp = np.zeros(79, dtype=float)
    cp[-2] = 0.9

    with patch("mtdata.core.regime._fetch_history", return_value=_sample_df(80)), patch(
        "mtdata.core.regime._resolve_denoise_base_col",
        return_value="close",
    ), patch(
        "mtdata.core.regime._format_time_minimal",
        side_effect=lambda x: f"T{x}",
    ), patch(
        "mtdata.utils.regime.bocpd_gaussian",
        return_value={"cp_prob": cp},
    ):
        out = raw(symbol="EURUSD", timeframe="H1", limit=80, method="bocpd", threshold=0.5, lookback=20)

    assert out.get("success") is True
    assert "summary" in out
    assert out["summary"].get("lookback") == 20
    assert "regimes" in out
