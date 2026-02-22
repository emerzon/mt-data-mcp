from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast import target_builder as tb


def _ohlc_df(n: int = 6) -> pd.DataFrame:
    close = np.linspace(10.0, 15.0, n)
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.4,
            "close": close,
            "volume": np.linspace(100, 200, n),
        }
    )


def test_resolve_alias_base_variants_and_missing_inputs():
    arrs = {
        "open": np.array([1.0, 2.0]),
        "high": np.array([3.0, 4.0]),
        "low": np.array([0.5, 1.5]),
        "close": np.array([2.0, 3.0]),
    }

    assert np.allclose(tb.resolve_alias_base(arrs, "typical"), np.array([(3.0 + 0.5 + 2.0) / 3.0, (4.0 + 1.5 + 3.0) / 3.0]))
    assert np.allclose(tb.resolve_alias_base(arrs, "hl2"), np.array([(3.0 + 0.5) / 2.0, (4.0 + 1.5) / 2.0]))
    assert np.allclose(tb.resolve_alias_base(arrs, "ohlc4"), np.array([(1.0 + 3.0 + 0.5 + 2.0) / 4.0, (2.0 + 4.0 + 1.5 + 3.0) / 4.0]))
    assert tb.resolve_alias_base({"high": arrs["high"]}, "hl2") is None
    assert tb.resolve_alias_base(arrs, "unknown") is None


def test_build_target_series_legacy_price_and_return():
    df = _ohlc_df(5)

    y_price, info_price = tb.build_target_series(df, base_col="close", target_spec=None, legacy_target="price")
    assert np.allclose(y_price, df["close"].to_numpy())
    assert info_price == {"mode": "price", "base": "close", "transform": "none"}

    y_ret, info_ret = tb.build_target_series(df, base_col="close", target_spec=None, legacy_target="return")
    assert y_ret.shape[0] == 5
    assert info_ret == {"mode": "return", "base": "close", "transform": "log_return"}


def test_build_target_series_custom_with_indicators_and_transforms(monkeypatch):
    df = _ohlc_df(8)
    calls = {"parse": 0, "apply": 0}

    monkeypatch.setattr(tb, "_parse_ti_specs_util", lambda spec: calls.__setitem__("parse", calls["parse"] + 1) or [{"ti": spec}])
    monkeypatch.setattr(tb, "_apply_ta_indicators_util", lambda *args, **kwargs: calls.__setitem__("apply", calls["apply"] + 1))

    y_diff, info_diff = tb.build_target_series(
        df,
        base_col="close",
        target_spec={"base": "close", "transform": "diff", "k": 1, "indicators": "sma:close:5"},
    )
    assert y_diff.shape[0] == len(df)
    assert info_diff["mode"] == "custom"
    assert info_diff["transform"] == "diff(k=1)"
    assert calls["parse"] == 1
    assert calls["apply"] == 1

    y_log, info_log = tb.build_target_series(
        df,
        base_col="close",
        target_spec={"base": "close", "transform": "log_return", "k": 1},
    )
    assert y_log.shape[0] == len(df)
    assert info_log["transform"] == "log_return(k=1)"

    y_alias, info_alias = tb.build_target_series(
        df,
        base_col="close",
        target_spec={"base": "typical", "transform": "none"},
    )
    assert y_alias.shape[0] == len(df)
    assert info_alias["base"] == "typical"


def test_build_target_series_invalid_base_raises_value_error():
    df = _ohlc_df(5)

    with pytest.raises(ValueError, match="Base column 'does_not_exist' not found"):
        tb.build_target_series(
            df,
            base_col="close",
            target_spec={"base": "does_not_exist", "transform": "none"},
        )


def test_aggregate_horizon_target_passthrough_modes():
    y = np.array([1.0, 2.0, 3.0], dtype=float)

    out, info = tb.aggregate_horizon_target(y, horizon=2, agg_spec=None)
    assert np.allclose(out, y)
    assert info == {"agg": "last", "normalize": "none"}

    for agg in ("mean", "sum", "slope", "vol", "unknown"):
        out, info = tb.aggregate_horizon_target(y, horizon=2, agg_spec=agg, normalize="per_bar")
        assert np.allclose(out, y)
        assert info["agg"] == agg
        assert info["normalize"] == "per_bar"
