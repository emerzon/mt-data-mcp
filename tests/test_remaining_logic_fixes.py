from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Iterator, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.modules.setdefault("MetaTrader5", MagicMock())

from mtdata.forecast.backtest import forecast_backtest
from mtdata.forecast.forecast import _create_dimred_reducer
from mtdata.forecast.forecast_engine import _apply_dimensionality_reduction
from mtdata.services.data_service import fetch_ticks
from mtdata.utils.utils import _format_time_minimal


@contextmanager
def _mock_symbol_ready_guard(*args: Any, **kwargs: Any) -> Iterator[Tuple[None, MagicMock]]:
    yield None, MagicMock()


def test_selectkbest_dimred_reduces_without_target_y() -> None:
    X = pd.DataFrame(
        {
            "a": [1.0, 1.0, 1.0, 1.0],
            "b": [1.0, 2.0, 3.0, 4.0],
            "c": [10.0, 20.0, 30.0, 40.0],
        }
    )
    out = _apply_dimensionality_reduction(X, "selectkbest", {"k": 2})
    assert out.shape[1] == 2


def test_wrapper_selectkbest_reducer_no_y_dependency() -> None:
    reducer, meta = _create_dimred_reducer("selectkbest", {"k": 1})
    arr = np.asarray([[1.0, 1.0], [2.0, 3.0], [3.0, 5.0]], dtype=float)
    out = reducer.fit_transform(arr)
    assert out.shape == (3, 1)
    assert meta.get("k") == 1


def test_backtest_vol_proxy_not_mutated_across_anchors() -> None:
    times = np.arange(1700000000, 1700000000 + 80 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 80, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})
    anchors = [_format_time_minimal(float(times[60])), _format_time_minimal(float(times[65]))]
    params_per_method = {"ewma": {"proxy": "abs_return"}}

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast_volatility",
        return_value={"horizon_sigma_return": 0.1},
    ) as mock_vol:
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=2,
            methods=["ewma"],
            anchors=anchors,
            quantity="volatility",
            params_per_method=params_per_method,
        )

    assert res.get("success") is True
    proxies = [c.kwargs.get("proxy") for c in mock_vol.call_args_list]
    assert proxies == ["abs_return", "abs_return"]
    assert params_per_method["ewma"]["proxy"] == "abs_return"


@patch("mtdata.services.data_service._symbol_ready_guard", _mock_symbol_ready_guard)
@patch("mtdata.services.data_service._mt5_copy_ticks_range")
def test_fetch_ticks_select_simplify_mode_no_nameerror(mock_copy_ticks: MagicMock) -> None:
    now = datetime.utcnow()
    ticks = []
    for i in range(12):
        t = now - timedelta(seconds=12 - i)
        ticks.append(
            {
                "time": t.timestamp(),
                "bid": 1.1 + i * 0.0001,
                "ask": 1.1001 + i * 0.0001,
                "last": 1.1 + i * 0.0001,
                "volume": 1.0,
                "flags": 1,
                "volume_real": 0.0,
            }
        )
    mock_copy_ticks.return_value = ticks

    res = fetch_ticks(
        symbol="EURUSD",
        limit=10,
        output="rows",
        simplify={"mode": "select", "points": 5},
    )

    assert res.get("success") is True
    assert "error" not in res
