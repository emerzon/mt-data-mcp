"""Smoke test for clustering-based regime detection on synthetic data.

Runs `mtdata.core.regime.regime_detect` with `method="clustering"` while patching
the MT5 history fetch to avoid external dependencies.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from mtdata.core.regime import regime_detect


def _mock_fetch_history(symbol: str, timeframe: str, limit: int, as_of=None) -> pd.DataFrame:
    # Create a synthetic price series with two volatility regimes.
    rng = np.random.default_rng(12345)
    n = int(limit)
    t = pd.date_range("2024-01-01", periods=n, freq="h")

    split = min(400, n)
    y = np.empty(n, dtype=float)
    y[:split] = 100.0 + np.cumsum(rng.normal(0.0, 0.5, split))
    if split < n:
        y[split:] = y[split - 1] + np.cumsum(rng.normal(0.0, 2.0, n - split))

    t_seconds = (t.astype("int64") // 10**9).astype(np.int64)
    return pd.DataFrame(
        {
            "time": t_seconds,
            "close": y,
            "open": y,
            "high": y,
            "low": y,
            "tick_volume": 100,
            "spread": 1,
            "real_volume": 100,
        }
    )


@patch("mtdata.core.regime._fetch_history", side_effect=_mock_fetch_history)
def main(_mocked_fetch) -> int:
    try:
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            limit=800,
            method="clustering",
            params={"window_size": 20, "k_regimes": 2},
            output="full",
            __cli_raw=True,
        )
    except ImportError as ex:
        print(f"SKIP: clustering deps missing ({ex})")
        return 0

    if isinstance(res, dict) and res.get("error"):
        print(f"FAILED: {res['error']}")
        return 1

    states = np.asarray(res.get("state", []), dtype=int)
    if states.size == 0:
        print("FAILED: missing state output")
        return 1

    valid = states[states >= 0]
    if valid.size == 0:
        print("FAILED: all states undefined (-1)")
        return 1

    unique = np.unique(valid)
    if unique.size < 2:
        print(f"FAILED: expected >=2 regimes, got {unique.tolist()}")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
