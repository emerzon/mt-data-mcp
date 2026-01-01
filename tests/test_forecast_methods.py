#!/usr/bin/env python3
"""Lightweight forecast test runner.

Usage:
  python tests/test_forecast_methods.py EURUSD H1 12

Writes JSON output into tests/test_results/.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

from mtdata.core.forecast import forecast_generate
from mtdata.forecast.helpers import default_seasonality_period


def _usage() -> str:
    return "Usage: python tests/test_forecast_methods.py SYMBOL TIMEFRAME HORIZON [method ...]"


def _resolve_methods(args: List[str]) -> List[str]:
    if args:
        return [m.strip() for m in args if m.strip()]
    # Keep defaults lightweight and dependency-free.
    return ["theta", "naive", "drift", "seasonal_naive"]


def _run(symbol: str, timeframe: str, horizon: int, methods: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon": int(horizon),
        "methods": {},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    m_eff = default_seasonality_period(timeframe)
    successes = 0
    for method in methods:
        params = {}
        if method == "seasonal_naive":
            params["seasonality"] = int(m_eff)
        res = forecast_generate(
            symbol=symbol,
            timeframe=timeframe,
            method=method,
            horizon=int(horizon),
            params=params or None,
            __cli_raw=True,
        )
        out["methods"][method] = res
        if isinstance(res, dict) and not res.get("error"):
            successes += 1
    out["successes"] = successes
    out["failures"] = max(0, len(methods) - successes)
    return out


def main() -> int:
    if len(sys.argv) < 4:
        print(_usage())
        return 2
    symbol = str(sys.argv[1]).strip()
    timeframe = str(sys.argv[2]).strip().upper()
    try:
        horizon = int(sys.argv[3])
    except ValueError:
        print("HORIZON must be an integer.")
        return 2
    methods = _resolve_methods(sys.argv[4:])

    result = _run(symbol, timeframe, horizon, methods)
    out_dir = os.path.join(os.path.dirname(__file__), "test_results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fname = f"{symbol}_{timeframe}_{horizon}_{ts}.json"
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True, indent=2)

    print(f"Wrote results to {out_path}")
    if result.get("successes", 0) <= 0:
        print("No successful methods.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
