#!/usr/bin/env python3
"""Lightweight forecast test runner.

Usage:
  python tests/test_forecast_methods.py EURUSD H1 12 [model ...]
  python tests/test_forecast_methods.py EURUSD H1 12 statsforecast:AutoARIMA

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
    return "Usage: python tests/test_forecast_methods.py SYMBOL TIMEFRAME HORIZON [model ...]"


def _resolve_models(args: List[str]) -> List[str]:
    if args:
        return [m.strip() for m in args if m.strip()]
    # Keep defaults lightweight and dependency-free.
    return ["theta", "naive", "drift", "seasonal_naive"]


def _parse_model_spec(spec: str) -> tuple[str, str]:
    parts = spec.split(":", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "native", spec.strip()


def _run(symbol: str, timeframe: str, horizon: int, models: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon": int(horizon),
        "models": {},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    m_eff = default_seasonality_period(timeframe)
    successes = 0
    for spec in models:
        library, model = _parse_model_spec(spec)
        params = {}
        if library in ("", "native") and model == "seasonal_naive":
            params["seasonality"] = int(m_eff)
        res = forecast_generate(
            symbol=symbol,
            timeframe=timeframe,
            library=library or "native",
            model=model,
            horizon=int(horizon),
            model_params=params or None,
            __cli_raw=True,
        )
        out["models"][spec] = res
        if isinstance(res, dict) and not res.get("error"):
            successes += 1
    out["successes"] = successes
    out["failures"] = max(0, len(models) - successes)
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
    models = _resolve_models(sys.argv[4:])

    result = _run(symbol, timeframe, horizon, models)
    out_dir = os.path.join(os.path.dirname(__file__), "test_results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fname = f"{symbol}_{timeframe}_{horizon}_{ts}.json"
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True, indent=2)

    print(f"Wrote results to {out_path}")
    if result.get("successes", 0) <= 0:
        print("No successful models.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
