#!/usr/bin/env python3
"""Lightweight live forecast runner.

Usage:
  python scripts/run_forecast_methods.py EURUSD H1 12 [method ...]
  python scripts/run_forecast_methods.py EURUSD H1 12 statsforecast:AutoARIMA

Writes JSON output into tests/test_results/.
This is not a pytest module; keep it out of tests/ so collection does not
import forecast_generate.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mtdata.core.forecast import forecast_generate
from mtdata.forecast.common import default_seasonality
from mtdata.forecast.requests import ForecastGenerateRequest

_ROOT = Path(__file__).resolve().parents[1]


def _usage() -> str:
    return "Usage: python scripts/run_forecast_methods.py SYMBOL TIMEFRAME HORIZON [method ...]"


def _resolve_methods(args: list[str]) -> list[str]:
    if args:
        return [m.strip() for m in args if m.strip()]
    # Keep defaults lightweight and dependency-free.
    return ["theta", "naive", "drift", "seasonal_naive"]


def _parse_method_spec(spec: str) -> tuple[str, str]:
    parts = spec.split(":", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "native", spec.strip()


def _run(symbol: str, timeframe: str, horizon: int, methods: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon": int(horizon),
        "methods": {},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    m_eff = default_seasonality(timeframe)
    successes = 0
    for spec in methods:
        library, method = _parse_method_spec(spec)
        params = {}
        if library in ("", "native") and method == "seasonal_naive":
            params["seasonality"] = int(m_eff)
        res = forecast_generate(
            request=ForecastGenerateRequest(
                symbol=symbol,
                timeframe=timeframe,
                library=library or "native",
                method=method,
                horizon=int(horizon),
                params=params or None,
            ),
            __cli_raw=True,
        )
        out["methods"][spec] = res
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
    out_dir = _ROOT / "tests" / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fname = f"{symbol}_{timeframe}_{horizon}_{ts}.json"
    out_path = out_dir / fname
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True, indent=2)

    print(f"Wrote results to {out_path}")
    if result.get("successes", 0) <= 0:
        print("No successful models.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
