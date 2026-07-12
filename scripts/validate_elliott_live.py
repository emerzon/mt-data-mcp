"""Read-only Elliott v2 live smoke validation against an MT5 terminal."""

from __future__ import annotations

import argparse
import math
from typing import Any

from mtdata.core.patterns import patterns_detect

DEFAULT_SYMBOLS = ("EURUSD", "USDJPY", "XAUUSD", "USTEC", "BTCUSD")


def _validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    details = row.get("details") if isinstance(row.get("details"), dict) else {}
    pivots = details.get("pivots") if isinstance(details.get("pivots"), list) else []
    kinds = [str(pivot.get("kind")) for pivot in pivots if isinstance(pivot, dict)]
    if any(left == right for left, right in zip(kinds, kinds[1:])):
        errors.append("pivot kinds do not alternate")
    for pivot in pivots:
        if not isinstance(pivot, dict):
            errors.append("pivot is not an object")
            continue
        price = pivot.get("price")
        if not isinstance(price, (int, float)) or not math.isfinite(float(price)):
            errors.append("pivot price is non-finite")
        confirmation = pivot.get("confirmation_index")
        if confirmation is not None and int(confirmation) < int(pivot.get("index", 0)):
            errors.append("confirmation precedes geometric pivot")
        if not isinstance(pivot.get("confirmed"), bool):
            errors.append("pivot confirmed flag is not boolean")
    for key in (
        "rule_valid",
        "fallback_candidate",
        "pattern_confirmed",
        "terminal_confirmed",
    ):
        if key in details and not isinstance(details[key], bool):
            errors.append(f"{key} is not boolean")
    if details.get("rule_valid") is False and float(
        details.get("structural_score") or 0.0
    ) != 0.0:
        errors.append("hard-invalid geometry has a positive structural score")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    failures = 0
    config = {
        "pivot_price_source": "ohlc",
        "scan_thresholds_pct": [0.3, 0.6, 1.0],
        "scan_min_distances": [3, 5],
    }
    for symbol in args.symbols:
        result = patterns_detect(
            symbol=symbol,
            timeframe=args.timeframe,
            mode="elliott",
            detail="full",
            limit=args.limit,
            top_k=10,
            include_confirmed=True,
            config=config,
            __cli_raw=True,
        )
        if result.get("error"):
            print(f"{symbol}: ERROR {result['error']}")
            failures += 1
            continue
        rows = result.get("patterns") if isinstance(result.get("patterns"), list) else []
        row_errors = [error for row in rows for error in _validate_row(row)]
        if row_errors:
            print(f"{symbol}: FAIL ({len(rows)} rows): {sorted(set(row_errors))}")
            failures += 1
        else:
            print(f"{symbol}: PASS ({len(rows)} rows)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
