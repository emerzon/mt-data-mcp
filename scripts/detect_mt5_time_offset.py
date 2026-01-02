from __future__ import annotations

import argparse
import os
import sys
import time
from statistics import median
from typing import Optional


def _try_load_dotenv() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore

        env_path = find_dotenv()
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
    except Exception:
        pass


def _initialize_mt5() -> bool:
    import MetaTrader5 as mt5  # type: ignore

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if login and password and server:
        try:
            login_i = int(login)
        except ValueError:
            login_i = None

        if login_i is not None and mt5.initialize(login=login_i, password=password, server=server):
            return True

    return bool(mt5.initialize())


def _get_tick_epoch_seconds(symbol: str) -> Optional[int]:
    import MetaTrader5 as mt5  # type: ignore

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    try:
        value = int(getattr(tick, "time"))
    except Exception:
        return None
    return value if value > 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate MT5 server epoch offset vs UTC by comparing the latest tick time to local time.\n"
            "Run this during active market hours so the tick time is current."
        )
    )
    parser.add_argument("--symbol", default="EURUSD", help="Symbol to sample ticks from (default: EURUSD)")
    parser.add_argument("--samples", type=int, default=15, help="Number of samples to take (default: 15)")
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds between samples (default: 0.2)")
    args = parser.parse_args()

    _try_load_dotenv()

    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception as exc:
        print(f"ERROR: Could not import MetaTrader5. Is it installed? ({exc})", file=sys.stderr)
        return 2

    if not _initialize_mt5():
        print(
            "ERROR: Failed to initialize MetaTrader5.\n"
            "- Ensure the MT5 terminal is installed, running, and logged in.\n"
            "- If you rely on credentials, set MT5_LOGIN/MT5_PASSWORD/MT5_SERVER in .env.",
            file=sys.stderr,
        )
        return 2

    symbol = str(args.symbol)
    samples = max(3, int(args.samples))
    sleep_s = max(0.0, float(args.sleep))

    deltas_sec: list[float] = []
    last_tick: Optional[int] = None
    for _ in range(samples):
        now = time.time()
        tick_time = _get_tick_epoch_seconds(symbol)
        if tick_time is not None:
            last_tick = tick_time
            deltas_sec.append(float(tick_time) - float(now))
        time.sleep(sleep_s)

    mt5.shutdown()

    if not deltas_sec or last_tick is None:
        print(
            f"ERROR: No tick data received for symbol '{symbol}'.\n"
            "- Ensure the symbol exists for your broker and is visible in Market Watch.\n"
            "- Try a different symbol (e.g., XAUUSD) or run during active market hours.",
            file=sys.stderr,
        )
        return 2

    delta = float(median(deltas_sec))
    offset_minutes = int(round(delta / 60.0))

    adjusted_tick = float(last_tick) - float(offset_minutes * 60)
    age_sec = float(time.time()) - adjusted_tick

    print(f"Symbol: {symbol}")
    print(f"Median(tick_time - local_time): {delta:.1f} seconds")
    print(f"Recommended MT5_TIME_OFFSET_MINUTES={offset_minutes}")
    if abs(age_sec) > 60.0:
        print(
            f"WARNING: After applying the offset, the last tick looks ~{age_sec/60.0:.1f} minutes old.\n"
            "This estimate may be unreliable if markets are closed or the symbol is inactive."
        )
    print()
    print("Tip: If your broker uses daylight saving time, prefer MT5_SERVER_TZ=<IANA name> for DST-aware conversion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

