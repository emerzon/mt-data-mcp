from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TickSample:
    observed_at: float
    tick_time: float

    @property
    def delta_sec(self) -> float:
        return self.tick_time - self.observed_at


def _extract_tick_epoch_seconds(tick: object) -> Optional[float]:
    if tick is None:
        return None
    for attr_name, scale in (("time_msc", 1000.0), ("time", 1.0)):
        raw_value = getattr(tick, attr_name, None)
        if raw_value is None:
            continue
        try:
            value = float(raw_value) / scale
        except Exception:
            continue
        if value > 0:
            return value
    return None


def _get_tick_epoch_seconds(symbol: str) -> Optional[float]:
    import MetaTrader5 as mt5  # type: ignore

    tick = mt5.symbol_info_tick(symbol)
    return _extract_tick_epoch_seconds(tick)


def _reconcile_tick_samples(
    samples: list[TickSample], *, freshness_slack_sec: float
) -> tuple[list[TickSample], bool]:
    if not samples:
        return [], False

    live_progress = any(curr.tick_time > prev.tick_time for prev, curr in zip(samples, samples[1:]))
    freshest_delta = max(sample.delta_sec for sample in samples)
    slack = max(0.0, float(freshness_slack_sec))
    reconciled = [sample for sample in samples if freshest_delta - sample.delta_sec <= slack]
    return reconciled, live_progress


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

    tick_samples: list[TickSample] = []
    for _ in range(samples):
        now = time.time()
        tick_time = _get_tick_epoch_seconds(symbol)
        if tick_time is not None:
            tick_samples.append(TickSample(observed_at=float(now), tick_time=float(tick_time)))
        time.sleep(sleep_s)

    mt5.shutdown()

    if not tick_samples:
        print(
            f"ERROR: No tick data received for symbol '{symbol}'.\n"
            "- Ensure the symbol exists for your broker and is visible in Market Watch.\n"
            "- Try a different symbol (e.g., XAUUSD) or run during active market hours.",
            file=sys.stderr,
        )
        return 2

    freshness_slack_sec = max(2.0, sleep_s * 3.0)
    reconciled_samples, live_progress = _reconcile_tick_samples(
        tick_samples, freshness_slack_sec=freshness_slack_sec
    )
    if not live_progress:
        print(
            f"ERROR: No advancing tick timestamps were observed for symbol '{symbol}'.\n"
            "- The sampled feed may be stale or inactive.\n"
            "- Try a more active symbol, increase --samples, or rerun during active market hours.",
            file=sys.stderr,
        )
        return 2

    delta = float(median(sample.delta_sec for sample in reconciled_samples))
    offset_minutes = int(round(delta / 60.0))

    freshest_sample = max(reconciled_samples, key=lambda sample: sample.delta_sec)
    adjusted_tick = float(freshest_sample.tick_time) - float(offset_minutes * 60)
    age_sec = float(time.time()) - adjusted_tick
    discarded_samples = len(tick_samples) - len(reconciled_samples)

    print(f"Symbol: {symbol}")
    if discarded_samples:
        print(
            f"Reconciled using {len(reconciled_samples)} freshest sample(s); "
            f"discarded {discarded_samples} older/noisy sample(s)."
        )
    else:
        print(f"Reconciled using {len(reconciled_samples)} live sample(s).")
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

