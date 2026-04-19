import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "detect_mt5_time_offset.py"

spec = importlib.util.spec_from_file_location("detect_mt5_time_offset", SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load script module from {SCRIPT_PATH}")
detect_mt5_time_offset = importlib.util.module_from_spec(spec)
sys.modules["detect_mt5_time_offset"] = detect_mt5_time_offset
spec.loader.exec_module(detect_mt5_time_offset)


def test_extract_tick_epoch_seconds_prefers_millisecond_timestamp() -> None:
    tick = SimpleNamespace(time=1_700_000_000, time_msc=1_700_000_123_456)

    result = detect_mt5_time_offset._extract_tick_epoch_seconds(tick)

    assert result == pytest.approx(1_700_000_123.456)


def test_reconcile_tick_samples_keeps_freshest_cluster() -> None:
    samples = [
        detect_mt5_time_offset.TickSample(observed_at=100.0, tick_time=7_290.0),
        detect_mt5_time_offset.TickSample(observed_at=100.2, tick_time=7_290.0),
        detect_mt5_time_offset.TickSample(observed_at=100.4, tick_time=7_300.3),
        detect_mt5_time_offset.TickSample(observed_at=100.6, tick_time=7_300.6),
    ]

    reconciled, live_progress = detect_mt5_time_offset._reconcile_tick_samples(
        samples, freshness_slack_sec=2.0
    )

    assert live_progress is True
    assert [sample.tick_time for sample in reconciled] == [7_300.3, 7_300.6]


def test_main_errors_when_ticks_never_advance(monkeypatch, capsys, mt5_module) -> None:
    ticks = [SimpleNamespace(time=1_700_000_000, time_msc=1_700_000_000_000)] * 4
    mt5_module.symbol_info_tick.side_effect = ticks

    times = iter([1_000.0, 1_000.2, 1_000.4, 1_000.6])
    monkeypatch.setattr(detect_mt5_time_offset.time, "time", lambda: next(times))
    monkeypatch.setattr(detect_mt5_time_offset.time, "sleep", lambda _: None)
    monkeypatch.setattr(detect_mt5_time_offset, "_try_load_dotenv", lambda: None)
    monkeypatch.setattr(detect_mt5_time_offset, "_initialize_mt5", lambda: True)
    monkeypatch.setattr(
        detect_mt5_time_offset.argparse,
        "ArgumentParser",
        lambda *args, **kwargs: SimpleNamespace(
            add_argument=lambda *a, **k: None,
            parse_args=lambda: SimpleNamespace(symbol="EURUSD", samples=4, sleep=0.0),
        ),
    )

    result = detect_mt5_time_offset.main()
    captured = capsys.readouterr()

    assert result == 2
    assert "No advancing tick timestamps were observed" in captured.err


def test_main_discards_stale_samples_before_estimating_offset(monkeypatch, capsys, mt5_module) -> None:
    mt5_module.symbol_info_tick.side_effect = [
        SimpleNamespace(time=7_290, time_msc=7_290_000),
        SimpleNamespace(time=7_290, time_msc=7_290_000),
        SimpleNamespace(time=7_300, time_msc=7_300_300),
        SimpleNamespace(time=7_300, time_msc=7_300_600),
    ]

    times = iter([100.0, 100.2, 100.4, 100.6, 101.0])
    monkeypatch.setattr(detect_mt5_time_offset.time, "time", lambda: next(times))
    monkeypatch.setattr(detect_mt5_time_offset.time, "sleep", lambda _: None)
    monkeypatch.setattr(detect_mt5_time_offset, "_try_load_dotenv", lambda: None)
    monkeypatch.setattr(detect_mt5_time_offset, "_initialize_mt5", lambda: True)
    monkeypatch.setattr(
        detect_mt5_time_offset.argparse,
        "ArgumentParser",
        lambda *args, **kwargs: SimpleNamespace(
            add_argument=lambda *a, **k: None,
            parse_args=lambda: SimpleNamespace(symbol="EURUSD", samples=4, sleep=0.0),
        ),
    )

    result = detect_mt5_time_offset.main()
    captured = capsys.readouterr()

    assert result == 0
    assert "discarded 2 older/noisy sample(s)" in captured.out
    assert "Recommended MT5_TIME_OFFSET_MINUTES=120" in captured.out
