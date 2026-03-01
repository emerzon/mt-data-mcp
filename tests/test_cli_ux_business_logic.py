from __future__ import annotations

from mtdata.core.cli import _build_usage_examples, _format_result_for_cli


def test_build_usage_examples_trade_risk_analyze_uses_symbol_flag() -> None:
    base, advanced = _build_usage_examples("trade_risk_analyze", {"params": []})

    assert "--symbol BTCUSD" in base
    assert "--desired-risk-pct 1" in base
    assert advanced is not None
    assert "--proposed-tp 69000" in advanced


def test_build_usage_examples_trade_modify_keeps_ticket_positional() -> None:
    base, advanced = _build_usage_examples("trade_modify", {"params": []})

    assert "trade_modify 123456789" in base
    assert "--price 61000" in base
    assert advanced is not None
    assert "--stop-loss 60500" in advanced


def test_patterns_detect_text_output_has_no_cli_specific_compaction() -> None:
    payload = {
        "success": True,
        "symbol": "BTCUSD",
        "timeframe": "H1",
        "mode": "candlestick",
        "count": 12,
        "data": [
            {"time": f"2026-03-01 0{i}:00", "pattern": "hammer" if i % 2 == 0 else "doji"}
            for i in range(12)
        ],
    }

    out = _format_result_for_cli(payload, fmt="text", verbose=False, cmd_name="patterns_detect")

    assert "data[12]" in out
    assert "recent_patterns[8]" not in out


def test_patterns_detect_verbose_output_keeps_full_rows() -> None:
    payload = {
        "success": True,
        "symbol": "BTCUSD",
        "timeframe": "H1",
        "mode": "candlestick",
        "count": 12,
        "data": [{"time": f"T{i}", "pattern": "hammer"} for i in range(12)],
    }

    out = _format_result_for_cli(payload, fmt="text", verbose=True, cmd_name="patterns_detect")

    assert "data[12]" in out


def test_regime_detect_text_output_has_no_cli_specific_compaction() -> None:
    payload = {
        "success": True,
        "symbol": "BTCUSD",
        "timeframe": "H1",
        "method": "hmm",
        "regimes": [
            {
                "start": f"2026-03-01 0{i}:00",
                "end": f"2026-03-01 0{i}:59",
                "bars": 10,
                "regime": i % 3,
                "avg_conf": 0.8,
            }
            for i in range(7)
        ],
    }

    out = _format_result_for_cli(payload, fmt="text", verbose=False, cmd_name="regime_detect")

    assert "regimes[7]" in out
    assert "recent_regimes[5]" not in out
