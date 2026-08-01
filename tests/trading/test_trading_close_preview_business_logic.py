from __future__ import annotations

from types import SimpleNamespace

from mtdata.core.trading.execution import _close_positions_dry_run_preview


def test_close_preview_discloses_stale_market_readiness() -> None:
    position = SimpleNamespace(
        ticket=7,
        symbol="EURUSD",
        type=0,
        volume=0.5,
        profit=12.0,
        price_open=1.09,
        price_current=1.10,
    )
    gateway = SimpleNamespace(
        symbol_info_tick=lambda symbol: SimpleNamespace(
            bid=1.0999,
            ask=1.1001,
            time=1,
        )
    )

    result = _close_positions_dry_run_preview(
        [position],
        symbol="EURUSD",
        magic=None,
        profit_only=False,
        loss_only=False,
        close_priority=None,
        mt5=gateway,
    )

    assert result["success"] is True
    assert result["preview_ok"] is False
    assert result["market_readiness"] == {
        "symbols_checked": 1,
        "usable_for_live_trading": False,
        "stale_or_unverified": 1,
    }
    assert result["matched_positions"][0]["quote_context"]["data_stale"] is True
