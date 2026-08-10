from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtdata.core.trading.safety import _account_is_demo


@pytest.mark.parametrize(
    ("account_info", "expected"),
    [
        (None, False),
        (SimpleNamespace(), False),
        (SimpleNamespace(trade_mode=None), False),
        (SimpleNamespace(trade_mode=0), True),
        (SimpleNamespace(trade_mode=1), False),
        (SimpleNamespace(trade_mode=2), False),
        (SimpleNamespace(trade_mode=" Demo "), True),
        (SimpleNamespace(trade_mode="real"), False),
        (SimpleNamespace(account_type="DEMO"), True),
        (SimpleNamespace(account_type="real", trade_mode=0), False),
        (SimpleNamespace(is_demo=True, account_type="real"), True),
        (SimpleNamespace(is_demo=False, trade_mode=0), False),
    ],
)
def test_account_demo_detection_is_conservative(account_info, expected: bool) -> None:
    assert _account_is_demo(account_info) is expected
