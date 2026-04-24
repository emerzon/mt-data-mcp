from __future__ import annotations

from typing import get_args, get_type_hints

import mtdata.core.causal as causal_mod
import mtdata.core.pivot as pivot_mod
from mtdata.core.report.requests import ReportGenerateRequest
from mtdata.core.trading.requests import TradeVarCvarRequest
from mtdata.shared.schema import AutoTimeframeLiteral, TimeframeLiteral


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_causal_tools_use_timeframe_literal_annotations() -> None:
    for fn in (
        causal_mod.causal_discover_signals,
        causal_mod.correlation_matrix,
        causal_mod.cointegration_test,
    ):
        raw = _unwrap(fn)
        assert get_type_hints(raw)["timeframe"] == TimeframeLiteral


def test_support_resistance_accepts_auto_timeframe_literal() -> None:
    raw = _unwrap(pivot_mod.support_resistance_levels)
    annotation = get_type_hints(raw)["timeframe"]

    assert annotation == AutoTimeframeLiteral
    assert any("auto" in get_args(arg) for arg in get_args(annotation))


def test_request_models_use_timeframe_literal_annotations() -> None:
    assert TradeVarCvarRequest.model_fields["timeframe"].annotation == TimeframeLiteral
    assert TimeframeLiteral in get_args(ReportGenerateRequest.model_fields["timeframe"].annotation)
