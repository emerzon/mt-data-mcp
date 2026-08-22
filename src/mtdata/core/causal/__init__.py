"""Predictive lead/lag discovery tools."""

from __future__ import annotations

from mtdata.core.causal import cointegration, common, correlation, cross, discover
from mtdata.core.causal.cointegration import cointegration_test
from mtdata.core.causal.correlation import correlation_matrix
from mtdata.core.causal.cross import cross_correlation
from mtdata.core.causal.discover import causal_discover_signals

for _tool in (
    causal_discover_signals,
    correlation_matrix,
    cross_correlation,
    cointegration_test,
):
    _tool.__module__ = "mtdata.core.causal"

__all__ = [
    "causal_discover_signals",
    "cointegration",
    "cointegration_test",
    "common",
    "correlation",
    "correlation_matrix",
    "cross",
    "cross_correlation",
    "discover",
]
