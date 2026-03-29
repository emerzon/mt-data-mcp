"""Forecast-domain exception types."""

from __future__ import annotations

from typing import Any


class ForecastError(RuntimeError):
    """Raised when forecast execution fails outside normal result handling."""


class ForecastResultError(ForecastError):
    """Raised when a legacy forecast payload encodes an error dict."""


def raise_if_error_result(result: Any) -> Any:
    """Raise ForecastResultError when a legacy forecast payload encodes an error dict."""
    if isinstance(result, dict):
        error = result.get("error")
        if isinstance(error, str) and error.strip():
            raise ForecastResultError(error)
    return result
