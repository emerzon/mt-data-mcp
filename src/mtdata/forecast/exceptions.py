"""Forecast-domain exception types."""

from __future__ import annotations


class ForecastError(RuntimeError):
    """Raised when forecast execution fails outside normal result handling."""
