"""
Forecast methods init file to ensure proper imports.
"""

from __future__ import annotations

# Import specific functions to avoid wildcard issues
from . import pretrained

# Re-export the main forecast functions
forecast_chronos_bolt = pretrained.forecast_chronos_bolt
forecast_timesfm = pretrained.forecast_timesfm
forecast_lag_llama = pretrained.forecast_lag_llama

__all__ = [
    'forecast_chronos_bolt',
    'forecast_timesfm', 
    'forecast_lag_llama',
]
