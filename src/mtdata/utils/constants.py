"""Shared constants that don't depend on external packages.

This module contains constants used by utils modules to avoid
circular import issues with core.constants.
"""

# Precision/formatting constants
PRECISION_REL_TOL = 1e-6     # relative tolerance for rounding optimization
PRECISION_ABS_TOL = 1e-12    # absolute tolerance for rounding optimization
PRECISION_MAX_LOSS_PCT = 1e-3  # max rounding loss as % of column range
PRECISION_MAX_DECIMALS = 10  # upper bound on decimal places
DISPLAY_MAX_DECIMALS = 8     # default display precision for numeric outputs

# Normalized datetime display format (UTC/local)
TIME_DISPLAY_FORMAT = "%Y-%m-%d %H:%M"

# Approximate seconds per bar for each timeframe (no MT5 dependency)
TIMEFRAME_SECONDS = {
    "M1": 60,
    "M2": 120,
    "M3": 180,
    "M4": 240,
    "M5": 300,
    "M6": 360,
    "M10": 600,
    "M12": 720,
    "M15": 900,
    "M20": 1200,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H3": 10800,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D1": 86400,
    "W1": 604800,
    # For months, use a rough average of 30 days
    "MN1": 2592000,
}
