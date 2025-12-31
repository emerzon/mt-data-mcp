"""Shared constants that don't depend on external packages.

This module contains constants used by utils modules to avoid
circular import issues with core.constants.
"""

# Precision/formatting constants
PRECISION_REL_TOL = 1e-6     # relative tolerance for rounding optimization
PRECISION_ABS_TOL = 1e-12    # absolute tolerance for rounding optimization
PRECISION_MAX_DECIMALS = 10  # upper bound on decimal places
DISPLAY_MAX_DECIMALS = 8     # default display precision for numeric outputs

# Normalized datetime display format (UTC/local)
TIME_DISPLAY_FORMAT = "%Y-%m-%d %H:%M"
