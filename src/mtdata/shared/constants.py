"""Canonical shared constants module."""

from __future__ import annotations

from ..utils.mt5 import mt5

# Precision/formatting constants
PRECISION_REL_TOL = 1e-6
PRECISION_ABS_TOL = 1e-12
PRECISION_MAX_LOSS_PCT = 1e-3
PRECISION_MAX_DECIMALS = 10
DISPLAY_MAX_DECIMALS = 8

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
    "MN1": 2592000,
}

# Constants (centralize defaults instead of hardcoding inline)
SERVICE_NAME = "MetaTrader5 Market Data Server"
GROUP_SEARCH_THRESHOLD = 5
TICKS_LOOKBACK_DAYS = 1
DATA_READY_TIMEOUT = 3.0
DATA_POLL_INTERVAL = 0.2
FETCH_RETRY_ATTEMPTS = 3
FETCH_RETRY_DELAY = 0.3
SANITY_BARS_TOLERANCE = 3
TI_NAN_WARMUP_FACTOR = 2
TI_NAN_WARMUP_MIN_ADD = 50

# Global parameter defaults
DEFAULT_TIMEFRAME = "H1"
DEFAULT_ROW_LIMIT = 25

# Simplification defaults
SIMPLIFY_DEFAULT_METHOD = "lttb"
SIMPLIFY_DEFAULT_MODE = "select"
SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT = 0.10
SIMPLIFY_DEFAULT_RATIO = 0.25
SIMPLIFY_DEFAULT_MIN_POINTS = 100
SIMPLIFY_DEFAULT_MAX_POINTS = 500

# Shared timeframe mapping (per MetaTrader5 docs)
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6,
    "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

