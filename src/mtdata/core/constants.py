import MetaTrader5 as mt5

# Import shared constants to avoid duplication
from ..utils.constants import PRECISION_REL_TOL, PRECISION_ABS_TOL, PRECISION_MAX_DECIMALS, TIME_DISPLAY_FORMAT

# Re-export precision/display constants from a single canonical location.
_PRECISION_EXPORTS = (
    PRECISION_REL_TOL,
    PRECISION_ABS_TOL,
    PRECISION_MAX_DECIMALS,
    TIME_DISPLAY_FORMAT,
)

# Constants (centralize defaults instead of hardcoding inline)
SERVICE_NAME = "MetaTrader5 Market Data Server"
GROUP_SEARCH_THRESHOLD = 5   # threshold for treating a search as group vs symbol search
TICKS_LOOKBACK_DAYS = 1      # lookback days for ticks when no start_datetime provided
DATA_READY_TIMEOUT = 3.0     # seconds to wait for feed to become ready after selection
DATA_POLL_INTERVAL = 0.2     # seconds between readiness polls
FETCH_RETRY_ATTEMPTS = 3     # attempts to fetch data if none returned
FETCH_RETRY_DELAY = 0.3      # delay between fetch retries
SANITY_BARS_TOLERANCE = 3    # acceptable lag in bars when checking freshness
TI_NAN_WARMUP_FACTOR = 2     # multiply warmup by this on retry
TI_NAN_WARMUP_MIN_ADD = 50   # at least add this many bars on retry

# Global parameter defaults
DEFAULT_TIMEFRAME = "H1"     # default timeframe parameter  
# Default output caps
DEFAULT_ROW_LIMIT = 25       # default row limit for large/tabular outputs
# Simplification defaults
SIMPLIFY_DEFAULT_METHOD = "lttb"  # default simplify method when not specified
SIMPLIFY_DEFAULT_MODE = "select"  # default simplify mode when not specified
SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT = 0.10  # default points as a fraction of --limit/--count when unspecified
SIMPLIFY_DEFAULT_RATIO = 0.25     # default ratio if no points/ratio provided
SIMPLIFY_DEFAULT_MIN_POINTS = 100 # minimum target points for default simplify
SIMPLIFY_DEFAULT_MAX_POINTS = 500 # maximum target points for default simplify

# Shared timeframe mapping (per MetaTrader5 docs)
TIMEFRAME_MAP = {
    # Minutes
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
    # Hours
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6,
    "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    # Days / Weeks / Months
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

# Approximate seconds per bar for timeframe window calculations
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
