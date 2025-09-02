"""Configuration settings for MetaTrader5 MCP Server"""

import os
from typing import Optional
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    # Load environment variables from a .env file if present
    _env_path = find_dotenv()
    if _env_path:
        load_dotenv(_env_path)
    else:
        # Fallback to default search (current working directory)
        load_dotenv()
except Exception:
    # dotenv is optional; environment can still be provided by the OS
    pass
try:
    import pytz  # type: ignore
except Exception:
    pytz = None  # optional

class MT5Config:
    """MetaTrader5 connection configuration"""
    
    def __init__(self):
        self.login = os.getenv("MT5_LOGIN")
        self.password = os.getenv("MT5_PASSWORD") 
        self.server = os.getenv("MT5_SERVER")
        self.timeout = int(os.getenv("MT5_TIMEOUT", "30"))
        # Timezone handling
        # Preferred: IANA tz (pytz) names for server/client
        self.server_tz_name = os.getenv("MT5_SERVER_TZ")  # e.g., "Europe/Lisbon"
        self.client_tz_name = os.getenv("CLIENT_TZ") or os.getenv("MT5_CLIENT_TZ")  # e.g., "America/New_York"
        # Legacy fallback: offset minutes
        # Timezone handling: MT5 server offset from UTC in minutes (can be negative).
        # Example: MT5_TIME_OFFSET_MINUTES=120 for UTC+2
        try:
            self.time_offset_minutes = int(os.getenv("MT5_TIME_OFFSET_MINUTES", "0"))
        except Exception:
            self.time_offset_minutes = 0
        
    def get_login(self) -> Optional[int]:
        """Get login as integer if available"""
        return int(self.login) if self.login else None
    
    def get_password(self) -> Optional[str]:
        """Get password"""
        return self.password
    
    def get_server(self) -> Optional[str]:
        """Get server name"""
        return self.server
    
    def has_credentials(self) -> bool:
        """Check if all credentials are available"""
        return all([self.login, self.password, self.server])

    def get_time_offset_seconds(self) -> int:
        """Return configured MT5 server time offset relative to UTC in seconds.

        Positive value means MT5 server time is ahead of UTC (UTC+X).
        All MT5 timestamps read from the terminal should be adjusted by subtracting this offset
        to normalize into UTC epochs.
        """
        try:
            return int(self.time_offset_minutes) * 60
        except Exception:
            return 0

    def get_server_tz(self):
        """Return a pytz timezone for the MT5 server, if configured and pytz is available."""
        try:
            if pytz and self.server_tz_name:
                return pytz.timezone(self.server_tz_name)
        except Exception:
            return None
        return None

    def get_client_tz(self):
        """Return a pytz timezone for the client, if configured and pytz is available."""
        try:
            if pytz and self.client_tz_name:
                return pytz.timezone(self.client_tz_name)
        except Exception:
            return None
        return None

# Global configuration instance
mt5_config = MT5Config()
