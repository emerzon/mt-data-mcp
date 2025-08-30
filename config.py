"""Configuration settings for MetaTrader5 MCP Server"""

import os
from typing import Optional

class MT5Config:
    """MetaTrader5 connection configuration"""
    
    def __init__(self):
        self.login = os.getenv("MT5_LOGIN")
        self.password = os.getenv("MT5_PASSWORD") 
        self.server = os.getenv("MT5_SERVER")
        self.timeout = int(os.getenv("MT5_TIMEOUT", "30"))
        
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

# Global configuration instance
mt5_config = MT5Config()
