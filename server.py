#!/usr/bin/env python3
"""
Convenience module exposing the mtdata core server tools at top-level.
Allows `from server import ...` for tests and scripts, and remains executable.
"""

# Re-export all public API from the core server
from src.mtdata.core.server import *  # noqa: F401,F403
# Explicitly import functions that need to be available
from src.mtdata.core.data import data_fetch_candles, data_fetch_ticks  # noqa: F401
from src.mtdata.utils.denoise import denoise_list_methods  # noqa: F401

if __name__ == "__main__":
    from src.mtdata.core.server import main
    main()
