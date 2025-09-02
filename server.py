#!/usr/bin/env python3
"""
Convenience module exposing the mtdata core server tools at top-level.
Allows `from server import ...` for tests and scripts, and remains executable.
"""

# Re-export all public API from the core server
from src.mtdata.core.server import *  # noqa: F401,F403

if __name__ == "__main__":
    from src.mtdata.core.server import main
    main()
