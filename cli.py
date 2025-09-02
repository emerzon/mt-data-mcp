#!/usr/bin/env python3
"""
Wrapper script for backwards compatibility.
Import the actual CLI from the new package structure.
"""

if __name__ == "__main__":
    from src.mtdata.core.cli import main
    main()