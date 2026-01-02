"""MetaTrader5 MCP Server - Market Data Provider"""

from __future__ import annotations

from pathlib import Path

# If the repo root is imported as the `mtdata` package (e.g. when running tests
# with the parent directory on `sys.path`), expose the real src-layout package.
_src_pkg = Path(__file__).resolve().parent / "src" / "mtdata"
if _src_pkg.is_dir():
    __path__.append(str(_src_pkg))  # type: ignore[name-defined]

__version__ = "0.1.0"
__author__ = "MCP MetaTrader5 Server"
