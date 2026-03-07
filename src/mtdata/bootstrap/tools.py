"""Explicit tool-module bootstrap for transport adapters."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Final

from ..core._mcp_instance import mcp
from ..core.schema import get_shared_enum_lists
from ..core.schema_attach import attach_schemas_to_tools

TOOL_MODULE_NAMES: Final[tuple[str, ...]] = (
    "mtdata.core.data",
    "mtdata.core.forecast",
    "mtdata.core.causal",
    "mtdata.core.indicators",
    "mtdata.core.market_depth",
    "mtdata.core.patterns",
    "mtdata.core.pivot",
    "mtdata.core.symbols",
    "mtdata.core.regime",
    "mtdata.core.labels",
    "mtdata.core.report",
    "mtdata.core.trading",
    "mtdata.core.temporal",
    "mtdata.core.finviz",
)

_BOOTSTRAPPED_MODULES: tuple[ModuleType, ...] = ()
_SCHEMAS_ATTACHED = False


def bootstrap_tools() -> tuple[ModuleType, ...]:
    """Load tool modules and attach shared schemas exactly once per process."""
    global _BOOTSTRAPPED_MODULES, _SCHEMAS_ATTACHED

    if not _BOOTSTRAPPED_MODULES:
        _BOOTSTRAPPED_MODULES = tuple(import_module(name) for name in TOOL_MODULE_NAMES)

    if not _SCHEMAS_ATTACHED:
        attach_schemas_to_tools(mcp, get_shared_enum_lists())
        _SCHEMAS_ATTACHED = True

    return _BOOTSTRAPPED_MODULES
