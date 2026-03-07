"""Compatibility alias for the canonical shared schema module."""

from __future__ import annotations

import sys

from ..shared import schema as _shared_schema

sys.modules[__name__] = _shared_schema
