"""Compatibility alias for the canonical shared constants module."""

from __future__ import annotations

import sys

from ..shared import constants as _shared_constants

sys.modules[__name__] = _shared_constants
