"""Compatibility alias for the canonical bootstrap settings module."""

from __future__ import annotations

import importlib
import sys

_bootstrap_settings = importlib.import_module("mtdata.bootstrap.settings")
sys.modules[__name__] = _bootstrap_settings
