"""Public regime detection package."""

from . import methods
from .api import regime_detect

__all__ = [
    "methods",
    "regime_detect",
]
