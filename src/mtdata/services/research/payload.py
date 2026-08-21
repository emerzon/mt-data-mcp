"""Shared payload stamps for research domain tools."""

from __future__ import annotations

from typing import Any


def stamp_provider(payload: Any, *, provider: str) -> Any:
    """Attach provider identity without clobbering an existing list."""
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    out.setdefault("provider", provider)
    used = out.get("providers_used")
    if not isinstance(used, list) or not used:
        out["providers_used"] = [provider]
    return out
