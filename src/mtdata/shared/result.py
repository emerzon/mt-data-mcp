"""Lightweight Result[T] type for internal domain boundaries.

Public tool and use-case functions still return ``dict`` — call
:func:`to_dict` at the boundary to convert.  This module is strictly
internal and must *not* appear in any MCP-facing signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, Union

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Successful result wrapping an arbitrary value."""

    value: T


@dataclass(frozen=True, slots=True)
class Err:
    """Failed result with a human-readable message and optional metadata."""

    message: str
    code: str = ""
    details: dict[str, Any] = field(default_factory=dict)


Result = Union[Ok[T], Err]


def is_ok(result: Result) -> bool:
    """Return *True* if *result* is an :class:`Ok`."""
    return isinstance(result, Ok)


def is_err(result: Result) -> bool:
    """Return *True* if *result* is an :class:`Err`."""
    return isinstance(result, Err)


def to_dict(result: Result) -> dict[str, Any]:
    """Convert a :class:`Result` to the existing dict error-envelope contract.

    * ``Ok(value)`` — if *value* is already a ``dict`` it is returned as-is
      (with ``"success": True`` set if absent); otherwise
      ``{"success": True, "value": value}``.
    * ``Err(message, code, details)`` — ``{"success": False, "error": message,
      "error_code": code, **details}``.
    """
    if isinstance(result, Ok):
        val = result.value
        if isinstance(val, dict):
            out = dict(val)
            out.setdefault("success", True)
            return out
        return {"success": True, "value": val}

    # Err
    payload: dict[str, Any] = {
        "success": False,
        "error": result.message,
    }
    if result.code:
        payload["error_code"] = result.code
    if result.details:
        payload.update(result.details)
    return payload
