"""Runtime annotation resolution helpers."""

from __future__ import annotations

import inspect
from typing import Any, Dict, get_type_hints

try:
    import annotationlib
except Exception:  # pragma: no cover - Python 3.14+ should provide this
    annotationlib = None


_ANNOTATION_VALUE_FORMAT = getattr(
    getattr(annotationlib, "Format", None),
    "VALUE",
    None,
)


def get_runtime_signature(obj: Any) -> inspect.Signature:
    """Resolve a signature with evaluated annotations when available."""
    if _ANNOTATION_VALUE_FORMAT is not None:
        try:
            return inspect.signature(
                obj,
                eval_str=True,
                annotation_format=_ANNOTATION_VALUE_FORMAT,
            )
        except Exception:
            pass
    return inspect.signature(obj)


def get_runtime_annotations(obj: Any) -> Dict[str, Any]:
    """Resolve annotations using the best runtime API available."""
    if annotationlib is not None and _ANNOTATION_VALUE_FORMAT is not None:
        try:
            resolved = annotationlib.get_annotations(
                obj,
                eval_str=True,
                format=_ANNOTATION_VALUE_FORMAT,
            )
            if isinstance(resolved, dict):
                return resolved
        except Exception:
            pass
    try:
        resolved = get_type_hints(obj)
        if isinstance(resolved, dict):
            return resolved
    except Exception:
        pass
    raw = getattr(obj, "__annotations__", None)
    return raw if isinstance(raw, dict) else {}
