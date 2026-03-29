from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from ..schema import DenoiseSpec
from .basic import template_basic


_MINIMAL_SECTION_KEYS = ("context", "forecast", "barriers")


def template_minimal(
    symbol: str,
    horizon: int,
    denoise: Optional[DenoiseSpec],
    params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    base = template_basic(symbol, horizon, denoise, params)

    if isinstance(base, str):
        return {"error": f"template_basic returned string: {base}"}
    if not isinstance(base, dict):
        return {"error": f"template_basic returned unexpected type: {type(base)}"}

    report = deepcopy(base)
    meta = report.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    meta["template"] = "minimal"
    report["meta"] = meta

    sections_in = report.get("sections")
    if not isinstance(sections_in, dict):
        report["sections"] = {}
        return report

    report["sections"] = {
        key: sections_in[key]
        for key in _MINIMAL_SECTION_KEYS
        if key in sections_in
    }
    return report
