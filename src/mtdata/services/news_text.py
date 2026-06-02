"""Text normalization helpers for news provider payloads."""

from __future__ import annotations

import re
from typing import Any

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_MOJIBAKE_MARKERS = ("\u00c3", "\u00c2", "\u00e2", "\u00d4", "\u00c7", "\ufffd")
_MOJIBAKE_ENCODINGS = ("cp1252", "latin1", "cp850", "cp858")


def _mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)


def _repair_news_mojibake(text: str) -> str:
    current = text
    for _ in range(3):
        current_score = _mojibake_score(current)
        if current_score == 0:
            break
        best = current
        best_score = current_score
        for encoding in _MOJIBAKE_ENCODINGS:
            try:
                candidate = current.encode(encoding).decode("utf-8")
            except UnicodeError:
                continue
            candidate_score = _mojibake_score(candidate)
            if candidate_score < best_score:
                best = candidate
                best_score = candidate_score
        if best == current:
            break
        current = best
    return current


def normalize_news_text(value: Any) -> Any:
    """Repair common provider mojibake and compact whitespace in news text."""
    if not isinstance(value, str):
        return value
    text = _repair_news_mojibake(value.strip())
    text = _CONTROL_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()
