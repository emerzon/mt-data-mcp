"""Tests for tools_list unknown-category validation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mtdata.core.tools import tools_list


def _call(**kwargs):
    fn = getattr(tools_list, "__wrapped__", tools_list)
    return fn(**kwargs)


def test_tools_list_unknown_category_warns():
    out = _call(category="definitely_not_a_category", detail="compact")
    assert out.get("count") == 0
    assert out.get("warning")
    assert "Unknown category" in out["warning"]


def test_tools_list_known_category_has_no_warning():
    catalog = _call(detail="compact")
    categories = catalog.get("categories") or {}
    if not categories:
        return
    known = sorted(categories.keys())[0]
    out = _call(category=known, detail="compact")
    assert "warning" not in out
