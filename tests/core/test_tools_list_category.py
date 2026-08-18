"""Tests for tools_list unknown-category validation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mtdata.bootstrap.tools import bootstrap_tools
from mtdata.core.tools import tools_list


def _call(**kwargs):
    fn = getattr(tools_list, "__wrapped__", tools_list)
    return fn(**kwargs)


def test_tools_list_unknown_category_fails():
    out = _call(category="definitely_not_a_category", detail="compact")
    assert out["success"] is False
    assert out["error_code"] == "invalid_category"
    assert "Unknown category" in out["error"]
    assert "forecast" in out["valid_categories"]


def test_tools_list_known_category_has_no_warning():
    catalog = _call(detail="compact")
    categories = catalog.get("categories") or {}
    if not categories:
        return
    known = sorted(categories.keys())[0]
    out = _call(category=known, detail="compact")
    assert "warning" not in out


def test_tools_list_news_category_matches_news_help_family():
    bootstrap_tools()

    out = _call(category="news", detail="compact", limit=50)

    assert out["success"] is True
    assert {row["name"] for row in out["tools"]} == {
        "finviz_market_news",
        "finviz_news",
        "news",
    }
