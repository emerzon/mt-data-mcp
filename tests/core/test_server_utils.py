"""Tests for src/mtdata/core/server_utils.py"""
from types import SimpleNamespace
from mtdata.core.server_utils import get_mcp_registry


class TestGetMcpRegistry:
    def test_with_tools_attr(self):
        mcp = SimpleNamespace(tools={"a": 1, "b": 2})
        result = get_mcp_registry(mcp)
        assert result == {"a": 1, "b": 2}

    def test_with_registry_attr(self):
        mcp = SimpleNamespace(registry={"x": 10})
        result = get_mcp_registry(mcp)
        assert result == {"x": 10}

    def test_no_matching_attr(self):
        mcp = SimpleNamespace()
        assert get_mcp_registry(mcp) is None

    def test_non_dict_tools(self):
        mcp = SimpleNamespace(tools=[1, 2, 3])
        assert get_mcp_registry(mcp) is None
