"""Tests for pattern status helpers and configuration."""

from types import SimpleNamespace

import pytest


# ── pattern status helpers ─────────────────────────────────────────────────

class TestPatternStatusHelpers:

    def test_visible_pattern_rows_and_counts_share_status_normalization(self):
        from mtdata.core.patterns_support import (
            _count_patterns_with_status,
            _visible_pattern_rows,
        )

        rows = [
            {"status": " forming "},
            {"status": "COMPLETED"},
            {"status": "other"},
        ]

        visible = _visible_pattern_rows(rows, include_completed=False)

        assert visible == [rows[0]]
        assert _count_patterns_with_status(rows, "forming") == 1
        assert _count_patterns_with_status(rows, "completed") == 1

    def test_resolve_elliott_pattern_status_uses_recent_window(self):
        from mtdata.core.patterns_support import _resolve_elliott_pattern_status

        assert _resolve_elliott_pattern_status(8, n_bars=10, recent_bars=3) == "forming"
        assert _resolve_elliott_pattern_status(6, n_bars=10, recent_bars=3) == "completed"


# ── _apply_config_to_obj ────────────────────────────────────────────────

class TestApplyConfigToObj:

    def _call(self, cfg, config):
        from mtdata.core.patterns import _apply_config_to_obj
        return _apply_config_to_obj(cfg, config)

    def test_sets_float_attr(self):
        obj = SimpleNamespace(min_prominence_pct=0.5)
        unknown = self._call(obj, {"min_prominence_pct": 0.8})
        assert obj.min_prominence_pct == pytest.approx(0.8)
        assert unknown == []

    def test_sets_int_attr(self):
        obj = SimpleNamespace(min_distance=5)
        unknown = self._call(obj, {"min_distance": 10})
        assert obj.min_distance == 10
        assert unknown == []

    def test_sets_bool_attr_from_string(self):
        obj = SimpleNamespace(use_robust_fit=False)
        unknown = self._call(obj, {"use_robust_fit": "true"})
        assert obj.use_robust_fit is True
        assert unknown == []

    def test_sets_bool_false_from_string(self):
        obj = SimpleNamespace(use_robust_fit=True)
        unknown = self._call(obj, {"use_robust_fit": "false"})
        assert obj.use_robust_fit is False
        assert unknown == []

    def test_sets_list_from_string(self):
        obj = SimpleNamespace(pattern_types=["impulse"])
        unknown = self._call(obj, {"pattern_types": "impulse,correction"})
        assert obj.pattern_types == ["impulse", "correction"]
        assert unknown == []

    def test_sets_list_from_list(self):
        obj = SimpleNamespace(pattern_types=["impulse"])
        unknown = self._call(obj, {"pattern_types": ["a", "b"]})
        assert obj.pattern_types == ["a", "b"]
        assert unknown == []

    def test_ignores_unknown_keys(self):
        obj = SimpleNamespace(x=1)
        unknown = self._call(obj, {"unknown_key": 99})
        assert not hasattr(obj, "unknown_key")
        assert unknown == ["unknown_key"]

    def test_none_config_noop(self):
        obj = SimpleNamespace(x=1)
        unknown = self._call(obj, None)
        assert obj.x == 1
        assert unknown == []

    def test_non_dict_config_noop(self):
        obj = SimpleNamespace(x=1)
        unknown = self._call(obj, "not a dict")
        assert obj.x == 1
        assert unknown == []
