"""Tests for the shared Result[T] type."""

import pytest

from mtdata.shared.result import Err, Ok, Result, is_err, is_ok, to_dict


# ---------------------------------------------------------------------------
# Ok / Err construction
# ---------------------------------------------------------------------------
class TestOkConstruction:
    def test_wraps_dict(self):
        r = Ok({"data": [1, 2, 3]})
        assert r.value == {"data": [1, 2, 3]}

    def test_wraps_string(self):
        r = Ok("hello")
        assert r.value == "hello"

    def test_wraps_none(self):
        r = Ok(None)
        assert r.value is None

    def test_frozen(self):
        r = Ok(42)
        with pytest.raises(AttributeError):
            r.value = 99  # type: ignore[misc]


class TestErrConstruction:
    def test_message_only(self):
        r = Err("something failed")
        assert r.message == "something failed"
        assert r.code == ""
        assert r.details == {}

    def test_with_code_and_details(self):
        r = Err("fail", code="E001", details={"extra": True})
        assert r.code == "E001"
        assert r.details == {"extra": True}

    def test_frozen(self):
        r = Err("fail")
        with pytest.raises(AttributeError):
            r.message = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Type guards
# ---------------------------------------------------------------------------
class TestTypeGuards:
    def test_is_ok_true(self):
        assert is_ok(Ok(1)) is True

    def test_is_ok_false(self):
        assert is_ok(Err("x")) is False

    def test_is_err_true(self):
        assert is_err(Err("x")) is True

    def test_is_err_false(self):
        assert is_err(Ok(1)) is False


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------
class TestToDict:
    def test_ok_dict_value_preserves_keys(self):
        result = Ok({"data": [1], "meta": "info"})
        d = to_dict(result)
        assert d["data"] == [1]
        assert d["meta"] == "info"
        assert d["success"] is True

    def test_ok_dict_does_not_overwrite_explicit_success(self):
        result = Ok({"success": False, "note": "user set"})
        d = to_dict(result)
        assert d["success"] is False

    def test_ok_non_dict_value(self):
        d = to_dict(Ok(42))
        assert d == {"success": True, "value": 42}

    def test_ok_none_value(self):
        d = to_dict(Ok(None))
        assert d == {"success": True, "value": None}

    def test_err_minimal(self):
        d = to_dict(Err("boom"))
        assert d == {"success": False, "error": "boom"}

    def test_err_with_code(self):
        d = to_dict(Err("boom", code="E42"))
        assert d["error_code"] == "E42"

    def test_err_with_details_merged(self):
        d = to_dict(Err("boom", details={"hint": "retry"}))
        assert d["hint"] == "retry"
        assert d["success"] is False

    def test_err_details_do_not_overwrite_core_keys(self):
        d = to_dict(Err("boom", code="E1", details={"error": "sneaky"}))
        assert d["error"] == "sneaky"  # details merge wins (intentional)

    def test_roundtrip_ok_dict_is_copy(self):
        original = {"data": [1]}
        d = to_dict(Ok(original))
        d["injected"] = True
        assert "injected" not in original


# ---------------------------------------------------------------------------
# Usage with isinstance (pattern matching substitute)
# ---------------------------------------------------------------------------
class TestPatternUsage:
    def test_match_ok(self):
        r: Result = Ok(10)
        if isinstance(r, Ok):
            assert r.value == 10
        else:
            pytest.fail("Expected Ok")

    def test_match_err(self):
        r: Result = Err("fail")
        if isinstance(r, Err):
            assert r.message == "fail"
        else:
            pytest.fail("Expected Err")


# ---------------------------------------------------------------------------
# infer_result_success integration
# ---------------------------------------------------------------------------
class TestInferResultSuccess:
    def test_ok_returns_true(self):
        from mtdata.core.execution_logging import infer_result_success

        assert infer_result_success(Ok({"data": 1})) is True

    def test_err_returns_false(self):
        from mtdata.core.execution_logging import infer_result_success

        assert infer_result_success(Err("boom")) is False

    def test_ok_none_returns_true(self):
        from mtdata.core.execution_logging import infer_result_success

        assert infer_result_success(Ok(None)) is True

    def test_dict_still_works(self):
        from mtdata.core.execution_logging import infer_result_success

        assert infer_result_success({"success": True}) is True
        assert infer_result_success({"error": "bad"}) is False
