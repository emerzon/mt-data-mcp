from mtdata.utils.coercion import UNPARSED_BOOL, is_explicit_false, parse_bool_like


def test_parse_bool_like_accepts_canonical_boolean_values():
    for value in (True, 1, "true", " YES ", "y", "on"):
        assert parse_bool_like(value) is True
    for value in (False, 0, "false", " NO ", "n", "off"):
        assert parse_bool_like(value) is False


def test_parse_bool_like_distinguishes_null_and_unrecognized_values():
    assert parse_bool_like(None) is UNPARSED_BOOL
    assert parse_bool_like("null") is UNPARSED_BOOL
    assert parse_bool_like([], allow_none=True) is UNPARSED_BOOL
    assert parse_bool_like(None, allow_none=True) is None
    assert parse_bool_like("null", allow_none=True) is None


def test_is_explicit_false_distinguishes_missing_from_falsey_values():
    assert is_explicit_false(None) is False
    assert is_explicit_false(False) is True
    assert is_explicit_false(0) is True
    assert is_explicit_false("") is True
    assert is_explicit_false([]) is True
    assert is_explicit_false(True) is False
    assert is_explicit_false(1) is False
