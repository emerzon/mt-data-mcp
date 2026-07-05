from mtdata.utils.coercion import is_explicit_false


def test_is_explicit_false_distinguishes_missing_from_falsey_values():
    assert is_explicit_false(None) is False
    assert is_explicit_false(False) is True
    assert is_explicit_false(0) is True
    assert is_explicit_false("") is True
    assert is_explicit_false([]) is True
    assert is_explicit_false(True) is False
    assert is_explicit_false(1) is False
