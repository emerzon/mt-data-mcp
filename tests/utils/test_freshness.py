from mtdata.utils.freshness import format_freshness_label


class _FalseLike:
    def __bool__(self):
        return False


class _TrueLike:
    def __bool__(self):
        return True


def test_format_freshness_label_accepts_bool_like_stale_flags():
    assert format_freshness_label(data_stale=_TrueLike()) == "stale"
    assert format_freshness_label(data_stale=_FalseLike()) == "fresh"


def test_format_freshness_label_ignores_textual_stale_flags():
    assert format_freshness_label(data_stale="false") is None
