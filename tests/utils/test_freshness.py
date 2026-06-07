from datetime import datetime, timezone

from mtdata.utils.freshness import closed_session_context, format_freshness_label


def test_closed_session_context_marks_weekend_fx_but_not_crypto():
    saturday = datetime(2026, 6, 6, 12, tzinfo=timezone.utc).timestamp()

    assert closed_session_context("EURUSD", now_epoch=saturday) == {
        "market_status": "closed",
        "market_status_reason": "weekend",
        "market_status_source": "standard_weekend_hours",
        "note": "Market is closed; showing the latest completed session tick.",
    }
    assert closed_session_context("BTCUSD", now_epoch=saturday) is None


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
