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


def test_closed_session_context_marks_other_non_crypto_weekend_markets() -> None:
    saturday = datetime(2026, 6, 6, 12, tzinfo=timezone.utc).timestamp()

    assert closed_session_context("US500", now_epoch=saturday)["market_status"] == "closed"
    assert closed_session_context("XAUUSD", now_epoch=saturday)["market_status"] == "closed"


def test_closed_session_context_allows_fx_after_sunday_utc_reopen() -> None:
    sunday_reopen = datetime(2026, 6, 14, 21, 30, tzinfo=timezone.utc).timestamp()

    assert closed_session_context("EURUSD", now_epoch=sunday_reopen) is None


def test_closed_session_context_does_not_relax_very_old_data():
    saturday = datetime(2026, 6, 6, 12, tzinfo=timezone.utc).timestamp()

    result = closed_session_context(
        "EURUSD",
        now_epoch=saturday,
        data_age_seconds=4 * 24 * 60 * 60,
    )

    assert result is not None
    assert result["freshness_policy_relaxed"] is False


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
