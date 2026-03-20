from types import SimpleNamespace

from mtdata.core import trading_positions


class _FakeMt5:
    ORDER_TYPE_BUY = 0
    POSITION_TYPE_BUY = 0

    def __init__(self, *, ticket_rows=None, fallback_rows=None):
        self._ticket_rows = ticket_rows or []
        self._fallback_rows = fallback_rows or []

    def positions_get(self, **kwargs):
        if "ticket" in kwargs:
            return list(self._ticket_rows)
        return list(self._fallback_rows)


def test_resolve_open_position_uses_candidate_ticket_when_direct_lookup_ticket_is_invalid():
    mt5 = _FakeMt5(
        ticket_rows=[
            SimpleNamespace(
                ticket=0,
                identifier=None,
                position_id=None,
                position=None,
                order=None,
                deal=None,
                symbol="EURUSD",
                type=0,
                volume=0.1,
                time_update_msc=1000,
            )
        ]
    )

    pos, resolved_ticket, info = trading_positions._resolve_open_position(
        mt5,
        ticket_candidates=[456],
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
    )

    assert pos is not None
    assert resolved_ticket == 456
    assert info["method"] == "positions_get(ticket)"


def test_resolve_open_position_uses_exact_match_value_when_ticket_field_is_invalid():
    mt5 = _FakeMt5(
        ticket_rows=[],
        fallback_rows=[
            SimpleNamespace(
                ticket=0,
                identifier=789,
                position_id=None,
                position=None,
                order=None,
                deal=None,
                symbol="EURUSD",
                type=0,
                volume=0.1,
                time_update_msc=1000,
            )
        ],
    )

    pos, resolved_ticket, info = trading_positions._resolve_open_position(
        mt5,
        ticket_candidates=[789],
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
    )

    assert pos is not None
    assert resolved_ticket == 789
    assert info["method"] == "positions_get(fallback_exact)"
    assert info["matched_field"] == "identifier"


def test_resolve_open_position_uses_other_ticket_like_fields_in_heuristic_path():
    mt5 = _FakeMt5(
        ticket_rows=[],
        fallback_rows=[
            SimpleNamespace(
                ticket=0,
                identifier=None,
                position_id=None,
                position=None,
                order=999,
                deal=None,
                symbol="EURUSD",
                type=0,
                volume=0.1,
                time_update_msc=1000,
            )
        ],
    )

    pos, resolved_ticket, info = trading_positions._resolve_open_position(
        mt5,
        ticket_candidates=[],
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
    )

    assert pos is not None
    assert resolved_ticket == 999
    assert info["method"] == "positions_get(fallback_heuristic)"
