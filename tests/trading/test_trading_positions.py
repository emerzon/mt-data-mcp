from types import SimpleNamespace

from mtdata.core.trading import positions


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


def test_position_side_matches_uses_numeric_defaults_when_mt5_constants_are_missing():
    mt5 = SimpleNamespace()

    assert positions._position_side_matches(SimpleNamespace(type=0), "BUY", mt5) is True
    assert positions._position_side_matches(SimpleNamespace(type=1), "BUY", mt5) is False
    assert positions._position_side_matches(SimpleNamespace(type=1), "SELL", mt5) is True


def test_resolve_open_position_respects_side_filter_when_mt5_constants_are_missing():
    class _NoConstantsMt5:
        def __init__(self, rows):
            self._rows = rows

        def positions_get(self, **kwargs):
            return list(self._rows)

    rows = [
        SimpleNamespace(
            ticket=100,
            identifier=100,
            position_id=None,
            position=None,
            order=None,
            deal=None,
            symbol="EURUSD",
            type=0,
            volume=0.1,
            time_update_msc=1000,
        ),
        SimpleNamespace(
            ticket=200,
            identifier=200,
            position_id=None,
            position=None,
            order=None,
            deal=None,
            symbol="EURUSD",
            type=1,
            volume=0.1,
            time_update_msc=2000,
        ),
    ]

    pos, resolved_ticket, info = positions._resolve_open_position(
        _NoConstantsMt5(rows),
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
    )

    assert pos is not None
    assert resolved_ticket == 100
    assert info["method"] == "positions_get(fallback_heuristic)"


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

    pos, resolved_ticket, info = positions._resolve_open_position(
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

    pos, resolved_ticket, info = positions._resolve_open_position(
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

    pos, resolved_ticket, info = positions._resolve_open_position(
        mt5,
        ticket_candidates=[],
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
    )

    assert pos is not None
    assert resolved_ticket == 999
    assert info["method"] == "positions_get(fallback_heuristic)"


# ---------------------------------------------------------------------------
# Hedged-account tests (magic-based disambiguation)
# ---------------------------------------------------------------------------


class _HedgedFakeMt5:
    """Simulates a hedged account with multiple positions for the same symbol."""
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1

    def __init__(self, all_positions):
        self._all = all_positions

    def positions_get(self, **kwargs):
        if "ticket" in kwargs:
            return [p for p in self._all if p.ticket == kwargs["ticket"]] or None
        if "symbol" in kwargs:
            sym = kwargs["symbol"].upper()
            return [p for p in self._all if str(getattr(p, "symbol", "")).upper() == sym] or None
        return list(self._all)


def test_select_position_candidate_magic_disambiguates():
    """Magic number narrows candidates in hedged accounts."""
    positions_list = [
        SimpleNamespace(ticket=100, symbol="EURUSD", type=0, volume=0.1, magic=1000, time_update_msc=5000),
        SimpleNamespace(ticket=200, symbol="EURUSD", type=0, volume=0.1, magic=2000, time_update_msc=6000),
        SimpleNamespace(ticket=300, symbol="EURUSD", type=0, volume=0.1, magic=1000, time_update_msc=7000),
    ]
    mt5 = _HedgedFakeMt5(positions_list)

    # Without magic: picks most recent (ticket 300)
    picked = positions._select_position_candidate(
        positions_list, symbol="EURUSD", side="BUY", volume=0.1, mt5=mt5,
    )
    assert picked.ticket == 300

    # With magic=2000: narrows to ticket 200
    picked = positions._select_position_candidate(
        positions_list, symbol="EURUSD", side="BUY", volume=0.1, magic=2000, mt5=mt5,
    )
    assert picked.ticket == 200

    # With magic=1000: two match, picks most recent (ticket 300)
    picked = positions._select_position_candidate(
        positions_list, symbol="EURUSD", side="BUY", volume=0.1, magic=1000, mt5=mt5,
    )
    assert picked.ticket == 300


def test_resolve_open_position_magic_filter_hedged():
    """_resolve_open_position uses magic to disambiguate hedged positions."""
    all_positions = [
        SimpleNamespace(ticket=100, identifier=100, position_id=None, position=None, order=None, deal=None,
                        symbol="EURUSD", type=0, volume=0.1, magic=1000, time_update_msc=5000),
        SimpleNamespace(ticket=200, identifier=200, position_id=None, position=None, order=None, deal=None,
                        symbol="EURUSD", type=0, volume=0.1, magic=2000, time_update_msc=6000),
    ]
    mt5 = _HedgedFakeMt5(all_positions)

    # Fallback heuristic without magic picks most recent
    pos, ticket, info = positions._resolve_open_position(
        mt5, symbol="EURUSD", side="BUY", volume=0.1,
    )
    assert pos.ticket == 200

    # With magic=1000, narrows to ticket 100
    pos, ticket, info = positions._resolve_open_position(
        mt5, symbol="EURUSD", side="BUY", volume=0.1, magic=1000,
    )
    assert pos.ticket == 100
    assert info.get("magic_filter") == 1000


def test_resolve_open_position_magic_no_match_falls_through():
    """If no position matches the magic filter, still resolves via other criteria."""
    all_positions = [
        SimpleNamespace(ticket=100, identifier=100, position_id=None, position=None, order=None, deal=None,
                        symbol="EURUSD", type=0, volume=0.1, magic=1000, time_update_msc=5000),
    ]
    mt5 = _HedgedFakeMt5(all_positions)

    # Magic 9999 doesn't match, but position still found via symbol/side/volume
    pos, ticket, info = positions._resolve_open_position(
        mt5, symbol="EURUSD", side="BUY", volume=0.1, magic=9999,
    )
    assert pos is not None
    assert pos.ticket == 100
