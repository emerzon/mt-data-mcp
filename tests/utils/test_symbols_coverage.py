"""Tests for core/symbols.py — symbols_list, _list_symbol_groups, symbols_describe.

Covers lines 20-199 by mocking MT5.
"""
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_symbol(name, path="Forex\\Majors", description="Euro vs US Dollar", visible=True):
    s = MagicMock()
    s.name = name
    s.path = path
    s.description = description
    s.visible = visible
    return s


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _get_symbols_list():
    from mtdata.core.symbols import symbols_list
    raw = _unwrap(symbols_list)

    def _call(*args, **kwargs):
        with patch("mtdata.core.symbols.ensure_mt5_connection_or_raise", return_value=None):
            return raw(*args, **kwargs)

    return _call


def _get_symbols_describe():
    from mtdata.core.symbols import symbols_describe
    raw = _unwrap(symbols_describe)

    def _call(*args, **kwargs):
        with patch("mtdata.core.symbols.ensure_mt5_connection_or_raise", return_value=None):
            return raw(*args, **kwargs)

    return _call


_MT5 = "mtdata.core.symbols.mt5"
_GROUP_PATH = "mtdata.core.symbols._extract_group_path_util"
_TABLE = "mtdata.core.symbols._table_from_rows"
_NORM_LIMIT = "mtdata.core.symbols._normalize_limit"


# ---------------------------------------------------------------------------
# symbols_list — no search
# ---------------------------------------------------------------------------


class TestSymbolsListNoSearch:

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(_GROUP_PATH, return_value="Forex\\Majors")
    @patch(f"{_MT5}.symbols_get")
    def test_all_visible(self, mock_get, mock_gp, mock_tbl, mock_lim):
        syms = [_make_symbol("EURUSD"), _make_symbol("GBPUSD")]
        mock_get.return_value = syms
        fn = _get_symbols_list()
        res = fn(search_term=None, limit=25)
        assert "data" in res
        assert res["headers"] == ["symbol", "group", "description"]
        assert len(res["data"]) == 2
        assert "rows" not in res
        assert "collection_kind" not in res
        assert "collection_contract_version" not in res

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(_GROUP_PATH, return_value="Forex\\Majors")
    @patch(f"{_MT5}.symbols_get")
    def test_standard_detail_includes_descriptions(self, mock_get, mock_gp, mock_tbl, mock_lim):
        mock_get.return_value = [_make_symbol("EURUSD")]
        fn = _get_symbols_list()

        res = fn(search_term=None, limit=25, detail="standard")

        assert res["headers"] == ["symbol", "group", "description"]
        assert res["data"] == [["EURUSD", "Forex\\Majors", "Euro vs US Dollar"]]

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(_GROUP_PATH, return_value="Forex\\Majors")
    @patch(f"{_MT5}.symbols_get")
    def test_compact_detail_includes_static_identification_fields(
        self, mock_get, mock_gp, mock_tbl, mock_lim
    ):
        symbol = _make_symbol("EURUSD")
        symbol.currency_base = "EUR"
        symbol.currency_profit = "USD"
        symbol.digits = 5
        symbol.spread_float = True
        mock_get.return_value = [symbol]
        fn = _get_symbols_list()

        res = fn(search_term=None, limit=25)

        assert res["headers"] == [
            "symbol",
            "group",
            "description",
            "currency_base",
            "currency_profit",
            "digits",
            "spread_float",
        ]
        assert res["data"] == [
            ["EURUSD", "Forex\\Majors", "Euro vs US Dollar", "EUR", "USD", 5, True]
        ]

    @patch(_NORM_LIMIT, return_value=2)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(_GROUP_PATH, return_value="Forex\\Majors")
    @patch(f"{_MT5}.symbols_get")
    def test_offset_pages_symbol_rows(self, mock_get, mock_gp, mock_tbl, mock_lim):
        mock_get.return_value = [
            _make_symbol("AUDUSD"),
            _make_symbol("EURUSD"),
            _make_symbol("GBPUSD"),
            _make_symbol("USDJPY"),
        ]
        fn = _get_symbols_list()

        res = fn(search_term=None, limit=2, offset=1)

        assert [row[0] for row in res["data"]] == ["EURUSD", "GBPUSD"]
        assert res["total_count"] == 4
        assert res["offset"] == 1
        assert res["limit"] == 2
        assert res["has_more"] is True

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_GROUP_PATH, return_value="Forex\\Majors")
    @patch(f"{_MT5}.symbols_get")
    def test_summary_detail_returns_counts_only(self, mock_get, mock_gp, mock_lim):
        mock_get.return_value = [_make_symbol("EURUSD"), _make_symbol("GBPUSD")]
        fn = _get_symbols_list()

        res = fn(search_term=None, limit=25, detail="summary")

        assert res == {
            "success": True,
            "list_mode": "symbols",
            "count": 2,
            "search_term": None,
            "search_mode": "auto",
            "limit": 25,
        }

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(_GROUP_PATH, return_value="Forex\\Majors")
    @patch(f"{_MT5}.symbols_get")
    def test_hidden_filtered(self, mock_get, mock_gp, mock_tbl, mock_lim):
        syms = [_make_symbol("EURUSD", visible=True), _make_symbol("HIDDEN", visible=False)]
        mock_get.return_value = syms
        fn = _get_symbols_list()
        res = fn(search_term=None, limit=25)
        assert len(res["data"]) == 1

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(_GROUP_PATH, return_value="Forex")
    @patch(f"{_MT5}.symbols_get")
    def test_none_symbols(self, mock_get, mock_gp, mock_tbl, mock_lim):
        mock_get.return_value = None
        fn = _get_symbols_list()
        res = fn(search_term=None)
        # None → empty list
        assert res is not None


# ---------------------------------------------------------------------------
# symbols_list — with search
# ---------------------------------------------------------------------------


class TestSymbolsListSearch:

    def _setup_syms(self):
        return [
            _make_symbol("EURUSD", path="Forex\\Majors", description="Euro vs Dollar"),
            _make_symbol("GBPUSD", path="Forex\\Majors", description="Pound vs Dollar"),
            _make_symbol("XAUUSD", path="Commodities\\Metals", description="Gold"),
            _make_symbol("USDJPY", path="Forex\\Majors", description="Dollar vs Yen"),
        ]

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_search_by_symbol_name(self, mock_get, mock_tbl, mock_lim):
        mock_get.return_value = self._setup_syms()
        with patch(_GROUP_PATH, side_effect=lambda s: s.path):
            fn = _get_symbols_list()
            res = fn(search_term="EUR", limit=25)
        assert "data" in res

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_search_stock_suffixes_include_session_type(self, mock_get, mock_tbl, mock_lim):
        mock_get.return_value = [
            _make_symbol("AAPL.NAS", path="Stock CFD's\\Nasdaq", description="Apple Inc CFD"),
            _make_symbol("AAPL.NAS-24", path="Stock CFD's\\Nasdaq\\24HR NAS", description="Apple Inc 24/5 CFD"),
        ]
        with patch(_GROUP_PATH, side_effect=lambda s: s.path):
            fn = _get_symbols_list()
            res = fn(search_term="AAPL", limit=25)

        assert res["headers"] == ["symbol", "group", "description", "session_type"]
        assert res["data"] == [
            ["AAPL.NAS", "Stock CFD's\\Nasdaq", "Apple Inc CFD", "regular"],
            [
                "AAPL.NAS-24",
                "Stock CFD's\\Nasdaq\\24HR NAS",
                "Apple Inc 24/5 CFD",
                "extended_24h",
            ],
        ]

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_search_trims_padded_symbol_query(self, mock_get, mock_tbl, mock_lim):
        mock_get.return_value = self._setup_syms()
        with patch(_GROUP_PATH, side_effect=lambda s: s.path):
            fn = _get_symbols_list()
            res = fn(search_term="  EUR  ", limit=25)
        assert [row[0] for row in res["data"]] == ["EURUSD"]

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_search_normalizes_slashed_pair_query(self, mock_get, mock_tbl, mock_lim):
        mock_get.return_value = self._setup_syms()
        with patch(_GROUP_PATH, side_effect=lambda s: s.path):
            fn = _get_symbols_list()
            res = fn(search_term="EUR/USD", limit=25)

        assert [row[0] for row in res["data"]] == ["EURUSD"]
        assert res["search"]["term"] == "EURUSD"
        assert res["search"]["normalized_from"] == "EUR/USD"
        assert res["top_match"] == {
            "symbol": "EURUSD",
            "match_reason": "exact_name",
            "group": "Forex\\Majors",
        }

    @patch(_NORM_LIMIT, return_value=5)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_short_currency_search_promotes_major_fx_pairs(self, mock_get, mock_tbl, mock_lim):
        mock_get.return_value = [
            _make_symbol("EURAUD", path="Forex\\Minors", description="Euro vs Aussie"),
            _make_symbol("EURBBL_M6", path="Bonds\\Euro", description="Euro bond"),
            _make_symbol("EURBND_M6", path="Bonds\\Euro", description="Euro bond"),
            _make_symbol("EURCAD", path="Forex\\Minors", description="Euro vs CAD"),
            _make_symbol("EURCHF", path="Forex\\Minors", description="Euro vs CHF"),
            _make_symbol("EURUSD", path="Forex\\Majors", description="Euro vs Dollar"),
        ]
        with patch(_GROUP_PATH, side_effect=lambda s: s.path):
            fn = _get_symbols_list()
            res = fn(search_term="EUR", limit=5)

        assert [row[0] for row in res["data"]][:3] == ["EURUSD", "EURCHF", "EURAUD"]
        assert res["top_match"] == {
            "symbol": "EURUSD",
            "match_reason": "name_prefix",
            "group": "Forex\\Majors",
        }

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_search_results_sorted_case_insensitively(self, mock_get, mock_tbl, mock_lim):
        mock_get.return_value = [
            _make_symbol("usdjpy", path="Forex\\Majors"),
            _make_symbol("EURUSD", path="Forex\\Majors"),
            _make_symbol("gbpusd", path="Forex\\Majors"),
        ]
        with patch(_GROUP_PATH, side_effect=lambda s: s.path):
            fn = _get_symbols_list()
            res = fn(search_term="USD", limit=25)
        assert [row[0] for row in res["data"]] == ["EURUSD", "gbpusd", "usdjpy"]

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_search_matches_group(self, mock_get, mock_tbl, mock_lim):
        """When search matches few groups → use group search."""
        syms = [
            _make_symbol("XAUUSD", path="Commodities\\Metals"),
            _make_symbol("XAGUSD", path="Commodities\\Metals"),
            _make_symbol("EURUSD", path="Forex\\Majors"),
        ]
        mock_get.return_value = syms
        with patch(_GROUP_PATH, side_effect=lambda s: s.path), \
             patch("mtdata.core.symbols.GROUP_SEARCH_THRESHOLD", 5):
            fn = _get_symbols_list()
            res = fn(search_term="Metal", limit=25)
        assert "data" in res

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_search_many_groups_fallback_to_name(self, mock_get, mock_tbl, mock_lim):
        """When search matches many groups → fall back to name search."""
        syms = []
        for i in range(10):
            syms.append(_make_symbol(f"USD{i}", path=f"Group{i}\\USD"))
        mock_get.return_value = syms
        with patch(_GROUP_PATH, side_effect=lambda s: s.path), \
             patch("mtdata.core.symbols.GROUP_SEARCH_THRESHOLD", 3):
            fn = _get_symbols_list()
            res = fn(search_term="USD", limit=25)
        assert "data" in res

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_search_description_fallback(self, mock_get, mock_tbl, mock_lim):
        """When no name or group match, fallback to description."""
        syms = [_make_symbol("SYM1", path="G1", description="Gold Spot")]
        mock_get.return_value = syms
        with patch(_GROUP_PATH, side_effect=lambda s: s.path), \
             patch("mtdata.core.symbols.GROUP_SEARCH_THRESHOLD", 5):
            fn = _get_symbols_list()
            res = fn(search_term="Gold", limit=25)
        assert "data" in res

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_search_path_fallback(self, mock_get, mock_tbl, mock_lim):
        """Description empty but path matches."""
        s = _make_symbol("SYM1", path="Metals\\Gold", description="")
        mock_get.return_value = [s]
        with patch(_GROUP_PATH, side_effect=lambda s: s.path), \
             patch("mtdata.core.symbols.GROUP_SEARCH_THRESHOLD", 5):
            fn = _get_symbols_list()
            res = fn(search_term="Gold", limit=25)
        assert "data" in res

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_search_mode_name_ignores_description_and_group(
        self, mock_get, mock_tbl, mock_lim
    ):
        mock_get.return_value = [
            _make_symbol("SYM1", path="Metals\\Gold", description="Gold Spot"),
        ]
        with patch(_GROUP_PATH, side_effect=lambda s: s.path):
            fn = _get_symbols_list()
            res = fn(search_term="Gold", search_mode="name", limit=25)
        assert res["data"] == []

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_search_mode_exact_requires_full_symbol_name(
        self, mock_get, mock_tbl, mock_lim
    ):
        mock_get.return_value = [
            _make_symbol("EURUSD", path="Forex\\Majors"),
            _make_symbol("EURUSDm", path="Forex\\Majors"),
        ]
        with patch(_GROUP_PATH, side_effect=lambda s: s.path):
            fn = _get_symbols_list()
            res = fn(search_term="EURUSD", search_mode="exact", limit=25)
        assert [row[0] for row in res["data"]] == ["EURUSD"]

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_search_mode_all_matches_name_description_and_group(
        self, mock_get, mock_tbl, mock_lim
    ):
        mock_get.return_value = [
            _make_symbol("GOLDMICRO", path="Metals\\Micro", description="Micro future"),
            _make_symbol("XAUUSD", path="Commodities\\Metals", description="Gold Spot"),
            _make_symbol("SILVER", path="Commodities\\Gold", description="Silver Spot"),
        ]
        with patch(_GROUP_PATH, side_effect=lambda s: s.path):
            fn = _get_symbols_list()
            res = fn(search_term="Gold", search_mode="all", limit=25)
        assert [row[0] for row in res["data"]] == ["GOLDMICRO", "SILVER", "XAUUSD"]

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_search_no_match(self, mock_get, mock_tbl, mock_lim):
        syms = [_make_symbol("EURUSD", path="Forex\\Majors", description="Euro vs Dollar")]
        mock_get.return_value = syms
        with patch(_GROUP_PATH, side_effect=lambda s: s.path), \
             patch("mtdata.core.symbols.GROUP_SEARCH_THRESHOLD", 5):
            fn = _get_symbols_list()
            res = fn(search_term="ZZZZZ", limit=25)
        assert "data" in res
        assert len(res["data"]) == 0

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_search_symbols_get_none(self, mock_get, mock_tbl, mock_lim):
        mock_get.return_value = None
        with patch(f"{_MT5}.last_error", return_value=(0, "no syms")):
            fn = _get_symbols_list()
            res = fn(search_term="EUR")
        assert "error" in res

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_search_with_hidden_symbols(self, mock_get, mock_tbl, mock_lim):
        """Hidden syms are included when search_term is provided."""
        syms = [_make_symbol("EURUSD", visible=False)]
        mock_get.return_value = syms
        with patch(_GROUP_PATH, side_effect=lambda s: s.path):
            fn = _get_symbols_list()
            res = fn(search_term="EUR", limit=25)
        # When searching, only_visible is False → hidden included
        assert "data" in res

    @patch(_NORM_LIMIT, return_value=25)
    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(f"{_MT5}.symbols_get")
    def test_whitespace_only_search_behaves_like_no_search(self, mock_get, mock_tbl, mock_lim):
        syms = [
            _make_symbol("EURUSD", visible=True),
            _make_symbol("HIDDEN", visible=False),
        ]
        mock_get.return_value = syms
        with patch(_GROUP_PATH, side_effect=lambda s: s.path):
            fn = _get_symbols_list()
            res = fn(search_term="   ", limit=25)
        assert [row[0] for row in res["data"]] == ["EURUSD"]


class TestSymbolsListModes:

    def test_invalid_list_mode(self):
        fn = _get_symbols_list()
        res = fn(list_mode="invalid")
        assert res == {"error": "list_mode must be 'symbols' or 'groups'."}

    @patch("mtdata.core.symbols._list_symbol_groups", return_value={"headers": ["group"], "data": []})
    def test_groups_mode(self, mock_lsg):
        fn = _get_symbols_list()
        res = fn(list_mode="groups")
        mock_lsg.assert_called_once()


class TestSymbolsListException:

    @patch(f"{_MT5}.symbols_get", side_effect=RuntimeError("boom"))
    def test_exception(self, mock_get):
        fn = _get_symbols_list()
        res = fn(search_term=None)
        assert "error" in res


def test_symbols_list_logs_finish_event(caplog):
    fn = _get_symbols_list()
    with patch(f"{_MT5}.symbols_get", return_value=[_make_symbol("EURUSD")]), \
         patch(_GROUP_PATH, side_effect=lambda s: s.path), \
         patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r}), \
         patch(_NORM_LIMIT, return_value=25), \
         caplog.at_level("DEBUG", logger="mtdata.core.symbols"):
        res = fn(search_term=None, limit=25)

    assert "data" in res
    assert any(
        "event=finish operation=symbols_list success=True" in record.message
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# _list_symbol_groups
# ---------------------------------------------------------------------------

from mtdata.core.symbols import _list_symbol_groups


class TestListSymbolGroups:

    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(_NORM_LIMIT, return_value=25)
    @patch(_GROUP_PATH, side_effect=lambda s: s.path)
    @patch(f"{_MT5}.symbols_get")
    def test_basic(self, mock_get, mock_gp, mock_lim, mock_tbl):
        syms = [_make_symbol("EURUSD", path="Forex\\Majors"),
                _make_symbol("GBPUSD", path="Forex\\Majors"),
                _make_symbol("XAUUSD", path="Commodities")]
        mock_get.return_value = syms
        res = _list_symbol_groups()
        assert "data" in res
        # Two unique groups
        assert len(res["data"]) == 2

    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(_NORM_LIMIT, return_value=25)
    @patch(_GROUP_PATH, side_effect=lambda s: s.path)
    @patch(f"{_MT5}.symbols_get")
    def test_with_search(self, mock_get, mock_gp, mock_lim, mock_tbl):
        syms = [_make_symbol("EURUSD", path="Forex\\Majors"),
                _make_symbol("XAUUSD", path="Commodities")]
        mock_get.return_value = syms
        res = _list_symbol_groups(search_term="Forex")
        assert "data" in res
        assert len(res["data"]) == 1

    @patch(f"{_MT5}.symbols_get")
    def test_none_symbols(self, mock_get):
        mock_get.return_value = None
        with patch(f"{_MT5}.last_error", return_value=(0, "")):
            res = _list_symbol_groups()
        assert "error" in res

    @patch(f"{_MT5}.symbols_get", side_effect=RuntimeError("fail"))
    def test_exception(self, mock_get):
        res = _list_symbol_groups()
        assert "error" in res

    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(_NORM_LIMIT, return_value=1)
    @patch(_GROUP_PATH, side_effect=lambda s: s.path)
    @patch(f"{_MT5}.symbols_get")
    def test_limit_applied(self, mock_get, mock_gp, mock_lim, mock_tbl):
        syms = [_make_symbol("A", path="G1"), _make_symbol("B", path="G2")]
        mock_get.return_value = syms
        res = _list_symbol_groups(limit=1)
        assert len(res["data"]) == 1

    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(_NORM_LIMIT, return_value=1)
    @patch(_GROUP_PATH, side_effect=lambda s: s.path)
    @patch(f"{_MT5}.symbols_get")
    def test_offset_pages_group_rows(self, mock_get, mock_gp, mock_lim, mock_tbl):
        mock_get.return_value = [
            _make_symbol("A1", path="G1"),
            _make_symbol("A2", path="G1"),
            _make_symbol("B1", path="G2"),
            _make_symbol("C1", path="G3"),
        ]

        res = _list_symbol_groups(limit=1, offset=1)

        assert res["data"] == [["G2", 1, 1, ["B1"]]]
        assert res["total_count"] == 3
        assert res["offset"] == 1
        assert res["limit"] == 1
        assert res["has_more"] is True

    @patch(_TABLE, side_effect=lambda h, r: {"headers": h, "data": r})
    @patch(_NORM_LIMIT, return_value=25)
    @patch(_GROUP_PATH, side_effect=lambda s: s.path)
    @patch(f"{_MT5}.symbols_get")
    def test_sorts_by_count_then_group_name(self, mock_get, mock_gp, mock_lim, mock_tbl):
        mock_get.return_value = [
            _make_symbol("A1", path="Zulu"),
            _make_symbol("A2", path="Zulu"),
            _make_symbol("A3", path="Zulu"),
            _make_symbol("B1", path="beta"),
            _make_symbol("B2", path="beta"),
            _make_symbol("C1", path="Alpha"),
            _make_symbol("C2", path="Alpha"),
        ]
        res = _list_symbol_groups()
        assert res["data"] == [
            ["Zulu", 3, 3, ["A1", "A2", "A3"]],
            ["Alpha", 2, 2, ["C1", "C2"]],
            ["beta", 2, 2, ["B1", "B2"]],
        ]


# ---------------------------------------------------------------------------
# symbols_describe
# ---------------------------------------------------------------------------


class TestSymbolsDescribe:

    @patch(f"{_MT5}.symbol_info")
    def test_symbol_not_found(self, mock_info):
        mock_info.return_value = None
        fn = _get_symbols_describe()
        res = fn("BAD")
        assert "error" in res
        assert "not found" in res["error"]
        assert res["success"] is False
        assert res["error_code"] == "symbol_not_found"
        assert res["operation"] == "symbols_describe"
        assert res["request_id"]

    @patch(f"{_MT5}.symbols_get")
    @patch(f"{_MT5}.symbol_info")
    def test_symbol_not_found_includes_suffix_suggestions(self, mock_info, mock_symbols):
        mock_info.return_value = None
        mock_symbols.return_value = [
            _make_symbol("AAPL.NAS", path="Stock CFD's\\Nasdaq", description="Apple Inc CFD"),
            _make_symbol("AAPL.NAS-24", path="Stock CFD's\\Nasdaq\\24HR NAS", description="Apple Inc 24/5 CFD"),
        ]
        fn = _get_symbols_describe()

        res = fn("AAPL")

        assert "symbols_list(search_term='AAPL')" in res["details"]["search_hint"]
        assert res["details"]["did_you_mean"] == [
            {
                "symbol": "AAPL.NAS",
                "group": "Stock CFD's\\Nasdaq",
                "description": "Apple Inc CFD",
                "session_type": "regular",
            },
            {
                "symbol": "AAPL.NAS-24",
                "group": "Stock CFD's\\Nasdaq\\24HR NAS",
                "description": "Apple Inc 24/5 CFD",
                "session_type": "extended_24h",
            },
        ]

    @patch(f"{_MT5}.symbol_info")
    def test_basic_describe(self, mock_info):
        info = MagicMock()
        # Provide some attributes
        info.__dir__ = lambda self: ["name", "digits", "trade_mode", "spread", "_priv", "method"]
        info.name = "EURUSD"
        info.digits = 5
        info.trade_mode = 2
        info.spread = 10  # excluded
        info._priv = "x"  # excluded (starts with _)
        info.method = lambda: None  # excluded (callable)
        mock_info.return_value = info
        fn = _get_symbols_describe()
        res = fn("EURUSD", detail="full")
        assert res["success"] is True
        assert res["symbol"] == "EURUSD"
        sd = res["details"]
        assert sd.get("digits") == 5
        assert "spread" not in sd  # excluded

    @patch(f"{_MT5}.symbol_info")
    def test_skip_none_values(self, mock_info):
        info = MagicMock()
        info.__dir__ = lambda self: ["name", "comment"]
        info.name = "EURUSD"
        info.comment = None  # skipped
        mock_info.return_value = info
        fn = _get_symbols_describe()
        res = fn("EURUSD")
        assert "comment" not in res["details"]

    @patch(f"{_MT5}.symbol_info")
    def test_skip_empty_string(self, mock_info):
        info = MagicMock()
        info.__dir__ = lambda self: ["name", "description"]
        info.name = "X"
        info.description = ""  # skipped
        mock_info.return_value = info
        fn = _get_symbols_describe()
        res = fn("X")
        assert res["symbol"] == "X"
        assert "description" not in res["details"]

    @patch(f"{_MT5}.symbol_info")
    def test_keeps_zero_numeric(self, mock_info):
        info = MagicMock()
        info.__dir__ = lambda self: ["name", "trade_mode"]
        info.name = "X"
        info.trade_mode = 0
        mock_info.return_value = info
        fn = _get_symbols_describe()
        res = fn("X", detail="full")
        assert res["details"]["trade_mode"] == 0

    @patch(f"{_MT5}.symbol_info")
    def test_decodes_enum_fields_with_labels(self, mock_info):
        info = MagicMock()
        info.__dir__ = lambda self: [
            "name",
            "trade_exemode",
            "trade_calc_mode",
            "order_mode",
            "expiration_mode",
            "filling_mode",
            "swap_mode",
        ]
        info.name = "EURUSD"
        info.trade_exemode = 2
        info.trade_calc_mode = 3
        info.order_mode = 3
        info.expiration_mode = 5
        info.filling_mode = 2
        info.swap_mode = 1
        mock_info.return_value = info

        import mtdata.core.symbols as symbols_mod

        symbols_mod.mt5.SYMBOL_TRADE_EXECUTION_INSTANT = 2
        symbols_mod.mt5.SYMBOL_CALC_MODE_CFD = 3
        symbols_mod.mt5.SYMBOL_ORDER_MARKET = 1
        symbols_mod.mt5.SYMBOL_ORDER_LIMIT = 2
        symbols_mod.mt5.SYMBOL_EXPIRATION_GTC = 1
        symbols_mod.mt5.SYMBOL_EXPIRATION_SPECIFIED = 4
        symbols_mod.mt5.ORDER_FILLING_IOC = 2
        symbols_mod.mt5.SYMBOL_SWAP_MODE_POINTS = 1

        fn = _get_symbols_describe()
        res = fn("EURUSD", detail="full")
        sd = res["details"]

        assert sd.get("trade_exemode_label") == "Market"
        assert "Cfd" in str(sd.get("trade_calc_mode_label"))
        assert "Market" in (sd.get("order_mode_labels") or [])
        assert "Limit" in (sd.get("order_mode_labels") or [])
        assert "GTC" in (sd.get("expiration_mode_labels") or [])
        assert "Specified" in (sd.get("expiration_mode_labels") or [])
        assert any(v in (sd.get("filling_mode_labels") or []) for v in ("IOC", "Return"))
        assert sd.get("swap_mode_label") == "Points"

    @patch(f"{_MT5}.symbol_info")
    def test_formats_time_and_excludes_internal_count_fields(self, mock_info):
        info = MagicMock()
        info.__dir__ = lambda self: ["name", "time", "n_fields", "n_sequence_fields"]
        info.name = "EURUSD"
        info.time = 1700000000
        info.n_fields = 96
        info.n_sequence_fields = 96
        mock_info.return_value = info

        fn = _get_symbols_describe()
        res = fn("EURUSD", detail="full")
        sd = res["details"]

        assert isinstance(sd.get("time"), str)
        assert ":" in sd.get("time")
        assert "T" in sd["time"]
        assert sd["time"].endswith("Z") or sd["time"][-6] in {"+", "-"}
        assert sd["time_epoch"] == 1700000000.0
        assert "n_fields" not in sd
        assert "n_sequence_fields" not in sd

    @patch("mtdata.core.symbols.time.time", return_value=1700000301.0)
    @patch(f"{_MT5}.symbol_info")
    def test_default_describe_uses_compact_detail(self, mock_info, mock_time):
        del mock_time
        info = MagicMock()
        info.__dir__ = lambda self: ["name", "time", "digits", "point"]
        info.name = "EURUSD"
        info.time = 1700000000
        info.digits = 5
        info.point = 0.00001
        mock_info.return_value = info

        fn = _get_symbols_describe()
        res = fn("EURUSD")
        sd = res["details"]

        assert res["symbol"] == "EURUSD"
        assert res["timezone"] == "UTC"
        assert sd["digits"] == 5
        assert sd["point"] == 0.00001
        assert sd["freshness"] == "stale, tick 5m 1s ago"
        assert "data_age_seconds" not in sd
        assert "data_stale" not in sd
        assert "stale_after_seconds" not in sd
        assert "Live quote timestamp" in sd["warning"]
        assert "quote_age_seconds" not in sd
        assert "time_epoch" not in sd

    @patch("mtdata.core.symbols.time.time", return_value=1779656400.0)
    @patch(f"{_MT5}.symbol_info")
    def test_describe_treats_weekend_gap_as_closed_market(self, mock_info, mock_time):
        del mock_time
        info = MagicMock()
        info.__dir__ = lambda self: ["name", "time", "digits", "point"]
        info.name = "EURUSD"
        info.time = 1779483360
        info.digits = 5
        info.point = 0.00001
        mock_info.return_value = info

        fn = _get_symbols_describe()
        res = fn("EURUSD")
        sd = res["details"]

        assert sd["freshness"].startswith("closed weekend, tick ")
        assert "data_stale" not in sd
        assert "stale_after_seconds" not in sd
        assert sd["market_status"] == "closed"
        assert sd["market_status_reason"] == "weekend"
        assert "latest completed session tick" in sd["note"]
        assert "warning" not in sd

    @patch(f"{_MT5}.symbol_info")
    def test_describe_warns_when_crypto_base_matches_profit_currency(self, mock_info):
        info = MagicMock()
        info.__dir__ = lambda self: ["name", "description", "currency_base", "currency_profit"]
        info.name = "BTCUSD"
        info.description = "Bitcoin (USD)"
        info.currency_base = "USD"
        info.currency_profit = "USD"
        mock_info.return_value = info

        fn = _get_symbols_describe()
        res = fn("BTCUSD")
        sd = res["details"]

        assert res["symbol"] == "BTCUSD"
        assert sd["currency_base"] == "USD"
        assert sd["currency_base_reported"] == "USD"
        assert sd["currency_base_source"] == "reported_by_mt5"
        assert sd["currency_profit"] == "USD"
        assert sd["currency_base_inferred"] == "BTC"
        assert sd["currency_base_inference_source"] == "inferred_from_symbol_name"
        assert "verify broker metadata" in sd["currency_base_warning"]
        assert res["warnings"] == [sd["currency_base_warning"]]
        assert res["trust"] == "verify_broker_metadata"

    @patch(f"{_MT5}.symbol_info")
    def test_full_detail_describe_adds_time_epoch(self, mock_info):
        info = MagicMock()
        info.__dir__ = lambda self: ["name", "time"]
        info.name = "EURUSD"
        info.time = 1700000000
        mock_info.return_value = info

        fn = _get_symbols_describe()
        res = fn("EURUSD", detail="full")

        assert res["details"]["time_epoch"] == 1700000000.0
        assert isinstance(res["details"]["time"], str)

    @patch(f"{_MT5}.symbol_info", side_effect=RuntimeError("fail"))
    def test_exception(self, mock_info):
        fn = _get_symbols_describe()
        res = fn("X")
        assert "error" in res

    @patch(f"{_MT5}.symbol_info")
    def test_omits_non_curated_symbol_fields(self, mock_info):
        info = MagicMock()
        info.__dir__ = lambda self: [
            "name",
            "digits",
            "price_greeks_delta",
            "volumehigh",
            "session_buy_orders",
            "trade_mode",
        ]
        info.name = "BTCUSD"
        info.digits = 2
        info.price_greeks_delta = 0.0
        info.volumehigh = 0.0
        info.session_buy_orders = 0
        info.trade_mode = 4
        mock_info.return_value = info

        fn = _get_symbols_describe()
        res = fn("BTCUSD", detail="full")
        sd = res["details"]

        assert res["symbol"] == "BTCUSD"
        assert sd["digits"] == 2
        assert sd["trade_mode"] == 4
        assert "price_greeks_delta" not in sd
        assert "volumehigh" not in sd
        assert "session_buy_orders" not in sd

    @patch(f"{_MT5}.symbol_info")
    def test_rounds_price_like_fields_to_symbol_digits(self, mock_info):
        info = MagicMock()
        info.__dir__ = lambda self: [
            "name",
            "digits",
            "bidlow",
            "bidhigh",
            "asklow",
            "askhigh",
            "price_change",
            "session_open",
            "session_close",
        ]
        info.name = "XAUUSD"
        info.digits = 2
        info.bidlow = 4744.295
        info.bidhigh = 4778.004
        info.asklow = 4744.305
        info.askhigh = 4778.014
        info.price_change = -0.7924
        info.session_open = 4750.126
        info.session_close = 4760.874
        mock_info.return_value = info

        fn = _get_symbols_describe()
        res = fn("XAUUSD", detail="full")
        sd = res["details"]

        assert sd["bidlow"] == 4744.3
        assert sd["bidhigh"] == 4778.0
        assert sd["asklow"] == 4744.31
        assert sd["askhigh"] == 4778.01
        assert sd["session_open"] == 4750.13
        assert sd["session_close"] == 4760.87
        assert sd["price_change_pct"] == -0.7924
        assert sd["price_change_pct_unit"] == "percentage_points (1.0 = 1%)"
        assert "price_change" not in sd

    @patch(f"{_MT5}.symbol_info")
    def test_compact_detail_keeps_canonical_fields_with_labels(self, mock_info):
        info = MagicMock()
        info.__dir__ = lambda self: [
            "name",
            "description",
            "digits",
            "trade_mode",
            "order_mode",
            "trade_tick_value",
            "trade_tick_value_profit",
            "time",
        ]
        info.name = "EURUSD"
        info.description = "Euro vs Dollar"
        info.digits = 5
        info.trade_mode = 2
        info.order_mode = 3
        info.trade_tick_value = 1.25
        info.trade_tick_value_profit = 1.3
        info.time = 1700000000
        mock_info.return_value = info

        import mtdata.core.symbols as symbols_mod

        symbols_mod.mt5.SYMBOL_TRADE_MODE_LONGONLY = 2
        symbols_mod.mt5.SYMBOL_ORDER_MARKET = 1
        symbols_mod.mt5.SYMBOL_ORDER_LIMIT = 2

        fn = _get_symbols_describe()
        res = fn("EURUSD", detail="compact")
        sd = res["details"]

        assert res["symbol"] == "EURUSD"
        assert "trade_mode" not in sd
        assert sd["trade_mode_label"] == "Longonly"
        assert "Market" in sd["order_mode_labels"]
        assert "Limit" in sd["order_mode_labels"]
        assert sd["trade_tick_value"] == 1.25
        assert "trade_tick_value_profit" not in sd
        assert "time_epoch" not in sd

    @patch("mtdata.core.symbols.time.time", return_value=1700000301.0)
    @patch(f"{_MT5}.symbol_info")
    def test_summary_detail_omits_trading_specs(self, mock_info, mock_time):
        del mock_time
        info = MagicMock()
        info.__dir__ = lambda self: [
            "name",
            "description",
            "currency_base",
            "currency_profit",
            "digits",
            "point",
            "trade_contract_size",
            "trade_tick_size",
            "trade_tick_value",
            "margin_initial",
            "margin_maintenance",
            "volume_min",
            "volume_max",
            "volume_step",
            "time",
        ]
        info.name = "EURUSD"
        info.description = "Euro vs Dollar"
        info.currency_base = "EUR"
        info.currency_profit = "USD"
        info.digits = 5
        info.point = 0.00001
        info.trade_contract_size = 100000.0
        info.trade_tick_size = 0.00001
        info.trade_tick_value = 1.0
        info.margin_initial = 1000.0
        info.margin_maintenance = 500.0
        info.volume_min = 0.01
        info.volume_max = 100.0
        info.volume_step = 0.01
        info.time = 1700000000
        mock_info.return_value = info

        fn = _get_symbols_describe()
        res = fn("EURUSD", detail="summary")
        sd = res["details"]

        assert res["symbol"] == "EURUSD"
        assert sd["description"] == "Euro vs Dollar"
        assert sd["currency_base"] == "EUR"
        assert sd["currency_profit"] == "USD"
        assert sd["freshness"] == "stale, tick 5m 1s ago"
        for trading_key in (
            "digits",
            "point",
            "trade_contract_size",
            "trade_tick_size",
            "trade_tick_value",
            "margin_initial",
            "margin_maintenance",
            "volume_min",
            "volume_max",
            "volume_step",
            "time_epoch",
        ):
            assert trading_key not in sd

    @patch(f"{_MT5}.symbol_info")
    def test_compact_detail_uses_full_detail_field_names(self, mock_info):
        info = MagicMock()
        info.__dir__ = lambda self: [
            "name",
            "digits",
            "point",
            "bidlow",
            "bidhigh",
            "asklow",
            "askhigh",
            "trade_contract_size",
            "trade_tick_size",
            "trade_tick_value",
            "margin_initial",
            "margin_maintenance",
            "volume_min",
            "volume_max",
            "volume_step",
            "order_mode",
        ]
        info.name = "EURUSD"
        info.digits = 5
        info.point = 0.00001
        info.bidlow = 1.16724
        info.bidhigh = 1.17223
        info.asklow = 1.16734
        info.askhigh = 1.17242
        info.trade_contract_size = 100000.0
        info.trade_tick_size = 0.00001
        info.trade_tick_value = 1.0
        info.margin_initial = 1000.0
        info.margin_maintenance = 500.0
        info.volume_min = 0.01
        info.volume_max = 100.0
        info.volume_step = 0.01
        info.order_mode = 3
        mock_info.return_value = info

        import mtdata.core.symbols as symbols_mod

        symbols_mod.mt5.SYMBOL_ORDER_MARKET = 1
        symbols_mod.mt5.SYMBOL_ORDER_LIMIT = 2

        fn = _get_symbols_describe()
        res = fn("EURUSD", detail="compact")
        sd = res["details"]

        assert res["symbol"] == "EURUSD"
        assert sd["digits"] == 5
        assert sd["point"] == 0.00001
        assert sd["trade_contract_size"] == 100000.0
        assert sd["trade_tick_size"] == 0.00001
        assert sd["trade_tick_value"] == 1.0
        assert sd["margin_initial"] == 1000.0
        assert sd["margin_maintenance"] == 500.0
        assert sd["volume_min"] == 0.01
        assert sd["volume_max"] == 100.0
        assert sd["volume_step"] == 0.01
        assert sd["order_mode_labels"] == ["Market", "Limit"]
        for raw_key in (
            "bidlow",
            "bidhigh",
            "asklow",
            "askhigh",
            "order_mode",
            "price_precision",
            "point_size",
            "contract_size",
            "tick_size",
            "tick_value",
            "min_volume",
            "max_volume",
        ):
            assert raw_key not in sd

    @patch(f"{_MT5}.symbol_info")
    def test_compact_detail_includes_core_contract_spec_fields(self, mock_info):
        info = MagicMock()
        info.__dir__ = lambda self: [
            "name",
            "trade_exemode",
            "trade_calc_mode",
            "filling_mode",
            "swap_mode",
            "swap_long",
            "swap_short",
            "trade_stops_level",
            "trade_freeze_level",
        ]
        info.name = "XAUUSD"
        info.trade_exemode = 2
        info.trade_calc_mode = 3
        info.filling_mode = 2
        info.swap_mode = 1
        info.swap_long = -54.5
        info.swap_short = 46.6
        info.trade_stops_level = 10
        info.trade_freeze_level = 5
        mock_info.return_value = info

        import mtdata.core.symbols as symbols_mod

        symbols_mod.mt5.SYMBOL_TRADE_EXECUTION_INSTANT = 2
        symbols_mod.mt5.SYMBOL_CALC_MODE_CFD = 3
        symbols_mod.mt5.ORDER_FILLING_IOC = 2
        symbols_mod.mt5.SYMBOL_SWAP_MODE_POINTS = 1

        fn = _get_symbols_describe()
        res = fn("XAUUSD", detail="compact")
        sd = res["details"]

        assert sd["trade_exemode_label"] == "Market"
        assert "Cfd" in sd["trade_calc_mode_label"]
        assert "IOC" in sd["filling_mode_labels"]
        assert sd["swap_mode_label"] == "Points"
        assert sd["swap_long"] == -54.5
        assert sd["swap_short"] == 46.6
        assert sd["trade_stops_level"] == 10
        assert sd["trade_freeze_level"] == 5
        for raw_key in (
            "trade_exemode",
            "trade_calc_mode",
            "filling_mode",
            "swap_mode",
        ):
            assert raw_key not in sd
