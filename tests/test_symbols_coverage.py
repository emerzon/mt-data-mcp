"""Tests for core/symbols.py — symbols_list, _list_symbol_groups, symbols_describe.

Covers lines 20-199 by mocking MT5.
"""
import pytest
from unittest.mock import patch, MagicMock


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
        assert len(res["data"]) == 2

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
        res = fn("EURUSD")
        assert res["success"] is True
        assert "symbol" in res
        sd = res["symbol"]
        assert sd.get("name") == "EURUSD"
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
        assert "comment" not in res["symbol"]

    @patch(f"{_MT5}.symbol_info")
    def test_skip_empty_string(self, mock_info):
        info = MagicMock()
        info.__dir__ = lambda self: ["name", "description"]
        info.name = "X"
        info.description = ""  # skipped
        mock_info.return_value = info
        fn = _get_symbols_describe()
        res = fn("X")
        assert "description" not in res["symbol"]

    @patch(f"{_MT5}.symbol_info")
    def test_skip_zero_numeric(self, mock_info):
        info = MagicMock()
        info.__dir__ = lambda self: ["name", "trade_mode"]
        info.name = "X"
        info.trade_mode = 0  # skipped
        mock_info.return_value = info
        fn = _get_symbols_describe()
        res = fn("X")
        assert "trade_mode" not in res["symbol"]

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
        res = fn("EURUSD")
        sd = res["symbol"]

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
        res = fn("EURUSD")
        sd = res["symbol"]

        assert sd.get("time_epoch") == 1700000000.0
        assert isinstance(sd.get("time"), str)
        assert ":" in sd.get("time")
        assert "n_fields" not in sd
        assert "n_sequence_fields" not in sd

    @patch(f"{_MT5}.symbol_info", side_effect=RuntimeError("fail"))
    def test_exception(self, mock_info):
        fn = _get_symbols_describe()
        res = fn("X")
        assert "error" in res
