from mtdata.utils.mt5 import MT5Adapter


def test_default_mt5_stub_models_disconnected_empty_state(mt5_module):
    adapter = MT5Adapter()

    assert adapter.initialize() is False
    assert adapter.account_info() is None
    assert adapter.symbol_info("UNKNOWN") is None
    assert adapter.positions_get() == ()
    assert adapter.orders_get() == ()
    assert adapter.copy_rates_from_pos("EURUSD", 60, 0, 10) is None
    assert adapter.order_send({"symbol": "EURUSD"}) is None
