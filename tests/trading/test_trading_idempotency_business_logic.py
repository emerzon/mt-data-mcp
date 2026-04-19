import threading
from unittest.mock import MagicMock

from mtdata.core.trading.idempotency import IdempotencyStore
from mtdata.core.trading.requests import TradeModifyRequest, TradePlaceRequest
from mtdata.core.trading.use_cases import run_trade_modify, run_trade_place


def test_run_trade_place_replays_duplicate_result_without_resending():
    store = IdempotencyStore()
    place_market_order = MagicMock(return_value={"success": True, "order_id": 7})
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="place-1",
    )

    first = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=MagicMock(),
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
        idempotency_store=store,
    )
    second = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=MagicMock(),
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
        idempotency_store=store,
    )

    assert first == {"success": True, "order_id": 7}
    assert second["duplicate"] is True
    assert second["success"] is True
    assert second["original_outcome"] == first
    place_market_order.assert_called_once()


def test_run_trade_place_rejects_idempotency_key_reuse_for_different_payload():
    store = IdempotencyStore()
    place_market_order = MagicMock(return_value={"success": True, "order_id": 7})

    first_request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="place-1",
    )
    second_request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.2,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="place-1",
    )

    first = run_trade_place(
        first_request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=MagicMock(),
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
        idempotency_store=store,
    )
    second = run_trade_place(
        second_request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=MagicMock(),
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
        idempotency_store=store,
    )

    assert first == {"success": True, "order_id": 7}
    assert "error" in second
    assert second["idempotency_conflict"] is True
    assert "different trade request" in second["error"]
    place_market_order.assert_called_once()


def test_run_trade_modify_replays_duplicate_result_without_resending():
    store = IdempotencyStore()
    modify_position = MagicMock(return_value={"success": True, "ticket": 123})
    modify_pending_order = MagicMock()
    request = TradeModifyRequest(
        ticket=123,
        stop_loss=1.0,
        idempotency_key="modify-1",
    )

    first = run_trade_modify(
        request,
        normalize_pending_expiration=lambda value: (value, False),
        modify_pending_order=modify_pending_order,
        modify_position=modify_position,
        idempotency_store=store,
    )
    second = run_trade_modify(
        request,
        normalize_pending_expiration=lambda value: (value, False),
        modify_pending_order=modify_pending_order,
        modify_position=modify_position,
        idempotency_store=store,
    )

    assert first == {"success": True, "ticket": 123}
    assert second["duplicate"] is True
    assert second["success"] is True
    assert second["original_outcome"] == first
    modify_position.assert_called_once()
    modify_pending_order.assert_not_called()


def test_run_trade_place_replays_inflight_duplicate_after_first_request_finishes():
    store = IdempotencyStore()
    release_first_call = threading.Event()
    first_call_started = threading.Event()
    call_count = 0
    call_count_lock = threading.Lock()

    def place_market_order(**kwargs):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            first_call_started.set()
            assert release_first_call.wait(timeout=1.0)
        return {"success": True, "order_id": 7}

    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="place-1",
    )
    results = [None, None]

    def _run(index: int) -> None:
        results[index] = run_trade_place(
            request,
            normalize_order_type_input=lambda value: ("BUY", None),
            normalize_pending_expiration=lambda value: (value, False),
            prevalidate_trade_place_market_input=lambda symbol, volume: None,
            place_market_order=place_market_order,
            place_pending_order=MagicMock(),
            close_positions=lambda **kwargs: {"closed_count": 1},
            safe_int_ticket=lambda value: value,
            idempotency_store=store,
        )

    first_thread = threading.Thread(target=_run, args=(0,))
    second_thread = threading.Thread(target=_run, args=(1,))

    first_thread.start()
    assert first_call_started.wait(timeout=1.0)
    second_thread.start()
    release_first_call.set()
    first_thread.join(timeout=1.0)
    second_thread.join(timeout=1.0)

    assert results[0] == {"success": True, "order_id": 7}
    assert results[1]["duplicate"] is True
    assert results[1]["original_outcome"] == results[0]
    assert call_count == 1


def test_run_trade_place_rejects_inflight_key_reuse_for_different_payload():
    store = IdempotencyStore()
    release_first_call = threading.Event()
    first_call_started = threading.Event()
    call_count = 0
    call_count_lock = threading.Lock()

    def place_market_order(**kwargs):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            first_call_started.set()
            assert release_first_call.wait(timeout=1.0)
        return {"success": True, "order_id": 7}

    first_request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="place-1",
    )
    second_request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.2,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="place-1",
    )
    results = [None, None]

    def _run(request, index: int) -> None:
        results[index] = run_trade_place(
            request,
            normalize_order_type_input=lambda value: ("BUY", None),
            normalize_pending_expiration=lambda value: (value, False),
            prevalidate_trade_place_market_input=lambda symbol, volume: None,
            place_market_order=place_market_order,
            place_pending_order=MagicMock(),
            close_positions=lambda **kwargs: {"closed_count": 1},
            safe_int_ticket=lambda value: value,
            idempotency_store=store,
        )

    first_thread = threading.Thread(target=_run, args=(first_request, 0))
    second_thread = threading.Thread(target=_run, args=(second_request, 1))

    first_thread.start()
    assert first_call_started.wait(timeout=1.0)
    second_thread.start()
    second_thread.join(timeout=1.0)
    release_first_call.set()
    first_thread.join(timeout=1.0)

    assert results[0] == {"success": True, "order_id": 7}
    assert "error" in results[1]
    assert results[1]["idempotency_conflict"] is True
    assert call_count == 1
