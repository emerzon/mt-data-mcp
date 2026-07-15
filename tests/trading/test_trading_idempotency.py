"""Tests for order idempotency store."""

import threading
import time

from src.mtdata.core.trading.idempotency import IdempotencyStore, SQLiteIdempotencyStore

# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------

def test_no_key_returns_none():
    store = IdempotencyStore()
    assert store.check(None) is None


def test_unknown_key_returns_none():
    store = IdempotencyStore()
    assert store.check("abc") is None


def test_record_and_check():
    store = IdempotencyStore()
    outcome = {"success": True, "ticket": 12345}
    store.record("key-1", outcome)
    dup = store.check("key-1")
    assert dup is not None
    assert dup["duplicate"] is True
    assert dup["idempotency_key"] == "key-1"
    assert dup["original_outcome"] == outcome


def test_record_and_check_request_signature():
    store = IdempotencyStore()
    store.record("key-1", {"success": True}, request_signature="sig-1")
    dup = store.check("key-1")
    assert dup is not None
    assert dup["request_signature"] == "sig-1"


def test_record_none_key_is_noop():
    store = IdempotencyStore()
    store.record(None, {"success": True})
    assert len(store) == 0


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------

def test_expired_entry_returns_none():
    store = IdempotencyStore(ttl_seconds=0.05)
    store.record("exp-key", {"success": True})
    assert store.check("exp-key") is not None
    time.sleep(0.1)
    assert store.check("exp-key") is None


# ---------------------------------------------------------------------------
# Overwrite
# ---------------------------------------------------------------------------

def test_overwrite_existing_key():
    store = IdempotencyStore()
    store.record("k", {"first": True})
    store.record("k", {"second": True})
    dup = store.check("k")
    assert dup["original_outcome"]["second"] is True


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

def test_clear():
    store = IdempotencyStore()
    store.record("a", {"x": 1})
    store.record("b", {"x": 2})
    assert len(store) == 2
    store.clear()
    assert len(store) == 0
    assert store.check("a") is None


# ---------------------------------------------------------------------------
# GC during record
# ---------------------------------------------------------------------------

def test_gc_on_record():
    store = IdempotencyStore(ttl_seconds=0.05)
    store.record("old", {"x": 1})
    time.sleep(0.1)
    store.record("new", {"x": 2})
    assert len(store) == 1
    assert store.check("old") is None
    assert store.check("new") is not None


def test_reserve_waits_for_inflight_request_and_replays_outcome():
    store = IdempotencyStore()
    assert store.reserve("key-1", request_signature="sig-1") is None

    duplicate = {}

    def _reserve_again() -> None:
        duplicate.update(store.reserve("key-1", request_signature="sig-1") or {})

    thread = threading.Thread(target=_reserve_again)
    thread.start()
    time.sleep(0.05)
    store.record("key-1", {"success": True}, request_signature="sig-1")
    thread.join(timeout=1.0)

    assert duplicate["original_outcome"] == {"success": True}


def test_stale_inflight_reservation_does_not_allow_retry():
    store = IdempotencyStore(ttl_seconds=0.05)
    assert store.reserve("key-1", request_signature="sig-1") is None

    second_result: dict[str, object] = {}

    def _reserve_again() -> None:
        second_result["value"] = store.reserve("key-1", request_signature="sig-1")

    thread = threading.Thread(target=_reserve_again, daemon=True)
    thread.start()
    time.sleep(0.1)
    assert thread.is_alive()

    store.record("key-1", {"success": True}, request_signature="sig-1")
    thread.join(timeout=1.0)
    assert second_result["value"]["original_outcome"] == {"success": True}


def test_record_throttles_gc_scans(monkeypatch):
    store = IdempotencyStore(ttl_seconds=300.0)
    gc_calls = 0
    original_gc = store._gc

    def _counted_gc(*, now=None):
        nonlocal gc_calls
        gc_calls += 1
        return original_gc(now=now)

    monkeypatch.setattr(store, "_gc", _counted_gc)

    store.record("one", {"success": True})
    store.record("two", {"success": True})

    assert gc_calls == 0


def test_release_clears_inflight_reservation():
    store = IdempotencyStore()
    assert store.reserve("key-1", request_signature="sig-1") is None

    store.release("key-1", request_signature="sig-1")

    assert store.check("key-1") is None
    assert store.reserve("key-1", request_signature="sig-1") is None


def test_sqlite_store_replays_outcome_across_instances(tmp_path):
    database = tmp_path / "idempotency.sqlite3"
    first = SQLiteIdempotencyStore(database)
    second = SQLiteIdempotencyStore(database)

    assert first.reserve("key-1", request_signature="sig-1") is None
    first.record("key-1", {"success": True, "order": 42}, request_signature="sig-1")

    duplicate = second.reserve("key-1", request_signature="sig-1")
    assert duplicate["original_outcome"] == {"success": True, "order": 42}
    assert second.scope == "sqlite"
    assert second.durable is True


def test_sqlite_store_detects_cross_process_signature_conflict(tmp_path):
    database = tmp_path / "idempotency.sqlite3"
    first = SQLiteIdempotencyStore(database)
    second = SQLiteIdempotencyStore(database)
    first.record("key-1", {"success": True}, request_signature="sig-1")

    duplicate = second.reserve("key-1", request_signature="sig-2")

    assert duplicate["request_signature"] == "sig-1"
    assert duplicate["original_outcome"] == {"success": True}


def test_sqlite_store_fails_closed_for_orphaned_reservation(tmp_path):
    database = tmp_path / "idempotency.sqlite3"
    first = SQLiteIdempotencyStore(database)
    second = SQLiteIdempotencyStore(database)
    assert first.reserve("key-1", request_signature="sig-1") is None

    duplicate = second.reserve("key-1", request_signature="sig-1")

    assert duplicate["in_progress"] is True
    assert duplicate["request_signature"] == "sig-1"


def test_sqlite_store_reserves_atomically_across_workers(tmp_path):
    database = tmp_path / "idempotency.sqlite3"
    stores = [SQLiteIdempotencyStore(database), SQLiteIdempotencyStore(database)]
    barrier = threading.Barrier(2)
    results = [None, None]

    def _reserve(index):
        barrier.wait()
        results[index] = stores[index].reserve("key-1", request_signature="sig-1")

    threads = [threading.Thread(target=_reserve, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert sum(result is None for result in results) == 1
    duplicate = next(result for result in results if result is not None)
    assert duplicate["in_progress"] is True


def test_sqlite_store_expires_completed_outcomes(tmp_path):
    store = SQLiteIdempotencyStore(tmp_path / "idempotency.sqlite3", ttl_seconds=0.05)
    store.record("key-1", {"success": True})
    time.sleep(0.1)

    assert store.check("key-1") is None
