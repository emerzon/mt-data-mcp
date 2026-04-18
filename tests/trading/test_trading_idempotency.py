"""Tests for order idempotency store."""

import time

from src.mtdata.core.trading.idempotency import IdempotencyStore

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
