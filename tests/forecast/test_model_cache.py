"""Tests for shared model cache."""

import threading
import time

import pytest

from src.mtdata.forecast.model_cache import ModelCache


# ---------------------------------------------------------------------------
# Basic get_or_load
# ---------------------------------------------------------------------------

def test_cache_miss_then_hit():
    cache = ModelCache(ttl_seconds=60, max_entries=4)
    load_count = 0

    def loader():
        nonlocal load_count
        load_count += 1
        return {"name": "test_model"}

    model, meta = cache.get_or_load("k1", loader)
    assert meta["cache"] == "miss"
    assert model["name"] == "test_model"
    assert load_count == 1

    model2, meta2 = cache.get_or_load("k1", loader)
    assert meta2["cache"] == "hit"
    assert model2 is model
    assert load_count == 1  # Not reloaded


def test_different_keys_load_separately():
    cache = ModelCache(ttl_seconds=60, max_entries=4)
    m1, _ = cache.get_or_load("a", lambda: "model_a")
    m2, _ = cache.get_or_load("b", lambda: "model_b")
    assert m1 == "model_a"
    assert m2 == "model_b"
    assert len(cache) == 2


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------

def test_ttl_expiry():
    cache = ModelCache(ttl_seconds=0.05, max_entries=4)
    cache.get_or_load("k", lambda: "v1")
    time.sleep(0.1)
    _, meta = cache.get_or_load("k", lambda: "v2")
    assert meta["cache"] == "miss"


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

def test_lru_eviction():
    cache = ModelCache(ttl_seconds=60, max_entries=2)
    cache.get_or_load("a", lambda: 1)
    cache.get_or_load("b", lambda: 2)
    # Access "a" to make it more recently used
    cache.get_or_load("a", lambda: 99)
    # Adding "c" should evict "b" (LRU)
    cache.get_or_load("c", lambda: 3)
    assert len(cache) == 2
    assert "a" in cache.keys()
    assert "c" in cache.keys()


# ---------------------------------------------------------------------------
# Invalidate / clear
# ---------------------------------------------------------------------------

def test_invalidate():
    cache = ModelCache(ttl_seconds=60, max_entries=4)
    cache.get_or_load("x", lambda: "val")
    assert cache.invalidate("x") is True
    assert cache.invalidate("x") is False
    assert len(cache) == 0


def test_clear():
    cache = ModelCache(ttl_seconds=60, max_entries=4)
    cache.get_or_load("a", lambda: 1)
    cache.get_or_load("b", lambda: 2)
    cache.clear()
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# Concurrent load dedup
# ---------------------------------------------------------------------------

def test_concurrent_loads_deduplicate():
    cache = ModelCache(ttl_seconds=60, max_entries=4)
    load_count = 0
    load_lock = threading.Lock()

    def slow_loader():
        nonlocal load_count
        time.sleep(0.1)
        with load_lock:
            load_count += 1
        return "shared_model"

    results = [None, None]

    def thread_fn(idx):
        m, meta = cache.get_or_load("shared", slow_loader)
        results[idx] = (m, meta)

    t1 = threading.Thread(target=thread_fn, args=(0,))
    t2 = threading.Thread(target=thread_fn, args=(1,))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    # Only one load should have occurred (init lock deduplicates)
    assert load_count == 1
    assert results[0][0] == "shared_model"
    assert results[1][0] == "shared_model"


# ---------------------------------------------------------------------------
# Hit count tracking
# ---------------------------------------------------------------------------

def test_hit_count_increments():
    cache = ModelCache(ttl_seconds=60, max_entries=4)
    cache.get_or_load("k", lambda: "v")
    _, meta1 = cache.get_or_load("k", lambda: "v")
    assert meta1["hit_count"] == 1
    _, meta2 = cache.get_or_load("k", lambda: "v")
    assert meta2["hit_count"] == 2
