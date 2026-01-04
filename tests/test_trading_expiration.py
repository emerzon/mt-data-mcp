from __future__ import annotations

import os
import sys
from datetime import datetime

import pytest

# Add src to path to ensure local package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from mtdata.core.trading import _normalize_pending_expiration, mt5_config


def _with_clean_tz_config():
    """Temporarily clear tz/offset config for deterministic tests."""
    original = (mt5_config.server_tz_name, mt5_config.client_tz_name, mt5_config.time_offset_minutes)
    mt5_config.server_tz_name = None
    mt5_config.client_tz_name = None
    mt5_config.time_offset_minutes = 0
    return original


def _restore_tz_config(original) -> None:
    mt5_config.server_tz_name, mt5_config.client_tz_name, mt5_config.time_offset_minutes = original


def test_normalize_pending_expiration_datetime_returns_int_timestamp() -> None:
    original = _with_clean_tz_config()
    try:
        exp, specified = _normalize_pending_expiration(datetime(2020, 1, 1, 0, 0, 0))
        assert specified is True
        assert exp == 1577836800
    finally:
        _restore_tz_config(original)


def test_normalize_pending_expiration_string_iso_returns_int_timestamp() -> None:
    original = _with_clean_tz_config()
    try:
        exp, specified = _normalize_pending_expiration("2020-01-01 00:00:00")
        assert specified is True
        assert exp == 1577836800
    finally:
        _restore_tz_config(original)


def test_normalize_pending_expiration_numeric_epoch_returns_int_timestamp() -> None:
    original = _with_clean_tz_config()
    try:
        exp, specified = _normalize_pending_expiration(1577836800)
        assert specified is True
        assert exp == 1577836800
    finally:
        _restore_tz_config(original)


def test_normalize_pending_expiration_gtc_tokens_clear_expiration() -> None:
    original = _with_clean_tz_config()
    try:
        exp, specified = _normalize_pending_expiration("GTC")
        assert specified is True
        assert exp is None
    finally:
        _restore_tz_config(original)


def test_normalize_pending_expiration_none_is_not_explicit() -> None:
    exp, specified = _normalize_pending_expiration(None)
    assert specified is False
    assert exp is None


def test_normalize_pending_expiration_applies_server_tz_offset() -> None:
    from mtdata.core import config as mt5_config_module

    if getattr(mt5_config_module, "pytz", None) is None:
        pytest.skip("pytz is not available")

    original = _with_clean_tz_config()
    try:
        mt5_config.server_tz_name = "Europe/Athens"
        exp, specified = _normalize_pending_expiration(1577836800)
        assert specified is True
        assert exp == 1577844000
    finally:
        _restore_tz_config(original)
