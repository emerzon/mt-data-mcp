from __future__ import annotations

import os
import sys

# Add src to path to ensure local package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from mtdata.forecast.methods.pretrained import _resolve_chronos_device_map


class _FakeCuda:
    def __init__(self, available: bool, count: int) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


class _FakeTorch:
    def __init__(self, available: bool, count: int) -> None:
        self.cuda = _FakeCuda(available=available, count=count)


def test_resolve_chronos_device_map_defaults_to_cuda0_when_available() -> None:
    torch = _FakeTorch(available=True, count=1)
    assert _resolve_chronos_device_map(None, torch) == "cuda:0"


def test_resolve_chronos_device_map_defaults_to_cpu_without_cuda() -> None:
    torch = _FakeTorch(available=False, count=0)
    assert _resolve_chronos_device_map(None, torch) == "cpu"


def test_resolve_chronos_device_map_auto_pins_multi_gpu_to_cuda0() -> None:
    torch = _FakeTorch(available=True, count=2)
    assert _resolve_chronos_device_map("auto", torch) == "cuda:0"


def test_resolve_chronos_device_map_auto_keeps_single_gpu_auto() -> None:
    torch = _FakeTorch(available=True, count=1)
    assert _resolve_chronos_device_map("auto", torch) == "auto"


def test_resolve_chronos_device_map_preserves_explicit_values() -> None:
    torch = _FakeTorch(available=True, count=2)
    assert _resolve_chronos_device_map("cpu", torch) == "cpu"
    assert _resolve_chronos_device_map("cuda:1", torch) == "cuda:1"
