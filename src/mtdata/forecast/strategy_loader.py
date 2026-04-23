from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml
from pydantic import ValidationError

from .contracts import DeclarativeStrategyContract

_SUPPORTED_STRATEGY_SUFFIXES = {".json", ".yaml", ".yml"}


def _normalize_strategy_payload(payload: Any, *, source: str) -> Dict[str, Any]:
    if payload is None:
        raise ValueError(f"{source} did not contain a strategy document")
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a mapping at the top level")
    return dict(payload)


def _strategy_document_from_text(
    text: str,
    *,
    source: str,
    suffix: str,
) -> Dict[str, Any]:
    suffix_l = str(suffix).lower().strip()
    if suffix_l == ".json":
        payload = json.loads(text)
    elif suffix_l in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        raise ValueError(
            f"Unsupported strategy file format '{suffix}'. Supported: {sorted(_SUPPORTED_STRATEGY_SUFFIXES)}"
        )
    return _normalize_strategy_payload(payload, source=source)


def _validate_contract_payload(payload: Dict[str, Any], *, source: str) -> DeclarativeStrategyContract:
    try:
        return DeclarativeStrategyContract.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid declarative strategy in {source}: {exc}") from exc


def load_declarative_strategy_file(path: str | Path) -> DeclarativeStrategyContract:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path_obj)
    if not path_obj.is_file():
        raise ValueError(f"{path_obj} is not a file")
    suffix = path_obj.suffix.lower()
    text = path_obj.read_text(encoding="utf-8")
    payload = _strategy_document_from_text(text, source=str(path_obj), suffix=suffix)
    return _validate_contract_payload(payload, source=str(path_obj))


def discover_declarative_strategy_files(directory: str | Path) -> list[Path]:
    directory_path = Path(directory)
    if not directory_path.exists():
        return []
    if not directory_path.is_dir():
        raise ValueError(f"{directory_path} is not a directory")
    return sorted(
        path
        for path in directory_path.iterdir()
        if path.is_file() and path.suffix.lower() in _SUPPORTED_STRATEGY_SUFFIXES
    )


def load_declarative_strategy_directory(
    directory: str | Path,
) -> Dict[str, DeclarativeStrategyContract]:
    loaded: Dict[str, DeclarativeStrategyContract] = {}
    for path in discover_declarative_strategy_files(directory):
        contract = load_declarative_strategy_file(path)
        if contract.name in loaded:
            raise ValueError(
                f"Duplicate declarative strategy name '{contract.name}' while loading {path}"
            )
        loaded[contract.name] = contract
    return loaded


def sample_strategy_directory() -> Path:
    sample_root = resources.files("mtdata.forecast.strategy_samples")
    return Path(str(sample_root))


def list_sample_strategy_files() -> list[Path]:
    return discover_declarative_strategy_files(sample_strategy_directory())


def load_sample_strategies() -> Dict[str, DeclarativeStrategyContract]:
    loaded: Dict[str, DeclarativeStrategyContract] = {}
    for path in list_sample_strategy_files():
        contract = load_declarative_strategy_file(path)
        if contract.name in loaded:
            raise ValueError(f"Duplicate bundled strategy name '{contract.name}'")
        loaded[contract.name] = contract
    return loaded


def iter_loaded_strategy_names(strategies: Dict[str, DeclarativeStrategyContract]) -> Iterable[str]:
    return tuple(sorted(strategies))


__all__ = [
    "discover_declarative_strategy_files",
    "iter_loaded_strategy_names",
    "list_sample_strategy_files",
    "load_declarative_strategy_directory",
    "load_declarative_strategy_file",
    "load_sample_strategies",
    "sample_strategy_directory",
]
