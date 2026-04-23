from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtdata.forecast.strategy_loader import (
    discover_declarative_strategy_files,
    iter_loaded_strategy_names,
    list_sample_strategy_files,
    load_declarative_strategy_directory,
    load_declarative_strategy_file,
    load_sample_strategies,
    sample_strategy_directory,
)


def test_load_declarative_strategy_file_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "strategy.yaml"
    path.write_text(
        "\n".join(
            [
                "name: temp-threshold",
                "version: 1",
                "entry:",
                "  type: forecast_threshold",
                "  long_above: 0.01",
                "  short_below: -0.01",
                "exits:",
                "  - type: time_stop",
                "    bars: 12",
            ]
        ),
        encoding="utf-8",
    )

    contract = load_declarative_strategy_file(path)

    assert contract.name == "temp-threshold"
    assert contract.entry.type == "forecast_threshold"
    assert contract.exits[0].type == "time_stop"


def test_load_declarative_strategy_file_from_json(tmp_path: Path) -> None:
    path = tmp_path / "strategy.json"
    path.write_text(
        json.dumps(
            {
                "name": "json-threshold",
                "version": 1,
                "entry": {"type": "forecast_sign"},
            }
        ),
        encoding="utf-8",
    )

    contract = load_declarative_strategy_file(path)

    assert contract.name == "json-threshold"
    assert contract.entry.type == "forecast_sign"


def test_discover_declarative_strategy_files_filters_supported_suffixes(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("name: a\nentry:\n  type: forecast_sign\n", encoding="utf-8")
    (tmp_path / "b.yml").write_text("name: b\nentry:\n  type: forecast_sign\n", encoding="utf-8")
    (tmp_path / "c.json").write_text('{"name":"c","entry":{"type":"forecast_sign"}}', encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    files = discover_declarative_strategy_files(tmp_path)

    assert [path.name for path in files] == ["a.yaml", "b.yml", "c.json"]


def test_load_declarative_strategy_directory_rejects_duplicate_names(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("name: same\nentry:\n  type: forecast_sign\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("name: same\nentry:\n  type: forecast_sign\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate declarative strategy name"):
        load_declarative_strategy_directory(tmp_path)


def test_sample_strategy_directory_exists() -> None:
    directory = sample_strategy_directory()

    assert directory.exists()
    assert directory.is_dir()


def test_load_sample_strategies_returns_discussed_examples() -> None:
    strategies = load_sample_strategies()

    assert set(iter_loaded_strategy_names(strategies)) == {
        "aggressive-threshold",
        "confidence-scaled",
        "conservative-threshold",
        "grid-threshold",
    }
    assert strategies["grid-threshold"].exits[0].type == "grid_take_profit"
    assert strategies["confidence-scaled"].position_sizing.type == "confidence_scaled"
    assert strategies["aggressive-threshold"].entry.type == "forecast_threshold"
    assert strategies["conservative-threshold"].position_sizing.type == "fixed_fraction"


def test_list_sample_strategy_files_matches_loaded_samples() -> None:
    files = list_sample_strategy_files()
    strategies = load_sample_strategies()

    assert len(files) == len(strategies)
    assert sorted(path.suffix for path in files) == [".yaml", ".yaml", ".yaml", ".yaml"]
