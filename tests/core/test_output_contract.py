from unittest.mock import patch

from mtdata.core.output_contract import (
    attach_collection_contract,
    ensure_common_meta,
    normalize_output_detail,
    normalize_output_verbosity_detail,
    resolve_output_contract,
    resolve_requested_output_verbosity,
)
from mtdata.shared.schema import CANONICAL_OUTPUT_DETAIL_ALIASES


def test_normalize_output_detail_preserves_summary_mode_aliases() -> None:
    assert (
        normalize_output_detail(
            " Summary_Only ",
            aliases=CANONICAL_OUTPUT_DETAIL_ALIASES,
        )
        == "summary"
    )


def test_normalize_output_detail_normalizes_alias_mapping_keys_and_values() -> None:
    assert (
        normalize_output_detail(
            " summary_only ",
            aliases={" Summary_Only ": " Summary "},
        )
        == "summary"
    )


def test_normalize_output_verbosity_detail_is_strict_compact_or_full() -> None:
    assert normalize_output_verbosity_detail(" summary ") == "compact"
    assert normalize_output_verbosity_detail(" standard ") == "compact"
    assert normalize_output_verbosity_detail(" Summary_Only ") == "compact"
    assert normalize_output_verbosity_detail(" FULL ") == "full"


def test_resolve_requested_output_verbosity_tracks_full_detail_only() -> None:
    assert resolve_requested_output_verbosity({"detail": " summary "}) is False
    assert resolve_requested_output_verbosity({"detail": " Summary_Only "}) is False
    assert resolve_requested_output_verbosity({"detail": " full "}) is True


def test_resolve_output_contract_maps_full_detail_to_full_state() -> None:
    state = resolve_output_contract({"detail": "full"})

    assert state.detail == "full"
    assert state.shape_detail == "full"
    assert state.verbose is True
    assert state.transport_verbose is True


def test_resolve_output_contract_preserves_tool_specific_detail_aliases() -> None:
    state = resolve_output_contract(
        {"detail": " summary_only "},
        aliases=CANONICAL_OUTPUT_DETAIL_ALIASES,
    )

    assert state.detail == "summary"
    assert state.shape_detail == "compact"
    assert state.verbose is False
    assert state.transport_verbose is False


def test_resolve_output_contract_prefers_explicit_verbose_when_detail_is_none() -> None:
    state = resolve_output_contract(
        {"detail": "summary"},
        detail=None,
        verbose=True,
        aliases=CANONICAL_OUTPUT_DETAIL_ALIASES,
    )

    assert state.detail == "full"
    assert state.shape_detail == "full"
    assert state.verbose is True


def test_ensure_common_meta_adds_tool_and_runtime_timezone() -> None:
    timezone_meta = {"used": {"tz": "UTC"}}
    with patch(
        "mtdata.core.output_contract.build_runtime_timezone_meta",
        return_value=timezone_meta,
    ):
        out = ensure_common_meta({"success": True}, tool_name="market_ticker")

    assert out["meta"]["tool"] == "market_ticker"
    assert out["meta"]["runtime"]["timezone"] == timezone_meta


def test_ensure_common_meta_preserves_existing_tool_and_timezone() -> None:
    existing_timezone = {"client": {"tz": "US/Central"}}
    with patch("mtdata.core.output_contract.build_runtime_timezone_meta") as build_meta:
        out = ensure_common_meta(
            {
                "meta": {
                    "tool": "existing_tool",
                    "runtime": {"timezone": existing_timezone},
                }
            },
            tool_name="market_ticker",
        )

    build_meta.assert_not_called()
    assert out["meta"]["tool"] == "existing_tool"
    assert out["meta"]["runtime"]["timezone"] == existing_timezone


def test_attach_collection_contract_adds_rows_without_replacing_legacy_data() -> None:
    rows = [{"symbol": "EURUSD"}]

    out = attach_collection_contract(
        {"success": True, "data": rows},
        collection_kind="table",
        rows=rows,
    )

    assert out["data"] == rows
    assert out["collection_kind"] == "table"
    assert out["collection_contract_version"] == "collection.v1"
    assert out["rows"] == rows


def test_attach_collection_contract_preserves_error_payloads() -> None:
    payload = {"error": "failed"}

    assert attach_collection_contract(payload, collection_kind="table", rows=[]) == payload
