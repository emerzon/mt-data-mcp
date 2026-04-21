from unittest.mock import patch

from mtdata.core.output_contract import (
    ensure_common_meta,
    normalize_output_detail,
    resolve_output_contract,
    normalize_output_verbosity_detail,
    resolve_requested_output_verbosity,
)


def test_normalize_output_detail_preserves_summary_mode_aliases() -> None:
    assert (
        normalize_output_detail(
            " Summary_Only ",
            aliases={"summary_only": "summary"},
        )
        == "summary"
    )


def test_normalize_output_verbosity_detail_is_strict_compact_or_full() -> None:
    assert normalize_output_verbosity_detail(" summary ") == "compact"
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
        aliases={"summary_only": "summary"},
    )

    assert state.detail == "summary"
    assert state.shape_detail == "compact"
    assert state.verbose is False
    assert state.transport_verbose is False


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
