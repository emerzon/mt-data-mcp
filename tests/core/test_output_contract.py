from mtdata.core.output_contract import (
    normalize_output_detail,
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


def test_normalize_output_verbosity_detail_compacts_legacy_summary_aliases() -> None:
    assert normalize_output_verbosity_detail(" summary ") == "compact"
    assert normalize_output_verbosity_detail(" Summary_Only ") == "compact"
    assert normalize_output_verbosity_detail(" FULL ") == "full"


def test_resolve_requested_output_verbosity_treats_summary_aliases_as_non_verbose() -> None:
    assert resolve_requested_output_verbosity({"detail": " summary "}) is False
    assert resolve_requested_output_verbosity({"detail": " Summary_Only "}) is False
    assert resolve_requested_output_verbosity({"detail": " full "}) is True
