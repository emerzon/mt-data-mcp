from __future__ import annotations

from mtdata.services.options_service import _options_error


def test_yahoo_auth_remediation_does_not_suggest_retrying() -> None:
    result = _options_error(
        "Failed to fetch options chain: Authentication error: Yahoo Finance options "
        "endpoint returned 401 Unauthorized."
    )

    assert result["error_code"] == "options_provider_auth"
    assert "retry later" not in result["remediation"].lower()
    assert "use another options data provider" in result["remediation"]
