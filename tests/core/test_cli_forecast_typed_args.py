import argparse
from unittest.mock import MagicMock, patch

import pytest

from mtdata.core.cli import _add_forecast_generate_args, main


class TestForecastTypedArgs:
    def test_parser_includes_typed_value_help_epilog(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)

        assert "Typed Value Formats:" in parser.epilog
        assert "--denoise PRESET|JSON" in parser.epilog

    def test_parser_allows_bare_presence_for_typed_flags(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)

        args = parser.parse_args(["BTCUSD", "--denoise", "--params"])

        assert args.denoise == "__PRESENT__"
        assert args.params == "__PRESENT__"

    @patch("mtdata.core.cli.discover_tools")
    def test_main_shows_targeted_error_for_bare_denoise(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value="ok")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }

        with (
            patch("sys.argv", ["cli.py", "forecast_generate", "BTCUSD", "--denoise"]),
            pytest.raises(SystemExit),
        ):
            main()

        err = capsys.readouterr().err
        assert "--denoise expects a value." in err
        assert "--denoise ema" in err
        mock_fn.assert_not_called()

    @patch("mtdata.core.cli.discover_tools")
    def test_main_allows_bare_denoise_when_set_supplies_value(self, mock_discover):
        mock_fn = MagicMock(return_value="ok")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."
        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }

        with patch(
            "sys.argv",
            [
                "cli.py",
                "forecast_generate",
                "BTCUSD",
                "--denoise",
                "--set",
                "denoise.method=ema",
            ],
        ):
            result = main()

        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.denoise == {"method": "ema"}
