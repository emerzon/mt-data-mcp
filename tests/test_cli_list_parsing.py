import argparse
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from typing import List, Optional

# Add src to path to ensure local package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from mtdata.core import cli


class TestCliListParsing(unittest.TestCase):
    def test_normalize_cli_list_value_supports_csv_space_and_json(self):
        self.assertEqual(
            cli._normalize_cli_list_value("arima,theta,mc_gbm"),
            ["arima", "theta", "mc_gbm"],
        )
        self.assertEqual(
            cli._normalize_cli_list_value("arima theta mc_gbm"),
            ["arima", "theta", "mc_gbm"],
        )
        self.assertEqual(
            cli._normalize_cli_list_value('["arima","theta","mc_gbm"]'),
            ["arima", "theta", "mc_gbm"],
        )

    def test_create_command_function_normalizes_list_args(self):
        captured = {}

        def fake_tool(**kwargs):
            captured.update(kwargs)
            return {"ok": True}

        func_info = {
            "func": fake_tool,
            "params": [
                {
                    "name": "methods",
                    "required": False,
                    "default": None,
                    "type": Optional[List[str]],
                }
            ],
        }

        command = cli.create_command_function(func_info, cmd_name="dummy")
        args = argparse.Namespace(methods="arima,theta,mc_gbm", format="json", verbose=False)

        with redirect_stdout(io.StringIO()):
            command(args)

        self.assertEqual(captured["methods"], ["arima", "theta", "mc_gbm"])
        self.assertTrue(captured["__cli_raw"])


if __name__ == "__main__":
    unittest.main()
