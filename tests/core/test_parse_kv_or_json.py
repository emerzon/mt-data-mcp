import unittest

import os
import sys

# Add src to path to ensure local package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from mtdata.utils.utils import parse_kv_or_json


class TestParseKvOrJson(unittest.TestCase):
    def test_dict_input(self):
        obj = {"a": 1, "b": "x"}
        out = parse_kv_or_json(obj)
        self.assertEqual(out, obj)
        self.assertIsNot(out, obj)

    def test_json_dict_string(self):
        out = parse_kv_or_json('{"a": 1, "b": "x"}')
        self.assertEqual(out, {"a": 1, "b": "x"})

    def test_json_list_of_pairs_string(self):
        out = parse_kv_or_json('[["a", 1], ["b", "x"]]')
        self.assertEqual(out, {"a": 1, "b": "x"})

    def test_kv_string_equals(self):
        out = parse_kv_or_json("a=1 b=x")
        self.assertEqual(out, {"a": "1", "b": "x"})

    def test_kv_string_comma_separated_assignments(self):
        out = parse_kv_or_json("a=1,b=2 c=3")
        self.assertEqual(out, {"a": "1", "b": "2", "c": "3"})

    def test_kv_string_preserves_commas_inside_value(self):
        out = parse_kv_or_json("methods=theta,naive,drift aggregation=mean")
        self.assertEqual(out, {"methods": "theta,naive,drift", "aggregation": "mean"})

    def test_kv_string_colon_token(self):
        out = parse_kv_or_json("a:1 b:x")
        self.assertEqual(out, {"a": "1", "b": "x"})

    def test_kv_string_colon_split_token(self):
        out = parse_kv_or_json("a: 1 b: x")
        self.assertEqual(out, {"a": "1", "b": "x"})

    def test_windows_path_token_is_not_key_value(self):
        out = parse_kv_or_json(r"C:\Users\Admin")
        self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main()
