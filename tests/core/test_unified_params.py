"""Tests for src/mtdata/core/unified_params.py"""

import argparse
from mtdata.core.unified_params import add_global_args_to_parser


class TestAddGlobalArgsToParser:
    def test_adds_timeframe(self):
        parser = argparse.ArgumentParser()
        add_global_args_to_parser(parser)
        args = parser.parse_args([])
        assert hasattr(args, "timeframe")

    def test_adds_verbose(self):
        parser = argparse.ArgumentParser()
        add_global_args_to_parser(parser)
        args = parser.parse_args([])
        assert args.verbose is False

    def test_adds_json(self):
        parser = argparse.ArgumentParser()
        add_global_args_to_parser(parser)
        args = parser.parse_args(["--json"])
        assert args.json is True

    def test_json_default_false(self):
        parser = argparse.ArgumentParser()
        add_global_args_to_parser(parser)
        args = parser.parse_args([])
        assert args.json is False

    def test_exclude_timeframe(self):
        parser = argparse.ArgumentParser()
        add_global_args_to_parser(parser, exclude_params=["timeframe"])
        args = parser.parse_args([])
        assert not hasattr(args, "timeframe")

    def test_exclude_verbose(self):
        parser = argparse.ArgumentParser()
        add_global_args_to_parser(parser, exclude_params=["verbose"])
        args = parser.parse_args([])
        assert not hasattr(args, "verbose")

    def test_exclude_json(self):
        parser = argparse.ArgumentParser()
        add_global_args_to_parser(parser, exclude_params=["json"])
        args = parser.parse_args([])
        assert not hasattr(args, "json")

    def test_suppress_defaults_hides_absent_global_flags(self):
        parser = argparse.ArgumentParser()
        add_global_args_to_parser(parser, suppress_defaults=True)
        args = parser.parse_args([])
        assert not hasattr(args, "timeframe")
        assert not hasattr(args, "verbose")
        assert not hasattr(args, "json")
