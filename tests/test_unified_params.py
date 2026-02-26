"""Tests for src/mtdata/core/unified_params.py"""
import argparse
import sys
import unittest.mock as mock
from mtdata.core.unified_params import add_global_args_to_parser


class TestAddGlobalArgsToParser:
    def test_adds_timeframe(self):
        parser = argparse.ArgumentParser()
        add_global_args_to_parser(parser)
        args = parser.parse_args([])
        assert hasattr(args, 'timeframe')

    def test_adds_verbose(self):
        parser = argparse.ArgumentParser()
        add_global_args_to_parser(parser)
        args = parser.parse_args([])
        assert args.verbose is False

    def test_adds_format(self):
        parser = argparse.ArgumentParser()
        add_global_args_to_parser(parser)
        args = parser.parse_args(['--format', 'json'])
        assert args.format == 'json'

    def test_exclude_timeframe(self):
        parser = argparse.ArgumentParser()
        add_global_args_to_parser(parser, exclude_params=['timeframe'])
        args = parser.parse_args([])
        assert not hasattr(args, 'timeframe')

    def test_exclude_verbose(self):
        parser = argparse.ArgumentParser()
        add_global_args_to_parser(parser, exclude_params=['verbose'])
        args = parser.parse_args([])
        assert not hasattr(args, 'verbose')

    def test_exclude_format(self):
        parser = argparse.ArgumentParser()
        add_global_args_to_parser(parser, exclude_params=['format'])
        args = parser.parse_args([])
        assert not hasattr(args, 'format')

    def test_format_normalized_to_lowercase(self):
        parser = argparse.ArgumentParser()
        add_global_args_to_parser(parser)
        args = parser.parse_args(['--format', 'JSON'])
        assert args.format == 'json'
