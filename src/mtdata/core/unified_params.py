#!/usr/bin/env python3
"""
Simple global parameter definitions for mtdata functions.

This module provides global parameters that work across all functions.
"""

import argparse
from typing import Optional

from .constants import DEFAULT_TIMEFRAME


def add_global_args_to_parser(
    parser: argparse.ArgumentParser,
    exclude_params: Optional[list] = None,
    *,
    suppress_defaults: bool = False,
) -> None:
    """Add all global parameters to an argument parser"""
    
    exclude_params = exclude_params or []
    
    # Timeframe
    if 'timeframe' not in exclude_params:
        timeframe_kwargs = {
            "help": "Timeframe for market data (H1, M30, D1, etc.)",
        }
        if suppress_defaults:
            timeframe_kwargs["default"] = argparse.SUPPRESS
        else:
            timeframe_kwargs["default"] = DEFAULT_TIMEFRAME
        parser.add_argument(
            '--timeframe',
            **timeframe_kwargs,
        )
    
    # Verbose flag
    if 'verbose' not in exclude_params:
        verbose_kwargs = {
            "action": "store_true",
            "help": "Show detailed metadata in output",
        }
        if suppress_defaults:
            verbose_kwargs["default"] = argparse.SUPPRESS
        parser.add_argument(
            '--verbose',
            **verbose_kwargs,
        )

    # Output format: formatted text by default, JSON when explicitly requested.
    if 'json' not in exclude_params:
        json_kwargs = {
            "action": "store_true",
            "dest": "json",
            "help": "Output raw JSON (default output is formatted text).",
        }
        if suppress_defaults:
            json_kwargs["default"] = argparse.SUPPRESS
        parser.add_argument(
            '--json',
            **json_kwargs,
        )
