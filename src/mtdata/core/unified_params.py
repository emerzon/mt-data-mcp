#!/usr/bin/env python3
"""
Simple global parameter definitions for mtdata functions.

This module provides global parameters that work across all functions.
"""

from typing import Optional
import argparse
from .constants import DEFAULT_TIMEFRAME

def add_global_args_to_parser(parser: argparse.ArgumentParser, exclude_params: Optional[list] = None) -> None:
    """Add all global parameters to an argument parser"""
    
    exclude_params = exclude_params or []
    
    # Timeframe
    if 'timeframe' not in exclude_params:
        parser.add_argument(
            '--timeframe',
            default=DEFAULT_TIMEFRAME,
            help='Timeframe for market data (H1, M30, D1, etc.)'
        )
    
    # Verbose flag
    if 'verbose' not in exclude_params:
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed metadata in output'
        )
