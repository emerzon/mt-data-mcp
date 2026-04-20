#!/usr/bin/env python3
"""Test to verify resolve_output_contract behavior."""
from mtdata.core.output_contract import resolve_output_contract

# Test with detail="compact"
contract_compact = resolve_output_contract(
    detail="compact",
    default_detail="full",
)
print(f"detail='compact' -> shape_detail={contract_compact.shape_detail!r}, detail={contract_compact.detail!r}")

# Test with detail="full"
contract_full = resolve_output_contract(
    detail="full",
    default_detail="full",
)
print(f"detail='full' -> shape_detail={contract_full.shape_detail!r}, detail={contract_full.detail!r}")

# Test with detail not specified (should use default)
contract_default = resolve_output_contract(
    default_detail="full",
)
print(f"detail=None -> shape_detail={contract_default.shape_detail!r}, detail={contract_default.detail!r}")
