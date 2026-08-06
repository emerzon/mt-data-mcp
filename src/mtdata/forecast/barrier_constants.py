"""Lightweight public constants shared by barrier tools and schema discovery."""

from typing import Literal

BarrierMethodLiteral = Literal[
    "mc_gbm",
    "mc_gbm_bb",
    "hmm_mc",
    "garch",
    "bootstrap",
    "heston",
    "jump_diffusion",
    "auto",
]

BARRIER_MONTE_CARLO_METHODS: tuple[BarrierMethodLiteral, ...] = (
    "mc_gbm",
    "mc_gbm_bb",
    "hmm_mc",
    "garch",
    "bootstrap",
    "heston",
    "jump_diffusion",
    "auto",
)
