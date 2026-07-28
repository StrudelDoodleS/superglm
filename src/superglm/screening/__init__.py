"""Interaction screening: rank candidate ti() pairs from cell moments.

Plan: docs/superpowers/plans/2026-07-28-interaction-screening.md.
"""

from superglm.screening._pair_moments import (
    pair_cell_moments,
    pair_score_curvature,
    working_score,
)

__all__ = [
    "pair_cell_moments",
    "pair_score_curvature",
    "working_score",
]
