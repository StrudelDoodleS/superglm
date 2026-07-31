"""Interaction screening: rank candidate ti() pairs from cell moments.

Plan: docs/superpowers/plans/2026-07-28-interaction-screening.md.
"""

from superglm.screening._numeric_margin import (
    numeric_numeric_moments,
    numeric_pair_moments,
)
from superglm.screening._pair_moments import (
    pair_cell_moments,
    pair_score_curvature,
    working_score,
)
from superglm.screening._score_stat import ScreenedPair, penalized_score_statistic

__all__ = [
    "ScreenedPair",
    "numeric_numeric_moments",
    "numeric_pair_moments",
    "pair_cell_moments",
    "pair_score_curvature",
    "penalized_score_statistic",
    "working_score",
]
