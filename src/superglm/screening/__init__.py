"""Interaction screening kernels: score a candidate pair as the block it would
refit as.

Five kinds screen.  The gridded ones — ``ti``, ``spline_cat`` and ``cat_cat``
— run through the cell moments of ``_pair_moments``, a factor margin gridding
on its fitted levels.  A numeric margin never grids: ``_numeric_margin``
accumulates its z-weighted moments over the other margin's cells instead,
which is the exact channel for ``numeric_cat`` and ``numeric_numeric``.

Plans: docs/superpowers/plans/2026-07-28-interaction-screening.md and
docs/superpowers/plans/2026-07-31-mixed-interaction-screening.md.
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
