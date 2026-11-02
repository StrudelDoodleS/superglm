"""Interaction screening kernels: score a candidate pair as the block it would refit as.

Five kinds screen. The gridded ones (``ti``, ``spline_cat``, ``cat_cat``) run through
the cell moments of ``_pair_moments``; a numeric margin never grids, and
``_pair_factor`` accumulates its z-weighted moments over the other margin's cells
(``numeric_cat``, ``numeric_numeric``).

This package is internal and ``__all__`` is not a public-API claim: it names what
these kernels offer ``screening_ops`` and the tests. The supported entry point is
``SuperGLM.screen_interactions``; the kernels take a reduced design factor whose layout
is internal.
"""

from superglm.screening._pair_factor import (
    PairFactor,
    numeric_cat_factor,
    numeric_numeric_factor,
    pair_design_factor,
)
from superglm.screening._pair_moments import (
    pair_cell_moments,
    working_score,
)
from superglm.screening._score_stat import (
    ScreenedPair,
    penalized_score_statistic_ladder,
)

__all__ = [
    "PairFactor",
    "ScreenedPair",
    "numeric_cat_factor",
    "numeric_numeric_factor",
    "pair_cell_moments",
    "pair_design_factor",
    "penalized_score_statistic_ladder",
    "working_score",
]
