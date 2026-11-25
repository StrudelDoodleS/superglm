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

import typing as _typing

from superglm._lazy import attach

if _typing.TYPE_CHECKING:
    from superglm.screening._pair_factor import (
        PairFactor,
        numeric_cat_factor,
        numeric_numeric_factor,
        pair_design_factor,
    )
    from superglm.screening._pair_moments import pair_cell_moments, working_score
    from superglm.screening._score_stat import ScreenedPair, penalized_score_statistic_ladder

# Each name is read from its module on first access (``superglm._lazy``): the package's body
# imports none of its modules, so a thread that imports one of them by path while another thread
# runs this body never waits on a sibling that thread holds.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm.screening._pair_factor": (
        "PairFactor",
        "numeric_cat_factor",
        "numeric_numeric_factor",
        "pair_design_factor",
    ),
    "superglm.screening._pair_moments": ("pair_cell_moments", "working_score"),
    "superglm.screening._score_stat": ("ScreenedPair", "penalized_score_statistic_ladder"),
}

__getattr__, __dir__ = attach(__name__, __path__, _LAZY_EXPORTS)

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
