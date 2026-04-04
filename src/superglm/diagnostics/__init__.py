"""Model diagnostics: term importance, spline redundancy, discretization.

# Internal submodules: import siblings directly, not through this __init__.
"""

from superglm.diagnostics.discretize import DiscretizationResult, discretization_impact
from superglm.diagnostics.spline_checks import SplineRedundancyReport, spline_redundancy
from superglm.diagnostics.term_diagnostics import (
    _drop_term_holdout,
    _drop_term_refit,
    term_drop_diagnostics,
    term_importance,
)

__all__ = [
    # spline_checks
    "SplineRedundancyReport",
    "spline_redundancy",
    # term_diagnostics
    "term_importance",
    "term_drop_diagnostics",
    "_drop_term_refit",
    "_drop_term_holdout",
    # discretize
    "DiscretizationResult",
    "discretization_impact",
]
