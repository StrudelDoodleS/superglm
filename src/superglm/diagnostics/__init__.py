"""Model diagnostics: term importance, spline redundancy, discretization.

# Internal submodules: import siblings directly, not through this __init__.
"""

import typing as _typing

from superglm._lazy import attach

if _typing.TYPE_CHECKING:
    from superglm.diagnostics.discretize import DiscretizationResult, discretization_impact
    from superglm.diagnostics.fit_report import (
        DiagnosticAction,
        DiagnosticEvidence,
        DiagnosticFinding,
        DiagnosticSubject,
        FitDiagnosticReport,
    )
    from superglm.diagnostics.spline_checks import SplineRedundancyReport, spline_redundancy
    from superglm.diagnostics.term_diagnostics import (
        _drop_term_holdout,
        _drop_term_refit,
        term_drop_diagnostics,
        term_importance,
    )

# Each name is read from its module on first access (``superglm._lazy``): importing the package, or a
# sibling such as ``diagnostics.convergence``, loads none of the others.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm.diagnostics.discretize": ("DiscretizationResult", "discretization_impact"),
    "superglm.diagnostics.fit_report": (
        "DiagnosticAction",
        "DiagnosticEvidence",
        "DiagnosticFinding",
        "DiagnosticSubject",
        "FitDiagnosticReport",
    ),
    "superglm.diagnostics.spline_checks": ("SplineRedundancyReport", "spline_redundancy"),
    "superglm.diagnostics.term_diagnostics": (
        "_drop_term_holdout",
        "_drop_term_refit",
        "term_drop_diagnostics",
        "term_importance",
    ),
}

__getattr__, __dir__ = attach(__name__, __path__, _LAZY_EXPORTS)

__all__ = [
    # fit_report
    "DiagnosticAction",
    "DiagnosticEvidence",
    "DiagnosticFinding",
    "DiagnosticSubject",
    "FitDiagnosticReport",
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
