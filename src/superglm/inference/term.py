"""Compatibility shell for term inference public helpers.

Keep this module as the canonical import surface while Wave 1 extraction moves
implementation details into smaller internal modules.
"""

from superglm.inference._term_covariance import (
    compute_coef_covariance,
    feature_se_from_cov,
    simultaneous_bands,
)
from superglm.inference._term_helpers import (
    _VALID_CENTERING,
    _recenter_term,
    _resolve_group_lambda,
    spline_group_enrichment,
)
from superglm.inference._term_model_ops import (
    drop1,
    refit_unpenalised,
    relativities,
)
from superglm.inference._term_ops import (
    term_inference,
)
from superglm.inference._term_types import (
    _MAX_LOG_REL,
    InteractionInference,
    SmoothCurve,
    SplineMetadata,
    TermInference,
    _safe_exp,
)

__all__ = [
    "_MAX_LOG_REL",
    "_VALID_CENTERING",
    "_recenter_term",
    "_resolve_group_lambda",
    "_safe_exp",
    "compute_coef_covariance",
    "drop1",
    "feature_se_from_cov",
    "InteractionInference",
    "refit_unpenalised",
    "relativities",
    "simultaneous_bands",
    "SmoothCurve",
    "spline_group_enrichment",
    "SplineMetadata",
    "TermInference",
    "term_inference",
]
