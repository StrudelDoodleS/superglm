"""Post-fit analysis: term inference, metrics, covariance, summaries.

# Internal submodules: import siblings directly, not through this __init__.
"""

from superglm.inference.coef_tables import build_basis_detail, build_coef_rows
from superglm.inference.covariance import (
    _penalised_xtwx_inv,
    _penalised_xtwx_inv_gram,
    _second_diff_penalty,
)
from superglm.inference.metrics import ModelMetrics
from superglm.inference.summary import (
    ModelSummary,
    _BasisDetailRow,
    _CoefRow,
    _compute_coef_stats,
)
from superglm.inference.term import (
    _MAX_LOG_REL,
    _VALID_CENTERING,
    InteractionInference,
    SmoothCurve,
    SplineMetadata,
    TermInference,
    _recenter_term,
    _resolve_group_lambda,
    _safe_exp,
    compute_coef_covariance,
    drop1,
    feature_se_from_cov,
    refit_unpenalised,
    relativities,
    simultaneous_bands,
    spline_group_enrichment,
    term_inference,
)

__all__ = [
    # covariance
    "_penalised_xtwx_inv",
    "_penalised_xtwx_inv_gram",
    "_second_diff_penalty",
    # metrics
    "ModelMetrics",
    "build_coef_rows",
    "build_basis_detail",
    # summary
    "ModelSummary",
    "_CoefRow",
    "_BasisDetailRow",
    "_compute_coef_stats",
    # term
    "TermInference",
    "SmoothCurve",
    "InteractionInference",
    "SplineMetadata",
    "_MAX_LOG_REL",
    "_VALID_CENTERING",
    "_recenter_term",
    "_resolve_group_lambda",
    "_safe_exp",
    "compute_coef_covariance",
    "drop1",
    "feature_se_from_cov",
    "refit_unpenalised",
    "relativities",
    "simultaneous_bands",
    "spline_group_enrichment",
    "term_inference",
]
