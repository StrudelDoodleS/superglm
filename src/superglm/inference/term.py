"""Former import surface for term inference; each name now lives in its owner module."""

from superglm.inference import _moved_attribute

_MOVED = {
    "InteractionInference": "superglm.inference._term_types",
    "SmoothCurve": "superglm.inference._term_types",
    "SplineMetadata": "superglm.inference._term_types",
    "TermInference": "superglm.inference._term_types",
    "drop1": "superglm.inference._term_model_ops",
    "feature_se_from_cov": "superglm.inference._term_covariance",
    "refit_unpenalised": "superglm.inference._term_model_ops",
    "relativities": "superglm.inference._term_model_ops",
    "spline_group_enrichment": "superglm.inference._term_helpers",
    "term_inference": "superglm.inference._term_ops",
}


def __getattr__(name: str):
    return _moved_attribute(__name__, name, _MOVED)
