"""Former import surface for term inference; each name now lives in its owner module."""

from superglm.inference import _MOVED as _PACKAGE_MOVED
from superglm.inference import _moved_attribute

_TYPES = ("InteractionInference", "SmoothCurve", "SplineMetadata", "TermInference")
_HELPERS = (
    "drop1",
    "feature_se_from_cov",
    "refit_unpenalised",
    "relativities",
    "spline_group_enrichment",
    "term_inference",
)
_MOVED = {
    **{name: ("superglm.inference", f"superglm.inference.{name}") for name in _TYPES},
    **{name: _PACKAGE_MOVED[name] for name in _HELPERS},
}


def __getattr__(name: str):
    return _moved_attribute(__name__, name, _MOVED)
