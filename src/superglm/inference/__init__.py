"""Post-fit analysis: term inference, metrics, covariance, summaries.

# Internal submodules: import siblings directly, not through this __init__.
"""

from importlib import import_module

# isort: off
# metrics loads the solver package, which _reporting_state (via factor_smooths)
# needs fully initialised; importing it first keeps that order.
from superglm.inference.metrics import ModelMetrics
from superglm.inference.factor_smooths import FactorSmoothResult
from superglm.inference.random_effects import RandomEffectResult
from superglm.inference.summary import ModelSummary

# isort: on
from superglm.inference._term_types import (
    InteractionInference,
    SmoothCurve,
    SplineMetadata,
    TermInference,
)

# Helpers this package used to re-export, by the module that owns them.  Each
# still resolves, with a FutureWarning naming the owner.
_MOVED = {
    "build_basis_detail": "superglm.inference.coef_tables",
    "build_coef_rows": "superglm.inference.coef_tables",
    "drop1": "superglm.inference._term_model_ops",
    "feature_se_from_cov": "superglm.inference._term_covariance",
    "refit_unpenalised": "superglm.inference._term_model_ops",
    "relativities": "superglm.inference._term_model_ops",
    "spline_group_enrichment": "superglm.inference._term_helpers",
    "term_inference": "superglm.inference._term_ops",
}


def _moved_attribute(module: str, name: str, moved: dict[str, str]):
    """Resolve a re-export this module no longer carries, warning where it lives now."""
    import warnings

    owner = moved.get(name)
    if owner is None:
        raise AttributeError(f"module {module!r} has no attribute {name!r}")
    warnings.warn(
        f"{module}.{name} is deprecated and will be removed; import it from {owner}.",
        FutureWarning,
        stacklevel=3,
    )
    return getattr(import_module(owner), name)


def __getattr__(name: str):
    return _moved_attribute(__name__, name, _MOVED)


__all__ = [
    "FactorSmoothResult",
    "InteractionInference",
    "ModelMetrics",
    "ModelSummary",
    "RandomEffectResult",
    "SmoothCurve",
    "SplineMetadata",
    "TermInference",
]
