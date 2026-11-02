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

# Helpers this package used to re-export: name -> (the module that owns it,
# its public replacement or None).  Each still resolves, with a FutureWarning
# naming the replacement, else the owner.
_MOVED: dict[str, tuple[str, str | None]] = {
    "build_basis_detail": (
        "superglm.inference.coef_tables",
        "superglm.inference.coef_tables.build_basis_detail",
    ),
    "build_coef_rows": (
        "superglm.inference.coef_tables",
        "superglm.inference.coef_tables.build_coef_rows",
    ),
    "drop1": ("superglm.inference._term_model_ops", "SuperGLM.drop1"),
    "feature_se_from_cov": ("superglm.inference._term_covariance", None),
    "refit_unpenalised": ("superglm.inference._term_model_ops", "SuperGLM.refit_unpenalised"),
    "relativities": ("superglm.inference._term_model_ops", "SuperGLM.relativities"),
    "spline_group_enrichment": ("superglm.inference._term_helpers", None),
    "term_inference": ("superglm.inference._term_ops", "SuperGLM.term_inference"),
}


def _moved_attribute(module: str, name: str, moved: dict[str, tuple[str, str | None]]):
    """Resolve a re-export this module no longer carries, warning what replaces it."""
    import warnings

    if name not in moved:
        raise AttributeError(f"module {module!r} has no attribute {name!r}")
    owner, replacement = moved[name]
    advice = (
        f"use {replacement}"
        if replacement is not None
        else f"it has no public replacement; its code is in {owner}"
    )
    warnings.warn(
        f"{module}.{name} is deprecated and will be removed; {advice}.",
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
