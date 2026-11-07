"""Post-fit analysis: term inference, metrics, covariance, summaries.

# Internal submodules: import siblings directly, not through this __init__.
"""

import typing as _typing
from functools import partial as _partial
from importlib import import_module

from superglm._lazy import attach

if _typing.TYPE_CHECKING:
    from superglm.inference._term_types import (
        InteractionInference,
        SmoothCurve,
        SplineMetadata,
        TermInference,
    )
    from superglm.inference.factor_smooths import FactorSmoothResult
    from superglm.inference.metrics import ModelMetrics
    from superglm.inference.random_effects import RandomEffectResult
    from superglm.inference.summary import ModelSummary

# Each name is read from its module on first access (``superglm._lazy``), so importing a sibling
# such as ``inference._term_types`` loads neither the summaries nor the design machinery.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm.inference._term_types": (
        "InteractionInference",
        "SmoothCurve",
        "SplineMetadata",
        "TermInference",
    ),
    "superglm.inference.factor_smooths": ("FactorSmoothResult",),
    "superglm.inference.metrics": ("ModelMetrics",),
    "superglm.inference.random_effects": ("RandomEffectResult",),
    "superglm.inference.summary": ("ModelSummary",),
}

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
    "simultaneous_bands": (
        "superglm.inference._term_covariance",
        "SuperGLM.simultaneous_bands",
    ),
    "spline_group_enrichment": ("superglm.inference._term_helpers", None),
    "term_inference": ("superglm.inference._term_ops", "SuperGLM.term_inference"),
}


def _moved_attribute(module: str, name: str, moved: dict[str, tuple[str, str | None]]):
    """Resolve a re-export this module no longer carries, warning what replaces it."""
    from superglm.profiling._scalar import warn_caller

    if name not in moved:
        raise AttributeError(f"module {module!r} has no attribute {name!r}")
    owner, replacement = moved[name]
    advice = (
        f"use {replacement}"
        if replacement is not None
        else f"it has no public replacement; its code is in {owner}"
    )
    warn_caller(f"{module}.{name} is deprecated and will be removed; {advice}.", FutureWarning)
    return getattr(import_module(owner), name)


__getattr__, __dir__ = attach(
    __name__, __path__, _LAZY_EXPORTS, missing=_partial(_moved_attribute, __name__, moved=_MOVED)
)


# The moved helpers stay listed so a star import reaches them through __getattr__,
# with its FutureWarning, until they are removed.
__all__ = [
    "FactorSmoothResult",
    "InteractionInference",
    "ModelMetrics",
    "ModelSummary",
    "RandomEffectResult",
    "SmoothCurve",
    "SplineMetadata",
    "TermInference",
    *_MOVED,
]
