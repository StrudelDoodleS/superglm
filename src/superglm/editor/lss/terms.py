"""SuperLSS terms in the shape ``EditableTerm`` is built from.

``term_from_inference`` reads a SuperGLM ``TermInference``. A SuperLSS term
arrives as a ``ParameterTermEffect``, so :class:`LSSTermInference` carries the
same fields, with the effect on the parameter's link scale in ``log_relativity``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.terms import ParameterTermEffect
from superglm.editor._types import EditableTerm
from superglm.editor.terms import term_from_inference, term_type_from_spec, term_weights_from_data
from superglm.features.spline import _SplineBase
from superglm.inference._term_helpers import _build_spline_metadata
from superglm.inference._term_types import SplineMetadata, _safe_exp


@dataclass(frozen=True)
class LSSTermInference:
    """The fields ``term_from_inference`` reads, for one term of one predictor.

    ``log_relativity`` is the link-scale effect. The interval is stored as
    ``exp`` of its link-scale ends, which ``term_from_inference`` reads back.
    """

    name: str
    kind: str
    x: NDArray[np.float64] | None
    levels: list[str] | None
    log_relativity: NDArray[np.float64]
    ci_lower: NDArray[np.float64] | None
    ci_upper: NDArray[np.float64] | None
    active: bool
    centering_mode: str
    edf: float
    smoothing_lambda: float | dict[str, float] | None
    monotone: str | None
    spline: SplineMetadata | None
    smooth_curve: None = None


def _smoothing_lambda(lambdas: Mapping[str, float]) -> float | dict[str, float] | None:
    if not lambdas:
        return None
    if len(lambdas) == 1:
        return float(next(iter(lambdas.values())))
    return {name: float(value) for name, value in lambdas.items()}


def lss_term_inference(
    key: str, effect: ParameterTermEffect, spec: Any, *, with_se: bool
) -> LSSTermInference:
    """``effect`` of the term ``key`` names, fitted as ``spec``, as the editor reads a term."""
    return LSSTermInference(
        name=key,
        kind=effect.kind,
        x=effect.x,
        levels=None if effect.levels is None else list(effect.levels),
        log_relativity=effect.effect,
        ci_lower=_safe_exp(effect.lower) if with_se else None,
        ci_upper=_safe_exp(effect.upper) if with_se else None,
        active=True,
        centering_mode="native",
        edf=effect.edf,
        smoothing_lambda=_smoothing_lambda(effect.lambdas),
        monotone=None,
        spline=_build_spline_metadata(spec) if isinstance(spec, _SplineBase) else None,
    )


def lss_editable_terms(
    view, keys, *, n_points: int, with_se: bool, train
) -> dict[str, EditableTerm]:
    """The editable terms for ``keys`` of ``view``, weighted by the ``train`` rows' exposure.

    Shape tools stage a shape-constrained refit, which a SuperLSS fit ignores, so
    no term carries the support those tools read.
    """
    editable: dict[str, EditableTerm] = {}
    for key in keys:
        term = term_from_inference(view.term_inference(key, with_se=with_se, n_points=n_points))
        term.metadata["term_type"] = term_type_from_spec(view._specs[key])
        term.metadata["shape_support"] = None
        term.weights = term_weights_from_data(
            train.X, train.sample_weight, view.column_of(key), term
        )
        editable[key] = term
    return editable
