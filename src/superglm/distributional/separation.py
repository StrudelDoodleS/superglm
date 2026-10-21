"""Build-time separated-cell scan for distributional predictors.

Reuses the scalar path's exact indicator-block scan
(:mod:`superglm.diagnostics.separation`) once per predictor, on the terms
each compiled predictor recorded while its ``Categorical`` specs learned
their levels, using the response boundaries the family declares for that
predictor.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from superglm.diagnostics.separation import (
    SeparatedTerm,
    SeparationError,
    SeparationWarning,
    emit_curve_separation,
    emit_level_curve_separation,
    format_factor_smooth_canonical,
    format_factor_smooth_separation,
    format_separation_message,
    scan_categorical_term,
    scan_interaction_term,
)
from superglm.distributional.family import DistributionalFamily, ResponseBoundaryFamily
from superglm.distributional.predictor import CompiledPredictor
from superglm.links import Link

SeparationPolicy = Literal["warn", "error", "ignore"]
SEPARATION_POLICIES: tuple[str, ...] = ("warn", "error", "ignore")


def validate_separation_policy(value: object) -> SeparationPolicy:
    """Return ``value`` unchanged when it names a supported separation policy."""

    if value not in SEPARATION_POLICIES:
        raise ValueError(f"separation must be one of {SEPARATION_POLICIES}; got {value!r}")
    return cast(SeparationPolicy, value)


def predictor_response_boundaries(
    family: DistributionalFamily,
    links: Sequence[Link],
) -> tuple[tuple[str, ...], ...]:
    """One response-boundary tuple per predictor; all empty without a declaration."""
    link_tuple = tuple(links)
    if not isinstance(family, ResponseBoundaryFamily):
        return tuple(() for _ in link_tuple)
    boundaries = tuple(tuple(boundary) for boundary in family.response_boundaries(link_tuple))
    if len(boundaries) != len(link_tuple):
        raise ValueError("family.response_boundaries must return one tuple per predictor")
    return boundaries


def scan_predictor_separation(
    compiled: Sequence[CompiledPredictor],
    y: NDArray,
    weights: NDArray,
    *,
    boundaries: Sequence[tuple[str, ...]],
) -> list[SeparatedTerm]:
    """Separated levels and crossed cells on every predictor that can separate.

    Each compiled predictor recorded its scannable terms with their built
    specs and raw columns; the scan itself is the scalar path's.  The
    varying-coefficient levels its build already held (spline-by-factor
    curves, and numeric or polynomial levels, which ``predict`` holds to
    their rows' range) are named in a ``SeparationWarning`` here, never
    refused: 0.40 fitted them without a scan.
    """
    compiled_tuple = tuple(compiled)
    boundary_tuple = tuple(tuple(boundary) for boundary in boundaries)
    if len(boundary_tuple) != len(compiled_tuple):
        raise ValueError("boundaries must supply one tuple per compiled predictor")
    response = np.asarray(y, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    findings: list[SeparatedTerm] = []
    curves: list = []
    level_curves: list = []
    for predictor, predictor_boundaries in zip(compiled_tuple, boundary_tuple, strict=True):
        if not predictor_boundaries:
            continue
        for record in predictor.compiled.separation_records:
            if record[0] == "curve":
                curves.extend(record[2])
            elif record[0] == "level_curve":
                level_curves.append((f"{predictor.name}:{record[1]}", record[2]))
            elif record[0] == "cat":
                _, term, spec, column = record
                findings.extend(
                    scan_categorical_term(
                        f"{predictor.name}:{term}",
                        spec,
                        column,
                        response,
                        weight_values,
                        predictor_boundaries,
                    )
                )
            else:
                _, term, left_spec, right_spec, left, right, left_name, right_name = record
                findings.extend(
                    scan_interaction_term(
                        f"{predictor.name}:{term}",
                        left_spec,
                        right_spec,
                        left,
                        right,
                        left_name,
                        right_name,
                        response,
                        weight_values,
                        predictor_boundaries,
                    )
                )
    emit_curve_separation(curves, "warn")
    emit_level_curve_separation(level_curves)
    return findings


def record_factor_smooth_levels(
    compiled: Sequence[CompiledPredictor],
    y: NDArray,
    weights: NDArray,
    *,
    boundaries: Sequence[tuple[str, ...]],
    policy: SeparationPolicy,
    stacklevel: int,
) -> None:
    """Record what the data identify of every ``sz`` term's levels, as SuperGLM's fit does.

    For each predictor's ``sz`` term without ``select=True``
    (``FactorSmooth._record_unidentified_levels``, read at prediction by
    ``prediction_design._score_interaction``): its thin and weightless
    levels, and, under the response boundaries the family declares for that
    predictor, the levels whose unpenalized line separates the response.  A
    dispersion predictor's walk runs the other way from a mean's (log phi to
    ``+inf`` on rows at zero), but its directions of recession are the
    negatives of a mean's on the same rows, with the same span.  Separated
    levels are named in a ``SeparationWarning`` unless the policy is
    ``"ignore"``; like SuperGLM it never refuses the fit.  A term no level of
    which identifies the population curve is named in a ``UserWarning``.
    """
    from superglm.features.factor_smooth import FactorSmooth, share_group_levels
    from superglm.group_matrix import FactorSmoothGroupMatrix

    response = np.asarray(y, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    for predictor, predictor_boundaries in zip(compiled, boundaries, strict=True):
        matrices = predictor.compiled.design.group_matrices
        for group, matrix in zip(predictor.compiled.groups, matrices, strict=True):
            spec = predictor.compiled.interaction_specs.get(group.feature_name)
            if not (
                isinstance(spec, FactorSmooth)
                and spec.basis == "sz"
                and not spec._selects_lines
                and isinstance(matrix, FactorSmoothGroupMatrix)
            ):
                continue
            name = f"{predictor.name}:{group.feature_name}"
            separated = spec._record_unidentified_levels(
                matrix, weight_values, response, tuple(predictor_boundaries)
            )
            if separated and policy != "ignore":
                labels = [spec._levels[code] for code in separated]
                warnings.warn(
                    format_factor_smooth_separation(name, labels, len(spec._levels)),
                    SeparationWarning,
                    stacklevel=stacklevel,
                )
            if spec._population_convention == "canonical":
                warnings.warn(
                    format_factor_smooth_canonical(name, bool(spec._separated_levels)),
                    UserWarning,
                    stacklevel=stacklevel,
                )
        interactions = predictor.compiled.interaction_specs
        share_group_levels(interactions[key] for key in predictor.compiled.interaction_order)


def apply_separation_policy(
    findings: Sequence[SeparatedTerm],
    policy: SeparationPolicy,
    *,
    stacklevel: int,
) -> None:
    """Warn, refuse, or do nothing, exactly as the scalar builder does."""
    if not findings or policy == "ignore":
        return
    message = format_separation_message(list(findings))
    if policy == "error":
        raise SeparationError(message)
    warnings.warn(message, SeparationWarning, stacklevel=stacklevel)


__all__ = [
    "SEPARATION_POLICIES",
    "SeparationPolicy",
    "apply_separation_policy",
    "predictor_response_boundaries",
    "record_factor_smooth_levels",
    "scan_predictor_separation",
    "validate_separation_policy",
]
