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
    specs and raw columns; the scan itself is the scalar path's.
    """
    compiled_tuple = tuple(compiled)
    boundary_tuple = tuple(tuple(boundary) for boundary in boundaries)
    if len(boundary_tuple) != len(compiled_tuple):
        raise ValueError("boundaries must supply one tuple per compiled predictor")
    response = np.asarray(y, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    findings: list[SeparatedTerm] = []
    for predictor, predictor_boundaries in zip(compiled_tuple, boundary_tuple, strict=True):
        if not predictor_boundaries:
            continue
        for record in predictor.compiled.separation_records:
            if record[0] == "cat":
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
    return findings


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
    "scan_predictor_separation",
    "validate_separation_policy",
]
