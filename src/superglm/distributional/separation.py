"""Build-time separated-cell scan for distributional predictors.

Reuses the scalar path's exact indicator-block scan
(:mod:`superglm.diagnostics.separation`) once per predictor, on the terms
each compiled predictor recorded while its ``Categorical`` specs learned
their levels, using the response boundaries the family declares for that
predictor.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from superglm._frame import EagerFrame
from superglm._predictor_compiler import CompiledPredictorDesign
from superglm.diagnostics.separation import (
    SeparatedTerm,
    SeparationError,
    SeparationWarning,
    _format_labels,
    emit_curve_separation,
    emit_level_curve_separation,
    format_factor_smooth_canonical,
    format_factor_smooth_separation,
    format_separation_message,
    scan_categorical_term,
    scan_interaction_term,
)
from superglm.distributional.family import (
    CensoredSeparationFamily,
    DistributionalFamily,
    ParameterSpec,
    ResponseBoundaryFamily,
)
from superglm.distributional.observation import LEFT_CENSORED, RIGHT_CENSORED, ResponseBounds
from superglm.distributional.predictor import CompiledPredictor
from superglm.features.categorical import Categorical
from superglm.features.interaction import CategoricalInteraction
from superglm.features.ordered_categorical import resolve_interaction_parent_of
from superglm.links import Link, resolve_link

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
                        predictor.compiled.scan_codes.pop(term, None),
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
    for predictor in compiled_tuple:
        predictor.compiled.scan_codes.clear()
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


def _without(findings: list[SeparatedTerm], pinned: object) -> list[SeparatedTerm]:
    """``findings`` less the ``pinned`` label; a term left with none is dropped."""
    kept = [replace(f, labels=[label for label in f.labels if label != pinned]) for f in findings]
    return [f for f in kept if f.labels]


def _scan_design_cells(
    design: CompiledPredictorDesign,
    frame: EagerFrame,
    off_side: NDArray,
    weights: NDArray,
    *,
    intercept: bool,
) -> list[SeparatedTerm]:
    """The response-boundary scan of one design's categorical terms, with ``off_side`` as response.

    The scan walks a reference level along the intercept, so without one the
    reference level, and the cell of two reference levels, are pinned at the
    offset and not reported.
    """
    findings: list[SeparatedTerm] = []
    for name, spec in design.specs.items():
        if not isinstance(spec, Categorical):
            continue
        found = scan_categorical_term(
            name, spec, frame.column_array(name), off_side, weights, ("zero",)
        )
        findings.extend(found if intercept else _without(found, spec._base_level))
    for name in design.interaction_order:
        ispec = design.interaction_specs[name]
        if not isinstance(ispec, CategoricalInteraction):
            continue
        p1, p2 = ispec.parent_names
        spec1, x1 = resolve_interaction_parent_of(
            ispec, design.specs.get(p1), frame.column_array(p1)
        )
        spec2, x2 = resolve_interaction_parent_of(
            ispec, design.specs.get(p2), frame.column_array(p2)
        )
        found = scan_interaction_term(
            name, spec1, spec2, x1, x2, p1, p2, off_side, weights, ("zero",)
        )
        pinned = (getattr(spec1, "_base_level", None), getattr(spec2, "_base_level", None))
        findings.extend(found if intercept else _without(found, pinned))
    return findings


def _default_link_class(parameter: ParameterSpec) -> type:
    default = parameter.default_link
    return type(default if isinstance(default, Link) else resolve_link(default, cast(Any, None)))


def censored_recession(
    family: DistributionalFamily, bounds: ResponseBounds, compiled: Sequence[CompiledPredictor]
) -> NDArray[np.bool_]:
    """One flag per row and predictor; all false without a declaration.

    A flag that is monotone in its parameter holds under any link onto the
    parameter's range: the level's likelihood is monotone in its coefficient,
    so its supremum is reached only as the predictor walks off.  A parameter
    the family names in ``censored_recession_link_sensitive`` has flags that
    are not monotone; there a bounded link, such as a logit on a scale, can
    stop short of the limit and leave the level a finite estimate, so those
    flags are read only on the parameter's default link class.
    """
    compiled_tuple = tuple(compiled)
    shape = (bounds.n_rows, len(compiled_tuple))
    if not isinstance(family, CensoredSeparationFamily):
        return np.zeros(shape, dtype=bool)
    flags = np.asarray(family.censored_recession(bounds))
    if flags.shape != shape or flags.dtype != np.bool_:
        raise ValueError("family.censored_recession must return one boolean per row and predictor")
    parameters = tuple(family.parameters)
    sensitive = getattr(family, "censored_recession_link_sensitive", ())
    if isinstance(sensitive, str) or not set(sensitive) <= {p.name for p in parameters}:
        raise ValueError(
            f"{type(family).__name__}.censored_recession_link_sensitive must be a tuple of "
            f"the family's parameter names, got {sensitive!r}"
        )
    readable = [
        parameters[predictor.parameter_index].name not in sensitive
        or type(predictor.link) is _default_link_class(parameters[predictor.parameter_index])
        for predictor in compiled_tuple
    ]
    return flags & np.asarray(readable, dtype=bool)


def scan_censored_separation(
    compiled: Sequence[CompiledPredictor],
    frame: EagerFrame,
    bounds: ResponseBounds,
    weights: NDArray,
    *,
    recession: NDArray[np.bool_],
) -> list[SeparatedTerm]:
    """Levels and crossed cells whose rows are all censored on one side and flagged.

    The data bound such a cell's response from one side only, so on a
    predictor along which each of its rows' bound probabilities nears its
    supremum only at infinity (``recession``, per row and predictor) its
    likelihood keeps rising as its effect walks off, as a cell on a response
    boundary does.  The scan is the response-boundary scan on the indicator of
    rows that are not both censored on that side and flagged.  When every
    weighted row is, a predictor's intercept is that level, holding every
    row, and the predictor itself is reported in place of its cells.
    """
    findings: list[SeparatedTerm] = []
    weighted = np.asarray(weights) > 0
    for value, label in ((RIGHT_CENSORED, "right-censored"), (LEFT_CENSORED, "left-censored")):
        censored = bounds.side == value
        for index, predictor in enumerate(compiled):
            on_side = censored & recession[:, index]
            if not np.any(on_side):
                continue
            if predictor.intercept and np.all(on_side[weighted]):
                findings.append(SeparatedTerm(predictor.name, "intercept", label, ["intercept"], 1))
                continue
            findings.extend(
                replace(finding, term=f"{predictor.name}:{finding.term}", boundary=label)
                for finding in _scan_design_cells(
                    predictor.compiled,
                    frame,
                    np.where(on_side, 0.0, 1.0),
                    weights,
                    intercept=predictor.intercept,
                )
            )
    return findings


def format_censored_separation_message(findings: list[SeparatedTerm]) -> str:
    """One actionable message naming every cell whose rows are all censored on one side."""
    n_total = sum(len(f.labels) for f in findings)
    what = (
        "categorical cell(s) or intercept(s)"
        if any(f.kind == "intercept" for f in findings)
        else "categorical cell(s)"
    )
    lines = [
        f"Censoring separation: {n_total} {what} have rows censored on one side "
        "only, so the data bound their response from that side and their maximum-likelihood "
        "effects on the predictors named below run to infinity. The fit stops at an arbitrary "
        "value for them, and their predictions are not estimates."
    ]
    for f in findings:
        if f.kind == "intercept":
            lines.append(
                f"  {f.term!r}: every row is {f.boundary}, so its intercept has no finite estimate"
            )
            continue
        unit = "occupied cells" if f.kind == "cells" else "occupied levels"
        lines.append(
            f"  {f.term!r}: {len(f.labels)} of {f.n_occupied} {unit} have only {f.boundary} "
            f"rows: {_format_labels(f.labels)}"
        )
    lines.append(
        "Remedies: collapse the affected levels into neighbours "
        "(collapse_levels / Categorical(grouping=...)), model the factor with RandomEffect "
        "(its ridge bounds every level), or add rows whose response was observed. Pass "
        "separation='error' to refuse such designs, or separation='ignore' to disable this check."
    )
    return "\n".join(lines)


def apply_separation_policy(
    findings: Sequence[SeparatedTerm],
    policy: SeparationPolicy,
    *,
    stacklevel: int,
    formatter: Callable[[list[SeparatedTerm]], str] = format_separation_message,
) -> None:
    """Warn, refuse, or do nothing, exactly as the scalar builder does."""
    if not findings or policy == "ignore":
        return
    message = formatter(list(findings))
    if policy == "error":
        raise SeparationError(message)
    warnings.warn(message, SeparationWarning, stacklevel=stacklevel)


__all__ = [
    "SEPARATION_POLICIES",
    "SeparationPolicy",
    "apply_separation_policy",
    "censored_recession",
    "format_censored_separation_message",
    "predictor_response_boundaries",
    "record_factor_smooth_levels",
    "scan_censored_separation",
    "scan_predictor_separation",
    "validate_separation_policy",
]
