"""The coefficient face a binding shape rule leaves, and the record of a shape-constrained refit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.layout import StackedLayout
from superglm.distributional.smoothing.penalty_face import (
    PenaltyFace,
    PenaltyFaceError,
    _readonly,
)


@dataclass(frozen=True, eq=False)
class ShapeFace(PenaltyFace):
    """The affine face ``{b : A_F b = A_F b*}`` of the shape rows binding at a fit ``b*``.

    A constrained fit is locally the fit with its binding rows held as equalities, so its
    degrees of freedom are those of the fit on this face, read from ``Z (Z' H Z)^+ Z'`` for
    ``Z`` the face's orthonormal ``null_basis`` (Meyer and Woodroofe 2000; Tibshirani and
    Taylor 2012, Theorem 3). It rides in the solver's ``coefficient_face`` slot, so every
    reader of a fit's face (EDF, rank certificate, the solver's own solves) reads it in reduced
    coordinates. The covariance a fit reports is not the face's: which rows bind is itself
    estimated, so the fit reports the covariance it had without the rules
    (``inference.compute_joint_inference``).
    ``component_names`` are those of the exact-infinity penalty face ``base`` it sits inside,
    empty without one. ``anchor`` is ``b*``: a step is projected onto the face through it, which
    a row with a nonzero bound needs.
    """

    unnamed_ok: ClassVar[bool] = True
    anchor: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    base: PenaltyFace | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        anchor = _readonly(self.anchor)
        if anchor.shape != (self.width,) or not np.all(np.isfinite(anchor)):
            raise PenaltyFaceError("shape face anchor does not match its layout")
        object.__setattr__(self, "anchor", anchor)

    def validate_layout(self, layout: StackedLayout) -> None:
        if not isinstance(layout, StackedLayout):
            raise TypeError("layout must be a StackedLayout")
        if tuple(layout.coefficient_names) != self.coefficient_names:
            raise PenaltyFaceError("shape face belongs to a different coefficient layout")
        if self.base is not None:
            self.base.validate_layout(layout)

    def project(self, coefficients: NDArray) -> NDArray[np.float64]:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (self.width,) or not np.all(np.isfinite(values)):
            raise ValueError("coefficients do not match the shape face")
        return _readonly(
            self.anchor + self.null_basis @ (self.null_basis.T @ (values - self.anchor))
        )


@dataclass(frozen=True)
class ShapeTermRecord:
    """One term's shape rules at the fit: how many rows, how many bind, how far they moved it.

    ``largest_change`` is the largest change, over the training rows, of the term's effect on
    the linear predictor from the smoothing-selected fit without the rules to the fit with them,
    after removing the constant change the predictor's intercept absorbs. ``log_scale`` says the
    predictor is on a log scale, so the change is a log relativity.
    """

    term: str
    rules: str
    n_rows: int
    n_binding: int
    largest_change: float
    log_scale: bool


@dataclass(frozen=True)
class ShapeRepairRecord:
    """One term's ``Constraint.postfit`` repair: the projection SuperGLM's repairer made.

    ``violation_before`` and ``violation_after`` are the largest amount by which the term's
    curve broke its shape before and after the projection, as the repairer's span-wise
    certificate measures it. ``largest_change`` is the largest change of the curve on the
    repair grid, on the linear-predictor scale; ``log_scale`` says that scale is a log, so the
    change is a log relativity. ``intercept_shift`` is how far the parameter's intercept moved
    when it was re-estimated with every term held at its repaired value.
    """

    term: str
    kind: str
    violation_before: float
    violation_after: float
    largest_change: float
    log_scale: bool
    intercept_shift: float = 0.0


def shape_repair_sentence(record: ShapeRepairRecord) -> str:
    """One plain sentence on what a post-fit repair did to a term."""
    return (
        f"The post-fit rule {record.kind!r} on {record.term} repaired the fitted curve, which "
        f"broke it by up to {record.violation_before:.3g}; the repair moved the curve by at most "
        f"{_moved(record.largest_change, record.log_scale)}, and the term's standard errors, "
        "bands and test are withheld."
    )


def _moved(change: float, log_scale: bool) -> str:
    if log_scale:
        return f"{100.0 * np.expm1(change):.1f}% ({change:.3g} on the log scale)"
    return f"{change:.3g} on the predictor's scale"


@dataclass(frozen=True)
class ShapeFit:
    """What a fit's shape rules did.

    ``applied`` is False when the fit without the rules already met every row, which is then the
    fit. Otherwise the rules were enforced by a refit at the smoothing the unconstrained search
    chose; ``converged`` and ``reason`` say how that refit stopped, ``n_binding`` is the number of
    rows that bind at the reported coefficients, and ``max_violation`` the largest scaled amount
    by which any row misses its bound there. ``unconstrained_coefficients`` are the coefficients
    the same smoothing gives without the rules; ``unruled`` is True when the refit could not start
    and those are the coefficients reported. ``covariance`` is the covariance the fit published:
    ``"rule_free"``, the fit's with no shape row held, or ``"face"``, conditional on the rows that
    bind, which a record saved before fits held no row has (``compute_joint_inference``).
    """

    terms: tuple[ShapeTermRecord, ...]
    applied: bool
    converged: bool
    reason: str
    iterations: int
    backtracks: int
    n_rows: int
    n_binding: int
    max_violation: float
    unconstrained_coefficients: NDArray[np.float64]
    unconstrained_penalized_log_likelihood: float
    penalized_log_likelihood: float
    seconds: float = 0.0
    unruled: bool = False
    # Absent on a record saved by 0.43.0, which reads the class default: that fit published the
    # face's covariance, and keeps it until it is refit.
    covariance: Literal["face", "rule_free"] = "face"

    def __post_init__(self) -> None:
        values = np.array(self.unconstrained_coefficients, dtype=np.float64, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "unconstrained_coefficients", values)
        object.__setattr__(self, "terms", tuple(self.terms))

    @property
    def binds(self) -> bool:
        """Whether any row binds at the reported coefficients."""
        return self.applied and self.n_binding > 0


_REASONS = {
    "max_iterations": "it ran out of iterations",
    "line_search_failed": "no step improved the likelihood while keeping the rules",
    "no_ascent_direction": "the fit's curvature gave no usable direction",
    "no_feasible_start": "it found no starting point that meets the rules",
}


def shape_term_sentence(term: ShapeTermRecord, shape: ShapeFit) -> str:
    """One plain sentence on what a term's shape rules did to its fit."""
    rules = f"The rule {term.rules!r} on {term.term}"
    if not shape.applied:
        return f"{rules} did not bind: the fitted curve already obeyed it, so the fit is unchanged."
    if shape.unruled:
        return (
            f"{rules} could not be applied: the refit found no start that meets it, so the curve is "
            f"the fit without it, which misses it by up to {shape.max_violation:.1e} of a row's scale."
        )
    if term.n_binding == 0:
        return (
            f"{rules} held the fit in place nowhere: no row of it binds at the fitted curve, "
            f"so the other terms' rules alone moved it, by at most {_change(term)}."
        )
    moved = _change(term)
    binds = (
        f"{rules} binds at {term.n_binding} of {term.n_rows} checkpoints along the curve; "
        f"it moved the curve by at most {moved} from the fit without the rule."
    )
    if shape.covariance == "face":
        return (
            f"{binds} Its standard errors treat those checkpoints as fixed, as the version that "
            "saved it did, so its bands there cover too rarely; refit it to get the bands of the "
            "fit without the rule."
        )
    return (
        f"{binds} Standard errors and bands are those of the fit without the rule, drawn "
        "around the ruled curve."
    )


def _change(term: ShapeTermRecord) -> str:
    return _moved(term.largest_change, term.log_scale)


def shape_disclosures(shape: ShapeFit | None) -> tuple[str, ...]:
    """The message a fit whose shape-rule refit did not converge publishes, else nothing."""
    if shape is None or not shape.applied or shape.converged:
        return ()
    terms = ", ".join(term.term for term in shape.terms)
    why = _REASONS.get(shape.reason, shape.reason)
    if shape.unruled:
        return (
            f"SuperLSS could not apply the shape rules on {terms}: {why}. The coefficients "
            f"returned are the fit without the rules, which misses them by up to "
            f"{shape.max_violation:.1e} of a row's scale, with that fit's standard errors. "
            "Refit with a larger max_inner_iter, or relax the rules, before reading the "
            "summary.",
        )
    return (
        f"SuperLSS could not certify the fit under the shape rules on {terms}: {why}. The "
        "coefficients returned are the last the refit reached; they meet the rules to within "
        f"{shape.max_violation:.1e} of each row's scale, but they may not be the best fit that "
        "does, and the standard errors "
        + (
            "treat only the rules that bind there as fixed"
            if shape.covariance == "face"
            else "are those of the fit without the rules"
        )
        + ". Refit with a larger max_inner_iter, or relax the rules, before reading the summary.",
    )
