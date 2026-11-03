"""Configuration and validation helpers for spline feature specs."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from superglm.features._spline_knots import QUANTILE_STRATEGIES
from superglm.features._spline_ranges import PolynomialRange
from superglm.features.constraint import (
    CONSTRAINT_KINDS,
    CONSTRAINT_MODES,
    RULE_KINDS,
    ConstraintRules,
    ConstraintSpec,
    postfit_token,
)
from superglm.types import LambdaPolicy


class SplineOptionWarning(UserWarning):
    """A spline option outside its accepted values, built as it always was."""


# Warnings about a declaration point at the first frame outside the package.
_PACKAGE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The enumerated spline options as types, for the private helpers between the
# public signatures (which spell each ``Literal`` out) and the spec.
type SplineKindName = Literal["cr", "ps", "bs", "ns", "cr_cardinal"]
type KnotStrategyName = Literal["uniform", "quantile", "quantile_rows", "quantile_tempered"]
type PenaltyName = Literal["ssp", "none"] | None
type ExtrapolationName = Literal["clip", "extend", "error"]

# The ``extrapolation=`` policies a spline takes.
EXTRAPOLATIONS = ("clip", "extend", "error")
# The ``penalty=`` values a spline takes: "ssp" reparametrizes the basis
# (``dm_builder``), and "none" (or None) builds it without.
PENALTIES = ("ssp", "none")
# The ``knot_strategy=`` rules a spline takes: "uniform", or a data-driven one.
KNOT_STRATEGIES = ("uniform", *QUANTILE_STRATEGIES)
# The ``shape_engine=`` values a spline takes. Every fit-time shape constraint
# fits on the QP; "scop" (the retired P-spline engine) is accepted with
# ``warn_scop_retired`` and fitted on the QP.
SHAPE_ENGINES = ("qp", "scop")


class SCOPRetiredWarning(UserWarning):
    """A shape constraint declared or saved on the retired SCOP engine, fitted on the QP."""


def warn_scop_retired(*, saved: bool = False, moves_fit: bool = True) -> None:
    """Warn that SCOP is retired, at the caller's line outside superglm.

    ``saved`` is a P-spline a model fitted on SCOP; otherwise the declaration
    passed ``shape_engine="scop"``. ``moves_fit`` is False where SCOP never
    fitted the term (no fit-time constraint, or a basis other than a
    P-spline), so the fit does not change.
    """
    if saved:
        subject = "This P-spline was fitted on the SCOP shape engine, which is retired"
        remedy = "Declare the spline again, without shape_engine, to silence this warning."
    else:
        subject = "shape_engine='scop' names the SCOP shape engine, which is retired"
        remedy = "Drop shape_engine='scop' to silence this warning."
    if saved or moves_fit:
        effect = (
            "its shape constraint fits on the QP engine instead, so the fitted curve "
            "and its smoothing parameter differ from those 0.41.0 gave"
        )
    else:
        effect = "it has no effect on this spline, which fits as it did in 0.41.0"
    message = f"{subject}: {effect}. {remedy}"
    warnings.warn(message, SCOPRetiredWarning, skip_file_prefixes=(_PACKAGE_DIRECTORY,))


def _iter_knots(knots: Any) -> list[Any]:
    """Flatten a knots argument to a plain list without numeric coercion.

    A scalar (plain number, numpy scalar, or 0-d array) is one knot, and a
    bare string is one name, never a character sequence.
    """
    if isinstance(knots, np.ndarray):
        return list(knots.ravel())
    if isinstance(knots, str) or not hasattr(knots, "__iter__"):
        return [knots]
    return list(knots)


def _knots_contain_names(knots: Any) -> bool:
    """Whether a knots argument states any knot as a level name."""
    return any(isinstance(entry, str) for entry in _iter_knots(knots))


def validate_m_orders(spec: Any) -> None:
    """Validate the configured derivative penalty orders."""
    if len(spec._m_orders) > 1 and not spec._multi_m_supported:
        raise NotImplementedError(
            f"{type(spec).__name__} does not support multi-order penalties "
            f"(m tuple). Use a single m value."
        )
    if spec._max_penalty_order is not None:
        for order in spec._m_orders:
            if order > spec._max_penalty_order:
                raise ValueError(
                    f"{type(spec).__name__} supports penalty orders "
                    f"up to {spec._max_penalty_order}, got m={order}."
                )


def validate_m_orders_build(spec: Any) -> None:
    """Validate m orders after knot placement fixes the basis dimension."""
    for order in spec._m_orders:
        if order >= spec._n_basis:
            raise ValueError(
                f"Penalty order m={order} requires at least {order + 1} basis functions, "
                f"but this spline has n_basis={spec._n_basis}. "
                f"Increase n_knots or reduce m."
            )


def validate_select(spec: Any) -> None:
    """Validate select=True against the current spline capability policy."""
    if not spec.select:
        return
    if not spec._select_compatible(spec._m_orders):
        if not spec._select_supported:
            raise NotImplementedError(f"select=True is not supported for {type(spec).__name__}.")
        raise NotImplementedError(
            f"select=True with m={spec._m_orders} is not supported for "
            f"{type(spec).__name__}. "
            f"This is a current capability policy, not a mathematical impossibility."
        )


def _normalize_constraint(
    constraint: ConstraintSpec | ConstraintRules | None,
) -> tuple[ConstraintRules, str | None, str]:
    """The rules, ``constraint_kind`` and ``constraint_mode`` a spline stores for ``constraint``.

    A single full-range shape rule keeps its own kind and mode. Any other
    collection (several rules, a stretch, ``flat``, ``cap``, ``max_step``) is
    a fit-time collection whose kind is its description.
    """
    if constraint is None:
        return ConstraintRules(()), None, "postfit"
    if isinstance(constraint, ConstraintSpec):
        constraint = ConstraintRules((constraint,))
    if not isinstance(constraint, ConstraintRules):
        raise TypeError(
            "constraint must be None or a Constraint token such as Constraint.fit.increasing"
        )
    for rule in constraint:
        if rule.kind not in RULE_KINDS:
            raise ValueError(f"Unsupported constraint kind: {rule.kind!r}")
        if rule.mode not in CONSTRAINT_MODES:
            raise ValueError(f"Unsupported constraint mode: {rule.mode!r}")
    constraint.validate()
    if not constraint:
        return constraint, None, "postfit"
    only = constraint.rules[0]
    if len(constraint) == 1 and only.kind in CONSTRAINT_KINDS and only.stretch == (None, None):
        return constraint, only.kind, only.mode
    return constraint, constraint.describe(), "fit"


def fit_time_select_message(rules: ConstraintRules) -> str:
    """Why ``select=True`` cannot meet fit-time ``rules``, and what to do instead."""
    refusal = "Fit-time shape constraints are not supported with select=True. "
    token = postfit_token(rules)
    if token:
        return f"{refusal}Use select=False or {token}."
    return (
        f"{refusal}Use select=False; Constraint.postfit repairs one whole-curve shape "
        f"and cannot take {rules.describe()}."
    )


def initialize_spec(
    spec: Any,
    *,
    n_knots: int,
    degree: int,
    knot_strategy: str,
    penalty: str | None,
    knots: ArrayLike | None,
    discrete: bool | None,
    n_bins: int | None,
    extrapolation: str,
    boundary: tuple[float, float] | None,
    knot_alpha: float,
    select: bool,
    constraint: ConstraintSpec | ConstraintRules | None,
    m: int | tuple[int, ...],
    lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None,
    polynomial_ranges: Sequence[PolynomialRange] | None = None,
) -> None:
    """Initialize a spline spec's public config and mutable build-time state."""
    spec._polynomial_ranges = tuple(polynomial_ranges or ())
    if spec._polynomial_ranges and select:
        raise ValueError("polynomial_ranges cannot be combined with select=True in this version.")
    if spec._polynomial_ranges and constraint is not None:
        raise ValueError(
            "polynomial_ranges cannot be combined with a shape constraint in this version: "
            "the constraint would move the pinned coefficients."
        )
    rules, constraint_kind, constraint_mode = _normalize_constraint(constraint)
    curvature = [r.kind for r in rules if r.kind in {"convex", "concave"}]
    if degree == 0 and curvature:
        raise ValueError(
            f"degree=0 splines are discontinuous and cannot represent a classical "
            f"{curvature[0]} curve; use degree>=1"
        )
    if constraint_mode == "fit" and curvature and degree > 3:
        token = postfit_token(rules)
        advice = (
            f"degree<=3 or {token}"
            if token
            else f"degree<=3 (Constraint.postfit cannot take {rules.describe()})"
        )
        raise NotImplementedError(
            "Fit-time convex/concave constraints require degree <= 3. "
            "For higher degrees, non-negative derivative control coefficients "
            f"are sufficient but not necessary for global curvature; use {advice}."
        )

    spec.constraint_rules = rules
    spec.constraint_kind = constraint_kind
    spec.constraint_mode = constraint_mode

    # Keep existing internals working while the build path is generalized.
    spec.monotone = constraint_kind
    spec.monotone_mode = constraint_mode
    spec.select = select

    spec._m_orders = (m,) if isinstance(m, int) else tuple(m)
    if not all(isinstance(order, int) and order >= 1 for order in spec._m_orders):
        raise ValueError(f"m must contain positive integers, got {m}")
    validate_m_orders(spec)
    validate_select(spec)

    spec.degree = degree
    # Any other value builds without SSP, as it always has (None among them);
    # a value outside the accepted ones also says so, since it may be a typo.
    if penalty is not None and penalty not in PENALTIES:
        warnings.warn(
            f"penalty={penalty!r} is not one of {PENALTIES}; it was treated as 'none': "
            "no SSP reparametrization.",
            SplineOptionWarning,
            skip_file_prefixes=(_PACKAGE_DIRECTORY,),
        )
    spec.penalty = penalty
    spec.discrete = discrete
    spec.n_bins = n_bins
    if extrapolation not in EXTRAPOLATIONS:
        raise ValueError(
            f"extrapolation must be one of ('clip', 'extend', 'error'), got {extrapolation!r}"
        )
    spec.extrapolation = extrapolation
    configure_knots(
        spec,
        knots=knots,
        n_knots=n_knots,
        knot_strategy=knot_strategy,
        knot_alpha=knot_alpha,
        boundary=boundary,
    )
    initialize_runtime_state(spec, knot_strategy, lambda_policy)


def configure_knots(
    spec: Any,
    *,
    knots: ArrayLike | None,
    n_knots: int,
    knot_strategy: str,
    knot_alpha: float,
    boundary: tuple[float, float] | None,
) -> None:
    """Set an unfitted spline spec's knot configuration: stated knots, or a count and a rule."""
    # Stated knots make the rule moot. An unknown rule places uniform knots and
    # says so, since it may be a typo.
    if knots is None and knot_strategy not in KNOT_STRATEGIES:
        warnings.warn(
            f"knot_strategy={knot_strategy!r} is not one of {KNOT_STRATEGIES}; it was "
            "treated as 'uniform': uniform knots everywhere, including a cr spline's "
            "interaction margins, where 'uniform' itself places quantile knots.",
            SplineOptionWarning,
            skip_file_prefixes=(_PACKAGE_DIRECTORY,),
        )
    if knots is not None and _knots_contain_names(knots):
        # Level-name knots resolve only against an OrderedCategorical's declared
        # levels, so the host resolves them on its own copy; a numeric-axis
        # build refuses them in ``place_knots``. Numeric entries stay axis values.
        named = [k for k in _iter_knots(knots)]
        spec.n_knots = len(named)
        spec._explicit_knots = None
        spec._named_knots = named
    elif knots is not None:
        explicit_knots = np.asarray(knots, dtype=np.float64).ravel()
        if explicit_knots.ndim != 1 or len(explicit_knots) < 1:
            raise ValueError("knots must be a non-empty 1D array")
        if not np.all(np.diff(explicit_knots) > 0):
            raise ValueError("knots must be strictly increasing")
        spec.n_knots = len(explicit_knots)
        spec._explicit_knots = explicit_knots
        spec._named_knots = None
    else:
        spec.n_knots = n_knots
        spec._explicit_knots = None
        spec._named_knots = None

    spec.knot_strategy = knot_strategy
    spec.knot_alpha = knot_alpha
    spec._explicit_boundary = _coerce_boundary(boundary, spec._explicit_knots)
    spec._knot_strategy_actual = knot_strategy


def _coerce_boundary(
    boundary: tuple[float, float] | None,
    explicit_knots: NDArray | None,
) -> tuple[float, float] | None:
    """Validate and normalize an explicit fit boundary, if any."""
    if boundary is None:
        return None

    lo_bound, hi_bound = float(boundary[0]), float(boundary[1])
    if lo_bound >= hi_bound:
        raise ValueError(f"boundary must satisfy lo < hi, got boundary=({lo_bound}, {hi_bound})")
    if explicit_knots is not None:
        if explicit_knots[0] <= lo_bound or explicit_knots[-1] >= hi_bound:
            raise ValueError(
                f"explicit knots must lie strictly inside boundary=({lo_bound}, {hi_bound}), "
                f"got knots in [{explicit_knots[0]}, {explicit_knots[-1]}]"
            )
    return (lo_bound, hi_bound)


def initialize_runtime_state(
    spec: Any,
    knot_strategy: str,
    lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None,
) -> None:
    """Initialize mutable build-time state on a spline spec."""
    spec._knots = np.array([])
    spec._base_interior_knots = None
    spec._n_basis = 0
    spec._lo = 0.0
    spec._hi = 1.0
    spec._knot_strategy_actual = knot_strategy
    spec._difference_penalty = {}
    spec._R_inv = None
    spec._interaction_projection = None
    spec._basis_lo = None
    spec._basis_hi = None
    spec._basis_d1_lo = None
    spec._basis_d1_hi = None
    spec._U_null = None
    spec._U_range = None
    spec._omega_range = None
    spec._penalty_components = None
    spec._lambda_policy = lambda_policy
    clear_scop_state(spec)


def clear_scop_state(spec: Any) -> None:
    """Drop the SCOP transform a model fitted before 0.42 left; every build starts from none.

    ``transform`` scores through ``_scop_Sigma`` whenever it is set, so a
    saved SCOP spec refitted on the QP would otherwise score its new
    coefficients through the old SCOP basis instead of its ``R_inv``.
    """
    for name in ("_scop_Sigma", "_scop_null_dim", "_scop_col_means"):
        spec.__dict__.pop(name, None)
