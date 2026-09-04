"""Per-parameter term inference for a fitted distributional model.

Every predictor of a location-scale-shape fit carries its own terms, and this
module reports each one on its own link scale: the fitted effect over the
term's grid or level set, its Bayesian standard error, pointwise and
simultaneous bands, the Wood (2013) test of "this term is zero", and one
summary table over all of them.

The band construction is the Bayesian posterior ``N(beta_hat, V)`` whose
across-the-function coverage is Nychka (1988) as extended component-wise by
Marra and Wood (2012), *Scandinavian Journal of Statistics* 39(1), 53-74; the
max-deviation simultaneous critical value is Ruppert, Wand and Carroll (2003),
*Semiparametric Regression*, section 6.5, and comes from
:func:`superglm.distributional.posterior.simultaneous_critical_value`.

The term test is Wood (2013), *Biometrika* 100(1), 221-228, "On p-values for
smooth components of an extended generalized additive model", evaluated by
:func:`superglm.stats.wood_pvalue.wood_test_smooth` -- the same routine the
scalar path uses, so the two report the same statistic.  Its statistic is
``T_r = f_hat' V_f^{r-} f_hat`` with ``V_f`` the Bayesian covariance of the
term evaluated on a set of points and ``V_f^{r-}`` a rank-``r`` pseudo-inverse.

**The rank rule.**  Wood (2013), section 2.2, on why the rank is neither the
numerical rank of ``V_f`` nor a rounded EDF:

    "One way to avoid both this dropping of important terms, and the
    overweighting of highly penalized terms, is to relax the requirement for
    integer degrees of freedom in the test statistic.  Instead use ``r = nu``
    in a generalized ``T_r``, which is well defined for non-integer ``r``,
    varies smoothly with ``r``, but recovers a conventional Wald statistic
    when ``r = nu`` is integer."

Section 2.4 states which effective degrees of freedom ``nu`` is, for one term
of a multi-term model:

    "Note that if ``F = V X^T W X / phi`` then ``nu_j``, the required
    effective degrees of freedom for ``f_hat_j``, can be obtained by summing
    the diagonal elements of ``2F - FF`` corresponding to ``beta_hat_j``."

That is Wood's alternative EDF, and it is *not* the model's reported EDF
(``tr F`` over the same block, which this module reports as ``edf``);
:func:`_alternative_edf` computes it from ``JointInference.influence``, which
is exactly the ``F`` of that sentence.  Section 2.3 gives the reference
distribution the fractional rank implies, with ``k = floor(r) + 1`` and
``nu = r - k + 1``:

    "It follows that in the large sample limit under (2) and the null
    hypothesis, ``T_r ~ chi^2_r``, if ``r`` is integer, while for non-integer
    ``r``, ``T_r ~ chi^2_{k-2} + lambda_1 chi^2_1 + lambda_2 chi^2_1``, where
    ``lambda_1 = {nu + 1 + (1 - nu^2)^{1/2}}/2`` and
    ``lambda_2 = nu + 1 - lambda_1``."

The scale of a distributional model is modelled rather than estimated as a
free dispersion, so the reference is that chi-square mixture and never an F:
``wood_test_smooth`` is called with the residual degrees of freedom set to its
"scale known" sentinel.

Section 2.4 also settles which rows ``f_hat`` is evaluated on: the statistic is
computed through the QR factor of the term's model matrix, and "for large
datasets, little is usually gained by using the whole of ``X_j`` to compute
``T_r``, and we might as well use a random sample of ``n_s`` of its rows".
A one-variable term is therefore evaluated on its own ``n_points`` grid across
the training range -- the same grid the reported effect uses -- and a term with
no one-dimensional grid (an interaction) on the training design itself.

**Two shapes a level term takes.**  A term whose free levels live in a second
block -- an ``OrderedCategorical`` with ``specials=``, whose special levels sit
beside the smooth in the qualified term ``<parameter>:<term>:special`` --
reports both blocks in one payload, so a special level reads its own
coefficient and standard error instead of the smooth's exact zero;
:attr:`ParameterTermEffect.special` says which levels those are.  And a level
term the fit cannot identify -- one exactly collinear with an interaction on
the same feature, which a factor smooth carrying its own main effect produces
-- has an EDF and a covariance block of zero while the min-norm solve still
writes a number into its coefficients; :func:`summary_table` annotates that row
rather than leaving a reader to price off it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.posterior import (
    CovarianceKind,
    _posterior_draw_count,
    _posterior_draws_from_covariance,
    posterior_covariance,
    simultaneous_critical_value,
)
from superglm.distributional.prediction_design import (
    _required_columns,
    build_joint_prediction_design,
)
from superglm.stats.wood_pvalue import wood_test_smooth

TermKind = Literal["spline", "categorical", "numeric", "other"]

#: Payload schema version shared by the term-result serializers.
_SCHEMA_VERSION = 1
#: ``wood_test_smooth`` reads a non-positive residual df as "scale known".
_SCALE_MODELLED_RESIDUAL_DF = -1.0
#: Probe used to recognise an inverse link of the form ``c + exp(eta)``.
_LOG_LINK_PROBE = np.linspace(-1.5, 1.5, 7)
#: Column order of :func:`summary_table`.
_SUMMARY_COLUMNS = (
    "parameter",
    "term",
    "edf",
    "lambda",
    "statistic",
    "rank",
    "p_value",
    "estimate",
    "se",
    "note",
)
#: Row label the summary gives a predictor's intercept.
_INTERCEPT = "(intercept)"
#: Suffix of the qualified term holding a level term's free (special) block.
_SPECIAL_SUFFIX = ":special"
#: An EDF at or below this counts as "this term has no estimable direction".
_ABSORBED_EDF = 1.0e-10


def _json_number(value: Any) -> float | None:
    """Return a JSON-safe float, mapping a non-finite value to ``None``."""
    number = float(value)
    return number if math.isfinite(number) else None


def _json_array(values: NDArray | None) -> list[float | None] | None:
    if values is None:
        return None
    return [_json_number(value) for value in np.asarray(values, dtype=np.float64)]


@dataclass(frozen=True)
class ParameterTermEffect:
    """One term of one predictor, reported on that predictor's link scale.

    ``special`` is aligned with ``levels`` and is true at each level whose
    effect comes from the term's free ``:special`` block rather than from its
    smooth; it is ``None`` for a term that has no such block, which is every
    term except an ``OrderedCategorical`` fitted with ``specials=``.
    """

    parameter: str
    link: str
    term: str
    kind: TermKind
    x: NDArray[np.float64] | None
    levels: tuple[str, ...] | None
    special: tuple[bool, ...] | None
    effect: NDArray[np.float64]
    se: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    lower_simultaneous: NDArray[np.float64] | None
    upper_simultaneous: NDArray[np.float64] | None
    critical_value: float | None
    multiplier: NDArray[np.float64] | None
    edf: float
    lambdas: Mapping[str, float]
    covariance_kind: str
    alpha: float
    schema_version: int = field(default=_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lambdas", MappingProxyType(dict(self.lambdas)))

    def to_json(self) -> dict[str, Any]:
        """Emit the payload as lists, floats, strings and ``None`` only."""
        return {
            "schema_version": int(self.schema_version),
            "parameter": self.parameter,
            "link": self.link,
            "term": self.term,
            "kind": self.kind,
            "x": _json_array(self.x),
            "levels": None if self.levels is None else list(self.levels),
            "special": None if self.special is None else [bool(value) for value in self.special],
            "effect": _json_array(self.effect),
            "se": _json_array(self.se),
            "lower": _json_array(self.lower),
            "upper": _json_array(self.upper),
            "lower_simultaneous": _json_array(self.lower_simultaneous),
            "upper_simultaneous": _json_array(self.upper_simultaneous),
            "critical_value": (
                None if self.critical_value is None else _json_number(self.critical_value)
            ),
            "multiplier": _json_array(self.multiplier),
            "edf": _json_number(self.edf),
            "lambdas": {name: float(value) for name, value in self.lambdas.items()},
            "covariance_kind": self.covariance_kind,
            "alpha": float(self.alpha),
        }


@dataclass(frozen=True)
class TermTest:
    """The Wood (2013) test of a term against zero, or its Wald analogue."""

    parameter: str
    term: str
    statistic: float
    rank: float
    p_value: float
    edf: float
    schema_version: int = field(default=_SCHEMA_VERSION)

    def to_json(self) -> dict[str, Any]:
        """Emit the payload as floats, strings and ``None`` only."""
        return {
            "schema_version": int(self.schema_version),
            "parameter": self.parameter,
            "term": self.term,
            "statistic": _json_number(self.statistic),
            "rank": _json_number(self.rank),
            "p_value": _json_number(self.p_value),
            "edf": _json_number(self.edf),
        }


@dataclass(frozen=True)
class _PreparedTermEffect:
    """Covariance-independent state of one fully validated effect panel."""

    parameter: str
    link: str
    term: str
    kind: TermKind
    x: NDArray[np.float64] | None
    levels: tuple[str, ...] | None
    special: tuple[bool, ...] | None
    design: NDArray[np.float64]
    columns: NDArray[np.intp]
    beta: NDArray[np.float64]
    effect: NDArray[np.float64]
    multiplier: NDArray[np.float64] | None
    edf: float
    lambdas: Mapping[str, float]
    covariance_kind: CovarianceKind
    alpha: float
    pointwise_critical: float
    draw_count: int | None
    seed: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lambdas", MappingProxyType(dict(self.lambdas)))


@dataclass(frozen=True)
class _PreparedTermTest:
    """Covariance-independent state of one fully validated term test."""

    qualified: str
    parameter: str
    term: str
    term_slice: slice
    beta: NDArray[np.float64]
    edf: float
    level_term: bool
    design: NDArray[np.float64] | None
    alternative_edf: float | None
    reconciliation_tolerance: float


def _level_domain(spec: Any) -> tuple[Any, ...] | None:
    """Return a term's fitted level domain, or ``None`` if it has no levels.

    A ``Categorical`` names it ``_levels``; an ``OrderedCategorical`` keeps the
    ordered domain in ``_ordered_levels``.  Reading the compiled spec is what
    makes the reference level and any level pinned to it come back on the same
    footing as the estimated ones: neither owns a design column, so both read
    back as an exact zero effect with a zero standard error.
    """
    for attribute in ("_levels", "_ordered_levels"):
        levels = getattr(spec, attribute, None)
        if levels is not None and len(levels) > 0:
            return tuple(levels)
    return None


def _special_domain(spec: Any) -> tuple[str, ...]:
    """Return the level labels a term reports through its free ``:special`` block.

    ``OrderedCategorical`` reports each special under the label its own domain
    spells it with, which is the namespace ``_level_domain`` returns them in, so
    the two line up by string without re-deriving either list.
    """
    return tuple(str(level) for level in getattr(spec, "_special_display", ()))


def _term_kind(spec: Any) -> TermKind:
    from superglm.features.numeric import Numeric
    from superglm.features.polynomial import Polynomial
    from superglm.features.spline import _SplineBase

    if _level_domain(spec) is not None:
        return "categorical"
    if isinstance(spec, _SplineBase | Polynomial):
        return "spline"
    if isinstance(spec, Numeric):
        return "numeric"
    return "other"


def _is_shifted_log_link(link: Any) -> bool:
    """True when the inverse link is ``c + exp(eta)`` for some constant ``c``.

    That is the whole family for which adding an effect on the link scale
    multiplies the natural parameter (above its floor) by ``exp(effect)``, so
    it is the exact condition under which a ``multiplier`` view means anything.
    Probing ``deriv_inverse`` -- a required method of the ``Link`` protocol --
    tests the property itself rather than a list of class names that a new
    family would have to remember to join.
    """
    try:
        derivative = np.asarray(link.deriv_inverse(_LOG_LINK_PROBE), dtype=np.float64)
    except (AttributeError, TypeError, ValueError, FloatingPointError):
        return False
    if derivative.shape != _LOG_LINK_PROBE.shape:
        return False
    return bool(np.allclose(derivative, np.exp(_LOG_LINK_PROBE), rtol=1.0e-10, atol=0.0))


def _resolve_term(fitted: Any, parameter: str, term: str) -> tuple[str, Any, slice]:
    """Return ``(qualified name, predictor state, global term slice)``."""
    state = fitted.layout.predictor(parameter)
    qualified = f"{parameter}:{term}"
    try:
        term_slice = fitted.layout.term_slices[qualified]
    except KeyError:
        known = sorted(
            name for name in fitted.layout.term_slices if name.startswith(f"{parameter}:")
        )
        raise KeyError(f"unknown term {qualified!r}; predictor {parameter!r} has {known}") from None
    return qualified, state, term_slice


def _compiled_predictor(fitted: Any, parameter: str) -> Any:
    """Return one predictor's compiled state."""
    for predictor in fitted.compiled_predictors:
        if predictor.name == parameter:
            return predictor
    raise KeyError(f"unknown predictor {parameter!r}")


def _compiled_spec(fitted: Any, parameter: str, term: str) -> Any:
    """Return the compiled feature spec of a term, or ``None`` for an interaction."""
    return _compiled_predictor(fitted, parameter).compiled.specs.get(term)


def _interaction_on(fitted: Any, parameter: str, feature: str) -> str | None:
    """Name the predictor's first interaction built on ``feature``, if any."""
    compiled = _compiled_predictor(fitted, parameter).compiled
    for name in compiled.interaction_order:
        if feature in compiled.interaction_specs[name].parent_names:
            return name
    return None


def _reference_row(frame: EagerFrame, columns: tuple[str, ...]) -> dict[str, Any]:
    """Hold every covariate at its training median (numeric) or mode (otherwise)."""
    row: dict[str, Any] = {}
    for name in columns:
        values = np.asarray(frame.column_array(name))
        if frame.column_kind(name) == "numeric":
            row[name] = float(np.nanmedian(values.astype(np.float64)))
        else:
            row[name] = pd.Series(values).value_counts().index[0]
    return row


def _sweep(
    fitted: Any,
    frame: EagerFrame,
    parameter: str,
    term: str,
    n_points: int,
) -> tuple[pd.DataFrame, NDArray[np.float64] | None, tuple[str, ...] | None]:
    """Build the one-term evaluation frame, and the grid or levels it sweeps."""
    spec = _compiled_spec(fitted, parameter, term)
    if spec is None:
        raise NotImplementedError(
            f"term {parameter}:{term!r} is an interaction and has no one-dimensional "
            "effect grid; report it as a surface instead"
        )
    columns = _required_columns(fitted.compiled_predictors)
    reference = _reference_row(frame, columns)

    levels = _level_domain(spec)
    if levels is not None:
        swept: Any = list(levels)
        grid = None
        names = tuple(str(level) for level in levels)
    else:
        values = np.asarray(frame.column_array(term), dtype=np.float64)
        grid = np.linspace(float(np.nanmin(values)), float(np.nanmax(values)), n_points)
        swept = grid
        names = None

    rows = len(swept)
    data = {name: pd.Series([reference[name]] * rows) for name in columns}
    data[term] = pd.Series(swept)
    return pd.DataFrame(data), grid, names


def _term_design(
    fitted: Any,
    evaluation_frame: FrameLike | EagerFrame,
    state: Any,
    *term_slices: slice,
) -> NDArray[np.float64]:
    """Return the term's own columns of its predictor's local design.

    Several slices are laid side by side in the order given, which is how a
    level term whose free levels live in a second block is evaluated as one
    function of its levels.
    """
    design = build_joint_prediction_design(
        evaluation_frame, fitted.compiled_predictors, fitted.layout
    )
    offset = state.coefficient_slice.start
    local = design.local[state.name]
    return np.hstack(
        [
            np.asarray(local[:, block.start - offset : block.stop - offset], dtype=np.float64)
            for block in term_slices
        ]
    )


def _coefficient_columns(*term_slices: slice) -> NDArray[np.intp]:
    """The global coefficient indices of a sequence of term slices, in order."""
    return np.concatenate(
        [np.arange(block.start, block.stop, dtype=np.intp) for block in term_slices]
    )


def _band_variance(design: NDArray[np.float64], block: NDArray[np.float64]) -> NDArray[np.float64]:
    variance = np.einsum("ij,jk,ik->i", design, block, design, optimize=True)
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(variance), initial=0.0)))
    if np.any(variance < -tolerance):
        raise ValueError("a term band variance is materially negative")
    return np.maximum(variance, 0.0)


def _term_lambdas(fitted: Any, qualified: str) -> dict[str, float]:
    return {
        name: float(fitted.lambdas[name])
        for name in fitted.layout.penalty_names
        if name.rpartition("#")[0] == qualified
    }


def _alternative_edf(fitted: Any, term_slice: slice, edf: float) -> float:
    """Wood's alternative EDF, ``sum diag(2F - FF)`` over the term's block.

    ``JointInference.influence`` is the ``F = V X'WX`` of Wood (2013), section
    2.4, so the sentence quoted in the module docstring translates directly:
    ``2 tr F_jj - tr(F_j. F_.j)`` restricted to the term's coefficients.
    """
    influence = np.asarray(fitted.inference.influence, dtype=np.float64)
    squared = float(np.einsum("ij,ji->", influence[term_slice, :], influence[:, term_slice]))
    return 2.0 * float(edf) - squared


def _block_wald(
    beta: NDArray[np.float64],
    block: NDArray[np.float64],
    tolerance: float,
) -> tuple[float, float, float]:
    """Chi-square Wald test of a level block against zero.

    A level term has no smooth to shrink towards a null space, so the level
    block gets the plain quadratic form the scalar summary reports for a
    parametric group, with the pseudo-inverse taken at the fit's own
    reconciliation tolerance relative to the largest eigenvalue.
    """
    values, vectors = np.linalg.eigh(0.5 * (block + block.T))
    cutoff = tolerance * max(float(np.max(values, initial=0.0)), 0.0)
    keep = values > cutoff
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        return 0.0, 0.0, 1.0
    projected = vectors[:, keep].T @ beta
    statistic = float(np.sum(projected**2 / values[keep]))
    return statistic, float(rank), float(stats.chi2.sf(statistic, rank))


def term_effect(
    fitted: Any,
    X_train: FrameLike | EagerFrame,
    parameter: str,
    term: str,
    *,
    covariance: CovarianceKind = "fixed",
    n_points: int = 200,
    alpha: float = 0.05,
    n_sim: int = 2000,
    seed: int = 42,
    simultaneous: bool = True,
) -> ParameterTermEffect:
    """Report one term of one predictor on that predictor's link scale.

    A one-variable term is swept over ``n_points`` points spanning its training
    range with every other covariate held at its training median or mode; a
    term with levels reports one row per fitted level, the reference level (and
    any level pinned to it) reading back as an exact zero.  Bands are the
    Bayesian pointwise interval and, unless ``simultaneous=False``, the
    max-deviation simultaneous band of Ruppert, Wand and Carroll (2003),
    section 6.5, over ``n_sim`` posterior draws.  ``multiplier`` is
    ``exp(effect)`` when the predictor's link is a log-type link, and ``None``
    otherwise, because only then does an additive link-scale effect read as a
    multiplicative one.

    A term whose free levels sit in a ``<parameter>:<term>:special`` block --
    an ``OrderedCategorical`` fitted with ``specials=`` -- reports both blocks
    together: each special level reads its own coefficient and standard error
    from that block and its covariance, ``special`` marks which levels those
    are, and ``edf`` and ``lambdas`` cover the union.  Read on the smooth block
    alone a special level would come back as an exact zero with no band, which
    is the one number the model does not claim for it.
    """
    prepared = _prepare_term_effect(
        fitted,
        X_train,
        parameter,
        term,
        covariance=covariance,
        n_points=n_points,
        alpha=alpha,
        n_sim=n_sim,
        seed=seed,
        simultaneous=simultaneous,
    )
    matrix = posterior_covariance(fitted, kind=covariance)
    return _term_effect_from_covariance(fitted, prepared, matrix)


def _prepare_term_effect(
    fitted: Any,
    X_train: FrameLike | EagerFrame,
    parameter: str,
    term: str,
    *,
    covariance: CovarianceKind = "fixed",
    n_points: int = 200,
    alpha: float = 0.05,
    n_sim: int = 2000,
    seed: int = 42,
    simultaneous: bool = True,
) -> _PreparedTermEffect:
    """Validate and retain every covariance-independent part of one panel."""
    points = int(n_points)
    if points < 2:
        raise ValueError("n_points must place at least two points on the term's grid")
    resolved_alpha = float(alpha)
    if not 0.0 < resolved_alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1)")

    qualified, state, term_slice = _resolve_term(fitted, parameter, term)
    frame = as_eager_frame(X_train)
    evaluation, grid, levels = _sweep(fitted, frame, parameter, term, points)
    special_name = f"{qualified}{_SPECIAL_SUFFIX}"
    special_slice = fitted.layout.term_slices.get(special_name)
    blocks = (term_slice,) if special_slice is None else (term_slice, special_slice)
    names = (qualified,) if special_slice is None else (qualified, special_name)
    design = _term_design(fitted, evaluation, state, *blocks)
    columns = _coefficient_columns(*blocks)
    beta = np.asarray(fitted.coefficients, dtype=np.float64)
    effect = design @ beta[columns]

    draw_count: int | None = None
    resolved_seed: int | None = None
    if simultaneous:
        draw_count = _posterior_draw_count(n_sim)
        resolved_seed = int(seed)

    spec = _compiled_spec(fitted, parameter, term)
    special: tuple[bool, ...] | None = None
    if special_slice is not None:
        marked = set(_special_domain(spec))
        special = tuple(str(level) in marked for level in levels or ())
    return _PreparedTermEffect(
        parameter=parameter,
        link=type(state.link).__name__,
        term=term,
        kind=_term_kind(spec),
        x=grid,
        levels=levels,
        special=special,
        design=design,
        columns=columns,
        beta=beta,
        effect=effect,
        multiplier=np.exp(effect) if _is_shifted_log_link(state.link) else None,
        edf=float(sum(fitted.inference.term_edf[name] for name in names)),
        lambdas={
            key: value for name in names for key, value in _term_lambdas(fitted, name).items()
        },
        covariance_kind=covariance,
        alpha=resolved_alpha,
        pointwise_critical=float(stats.norm.isf(0.5 * resolved_alpha)),
        draw_count=draw_count,
        seed=resolved_seed,
    )


def _term_effect_from_covariance(
    fitted: Any,
    prepared: _PreparedTermEffect,
    matrix: NDArray[np.float64],
) -> ParameterTermEffect:
    """Evaluate one fully prepared term effect from a resolved covariance."""
    design = prepared.design
    columns = prepared.columns
    effect = prepared.effect
    se = np.sqrt(_band_variance(design, matrix[np.ix_(columns, columns)]))

    critical: float | None = None
    lower_simultaneous: NDArray[np.float64] | None = None
    upper_simultaneous: NDArray[np.float64] | None = None
    if prepared.draw_count is not None:
        assert prepared.seed is not None
        draws = _posterior_draws_from_covariance(
            fitted,
            matrix,
            prepared.draw_count,
            covariance=prepared.covariance_kind,
            seed=prepared.seed,
        )
        critical = simultaneous_critical_value(
            design,
            columns,
            draws,
            prepared.beta,
            se,
            alpha=prepared.alpha,
        )
        lower_simultaneous = effect - critical * se
        upper_simultaneous = effect + critical * se

    return ParameterTermEffect(
        parameter=prepared.parameter,
        link=prepared.link,
        term=prepared.term,
        kind=prepared.kind,
        x=prepared.x,
        levels=prepared.levels,
        special=prepared.special,
        effect=effect,
        se=se,
        lower=effect - prepared.pointwise_critical * se,
        upper=effect + prepared.pointwise_critical * se,
        lower_simultaneous=lower_simultaneous,
        upper_simultaneous=upper_simultaneous,
        critical_value=critical,
        multiplier=prepared.multiplier,
        edf=prepared.edf,
        lambdas=prepared.lambdas,
        covariance_kind=prepared.covariance_kind,
        alpha=prepared.alpha,
    )


def term_test(
    fitted: Any,
    X_train: FrameLike | EagerFrame,
    parameter: str,
    term: str,
    *,
    covariance: CovarianceKind = "fixed",
    n_points: int = 200,
) -> TermTest:
    """Test one term against zero.

    A term with a one-dimensional grid gets the Wood (2013) statistic quoted in
    the module docstring, at rank ``nu = sum diag(2F - FF)`` over the term's
    coefficients and against the chi-square mixture of section 2.3 -- the scale
    is modelled here, so the reference is never an F.  An interaction has no
    such grid and is evaluated on the training design instead, which is the
    construction of section 2.4 without the subsample.  A term with levels gets
    the chi-square Wald test of its level block.
    """
    prepared = _prepare_term_test(
        fitted,
        X_train,
        parameter,
        term,
        n_points=n_points,
    )
    matrix = posterior_covariance(fitted, kind=covariance)
    return _term_test_from_covariance(prepared, matrix)


def _prepare_term_test(
    fitted: Any,
    X_train: FrameLike | EagerFrame,
    parameter: str,
    term: str,
    *,
    n_points: int,
) -> _PreparedTermTest:
    """Validate and retain every covariance-independent part of one term test."""

    points = int(n_points)
    if points < 2:
        raise ValueError("n_points must place at least two points on the term's grid")

    qualified, state, term_slice = _resolve_term(fitted, parameter, term)
    frame = as_eager_frame(X_train)
    beta = np.asarray(fitted.coefficients, dtype=np.float64)[term_slice]
    edf = float(fitted.inference.term_edf[qualified])

    spec = _compiled_spec(fitted, parameter, term)
    level_term = spec is not None and _level_domain(spec) is not None
    design: NDArray[np.float64] | None = None
    if not level_term:
        if spec is None:
            evaluation: FrameLike | EagerFrame = frame
        else:
            evaluation, _, _ = _sweep(fitted, frame, parameter, term, points)
        design = _term_design(fitted, evaluation, state, term_slice)

    return _PreparedTermTest(
        qualified=qualified,
        parameter=parameter,
        term=term,
        term_slice=term_slice,
        beta=beta,
        edf=edf,
        level_term=level_term,
        design=design,
        alternative_edf=None if level_term else _alternative_edf(fitted, term_slice, edf),
        reconciliation_tolerance=float(fitted.inference.reconciliation_tolerance),
    )


def _term_test_from_covariance(
    prepared: _PreparedTermTest,
    matrix: NDArray[np.float64],
) -> TermTest:
    """Evaluate one fully prepared term test from a resolved covariance."""
    block = matrix[prepared.term_slice, prepared.term_slice]
    if prepared.level_term:
        statistic, rank, p_value = _block_wald(
            prepared.beta,
            block,
            prepared.reconciliation_tolerance,
        )
    else:
        assert prepared.design is not None
        assert prepared.alternative_edf is not None
        statistic, p_value, rank = wood_test_smooth(
            prepared.beta,
            prepared.design,
            block,
            prepared.alternative_edf,
            _SCALE_MODELLED_RESIDUAL_DF,
        )

    return TermTest(
        parameter=prepared.parameter,
        term=prepared.term,
        statistic=float(statistic),
        rank=float(rank),
        p_value=float(p_value),
        edf=prepared.edf,
    )


def _intercept_row(
    state: Any,
    beta: NDArray[np.float64],
    matrix: NDArray[np.float64],
    intercept_edf: Mapping[str, float],
) -> dict[str, Any]:
    index = state.intercept_index
    estimate = float(beta[index])
    se = math.sqrt(max(float(matrix[index, index]), 0.0))
    statistic = (estimate / se) ** 2 if se > 0.0 else float("nan")
    return {
        "parameter": state.name,
        "term": _INTERCEPT,
        "edf": float(intercept_edf[f"{state.name}:{_INTERCEPT}"]),
        "lambda": float("nan"),
        "statistic": statistic,
        "rank": 1.0,
        "p_value": float(stats.chi2.sf(statistic, 1.0)),
        "estimate": estimate,
        "se": se,
        "note": "",
    }


def _summary_label(term: str) -> str:
    """The row label of a term: a free level block says so instead of its suffix."""
    if term.endswith(_SPECIAL_SUFFIX):
        return f"{term[: -len(_SPECIAL_SUFFIX)]} (special level)"
    return term


def _absorption_note(
    fitted: Any,
    parameter: str,
    term: str,
    edf: float,
    coefficients: NDArray[np.float64],
) -> str:
    """Say when a level term's whole contribution has gone to an interaction.

    A level block exactly collinear with an interaction on the same feature --
    what a factor smooth carrying its own main effect produces -- leaves the
    fit with no estimable direction of its own: its EDF and its covariance
    block are zero, so its statistic is zero and its p-value one, while the
    min-norm solve still writes a non-zero number into its coefficients.  That
    number is not a relativity anyone may price off, so the row says where the
    effect went instead of standing there unexplained.
    """
    if abs(float(edf)) > _ABSORBED_EDF or not np.any(coefficients != 0.0):
        return ""
    spec = _compiled_spec(fitted, parameter, term)
    if spec is None or _level_domain(spec) is None:
        return ""
    interaction = _interaction_on(fitted, parameter, term)
    return "" if interaction is None else f"absorbed by {interaction}"


def summary_table(
    fitted: Any,
    X_train: FrameLike | EagerFrame,
    *,
    covariance: CovarianceKind = "fixed",
) -> pd.DataFrame:
    """One table of every predictor's intercept and terms.

    Rows follow the fit's own layout order: each predictor's intercept, then
    its terms.  ``statistic``, ``rank`` and ``p_value`` come from
    :func:`term_test`; ``estimate`` and ``se`` are filled for an intercept and
    for a term holding a single coefficient, and are NaN for a term that holds
    a block.  ``lambda`` is the term's smoothing parameter when it has exactly
    one penalty component and NaN otherwise -- a term with several carries them
    all in ``term_effect(...).lambdas``.

    A term's free level block reads as ``"<term> (special level)"`` rather than
    under its raw ``:special`` suffix, and ``note`` carries the one reading a
    row cannot state in numbers: ``"absorbed by <interaction>"`` where a level
    term has no estimable direction left because an interaction on the same
    feature spans it, and the empty string everywhere else.
    """
    beta = np.asarray(fitted.coefficients, dtype=np.float64)
    frame = as_eager_frame(X_train)
    prepared_tests: dict[str, _PreparedTermTest] = {}
    has_intercept = False
    for state in fitted.layout.predictors:
        has_intercept = has_intercept or state.intercept_index is not None
        for qualified in fitted.layout.term_slices:
            namespace, _, term = qualified.partition(":")
            if namespace != state.name:
                continue
            prepared = _prepare_term_test(
                fitted,
                frame,
                state.name,
                term,
                n_points=200,
            )
            prepared_tests[prepared.qualified] = prepared

    if not has_intercept and not prepared_tests:
        return pd.DataFrame(columns=list(_SUMMARY_COLUMNS))
    matrix = posterior_covariance(fitted, kind=covariance)

    rows: list[dict[str, Any]] = []
    for state in fitted.layout.predictors:
        if state.intercept_index is not None:
            rows.append(
                _intercept_row(
                    state,
                    beta,
                    matrix,
                    fitted.inference.intercept_edf,
                )
            )
        for qualified, term_slice in fitted.layout.term_slices.items():
            namespace, _, term = qualified.partition(":")
            if namespace != state.name:
                continue
            outcome = _term_test_from_covariance(
                prepared_tests[qualified],
                matrix,
            )
            lambdas = _term_lambdas(fitted, qualified)
            width = term_slice.stop - term_slice.start
            single = width == 1
            rows.append(
                {
                    "parameter": state.name,
                    "term": _summary_label(term),
                    "edf": outcome.edf,
                    "lambda": (
                        float(next(iter(lambdas.values()))) if len(lambdas) == 1 else float("nan")
                    ),
                    "statistic": outcome.statistic,
                    "rank": outcome.rank,
                    "p_value": outcome.p_value,
                    "estimate": float(beta[term_slice][0]) if single else float("nan"),
                    "se": (
                        math.sqrt(max(float(matrix[term_slice, term_slice][0, 0]), 0.0))
                        if single
                        else float("nan")
                    ),
                    "note": _absorption_note(
                        fitted, state.name, term, outcome.edf, beta[term_slice]
                    ),
                }
            )
    return pd.DataFrame(rows, columns=list(_SUMMARY_COLUMNS))


__all__ = [
    "ParameterTermEffect",
    "TermTest",
    "summary_table",
    "term_effect",
    "term_test",
]
