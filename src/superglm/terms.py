"""Declare linear, smooth, categorical and interaction terms for predictors."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from numpy.typing import ArrayLike

from superglm.features import Categorical, LevelGrouping, Numeric, RandomEffect, Spline
from superglm.features.constraint import ConstraintSpec
from superglm.features.factor_smooth import FactorSmooth
from superglm.features.interaction import (
    CategoricalInteraction,
    NumericCategorical,
    NumericInteraction,
    PolynomialCategorical,
    PolynomialInteraction,
    SplineCategorical,
    TensorInteraction,
)
from superglm.features.spline import _SplineBase
from superglm.types import FeatureSpec, LambdaPolicy

type InteractionSpec = (
    CategoricalInteraction
    | NumericCategorical
    | NumericInteraction
    | PolynomialCategorical
    | PolynomialInteraction
    | SplineCategorical
    | TensorInteraction
    | FactorSmooth
)

_INTERACTION_TYPES = (
    CategoricalInteraction,
    NumericCategorical,
    NumericInteraction,
    PolynomialCategorical,
    PolynomialInteraction,
    SplineCategorical,
    TensorInteraction,
    FactorSmooth,
)


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Term and column names must be non-empty strings")


@dataclass(frozen=True)
class BoundTerm:
    """A feature specification attached to a named input column.

    Create these declarations with ``s``, ``cat``, ``re`` or ``term`` and pass
    them to a family predictor helper. A declaration stores a copy of the
    specification. It reads no data and has no fitted coefficients.
    """

    column: str
    _spec: FeatureSpec = field(repr=False)

    def __post_init__(self) -> None:
        _validate_name(self.column)
        if isinstance(self._spec, _INTERACTION_TYPES):
            raise TypeError("Use interaction(spec) to bind an interaction specification")
        if not isinstance(self._spec, FeatureSpec):
            raise TypeError("term() requires a FeatureSpec")
        object.__setattr__(self, "_spec", deepcopy(self._spec))

    @property
    def spec(self) -> FeatureSpec:
        """Return an independent copy of the feature configuration."""
        return deepcopy(self._spec)


@dataclass(frozen=True)
class BoundInteraction:
    """An interaction declaration for use inside a family predictor.

    Create one with ``ti`` or ``interaction``. Its parent terms must also be
    declared in that predictor. ``name`` identifies the interaction in results.
    """

    name: str
    _spec: InteractionSpec = field(repr=False)
    _requires_spline_parents: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if not isinstance(self._spec, _INTERACTION_TYPES):
            raise TypeError("interaction() requires a supported explicit interaction specification")
        for parent in self._spec.parent_names:
            _validate_name(parent)
        object.__setattr__(self, "_spec", deepcopy(self._spec))

    @property
    def spec(self) -> InteractionSpec:
        """Return an independent copy of the interaction configuration."""
        return deepcopy(self._spec)


type TermInput = str | BoundTerm | BoundInteraction


@dataclass(frozen=True)
class NormalizedTerms:
    """Owned compiler inputs in main-effect and interaction declaration order."""

    features: dict[str, FeatureSpec]
    interaction_specs: dict[str, InteractionSpec]
    interaction_order: tuple[str, ...]


def term(column: str, spec: FeatureSpec) -> BoundTerm:
    """Attach an existing feature specification to an input column.

    Use this for specifications without a shorthand, such as ``Polynomial``,
    or to reuse a configured specification. The declaration owns a copy;
    building the model's design happens during fitting.

    Parameters
    ----------
    column : str
        Name of a column in the fit and prediction frames.
    spec : FeatureSpec
        A main-effect specification. Use ``interaction`` for interactions.

    Returns
    -------
    BoundTerm
        A declaration accepted by family predictor helpers.

    Examples
    --------
    >>> from superglm import Numeric, term
    >>> density = term("density", Numeric())
    >>> density.column
    'density'
    """
    return BoundTerm(column, spec)


def s(
    column: str,
    *,
    kind: str = "ps",
    k: int | None = None,
    n_knots: int | None = None,
    degree: int = 3,
    knot_strategy: str = "uniform",
    penalty: str = "ssp",
    select: bool = False,
    knots: ArrayLike | None = None,
    discrete: bool | None = None,
    n_bins: int | None = None,
    extrapolation: str = "clip",
    boundary: tuple[float, float] | None = None,
    knot_alpha: float = 0.2,
    constraint: ConstraintSpec | None = None,
    m: int | tuple[int, ...] = 2,
    lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
) -> BoundTerm:
    """Describe a smooth effect of one numeric column.

    For example, ``s("age", kind="cr", k=10)`` declares a cubic regression
    spline inside a family predictor. The model learns its basis and
    coefficients when fitted. All spline options have the same meaning as
    in ``Spline``.

    Parameters
    ----------
    column : str
        Name of the numeric input column.
    kind : str, default="ps"
        Spline basis. Common choices are ``"ps"`` for P-splines and ``"cr"``
        for cubic regression splines. See ``Spline`` for the other bases.
    k : int, optional
        Public basis size. This limits flexibility; it is not the fitted
        effective degrees of freedom. Supply either ``k`` or ``n_knots``.
    n_knots : int, optional
        Number of interior knots, as an alternative to ``k``.
    degree : int, default=3
        Polynomial degree for bases that support this option. Cubic
        regression splines remain cubic.
    knot_strategy : str, default="uniform"
        Knot-placement rule. ``"uniform"`` spaces knots evenly;
        ``"quantile_rows"`` follows the training data and
        ``"quantile_tempered"`` tempers that concentration.
    penalty : {"ssp", "none"}, default="ssp"
        Enable SSP basis reparametrization, or disable it with ``"none"``.
    select : bool, default=False
        Add a penalty on the spline's null space so that smoothing can also
        shrink directions left unpenalized by the ordinary wiggle penalty.
    knots : array-like, optional
        Explicit interior knot positions, replacing automatic placement.
    discrete : bool, optional
        Request discrete evaluation for this term. ``None`` inherits the
        model setting.
    n_bins : int, optional
        Bin count for discrete evaluation. ``None`` inherits the model
        setting.
    extrapolation : {"clip", "extend", "error"}, default="clip"
        Prediction outside the fitted boundaries: hold the boundary value,
        continue the basis, or raise an error.
    boundary : tuple of float, optional
        Explicit lower and upper spline boundaries. Otherwise use the
        training range.
    knot_alpha : float, default=0.2
        Tempering control for ``"quantile_tempered"`` knot placement.
    constraint : ConstraintSpec, optional
        Requested shape constraint. See the SuperLSS limitation below.
    m : int or tuple of int, default=2
        Penalty order: a difference order for P-splines or a derivative order
        for derivative-penalty bases. Multiple orders require a basis that
        supports multiple penalty components.
    lambda_policy : LambdaPolicy or dict of str to LambdaPolicy, optional
        Control whether smoothing penalties are estimated or fixed. A mapping
        sets policies for the spline's individual penalty components.

    Returns
    -------
    BoundTerm
        A spline declaration for the named column.

    Notes
    -----
    ``SuperLSS`` currently warns and fits unconstrained when a term requests
    shape constraints. A ``constraint`` argument does not enforce them there.

    See Also
    --------
    Spline : Basis, knot, penalty and extrapolation options.
    ti : An interaction between two declared spline terms.

    Examples
    --------
    >>> from superglm import s
    >>> age = s("age", kind="cr", k=10)
    >>> age.column
    'age'
    """
    return term(
        column,
        Spline(
            kind=kind,
            k=k,
            n_knots=n_knots,
            degree=degree,
            knot_strategy=knot_strategy,
            penalty=penalty,
            select=select,
            knots=knots,
            discrete=discrete,
            n_bins=n_bins,
            extrapolation=extrapolation,
            boundary=boundary,
            knot_alpha=knot_alpha,
            constraint=constraint,
            m=m,
            lambda_policy=lambda_policy,
        ),
    )


def cat(
    column: str,
    *,
    base: str = "most_exposed",
    grouping: LevelGrouping | None = None,
    levels: Any = None,
    unseen: Literal["error", "base"] = "error",
) -> BoundTerm:
    """Declare a categorical effect with a reference level.

    Use ``cat("region")`` for categories, including categories stored as
    numbers. A bare string in a predictor always declares a numeric linear
    term and does not infer categorical encoding from the column's dtype.

    Parameters
    ----------
    column : str
        Name of the categorical input column.
    base : str, default="most_exposed"
        Reference level. Use the level with the greatest total sample weight,
        ``"first"`` for the first level, or a specific level name.
    grouping : LevelGrouping, optional
        Combine input levels into groups before encoding.
    levels : sequence, data column or categorical dtype, optional
        Declare the allowed input levels. With ``grouping``, these are the
        original levels before grouping.
    unseen : {"error", "base"}, default="error"
        Prediction policy for levels outside the fitted level universe.
        ``"base"`` uses the reference level and emits a warning.

    Returns
    -------
    BoundTerm
        A categorical declaration for the named column.

    See Also
    --------
    Categorical : Encoding, grouping and level-universe rules.
    re : A penalized effect with a coefficient for every level.
    """
    return term(column, Categorical(base=base, grouping=grouping, levels=levels, unseen=unseen))


def re(
    column: str,
    *,
    levels: Any = None,
    unseen: Literal["population", "error"] = "population",
    missing: Literal["error"] = "error",
    lambda_policy: LambdaPolicy | None = None,
) -> BoundTerm:
    """Declare a random effect with a coefficient for every group level.

    For example, ``re("broker")`` lets broker effects shrink toward the
    population value. Use ``fit_reml`` to estimate the variance component.
    This encoding does not drop a reference level.

    Parameters
    ----------
    column : str
        Name of the grouping column.
    levels : sequence, data column or categorical dtype, optional
        Declare the allowed levels, including levels with no training rows.
    unseen : {"population", "error"}, default="population"
        Prediction policy for unknown levels. ``"population"`` gives the
        random effect a contribution of zero on the predictor's link scale.
    missing : {"error"}, default="error"
        Missing group labels raise an error.
    lambda_policy : LambdaPolicy, optional
        Set the policy for the random effect's smoothing penalty.

    Returns
    -------
    BoundTerm
        A random-effect declaration for the named column.

    See Also
    --------
    RandomEffect : Level handling and variance-component estimation.
    cat : Categorical encoding relative to a reference level.
    """
    return term(
        column,
        RandomEffect(
            levels=levels,
            unseen=unseen,
            missing=missing,
            lambda_policy=lambda_policy,
        ),
    )


def interaction(spec: InteractionSpec, *, name: str | None = None) -> BoundInteraction:
    """Use an existing interaction specification inside a predictor.

    Declare the specification's parent terms in the same predictor. Their
    order within the specification is preserved, even when the predictor's
    declarations appear in a different order.

    Parameters
    ----------
    spec : InteractionSpec
        A supported interaction, such as ``SplineCategorical`` or
        ``FactorSmooth``.
    name : str, optional
        Name used in model results. By default, use the factor smooth's own
        name or join a two-parent interaction's column names with ``":"``.

    Returns
    -------
    BoundInteraction
        An interaction declaration accepted by family predictor helpers.

    See Also
    --------
    ti : Shorthand for a two-spline interaction-only tensor.
    """
    if name is None and isinstance(spec, FactorSmooth):
        name = spec.name
    if name is None:
        parents = getattr(spec, "parent_names", None)
        if not isinstance(parents, tuple) or len(parents) != 2:
            raise TypeError("interaction() requires a supported explicit interaction specification")
        for parent in parents:
            _validate_name(parent)
        name = ":".join(parents)
    return BoundInteraction(name, spec)


def ti(
    left: str,
    right: str,
    *,
    n_knots: tuple[int, int] | None = None,
    decompose: bool = False,
) -> BoundInteraction:
    """Declare a tensor interaction between two smooth effects.

    Both columns must have spline terms in the same predictor. For example,
    use ``s("age"), s("value"), ti("age", "value")`` together. The tensor
    describes their interaction; the two ``s`` terms provide the main effects.

    Parameters
    ----------
    left, right : str
        Names of two columns with declared spline terms.
    n_knots : tuple of int, optional
        Interior-knot counts for the left and right tensor margins. By
        default, inherit the parent terms' counts.
    decompose : bool, default=False
        Separate the bilinear direction from the wiggly interaction so their
        penalties can be controlled separately.

    Returns
    -------
    BoundInteraction
        A declaration named ``"left:right"`` using the supplied column names.

    See Also
    --------
    TensorInteraction : Tensor construction and penalty details.
    interaction : Other supported interaction specifications.
    """
    _validate_name(left)
    _validate_name(right)
    return BoundInteraction(
        f"{left}:{right}",
        TensorInteraction(left, right, n_knots=n_knots, decompose=decompose),
        _requires_spline_parents=True,
    )


def normalize_terms(terms: Sequence[TermInput]) -> NormalizedTerms:
    """Resolve declarations without reading data or constructing row designs."""
    if isinstance(terms, str | bytes) or not isinstance(terms, Sequence):
        raise TypeError("terms must be a sequence of term declarations")
    features: dict[str, FeatureSpec] = {}
    interactions: dict[str, InteractionSpec] = {}
    spline_parent_interactions: set[str] = set()
    # Copy the complete graph once, retaining aliases within the owned graph.
    for declaration in deepcopy(tuple(terms)):
        if isinstance(declaration, str):
            _validate_name(declaration)
            name, spec = declaration, Numeric()
        elif isinstance(declaration, BoundTerm):
            name, spec = declaration.column, declaration._spec
        elif isinstance(declaration, BoundInteraction):
            if declaration.name in interactions:
                raise ValueError(f"Duplicate interaction declaration: {declaration.name!r}")
            interactions[declaration.name] = declaration._spec
            if declaration._requires_spline_parents:
                spline_parent_interactions.add(declaration.name)
            continue
        else:
            raise TypeError("Expected a numeric column string or a bound term declaration")
        if name in features:
            raise ValueError(f"Duplicate main term declaration for column {name!r}")
        features[name] = spec

    for name, ispec in interactions.items():
        if name in features:
            raise ValueError(
                f"Term name {name!r} identifies both a main feature and an interaction"
            )
        for parent in ispec.parent_names:
            if parent not in features:
                raise ValueError(f"Interaction {name!r} requires declared parent {parent!r}")
            if name in spline_parent_interactions and not isinstance(features[parent], _SplineBase):
                raise TypeError(
                    f"ti parent {parent!r} must be a declared spline; use s({parent!r})"
                )
    return NormalizedTerms(features, interactions, tuple(interactions))
