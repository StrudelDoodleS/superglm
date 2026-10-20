"""Declare linear, smooth, categorical and interaction terms for predictors."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, cast

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
from superglm.features.ordered_categorical import OrderedCategorical
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
    """An interaction's name: a non-empty string, as results name it by text."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Interaction names must be non-empty strings, got {name!r}")


def check_column_label(label: object) -> None:
    """Refuse a main-term column label that cannot name a column.

    A label is any hashable value, as a ``features=`` key is: a string, or an
    integer, tuple or even ``None`` a pandas frame names its columns by.
    """
    try:
        hash(label)
    except TypeError:
        raise TypeError(
            f"A column label must be hashable, as a dict key is; got {type(label).__name__}"
        ) from None


def check_column_texts(labels: Iterable[Hashable]) -> None:
    """Refuse two column labels that read as the same text, such as ``0`` and ``"0"``.

    Results name a term by its label's text: coefficient groups, smoothing
    parameters, summaries and structure files. Two labels with one text would
    name two terms alike, so both forms refuse them where they are declared.
    """
    seen: dict[str, Hashable] = {}
    for label in labels:
        first = seen.setdefault(str(label), label)
        if first is not label and first != label:
            raise ValueError(
                f"Columns {first!r} and {label!r} both read as {str(label)!r}, which is how "
                "results name a term; rename one of the columns"
            )


@dataclass(frozen=True)
class BoundTerm:
    """A feature specification attached to a named input column.

    Create these declarations with ``s``, ``cat``, ``re`` or ``term`` and pass
    them to a family predictor helper. A declaration stores a copy of the
    specification. It reads no data and has no fitted coefficients.
    """

    column: Hashable
    _spec: FeatureSpec = field(repr=False)

    def __post_init__(self) -> None:
        check_column_label(self.column)
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
    declared in that predictor, except a ``FactorSmooth``'s, which reads its
    two columns itself. ``name`` identifies the interaction in results.
    """

    name: str
    _spec: InteractionSpec = field(repr=False)
    _requires_spline_parents: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if not isinstance(self._spec, _INTERACTION_TYPES):
            raise TypeError("interaction() requires a supported explicit interaction specification")
        for parent in self._spec.parent_names:
            check_column_label(parent)
        object.__setattr__(self, "_spec", deepcopy(self._spec))

    @property
    def spec(self) -> InteractionSpec:
        """Return an independent copy of the interaction configuration."""
        return deepcopy(self._spec)


type TermInput = str | BoundTerm | BoundInteraction | Mapping[Hashable, FeatureSpec]


@dataclass(frozen=True)
class NormalizedTerms:
    """Owned compiler inputs in main-effect and interaction declaration order."""

    features: dict[Hashable, FeatureSpec]
    interaction_specs: dict[str, InteractionSpec]
    interaction_order: tuple[str, ...]


def term(column: Hashable, spec: FeatureSpec) -> BoundTerm:
    """Attach an existing feature specification to an input column.

    Use this for specifications without a shorthand, such as ``Polynomial``,
    or to reuse a configured specification. The declaration owns a copy;
    building the model's design happens during fitting.

    Parameters
    ----------
    column : hashable
        Label of a column in the fit and prediction frames: a string, or any
        other label a ``features=`` key may be, such as an integer.
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
    column: Hashable,
    *,
    kind: Literal["cr", "ps", "bs", "ns", "cr_cardinal"] = "cr",
    k: int | None = None,
    n_knots: int | None = None,
    degree: int = 3,
    knot_strategy: Literal["uniform", "quantile", "quantile_rows", "quantile_tempered"] = "uniform",
    penalty: Literal["ssp", "none"] | None = "ssp",
    select: bool = False,
    knots: ArrayLike | None = None,
    discrete: bool | None = None,
    n_bins: int | None = None,
    extrapolation: Literal["clip", "extend", "error"] = "clip",
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
    column : hashable
        Label of the numeric input column, usually its name.
    kind : {"cr", "ps", "bs", "ns", "cr_cardinal"}, default="cr"
        Spline basis. Common choices are ``"cr"`` for cubic regression splines
        and ``"ps"`` for P-splines. See ``Spline`` for the other bases.
    k : int, optional
        Public basis size. This limits flexibility; it is not the fitted
        effective degrees of freedom. Supply either ``k`` or ``n_knots``.
    n_knots : int, optional
        Number of interior knots, as an alternative to ``k``.
    degree : int, default=3
        Polynomial degree for bases that support this option. Cubic
        regression splines remain cubic.
    knot_strategy : {"uniform", "quantile", "quantile_rows", "quantile_tempered"}, \
default="uniform"
        Knot-placement rule. ``"uniform"`` spaces knots evenly;
        ``"quantile"`` places them at quantiles of the distinct values,
        ``"quantile_rows"`` follows the training data and
        ``"quantile_tempered"`` tempers that concentration.
    penalty : {"ssp", "none"} or None, default="ssp"
        Enable SSP basis reparametrization, or disable it with ``"none"``
        (``None`` also disables it).
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
    column: Hashable,
    *,
    base: Literal["most_exposed", "first"] | str = "most_exposed",
    grouping: LevelGrouping | None = None,
    levels: Any = None,
    unseen: Literal["error", "base"] | str = "error",
) -> BoundTerm:
    """Declare a categorical effect with a reference level.

    Use ``cat("region")`` for categories, including categories stored as
    numbers. A bare string in a predictor always declares a numeric linear
    term and does not infer categorical encoding from the column's dtype.

    Parameters
    ----------
    column : hashable
        Label of the categorical input column, usually its name.
    base : {"most_exposed", "first"} or str, default="most_exposed"
        Reference level. Use the level with the greatest total sample weight,
        ``"first"`` for the first level, or a specific level name.
    grouping : LevelGrouping, optional
        Combine input levels into groups before encoding.
    levels : sequence, data column or categorical dtype, optional
        Declare the allowed input levels. With ``grouping``, these are the
        original levels before grouping.
    unseen : {"error", "base"} or str, default="error"
        Prediction policy for levels outside the fitted level universe.
        ``"base"`` uses the reference level and emits a warning. The label of
        a group of ``grouping`` gives them that group's effect, also with a
        warning.

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
    column: Hashable,
    *,
    levels: Any = None,
    unseen: Literal["population", "error"] = "population",
    missing: Literal["error"] = "error",
    lambda_policy: LambdaPolicy | None = None,
    nested_in: Hashable | None = None,
) -> BoundTerm:
    """Declare a random effect with a coefficient for every group level.

    For example, ``re("broker")`` lets broker effects shrink toward the
    population value. Use ``fit_reml`` to estimate the variance component.
    This encoding does not drop a reference level.

    Parameters
    ----------
    column : hashable
        Label of the grouping column, usually its name.
    levels : sequence, data column or categorical dtype, optional
        Declare the allowed levels, including levels with no training rows.
    unseen : {"population", "error"}, default="population"
        Prediction policy for unknown levels. ``"population"`` gives the
        random effect a contribution of zero on the predictor's link scale.
    missing : {"error"}, default="error"
        Missing group labels raise an error.
    lambda_policy : LambdaPolicy, optional
        Set the policy for the random effect's smoothing penalty.
    nested_in : str, optional
        Name of another random effect in the same predictor that this one is
        nested in, such as a region within a country. Each training row is
        checked against it.

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
            nested_in=nested_in,
        ),
    )


def interaction(spec: InteractionSpec, *, name: str | None = None) -> BoundInteraction:
    """Use an existing interaction specification inside a predictor.

    Declare the specification's parent terms in the same predictor. Their
    order within the specification is preserved, even when the predictor's
    declarations appear in a different order. A ``FactorSmooth`` needs no
    parent terms: it reads its variable and group columns itself, as it does
    under ``interactions=``, so ``interaction(FactorSmooth("age",
    group="region"))`` can stand alone in a predictor.

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
            check_column_label(parent)
        name = ":".join(map(str, parents))
    return BoundInteraction(name, spec)


def ti(
    left: Hashable,
    right: Hashable,
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
    left, right : hashable
        Labels of two columns with declared spline terms.
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
    check_column_label(left)
    check_column_label(right)
    return BoundInteraction(
        f"{left}:{right}",
        # Typed by the interaction classes' str convention; any label works.
        TensorInteraction(cast(str, left), cast(str, right), n_knots=n_knots, decompose=decompose),
        _requires_spline_parents=True,
    )


def _expand_mappings(terms: Sequence[TermInput]) -> tuple[object, ...]:
    """Replace each ``{column: spec}`` mapping by ``term(column, spec)`` in its order.

    ``term`` copies each specification on its own, so one specification object
    placed under several columns gives each column an independent term, as
    ``SuperGLM(features=...)`` does.
    """
    expanded: list[object] = []
    for declaration in terms:
        if isinstance(declaration, Mapping):
            expanded.extend(term(column, spec) for column, spec in declaration.items())
        else:
            expanded.append(declaration)
    return tuple(expanded)


def resolve_label(key: object, labels: Iterable[Hashable]) -> object:
    """The label among ``labels`` that ``key`` names: ``key`` itself, or the one whose text it is.

    Results name a term by its label's text, so a term can be looked up by
    either: ``0`` and ``"0"`` find the term on column ``0``. A model holds no
    two labels with one text, so the text names at most one. A key naming
    nothing comes back as it is, for the caller's own not-found error.
    """
    try:
        if key in labels:
            return key
    except TypeError:
        return key
    text = str(key)
    return next((label for label in labels if str(label) == text), key)


class TermResults(dict):
    """Per-term results keyed by label, which also find a term by its label's text."""

    def __missing__(self, key: object) -> Any:
        label = resolve_label(key, dict.keys(self))
        if label is key:
            raise KeyError(key)
        return dict.__getitem__(self, label)

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, resolve_label(key, dict.keys(self)))

    def get(self, key: object, default: Any = None) -> Any:
        return dict.get(self, resolve_label(key, dict.keys(self)), default)


def declaration_text(column: Hashable, spec: object) -> str:
    """How a main term on ``column`` reads in a terms list, for messages."""
    from superglm.features.categorical import Categorical as _Categorical

    if isinstance(spec, RandomEffect):
        return f"re({column!r})"
    if isinstance(spec, _Categorical) and not isinstance(spec, OrderedCategorical):
        return f"cat({column!r})"
    if isinstance(spec, _SplineBase):
        return f"s({column!r})"
    return f"term({column!r}, {type(spec).__name__}(...))"


def _ti_parent_message(parent: str, spec: FeatureSpec, parents: tuple[str, str]) -> str:
    """Why ``ti`` refuses ``parent``, and what declares the intended interaction."""
    if isinstance(spec, OrderedCategorical) and spec.basis_kind == "spline":
        left, right = parents
        return (
            f"ti takes two s(...) splines; parent {parent!r} is an OrderedCategorical "
            "with a spline basis, so declare its tensor as "
            f"interaction(TensorInteraction({left!r}, {right!r}))"
        )
    if isinstance(spec, OrderedCategorical):
        return (
            f"ti takes two s(...) splines; parent {parent!r} is an OrderedCategorical "
            "without a spline basis, which cannot parent a tensor"
        )
    return f"ti parent {parent!r} must be a declared spline; use s({parent!r})"


def normalize_terms(terms: Sequence[TermInput]) -> NormalizedTerms:
    """Resolve declarations without reading data or constructing row designs.

    A ``{column: spec}`` mapping among the declarations stands for
    ``term(column, spec)`` for each entry, in the mapping's order. Main terms
    keep their declaration order, and interactions keep theirs.
    """
    if isinstance(terms, Mapping):
        raise TypeError(
            "terms must be a sequence of term declarations; put a {column: spec} "
            "mapping inside it, as in [features, 'density']"
        )
    if isinstance(terms, str | bytes) or not isinstance(terms, Sequence):
        raise TypeError("terms must be a sequence of term declarations")
    features: dict[Hashable, FeatureSpec] = {}
    interactions: dict[str, InteractionSpec] = {}
    spline_parent_interactions: set[str] = set()
    # One interaction of a class per unordered parent pair: ti("x", "z") and
    # ti("z", "x") span one column space, as the (left, right) tuple form,
    # which keeps the first pair, already assumes.
    pair_owners: dict[tuple[type, frozenset[str]], str] = {}
    # Copy the complete graph once, retaining aliases within the owned graph.
    for declaration in deepcopy(_expand_mappings(terms)):
        if isinstance(declaration, str):
            name, spec = declaration, Numeric()
        elif isinstance(declaration, BoundTerm):
            name, spec = declaration.column, declaration._spec
        elif isinstance(declaration, BoundInteraction):
            if declaration.name in interactions:
                raise ValueError(f"Duplicate interaction declaration: {declaration.name!r}")
            pair = (type(declaration._spec), frozenset(declaration._spec.parent_names))
            if pair in pair_owners:
                raise ValueError(
                    f"Interactions {pair_owners[pair]!r} and {declaration.name!r} are both "
                    f"{pair[0].__name__} over the same two columns, in either order; "
                    "declare one"
                )
            pair_owners[pair] = declaration.name
            interactions[declaration.name] = declaration._spec
            if declaration._requires_spline_parents:
                spline_parent_interactions.add(declaration.name)
            continue
        elif isinstance(declaration, _INTERACTION_TYPES):
            raise TypeError(
                f"A {type(declaration).__name__} in terms= is declared as interaction(spec)"
            )
        elif isinstance(declaration, FeatureSpec):
            raise TypeError(
                f"A {type(declaration).__name__} in terms= needs its column: write "
                "term(column, spec), or put it in a {column: spec} mapping"
            )
        else:
            raise TypeError(
                "A bare entry in terms= is a numeric column named by a string, got "
                f"{declaration!r}; declare another label as term({declaration!r}, Numeric()) "
                f"or in a mapping, {{{declaration!r}: Numeric()}}"
            )
        if name in features:
            raise ValueError(f"Duplicate main term declaration for column {name!r}")
        features[name] = spec
    check_column_texts(features)

    for name, ispec in interactions.items():
        if name in features:
            raise ValueError(
                f"Term name {name!r} identifies both a main feature and an interaction"
            )
        if isinstance(ispec, FactorSmooth):
            # A factor smooth reads its variable and group columns itself, as
            # under interactions=, so neither needs a main term. SuperGLM's own
            # rules on those terms (a global spline for basis="sz", no main
            # effect on the group) are the model's, checked where it is built.
            continue
        for parent in ispec.parent_names:
            if parent not in features:
                raise ValueError(
                    f"Interaction {name!r} requires declared parent {parent!r}; declare it "
                    f"in the same list, as in s({parent!r}) or cat({parent!r})"
                )
            parent_spec = features[parent]
            if name in spline_parent_interactions and not isinstance(parent_spec, _SplineBase):
                raise TypeError(_ti_parent_message(parent, parent_spec, ispec.parent_names))
    return NormalizedTerms(features, interactions, tuple(interactions))
