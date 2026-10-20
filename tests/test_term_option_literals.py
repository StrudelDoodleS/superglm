"""The ``Literal`` types on enumerated term options match what the runtime takes.

An IDE offers a ``Literal``'s values and a type checker catches a typo, so a
``Literal`` that drifts from the runtime either hides a value the runtime takes
or offers one it refuses. Each case compares a public signature's ``Literal``
with the runtime's own set: the constant its validation reads.
"""

from __future__ import annotations

import inspect
import types
import typing

import pytest

import superglm
import superglm.sklearn as sklearn_wrappers
from superglm.distributional.binding import bind_predictor
from superglm.distributional.families import _predictors
from superglm.editor.knots import STRATEGIES as EDITOR_KNOT_STRATEGIES
from superglm.features import (
    _spline_factory,
    categorical,
    factor_smooth,
    piecewise,
    random_effect,
    spline,
)
from superglm.features._spline_config import EXTRAPOLATIONS, KNOT_STRATEGIES, PENALTIES
from superglm.features._spline_ranges import JOINS
from superglm.features.constraint import CONSTRAINT_KINDS, CONSTRAINT_MODES, ConstraintSpec
from superglm.features.rebuild import SYMBOLIC_BASE_POLICIES
from superglm.links import _LINK_SHORTCUTS, Link
from superglm.types import LAMBDA_POLICY_MODES

_SPLINE_KINDS = (
    spline.PSpline,
    spline.BSplineSmooth,
    spline.NaturalSpline,
    spline.CubicRegressionSpline,
    spline.CardinalCRSpline,
)
_FAMILY_HELPERS = tuple(
    method
    for _, cls in inspect.getmembers(_predictors, inspect.isclass)
    if cls.__module__ == _predictors.__name__
    for name, method in vars(cls).items()
    if inspect.isfunction(method) and not name.startswith("_")
)

_KNOT_STRATEGIES = set(KNOT_STRATEGIES)

# (callables, option, the runtime's set, whether other strings are also taken)
_CASES = [
    ((superglm.s, superglm.Spline), "kind", set(_spline_factory._VALID_SPLINE_KINDS), False),
    ((superglm.s, superglm.Spline, *_SPLINE_KINDS), "knot_strategy", _KNOT_STRATEGIES, False),
    ((superglm.s, superglm.Spline, *_SPLINE_KINDS), "penalty", set(PENALTIES), False),
    ((superglm.s, superglm.Spline, *_SPLINE_KINDS), "extrapolation", set(EXTRAPOLATIONS), False),
    ((superglm.cat, superglm.Categorical), "base", set(categorical._BASE_POLICIES), True),
    ((superglm.cat, superglm.Categorical), "unseen", set(categorical._UNSEEN_POLICIES), True),
    ((superglm.re, superglm.RandomEffect), "unseen", set(random_effect.UNSEEN_POLICIES), False),
    ((superglm.re, superglm.RandomEffect), "missing", set(random_effect.MISSING_POLICIES), False),
    ((superglm.FactorSmooth,), "basis", set(factor_smooth.BASES), False),
    ((superglm.FactorSmooth,), "kind", set(factor_smooth.KINDS), False),
    ((superglm.FactorSmooth,), "unseen", set(factor_smooth.UNSEEN_POLICIES), False),
    ((superglm.FactorSmooth,), "missing", set(factor_smooth.MISSING_POLICIES), False),
    ((*_FAMILY_HELPERS, bind_predictor), "link", set(_LINK_SHORTCUTS), False),
    ((superglm.OrderedCategorical,), "base", set(categorical._BASE_POLICIES), True),
    ((superglm.Piecewise,), "base", set(categorical._BASE_POLICIES), True),
    ((superglm.Piecewise,), "strategy", set(piecewise._STRATEGIES), False),
    ((superglm.Piecewise,), "extrapolation", set(EXTRAPOLATIONS), False),
    ((superglm.PolynomialRange,), "join", set(JOINS), False),
    ((ConstraintSpec,), "mode", set(CONSTRAINT_MODES), False),
    ((ConstraintSpec,), "kind", set(CONSTRAINT_KINDS), False),
    ((superglm.LambdaPolicy,), "mode", set(LAMBDA_POLICY_MODES), False),
    (
        (sklearn_wrappers.SuperGLMRegressor, sklearn_wrappers.SuperGLMClassifier),
        "smoothing",
        set(sklearn_wrappers._SMOOTHING_METHODS),
        False,
    ),
]


def _annotation(target, option: str):
    """The option's annotation, evaluated alone: the modules postpone annotations."""
    function = target.__init__ if inspect.isclass(target) else target
    annotation = inspect.signature(function).parameters[option].annotation
    if not isinstance(annotation, str):
        return annotation
    return eval(annotation, {"Literal": typing.Literal, "Link": Link})


def _literal_values(annotation) -> tuple[set[str], bool]:
    """The ``Literal`` strings in ``annotation``, and whether it also takes other strings."""
    members = (
        typing.get_args(annotation)
        if typing.get_origin(annotation) in (typing.Union, types.UnionType)
        else (annotation,)
    )
    values: set[str] = set()
    for member in members:
        if typing.get_origin(member) is typing.Literal:
            values.update(typing.get_args(member))
    return values, str in members


@pytest.mark.parametrize(
    ("targets", "option", "accepted", "open_set"),
    _CASES,
    ids=[f"{targets[0].__name__}-{option}" for targets, option, _, _ in _CASES],
)
def test_option_literal_matches_the_values_the_runtime_takes(targets, option, accepted, open_set):
    for target in targets:
        values, other_strings = _literal_values(_annotation(target, option))
        assert values == accepted, f"{target.__qualname__}({option}=...)"
        # An option that also takes a level or group label keeps str beside the Literal.
        assert other_strings is open_set, f"{target.__qualname__}({option}=...)"


def test_every_family_helper_is_checked():
    assert len(_FAMILY_HELPERS) == 9


def test_knot_strategy_literal_matches_the_editor_list():
    """The editor validates a chosen strategy against its own list; the two agree."""
    assert _KNOT_STRATEGIES == set(EDITOR_KNOT_STRATEGIES)


def test_base_literal_matches_the_policies_a_rebuild_reads():
    assert set(categorical._BASE_POLICIES) == set(SYMBOLIC_BASE_POLICIES)


@pytest.mark.parametrize("option", ["kind", "knot_strategy", "penalty", "extrapolation"])
def test_every_spline_literal_value_builds(option):
    import numpy as np

    x = np.linspace(0.0, 1.0, 200)
    values, _ = _literal_values(_annotation(superglm.Spline, option))
    for value in sorted(values):
        superglm.Spline(**{option: value}, k=8).build(x)


def test_every_factor_smooth_and_link_literal_value_is_taken():
    for basis in ("fs", "sz"):
        superglm.FactorSmooth("x", group="g", basis=basis, kind="ps")
    family = superglm.Poisson()
    for name in _literal_values(_annotation(bind_predictor, "link"))[0]:
        assert isinstance(superglm.links.resolve_link(name, family), superglm.links.Link)


@pytest.mark.parametrize(
    ("option", "value", "treated_as"),
    [
        ("knot_strategy", "quantile_row", "uniform knots"),
        ("knot_strategy", "Uniform", "uniform knots"),
        ("knot_strategy", "explicit", "uniform knots"),
        ("penalty", "SSP", "no SSP reparametrization"),
        ("penalty", "off", "no SSP reparametrization"),
        ("penalty", "", "no SSP reparametrization"),
    ],
)
def test_an_unknown_spline_option_builds_as_before_and_says_so(option, value, treated_as):
    """0.40 built these silently; they build the same and warn (master warned nothing)."""
    import numpy as np

    from superglm.features._spline_config import SplineOptionWarning

    values, _ = _literal_values(_annotation(superglm.Spline, option))
    x = np.linspace(0.0, 1.0, 200)
    for build in (superglm.Spline, lambda **kw: superglm.s("x", **kw).spec, spline.PSpline):
        with pytest.warns(SplineOptionWarning, match=rf"{option}={value!r}") as caught:
            spec = build(**{option: value})
        message = str(caught[0].message)
        assert treated_as in message
        assert all(repr(accepted) in message for accepted in values)
    spec.build(x)


def test_what_040_fitted_as_intended_still_builds():
    """penalty=None disabled SSP as "none" does, and stated knots never read the rule."""
    import numpy as np

    x = np.linspace(0.0, 1.0, 200)
    superglm.Spline(penalty=None, k=8).build(x)
    superglm.Spline(knots=[0.25, 0.5, 0.75], knot_strategy="explicit").build(x)
