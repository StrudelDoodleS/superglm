"""A spline with no kind that ``cr`` cannot serve builds as ``ps``, as on 0.39.

0.40 made a spline with no kind a cubic regression spline (``cr``) and an
ordered term with no basis a ``cr`` smooth. Five declarations that 0.39 fitted
as P-splines were then refused: a degree other than 3, a penalty order above 3,
a column with one value, an ordered term with one observed level, and a
constrained spline beside a ``ps`` constrained term (SCOP and the QP cannot
share a model). Each now builds as the ``ps`` spline 0.39 built, with a
``UserWarning`` naming the choice, and fits bit for bit as the same model
declared with ``kind="ps"``. Every test of the fallback fails on e7b0df17,
where these declarations raise.
"""

from __future__ import annotations

import copy
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Constraint, OrderedCategorical, Spline, SuperGLM
from superglm.features.spline import CubicRegressionSpline, PSpline
from superglm.terms import s

LEVELS = list("ABCDE")
NAMES_NO_KIND = "names no kind"


def _frame(n: int = 400, seed: int = 20261020) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    band = rng.choice(LEVELS, n)
    eta = 0.3 + np.sin(3.0 * x) + 0.8 * z + 0.2 * np.array([LEVELS.index(b) for b in band])
    y = rng.poisson(np.exp(eta)).astype(float)
    X = pd.DataFrame({"x": x, "z": z, "band": band, "const": np.full(n, 2.5)})
    return X, y


def _fit(features: dict, X: pd.DataFrame, y: np.ndarray, method: str, **model) -> SuperGLM:
    glm = SuperGLM(family="poisson", selection_penalty=0.0, features=features, **model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(glm, method)(X, y)
    return glm


def _assert_same_fit(left: SuperGLM, right: SuperGLM, X: pd.DataFrame) -> None:
    """The two fits agree to the bit: coefficients, intercept and predictions."""
    np.testing.assert_array_equal(left.result.beta, right.result.beta)
    assert float(left.result.intercept) == float(right.result.intercept)
    np.testing.assert_array_equal(left.predict(X), right.predict(X))


def _warned(record, *needles: str) -> list[str]:
    messages = [str(w.message) for w in record if issubclass(w.category, UserWarning)]
    return [m for m in messages if all(needle in m for needle in needles)]


# -- Construction: a degree or penalty order cr cannot take -----------------------


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        ({"degree": 1}, "is always cubic, so it cannot take degree=1"),
        ({"degree": 2}, "is always cubic, so it cannot take degree=2"),
        ({"m": 4}, "takes penalty orders up to 3, so it cannot take m=4"),
        ({"m": (2, 4)}, "takes penalty orders up to 3, so it cannot take m=4"),
    ],
    ids=["degree1", "degree2", "m4", "m24"],
)
def test_a_degree_or_order_cr_cannot_take_builds_the_ps_spline(arguments, reason) -> None:
    with pytest.warns(UserWarning, match=NAMES_NO_KIND) as record:
        spec = Spline(n_knots=8, **arguments)
    message = _warned(record, NAMES_NO_KIND)[0]
    assert reason in message
    assert "Pass kind=" in message and "kind='ps' keeps this fit" in message
    assert isinstance(spec, PSpline)
    explicit = Spline(kind="ps", n_knots=8, **arguments)
    assert spec.degree == explicit.degree and spec._m_orders == explicit._m_orders


@pytest.mark.parametrize("method", ["fit", "fit_reml"])
def test_a_degree_two_spline_fits_as_the_explicit_ps(method) -> None:
    X, y = _frame()
    with pytest.warns(UserWarning, match=NAMES_NO_KIND):
        kindless = Spline(n_knots=8, degree=2)
    left = _fit({"x": kindless, "z": Spline(kind="cr", k=6)}, X, y, method)
    right = _fit(
        {"x": Spline(kind="ps", n_knots=8, degree=2), "z": Spline(kind="cr", k=6)}, X, y, method
    )
    _assert_same_fit(left, right, X)


def test_s_takes_the_same_rule() -> None:
    with pytest.warns(UserWarning, match=NAMES_NO_KIND):
        assert isinstance(s("x", n_knots=8, m=4).spec, PSpline)


def test_a_named_kind_is_never_replaced() -> None:
    """``kind="cr"`` keeps its refusal: the rule is for a spline that named no kind."""
    with pytest.raises(ValueError, match="always cubic"):
        Spline(kind="cr", n_knots=8, degree=2)
    with pytest.raises(ValueError, match="penalty orders up to 3"):
        Spline(kind="cr", m=4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert isinstance(Spline(n_knots=8), CubicRegressionSpline)


# -- Build: a column with one value -----------------------------------------------


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "binned"])
@pytest.mark.parametrize("method", ["fit", "fit_reml"])
def test_a_one_valued_column_builds_the_ps_spline(method, discrete) -> None:
    X, y = _frame()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        glm = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=discrete,
            features={"x": Spline(kind="ps", n_knots=8), "const": Spline(n_knots=5)},
        )
        getattr(glm, method)(X, y)
    message = _warned(record, "Feature 'const'", NAMES_NO_KIND)
    assert message and "every value is 2.5" in message[0]
    assert isinstance(glm._specs["const"], PSpline)

    explicit = _fit(
        {"x": Spline(kind="ps", n_knots=8), "const": Spline(kind="ps", n_knots=5)},
        X,
        y,
        method,
        discrete=discrete,
    )
    _assert_same_fit(glm, explicit, X)


def test_a_one_valued_column_still_refuses_a_named_cr() -> None:
    X, y = _frame()
    glm = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"x": Spline(kind="ps", n_knots=8), "const": Spline(kind="cr", n_knots=5)},
    )
    with pytest.raises(ValueError, match="no range to place its knots on"):
        glm.fit(X, y)


def test_a_spec_without_the_record_keeps_the_040_refusal() -> None:
    """A kindless spec pickled by 0.40 carries no record of its declaration, so it
    stays the cr spline 0.40 fitted and predicted with, refusal included."""
    X, y = _frame()
    spec = Spline(n_knots=5)
    del spec._kindless_args
    glm = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"x": Spline(kind="ps", n_knots=8), "const": spec},
    )
    with pytest.raises(ValueError, match="no range to place its knots on"):
        glm.fit(X, y)


def test_the_declaration_survives_copy_pickle_and_a_fresh_fit() -> None:
    """Each fit resolves the declaration again: a model fitted on a one-valued
    column refits as cr once the column varies."""
    X, y = _frame()
    spec = Spline(n_knots=5)
    for twin in (copy.deepcopy(spec), pickle.loads(pickle.dumps(spec))):
        assert isinstance(twin, CubicRegressionSpline)
        assert twin._kindless_args == spec._kindless_args

    glm = _fit({"x": Spline(kind="ps", n_knots=8), "const": spec}, X, y, "fit")
    assert isinstance(glm._specs["const"], PSpline)
    restored = pickle.loads(pickle.dumps(glm))
    np.testing.assert_array_equal(restored.predict(X), glm.predict(X))

    varying = X.assign(const=X["z"])
    glm.fit(varying, y)
    assert isinstance(glm._specs["const"], CubicRegressionSpline)
    assert isinstance(glm.clone_unfitted()._specs["const"], CubicRegressionSpline)


def test_sklearn_clone_keeps_the_declaration() -> None:
    from sklearn.base import clone

    from superglm import SuperGLMRegressor

    X, y = _frame()
    regressor = SuperGLMRegressor(
        family="poisson",
        selection_penalty=0.0,
        features={"x": Spline(kind="ps", n_knots=8), "const": Spline(n_knots=5)},
    )
    twin = clone(regressor)
    assert twin.get_params()["features"]["const"]._kindless_args is not None
    with pytest.warns(UserWarning, match=NAMES_NO_KIND):
        twin.fit(X, y)


# -- Build: an ordered term with one observed level -------------------------------


@pytest.mark.parametrize("method", ["fit", "fit_reml"])
def test_an_ordered_term_with_one_observed_level_builds_the_ps_default(method) -> None:
    X, y = _frame()
    X = X.assign(band="C")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        glm = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"band": OrderedCategorical(order=LEVELS), "x": Spline(kind="cr", k=8)},
        )
        getattr(glm, method)(X, y)
    message = _warned(record, "Feature 'band'", "one level")
    assert message and "basis=Spline(kind='ps', n_knots=5) keeps this fit" in message[0]
    assert isinstance(glm._specs["band"]._spline, PSpline)

    explicit = _fit(
        {
            "band": OrderedCategorical(order=LEVELS, basis=Spline(kind="ps", n_knots=5)),
            "x": Spline(kind="cr", k=8),
        },
        X,
        y,
        method,
    )
    _assert_same_fit(glm, explicit, X)


def test_an_ordered_term_with_a_named_cr_basis_keeps_its_refusal() -> None:
    X, y = _frame()
    X = X.assign(band="C")
    glm = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(order=LEVELS, basis=Spline(kind="cr", n_knots=4)),
            "x": Spline(kind="cr", k=8),
        },
    )
    with pytest.raises(ValueError, match="one level"):
        glm.fit(X, y)


# -- Model: a constrained spline beside a ps (SCOP) constrained term ---------------


@pytest.mark.parametrize("method", ["fit", "fit_reml"])
def test_a_constrained_spline_beside_a_scop_term_builds_the_ps_spline(method) -> None:
    X, y = _frame()
    increasing = Constraint.fit.increasing
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        glm = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "z": Spline(constraint=increasing),
                "x": Spline(kind="ps", k=8, constraint=increasing),
            },
        )
        getattr(glm, method)(X, y)
    message = _warned(record, "Feature 'z'", NAMES_NO_KIND)
    assert message and "'x' puts its constraint on SCOP" in message[0]
    assert isinstance(glm._specs["z"], PSpline)

    explicit = _fit(
        {
            "z": Spline(kind="ps", constraint=increasing),
            "x": Spline(kind="ps", k=8, constraint=increasing),
        },
        X,
        y,
        method,
    )
    _assert_same_fit(glm, explicit, X)


def test_a_constrained_spline_beside_an_ordered_scop_term_builds_the_ps_spline() -> None:
    X, y = _frame()
    increasing = Constraint.fit.increasing
    ordered = OrderedCategorical(order=LEVELS, basis=Spline(kind="ps", k=5, constraint=increasing))
    with pytest.warns(UserWarning, match=NAMES_NO_KIND):
        glm = _fit_warned({"z": Spline(constraint=increasing), "band": ordered}, X, y)
    explicit = _fit(
        {"z": Spline(kind="ps", constraint=increasing), "band": copy.deepcopy(ordered)},
        X,
        y,
        "fit",
    )
    _assert_same_fit(glm, explicit, X)


def _fit_warned(features: dict, X: pd.DataFrame, y: np.ndarray) -> SuperGLM:
    glm = SuperGLM(family="poisson", selection_penalty=0.0, features=features)
    glm.fit(X, y)
    return glm


def test_named_kinds_keep_the_scop_qp_refusal_and_qp_alone_stays_cr() -> None:
    X, y = _frame()
    increasing = Constraint.fit.increasing
    mixed = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "z": Spline(kind="cr", constraint=increasing),
            "x": Spline(kind="ps", k=8, constraint=increasing),
        },
    )
    with pytest.raises(NotImplementedError, match=r"SCOP \+ QP"):
        mixed.fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        alone = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "z": Spline(constraint=increasing),
                "x": Spline(kind="bs", k=8, constraint=increasing),
            },
        )
        alone.fit(X, y)
    assert isinstance(alone._specs["z"], CubicRegressionSpline)
