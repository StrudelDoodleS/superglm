from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Constraint, OrderedCategorical, Spline
from superglm.distributional import GaussianLS, NegativeBinomialLS, Predictor
from superglm.distributional.predictor import ShapeConstraintIgnoredWarning
from tests.bound_predictor_fixtures import model_from_templates


def _data(n: int = 200, seed: int = 4):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = -1.5 * x + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"x": x}), y


@pytest.mark.parametrize("constraint", [Constraint.fit.increasing, Constraint.postfit.increasing])
def test_a_shape_constraint_on_the_distributional_path_warns_and_fits_unconstrained(constraint):
    frame, y = _data()
    spec = Spline(kind="ps", n_knots=6, constraint=constraint)
    with pytest.warns(ShapeConstraintIgnoredWarning, match="location:x"):
        model = model_from_templates(
            family=GaussianLS(),
            predictors=(Predictor("location", {"x": spec}), Predictor("scale", {})),
        ).fit_reml(frame, y, method="efs")
    grid = pd.DataFrame({"x": np.linspace(0.05, 0.95, 20)})
    location = model.predict_parameters(grid)["location"].to_numpy()
    assert location[0] > location[-1], "the decreasing truth is fitted, not the requested increase"


def test_an_unconstrained_spline_does_not_warn():
    frame, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShapeConstraintIgnoredWarning)
        model_from_templates(
            family=GaussianLS(),
            predictors=(
                Predictor("location", {"x": Spline(kind="ps", n_knots=6)}),
                Predictor("scale", {}),
            ),
        ).fit_reml(frame, y, method="efs")


C = Constraint.fit
_BANDS = [f"B{i}" for i in range(6)]


def _lss_fit(frame, y, feature, spec, *, discrete=False):
    model = model_from_templates(
        family=NegativeBinomialLS(),
        predictors=(Predictor("mean", {feature: spec}), Predictor("theta", {})),
        discrete=discrete,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = model.fit_reml(frame, y, method="efs")
    ignored = [w for w in caught if issubclass(w.category, ShapeConstraintIgnoredWarning)]
    return fitted, ignored


def _lss_rows(n: int = 300, seed: int = 4):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 100.0, n)
    band = rng.choice(_BANDS, n)
    region = rng.choice(["A", "B", "C"], n)
    mean = np.exp(0.3 + 0.04 * x + 0.1 * np.array([_BANDS.index(b) for b in band]))
    frame = pd.DataFrame({"x": x, "band": band, "region": region})
    return frame, rng.negative_binomial(3, 3.0 / (3.0 + mean))


def test_lss_warning_names_every_rule():
    frame, y = _lss_rows()
    rules = C.decreasing.below(25) & C.flat.within(25, 40)
    spec = Spline(kind="ps", n_knots=6, constraint=rules)
    _, ignored = _lss_fit(frame, y, "x", spec)
    assert len(ignored) == 1
    assert "mean:x asked for decreasing.below(25) & flat.within(25, 40) (fit)" in str(
        ignored[0].message
    )


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
def test_lss_with_cap_warns_and_fits(discrete):
    """SuperLSS does not enforce a cap: it warns once and fits the curve unconstrained."""
    frame, y = _lss_rows()
    spec = Spline(kind="ps", n_knots=6, constraint=C.increasing & C.cap(0.6, 1.8))
    model, ignored = _lss_fit(frame, y, "x", spec, discrete=discrete)
    assert len(ignored) == 1
    assert "increasing & cap(0.6, 1.8) (fit)" in str(ignored[0].message)
    grid = pd.DataFrame({"x": np.linspace(2.0, 98.0, 25)})
    mean = model.predict_parameters(grid)["mean"].to_numpy()
    # A cap of (0.6, 1.8) holds the fitted mean within a factor of 3.
    assert mean.max() / mean.min() > 3.0


@pytest.mark.parametrize(
    ("feature", "spec", "line"),
    [
        (
            "band",
            OrderedCategorical(order=_BANDS, basis=Spline(kind="ps", k=5, constraint=C.increasing)),
            "increasing (fit)",
        ),
        (
            "band",
            OrderedCategorical(
                order=_BANDS,
                basis=Spline(
                    kind="ps",
                    k=5,
                    constraint=C.increasing & C.cap(0.6, 1.8),
                ),
            ),
            "increasing & cap(0.6, 1.8) (fit)",
        ),
        ("region", Categorical(constraint=C.cap(0.6, 1.8)), "cap(0.6, 1.8) (fit)"),
    ],
    ids=["ordered", "ordered-cap", "categorical-cap"],
)
def test_lss_warns_for_ordered_and_categorical_terms(feature, spec, line):
    """An ordered term hosts its rules on a basis; the warning still names them."""
    frame, y = _lss_rows()
    _, ignored = _lss_fit(frame, y, feature, spec)
    assert len(ignored) == 1
    assert f"mean:{feature} asked for {line}" in str(ignored[0].message)
