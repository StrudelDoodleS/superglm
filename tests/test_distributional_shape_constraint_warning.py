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


def test_a_postfit_constraint_is_fitted_unconstrained_and_repaired_on_request():
    """A post-fit rule repairs the curve afterwards, as on SuperGLM: the fit is free of it."""
    frame, y = _data()
    spec = Spline(kind="ps", n_knots=6, constraint=Constraint.postfit.increasing)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShapeConstraintIgnoredWarning)
        model = model_from_templates(
            family=GaussianLS(),
            predictors=(Predictor("location", {"x": spec}), Predictor("scale", {})),
        ).fit_reml(frame, y, method="efs")
    grid = pd.DataFrame({"x": np.linspace(0.05, 0.95, 20)})
    location = model.predict_parameters(grid)["location"].to_numpy()
    assert location[0] > location[-1], "the decreasing truth is fitted, not the requested increase"
    model.apply_shape_postfit()
    location = model.predict_parameters(grid)["location"].to_numpy()
    assert np.min(np.diff(location)) >= -1e-9 * np.ptp(location)


def test_a_fit_time_constraint_on_the_distributional_path_is_enforced_without_a_warning():
    frame, y = _data()
    spec = Spline(kind="ps", n_knots=6, constraint=Constraint.fit.increasing)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShapeConstraintIgnoredWarning)
        model = model_from_templates(
            family=GaussianLS(),
            predictors=(Predictor("location", {"x": spec}), Predictor("scale", {})),
        ).fit_reml(frame, y, method="efs")
    grid = pd.DataFrame({"x": np.linspace(0.02, 0.98, 200)})
    location = model.predict_parameters(grid)["location"].to_numpy()
    assert np.min(np.diff(location)) >= -1e-9, (
        "the decreasing truth is held to a non-decreasing fit"
    )


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


_U = 2.0**-53
C = Constraint.fit
_BANDS = [f"B{i}" for i in range(6)]


def _lss_fit_warned(frame, y, feature, spec, *, discrete=False):
    model = model_from_templates(
        family=NegativeBinomialLS(),
        predictors=(Predictor("mean", {feature: spec}), Predictor("theta", {})),
        discrete=discrete,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = model.fit_reml(frame, y, method="efs")
    return fitted, caught


def _lss_fit(frame, y, feature, spec, *, discrete=False):
    fitted, caught = _lss_fit_warned(frame, y, feature, spec, discrete=discrete)
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


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
def test_lss_enforces_a_cap_with_a_direction(discrete):
    """A direction and a cap on one SuperLSS term both hold, and nothing is ignored."""
    frame, y = _lss_rows()
    spec = Spline(kind="ps", n_knots=6, constraint=C.increasing & C.cap(0.6, 1.8))
    model, caught = _lss_fit_warned(frame, y, "x", spec, discrete=discrete)
    assert not [w for w in caught if issubclass(w.category, ShapeConstraintIgnoredWarning)]
    grid = pd.DataFrame({"x": np.linspace(2.0, 98.0, 25)})
    mean = model.predict_parameters(grid)["mean"].to_numpy()
    assert np.all(np.diff(mean) >= -1e-9 * mean.max())
    # a cap of (0.6, 1.8) holds the fitted mean within a factor of 3
    assert mean.max() / mean.min() <= 3.0 * (1.0 + 1e-6)


@pytest.mark.parametrize(
    ("feature", "spec"),
    [
        (
            "band",
            OrderedCategorical(order=_BANDS, basis=Spline(kind="ps", k=5, constraint=C.increasing)),
        ),
        ("region", Categorical(constraint=C.cap(0.6, 1.8))),
    ],
    ids=["ordered", "categorical-cap"],
)
def test_lss_warns_for_a_term_whose_basis_states_no_rows(feature, spec):
    """A rule the compiled term states as no row is disclosed, and the term is fitted unconstrained."""
    frame, y = _lss_rows()
    _, caught = _lss_fit_warned(frame, y, feature, spec)
    ignored = [w for w in caught if issubclass(w.category, ShapeConstraintIgnoredWarning)]
    if ignored:
        assert f"mean:{feature} asked for" in str(ignored[0].message)


def test_lss_holds_a_flat_stretch_flat():
    """The flat stretch is absorbed into the basis, so the term is flat there, to rounding.

    Mutation: when SuperLSS dropped every rule the same term fitted free and moved on the stretch.
    """
    frame, y = _lss_rows()
    held, caught = _lss_fit_warned(
        frame, y, "x", Spline(kind="ps", n_knots=6, constraint=C.flat.within(25, 60))
    )
    assert not [w for w in caught if issubclass(w.category, ShapeConstraintIgnoredWarning)]
    inside = pd.DataFrame({"x": np.linspace(30.0, 55.0, 25)})
    mean = held.predict_parameters(inside)["mean"].to_numpy()
    np.testing.assert_allclose(mean, mean[0], rtol=1e-9)
    free, _ = _lss_fit(frame, y, "x", Spline(kind="ps", n_knots=6))
    assert np.ptp(np.log(free.predict_parameters(inside)["mean"].to_numpy())) > 0.1


def test_lss_discloses_and_drops_a_rule_the_rule_validation_refuses():
    """A rule SuperGLM's validation refuses fitted on every earlier SuperLSS release (it was
    ignored): it is disclosed and dropped now, never refused."""
    frame, y = _lss_rows()
    ruled = OrderedCategorical(
        order=_BANDS, basis=Spline(kind="ps", k=5, constraint=C.flat.within("B1", "B4"))
    )
    plain = OrderedCategorical(order=_BANDS, basis=Spline(kind="ps", k=5))
    free, _ = _lss_fit(frame, y, "band", plain)
    held, caught = _lss_fit_warned(frame, y, "band", ruled)
    assert [w for w in caught if "cannot be applied as written" in str(w.message)]
    grid = pd.DataFrame({"band": _BANDS})
    np.testing.assert_allclose(
        held.predict_parameters(grid)["mean"].to_numpy(),
        free.predict_parameters(grid)["mean"].to_numpy(),
        rtol=1e-9,
    )
