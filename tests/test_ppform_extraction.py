import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.export._ppform import PpformNotExactError, PpformSegments, extract_ppform
from superglm.features import Spline
from superglm.features.constraint import Constraint


def _fit(kind: str, **spline_kwargs):
    rng = np.random.default_rng(3)
    x = rng.uniform(18.0, 80.0, 4000)
    eta = -1.0 + 0.9 * np.sin(x / 14.0) - 0.02 * (x - 45.0) ** 2 / 50.0
    y = rng.poisson(np.exp(eta)).astype(float)
    X = pd.DataFrame({"age": x})
    model = SuperGLM(
        family="poisson",
        features={"age": Spline(kind=kind, n_knots=8, **spline_kwargs)},
    )
    model.fit(X, y)
    return model, X


@pytest.mark.parametrize("kind", ["ps", "bs", "ns", "cr", "cr_cardinal"])
def test_every_spline_kind_converts_to_an_exact_ppform(kind):
    model, _X = _fit(kind)

    segments = extract_ppform(model, "age")

    assert isinstance(segments, PpformSegments)
    assert segments.coefficients.shape == (len(segments.breaks) - 1, 4)
    # The point of the feature: the segments ARE the curve, not an approximation.
    assert segments.residual < 1e-13


@pytest.mark.parametrize("kind", ["ps", "bs", "ns", "cr", "cr_cardinal"])
def test_the_segments_evaluate_to_the_fitted_curve(kind):
    model, _X = _fit(kind)
    ti = model.term_inference("age", n_points=997, with_se=False)

    segments = extract_ppform(model, "age")
    evaluated = segments.evaluate(np.asarray(ti.x, dtype=np.float64))

    # Not "close to" -- the same function.  A normalisation or power-order bug
    # in the coefficient rescaling shows up here and nowhere else.
    assert np.abs(evaluated - np.asarray(ti.log_relativity)).max() < 1e-12


@pytest.mark.parametrize(
    "label, kwargs",
    [
        ("monotone", {"constraint": Constraint.fit.increasing}),
        ("convex", {"constraint": Constraint.fit.convex}),
        ("select", {"select": True}),
        ("degree2", {"degree": 2}),
        ("quantile", {"knot_strategy": "quantile"}),
    ],
)
def test_constrained_and_reparameterised_fits_still_convert(label, kwargs):
    model, _X = _fit("ps", **kwargs)
    ti = model.term_inference("age", n_points=997, with_se=False)

    segments = extract_ppform(model, "age")

    assert np.abs(segments.evaluate(np.asarray(ti.x)) - np.asarray(ti.log_relativity)).max() < 1e-12


def test_the_segments_reproduce_predict_not_just_the_plotted_curve():
    """The workbook has to match what the model SCORES, which is predict().

    term_inference and predict could in principle disagree; asserting only
    against the plotting curve would not notice.
    """
    model, _X = _fit("ps")
    ti = model.term_inference("age", n_points=101, with_se=False)
    lo, hi = ti.spline.boundary
    grid = np.linspace(lo, hi, 997)

    segments = extract_ppform(model, "age")
    log_predict = np.log(model.predict(pd.DataFrame({"age": grid})))
    evaluated = segments.evaluate(grid)

    # The intercept and centering constant live outside the term, so compare
    # the SHAPE: predict and the segments may differ by one constant, and by
    # nothing else.
    shift = np.mean(log_predict - evaluated)
    assert np.abs((log_predict - shift) - evaluated).max() < 1e-12


def test_an_underdetermined_solve_is_refused_rather_than_answered():
    """``lstsq`` ANSWERS a rank-deficient system; it does not refuse one.

    With fewer curve samples than basis columns it returns the minimum-norm
    solution, which reproduces the sampled points exactly -- so ``residual``
    stays at round-off and the exactness check certifies nothing about the
    curve BETWEEN those points.  Measured on this 8-knot ``ps`` fit read at 5
    points: rank 5 of 12 columns, residual 2.2e-16, and the recovered pieces
    mis-rate the fitted curve by up to 2.41x.  A knot interval holding no grid
    point is the same failure arriving by a subtler route, which is why the
    guard is on the rank rather than on ``n_points``.
    """
    model, _X = _fit("ps")

    with pytest.raises(PpformNotExactError, match="'age'") as excinfo:
        extract_ppform(model, "age", n_points=5)

    assert "rank" in str(excinfo.value)
