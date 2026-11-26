"""One fitted row's law on SplitSplice: the density's two pieces, the mean excess and the curve.

The splice's parameters are pinned to known values (as ``test_split_splice.py`` pins them),
and every number is assembled here from ``scipy.stats`` or integrated by quadrature.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

import superglm.plotting.distributional as mpl_renderers
from superglm import GammaLS, GeneralizedParetoLSS, LogNormalLS, SplitSplice, SuperGLM, SuperLSS
from superglm._experimental import ExperimentalWarning
from superglm.distributional.row_law import DistributionCurve

U = 2.0**-53
THRESHOLD = 100.0


@pytest.fixture(autouse=True)
def _label_already_shown(monkeypatch):
    monkeypatch.setattr("superglm._experimental._announced", True)


def _components(family, semantics="prior"):
    tail = GeneralizedParetoLSS()
    return (
        SuperLSS(family, family.mean(), family.scale(), weight_semantics=semantics),
        SuperGLM(family="binomial", features={}, weight_semantics=semantics),
        SuperLSS(tail, tail.scale(), tail.shape(), weight_semantics=semantics),
    )


def _sample(n, rng):
    large = rng.random(n) < 0.25
    y = stats.gamma.ppf(rng.random(n) * stats.gamma.cdf(100, 3, scale=20), 3, scale=20)
    y[large] = 100 + stats.genpareto.rvs(0.25, scale=35, size=large.sum(), random_state=rng)
    return pd.DataFrame({"x": np.linspace(-1, 1, n)}), y


def _fit(family=None, sample_weight=None, semantics="prior", n=3000):
    X, y = _sample(n, np.random.default_rng(570))
    body, exceedance, tail = _components(GammaLS() if family is None else family, semantics)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        splice = SplitSplice.fit(
            X,
            y,
            threshold=THRESHOLD,
            body=body,
            exceedance=exceedance,
            tail=tail,
            sample_weight=sample_weight,
        )
    return splice, X


@pytest.fixture(scope="module")
def fitted():
    return _fit()


#: A body whose F(u) is well below one, a tail and an exceedance probability other than 1/2.
BODY = (150.0, 0.6)
TAIL = (35.0, 0.25)
Q = 0.3


def _pin(monkeypatch, splice, body=BODY, tail=TAIL, q=Q):
    """Keep the verified fit, and give every row known component parameters."""
    monkeypatch.setattr(
        splice.body,
        "predict_parameters",
        lambda X, **kw: pd.DataFrame(np.broadcast_to(body, (len(X), 2)), columns=["mean", "scale"]),
    )
    monkeypatch.setattr(
        splice.tail,
        "predict_parameters",
        lambda X, **kw: pd.DataFrame(
            np.broadcast_to(tail, (len(X), 2)), columns=["scale", "shape"]
        ),
    )
    monkeypatch.setattr(splice.exceedance, "predict", lambda X, **kw: np.broadcast_to(q, (len(X),)))


def _gamma(mean, cv):
    return stats.gamma(1 / cv**2, scale=mean * cv**2)


def _expected_density(y, body=BODY, tail=TAIL, q=Q, u=THRESHOLD, law=_gamma):
    b, g = law(*body), stats.genpareto(tail[1], scale=tail[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        below = (1 - q) * b.pdf(y) / b.cdf(u)
        above = q * g.pdf(y - u)
    return np.where(y <= 0, 0.0, np.where(y <= u, below, np.where(np.isfinite(y), above, 0.0)))


# --------------------------------------------------------------------------- #
# A1. The density
# --------------------------------------------------------------------------- #


def test_the_density_is_the_body_below_the_threshold_and_the_tail_above(monkeypatch, fitted):
    splice, _ = fitted
    _pin(monkeypatch, splice)
    assert _gamma(*BODY).cdf(THRESHOLD) < 0.5
    u = THRESHOLD
    y = np.array(
        [-1.0, 0.0, 1e-3, 40.0, np.nextafter(u, 0), u, np.nextafter(u, np.inf), 180, 1e4, np.inf]
    )
    X = pd.DataFrame({"x": np.zeros(len(y))})
    np.testing.assert_allclose(splice.predict_pdf(X, y), _expected_density(y), rtol=1e-12, atol=0)
    # a loss equal to the threshold is a body loss: the left limit
    left = (1 - Q) * _gamma(*BODY).pdf(u) / _gamma(*BODY).cdf(u)
    assert splice.predict_pdf(X.iloc[:1], u)[0] == pytest.approx(left, rel=1e-12)


def test_the_two_pieces_integrate_to_their_own_masses(monkeypatch, fitted):
    splice, _ = fitted
    _pin(monkeypatch, splice)
    row = pd.DataFrame({"x": [0.0]})
    density = lambda t: splice.predict_pdf(row, t)[0]  # noqa: E731
    body, _ = integrate.quad(density, 0, THRESHOLD, epsabs=1e-14, limit=200)
    tail, _ = integrate.quad(density, THRESHOLD, np.inf, epsabs=1e-14, limit=200)
    assert body == pytest.approx(1 - Q, abs=1e-10)
    assert tail == pytest.approx(Q, abs=1e-10)


def test_a_lognormal_body_is_its_own_truncated_density(monkeypatch):
    splice, _ = _fit(LogNormalLS())
    body = (80.0, 0.7)
    _pin(monkeypatch, splice, body=body)

    def law(mean, sigma):
        return stats.lognorm(sigma, scale=mean * np.exp(-(sigma**2) / 2))

    y = np.array([5.0, 60.0, 100.0, 140.0])
    X = pd.DataFrame({"x": np.zeros(len(y))})
    np.testing.assert_allclose(
        splice.predict_pdf(X, y), _expected_density(y, body=body, law=law), rtol=1e-12
    )


@pytest.mark.parametrize("family", [GammaLS(), LogNormalLS()], ids=["gamma", "lognormal"])
def test_a_frequency_weighted_splice_reports_the_unit_law_of_each_row(family):
    """Both body families: the density at the fitted components is the unit-weight law's."""
    n = 2000
    weights = np.random.default_rng(3).integers(1, 5, size=n).astype(float)
    splice, X = _fit(family, sample_weight=weights, semantics="frequency", n=n)
    rows = X.iloc[[0, 500, 1999, 7, 8]]
    y = np.array([20.0, 70.0, 100.0, 150.0, 400.0])
    body = splice.body.predict_parameters(rows).to_numpy()
    tail = splice.tail.predict_parameters(rows).to_numpy()
    q = splice.exceedance.predict(rows)
    expected = []
    for t, b, g, p in zip(y, body, tail, q, strict=True):
        if isinstance(family, GammaLS):
            law = _gamma(*b)
        else:
            law = stats.lognorm(b[1], scale=b[0] * np.exp(-(b[1] ** 2) / 2))
        expected.append(
            _expected_density(np.array([t]), body=b, tail=g, q=p, law=lambda *a: law)[0]
        )
    np.testing.assert_allclose(splice.predict_pdf(rows, y), expected, rtol=1e-11)


def test_a_missing_amount_is_refused_and_infinity_reads_zero(fitted):
    splice, X = fitted
    with pytest.raises(ValueError, match=r"predict_pdf needs an amount for every row; rows \[1\]"):
        splice.predict_pdf(X.iloc[:2], np.array([1.0, np.nan]))
    assert splice.predict_pdf(X.iloc[:2], np.array([np.inf, -np.inf])).tolist() == [0.0, 0.0]


# --------------------------------------------------------------------------- #
# A2. The mean excess
# --------------------------------------------------------------------------- #


def test_above_the_threshold_the_mean_excess_is_the_tails_line(monkeypatch, fitted):
    splice, _ = fitted
    _pin(monkeypatch, splice)
    v = np.array([THRESHOLD, 130.0, 1e4])
    X = pd.DataFrame({"x": np.zeros(len(v))})
    scale, shape = TAIL
    expected = (scale + shape * (v - THRESHOLD)) / (1 - shape)
    np.testing.assert_allclose(splice.predict_mean_excess(X, v), expected, rtol=16 * U)


@pytest.mark.parametrize("v", [5.0, 60.0, 99.0, 100.0, 140.0])
def test_the_mean_excess_is_the_integrated_survival_on_both_sides(monkeypatch, fitted, v):
    splice, _ = fitted
    _pin(monkeypatch, splice)
    row = pd.DataFrame({"x": [0.0]})
    survival = lambda t: splice.predict_survival(row, t)[0]  # noqa: E731
    start = max(v, THRESHOLD)
    area = integrate.quad(survival, start, np.inf, epsabs=0, epsrel=1e-13, limit=200)[0]
    if v < THRESHOLD:
        area += integrate.quad(survival, v, THRESHOLD, epsabs=0, epsrel=1e-13, limit=200)[0]
    expected = area / survival(v)
    assert splice.predict_mean_excess(row, v)[0] == pytest.approx(expected, rel=1e-10)


def test_the_mean_excess_matches_a_layer_whose_residual_is_below_rounding(monkeypatch, fitted):
    """The layer omits ``q S_g(t) (scale + shape t) / (1 - shape)`` beyond ``t = v + L - u``."""
    splice, _ = fitted
    _pin(monkeypatch, splice)
    scale, shape = TAIL
    v = np.array([0.0, 40.0, 100.0, 150.0])
    X = pd.DataFrame({"x": np.zeros(len(v))})
    limit = 1e9
    layer = splice.predict_layer(X, v, limit)
    beyond = v + limit - THRESHOLD
    omitted = (
        Q * stats.genpareto(shape, scale=scale).sf(beyond) * (scale + shape * beyond) / (1 - shape)
    )
    assert np.all(omitted < 1e-14 * layer)
    expected = layer / splice.predict_survival(X, v)
    np.testing.assert_allclose(splice.predict_mean_excess(X, v), expected, rtol=1e-12)
    assert splice.predict_mean_excess(X[:1], 0.0)[0] == pytest.approx(
        splice.predict(X[:1])[0], rel=16 * U
    )


def test_below_zero_the_mean_excess_is_the_mean_less_the_amount(fitted):
    splice, X = fitted
    rows = X.iloc[[0, 1500]]
    v = np.array([-10.0, -1e3])
    np.testing.assert_allclose(
        splice.predict_mean_excess(rows, v), splice.predict(rows) - v, rtol=16 * U
    )


def test_the_mean_excess_refuses_a_missing_or_infinite_amount(fitted):
    splice, X = fitted
    with pytest.raises(ValueError, match=r"finite amount for every row; rows \[0, 2\] are missing"):
        splice.predict_mean_excess(X.iloc[:3], np.array([np.nan, 1.0, -np.inf]))


# --------------------------------------------------------------------------- #
# A3. The curve
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["density", "survival", "mean_excess"])
def test_the_curve_holds_the_threshold_twice_with_its_two_limits(monkeypatch, fitted, kind):
    splice, _ = fitted
    _pin(monkeypatch, splice)
    row = pd.DataFrame({"x": [0.0]})
    curve = splice.distribution_curve(row, kind=kind, n_points=40)
    u = THRESHOLD
    pieces = np.array(curve.piece)
    assert (pieces == "body").sum() == 20 and (pieces == "tail").sum() == 20
    assert curve.y[19] == u == curve.y[20] and np.all(np.diff(curve.y) >= 0)
    assert np.all(curve.y[pieces == "body"] <= u) and np.all(curve.y[pieces == "tail"] >= u)
    rows = pd.DataFrame({"x": np.zeros(len(curve.y))})
    tail_g = stats.genpareto(TAIL[1], scale=TAIL[0])
    if kind == "density":
        expected = splice.predict_pdf(rows, curve.y)
        expected[20] = Q * tail_g.pdf(0.0)
        assert curve.values[19] == splice.predict_pdf(row, u)[0]
    elif kind == "survival":
        expected = splice.predict_survival(rows, curve.y)
    else:
        expected = splice.predict_mean_excess(rows, curve.y)
    np.testing.assert_allclose(curve.values, expected, rtol=1e-15)
    if kind != "density":
        assert curve.values[19] == curve.values[20]
    assert curve.threshold == u and curve.mean == splice.predict(row)[0]
    assert curve.family == "SplitSplice(GammaLS, GeneralizedParetoLSS)"
    assert dict(curve.parameters) == {
        "body_mean": BODY[0],
        "body_scale": BODY[1],
        "exceedance": Q,
        "tail_scale": TAIL[0],
        "tail_shape": TAIL[1],
    }
    assert curve.amount_scale == ("log" if kind == "survival" else "linear")


def test_a_rare_tail_still_gets_its_own_stretch_of_the_default_grid(monkeypatch, fitted):
    """R8: at q = 1e-4 the tail is drawn from the threshold to its own 0.999 quantile."""
    splice, _ = fitted
    _pin(monkeypatch, splice, q=1e-4)
    row = pd.DataFrame({"x": [0.0]})
    curve = splice.distribution_curve(row, n_points=20)
    body_law, tail_law = _gamma(*BODY), stats.genpareto(TAIL[1], scale=TAIL[0])
    tail = np.array(curve.piece) == "tail"
    assert tail.sum() == 10
    assert curve.y[tail][-1] == pytest.approx(THRESHOLD + tail_law.ppf(0.999), rel=1e-12)
    low = body_law.ppf(1e-3 * body_law.cdf(THRESHOLD))
    assert curve.y[0] == pytest.approx(low, rel=1e-10)
    assert np.all(curve.values[tail] > 0)
    left = (1 - 1e-4) * body_law.pdf(THRESHOLD) / body_law.cdf(THRESHOLD)
    assert curve.values[9] == pytest.approx(left, rel=1e-12)
    assert curve.values[10] == pytest.approx(1e-4 * tail_law.pdf(0.0), rel=1e-12)


def test_a_given_grid_gains_the_threshold_twice_only_when_it_spans_it(fitted):
    splice, X = fitted
    row = X.iloc[[0]]
    spanning = splice.distribution_curve(row, y=np.array([10.0, 100.0, 150.0]))
    assert spanning.y.tolist() == [10.0, 100.0, 100.0, 150.0]
    assert spanning.piece == ("body", "body", "tail", "tail")
    below = splice.distribution_curve(row, y=np.array([10.0, 50.0, 100.0]))
    assert below.y.tolist() == [10.0, 50.0, 100.0] and set(below.piece) == {"body"}
    above = splice.distribution_curve(row, y=np.array([120.0, 150.0]))
    assert above.piece == ("tail", "tail")


def test_the_payload_is_json_and_the_plot_hands_it_to_the_renderer(monkeypatch, fitted):
    splice, X = fitted
    row = X.iloc[[0]]
    payload = splice.distribution_curve(row, kind="survival", n_points=6).to_json()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert payload["threshold"] == THRESHOLD and payload["piece"].count("tail") == 3
    monkeypatch.setattr(
        mpl_renderers, "plot_distribution_curve", lambda c: ("drawn", c), raising=False
    )
    drawn, curve = splice.plot_distribution(row, kind="survival", n_points=6)
    assert drawn == "drawn" and isinstance(curve, DistributionCurve)
    assert curve.to_json() == payload


@pytest.mark.parametrize("method", ["distribution_curve", "plot_distribution"])
def test_a_curve_needs_one_row(fitted, method):
    splice, X = fitted
    with pytest.raises(ValueError, match=rf"^{method} \w+ one row's law and was given 0 rows"):
        getattr(splice, method)(X.iloc[:0])


# --------------------------------------------------------------------------- #
# F. The experimental label
# --------------------------------------------------------------------------- #


_ENTRY_POINTS = {
    "predict_pdf": lambda splice, row: splice.predict_pdf(row, 50.0),
    "predict_mean_excess": lambda splice, row: splice.predict_mean_excess(row, 50.0),
    "distribution_curve": lambda splice, row: splice.distribution_curve(row, n_points=4),
    "plot_distribution": lambda splice, row: splice.plot_distribution(row, n_points=4),
}


@pytest.mark.parametrize("name", sorted(_ENTRY_POINTS))
def test_each_entry_point_warns_on_its_first_call(monkeypatch, fitted, name):
    monkeypatch.setattr(
        mpl_renderers, "plot_distribution_curve", lambda curve: curve, raising=False
    )
    splice, X = fitted
    monkeypatch.setattr("superglm._experimental._announced", False)
    with pytest.warns(ExperimentalWarning, match=rf"^{name} is experimental") as caught:
        _ENTRY_POINTS[name](splice, X.iloc[[0]])
    assert caught[0].filename == __file__
