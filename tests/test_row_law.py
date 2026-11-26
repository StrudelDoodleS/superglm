"""One fitted row's law on SuperLSS: density, mean excess and the curve behind its plot.

Every number is checked against SciPy's closed forms (or a two-piece density written here
from the family's stated law), quadrature, or ``predict_cdf`` windows on ordinary laws.
"""

from __future__ import annotations

import functools
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special, stats

import superglm.plotting.distributional as mpl_renderers
import superglm.plotting.distributional_plotly as plotly_renderers
from superglm import (
    DirichletLSS,
    GammaLS,
    GaussianLS,
    GeneralizedGammaLSS,
    GeneralizedParetoLSS,
    LogLogisticLSS,
    LogNormalLS,
    LomaxLSS,
    NegativeBinomialLS,
    SuperLSS,
    TweedieLSS,
    TwoPieceLogNormalLSS,
    TwoPieceNormalLSS,
)
from superglm._experimental import ExperimentalWarning
from superglm.distributional.row_law import DistributionCurve
from tests._no_mean_fixture import CELLS, negative_power_lss

U = 2.0**-53
N = 1500


@pytest.fixture(autouse=True)
def _label_already_shown(monkeypatch):
    monkeypatch.setattr("superglm._experimental._announced", True)


def _two_piece(rng, n, location, scale, skew):
    """Epsilon-skew two-piece normal draws: the left piece has mass (1 - skew) / 2."""
    left = rng.random(n) < (1 - skew) / 2
    half = np.abs(rng.standard_normal(n))
    return location + scale * np.where(left, -(1 - skew) * half, (1 + skew) * half)


def _gengamma(location, sigma, q):
    return stats.gengamma(a=1 / q**2, c=q / sigma, scale=np.exp(location) * (q * q) ** (sigma / q))


#: name -> (family, sampler(rng, n)); intercept-only fits of ordinary laws.
LAWS = {
    "gaussian": (GaussianLS(), lambda r, n: stats.norm(1, 2).rvs(n, random_state=r)),
    "gamma": (GammaLS(), lambda r, n: stats.gamma(2, scale=3).rvs(n, random_state=r)),
    "lognormal": (
        LogNormalLS(),
        lambda r, n: stats.lognorm(0.6, scale=np.e).rvs(n, random_state=r),
    ),
    "lognormal_location": (
        LogNormalLS(parametrisation="location"),
        lambda r, n: stats.lognorm(0.6, scale=np.e).rvs(n, random_state=r),
    ),
    "gengamma": (GeneralizedGammaLSS(), lambda r, n: _gengamma(1, 0.5, 0.7).rvs(n, random_state=r)),
    "gengamma_location": (
        GeneralizedGammaLSS(parametrisation="location"),
        lambda r, n: _gengamma(1, 0.5, 0.7).rvs(n, random_state=r),
    ),
    "gpd": (
        GeneralizedParetoLSS(),
        lambda r, n: stats.genpareto(0.3, scale=2).rvs(n, random_state=r),
    ),
    "lomax": (LomaxLSS(), lambda r, n: stats.lomax(3, scale=2).rvs(n, random_state=r)),
    "loglogistic": (LogLogisticLSS(), lambda r, n: stats.fisk(3, scale=2).rvs(n, random_state=r)),
    "two_piece_normal": (TwoPieceNormalLSS(), lambda r, n: _two_piece(r, n, 1, 2, 0.3)),
    "two_piece_lognormal": (
        TwoPieceLogNormalLSS(),
        lambda r, n: np.exp(_two_piece(r, n, 1, 0.5, 0.3)),
    ),
    "two_piece_lognormal_location": (
        TwoPieceLogNormalLSS(parametrisation="location"),
        lambda r, n: np.exp(_two_piece(r, n, 1, 0.5, 0.3)),
    ),
}


def _estimator(family, **kwargs):
    return SuperLSS(family, *(getattr(family, p.name)() for p in family.parameters), **kwargs)


def _quiet_fit(model, X, y, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(X, y, *args, **kwargs)


@functools.cache
def _fitted(name):
    family, sample = LAWS[name]
    y = sample(np.random.default_rng(sorted(LAWS).index(name)), N)
    X = pd.DataFrame({"x": np.zeros(N)})
    return _quiet_fit(_estimator(family), X, y), X


class _TwoPieceLaw:
    """The epsilon-skew two-piece law on the line, or of ``log Y``, as the family states it.

    Density ``phi(z) / scale`` with ``z = (t - location) / scale_side``, where the left
    scale is ``scale (1 - skew)`` and the right ``scale (1 + skew)``.
    """

    def __init__(self, location, scale, skew, log):
        self.location, self.left, self.right, self.skew, self.log = (
            location,
            scale * (1 - skew),
            scale * (1 + skew),
            skew,
            log,
        )
        self.scale = scale

    def _t(self, y):
        with np.errstate(divide="ignore"):
            return np.log(y) if self.log else np.asarray(y, float)

    def logpdf(self, y):
        t = self._t(y)
        side = np.where(t < self.location, self.left, self.right)
        out = stats.norm.logpdf((t - self.location) / side) - np.log(self.scale)
        return out - t if self.log else out

    def logsf(self, y):
        t = self._t(y)
        right = np.log1p(self.skew) + stats.norm.logsf((t - self.location) / self.right)
        left = np.log1p(-(1 - self.skew) * stats.norm.cdf((t - self.location) / self.left))
        return np.where(t >= self.location, right, left)


class _LogLogistic(type(stats.fisk(1.0))):
    """SciPy's log-logistic with ``log S = -log1p((y / scale)^shape)``, exact in the far tail."""

    def __init__(self, shape, scale):
        super().__init__(stats.fisk, shape, scale=scale)
        self.shape_, self.scale_ = shape, scale

    def logsf(self, y):
        return -np.log1p((np.asarray(y, float) / self.scale_) ** self.shape_)


def _law(name, theta):
    """The SciPy law (or the two-piece law above) of one row's fitted parameters."""
    a, b, *c = theta
    if name == "gaussian":
        return stats.norm(a, b)
    if name == "gamma":
        return stats.gamma(1 / b**2, scale=a * b**2)
    if name == "lognormal":
        return stats.lognorm(b, scale=a * np.exp(-(b**2) / 2))
    if name == "lognormal_location":
        return stats.lognorm(b, scale=np.exp(a))
    if name.startswith("gengamma"):
        (q,) = c
        if name == "gengamma":
            # The mean form: E[Y] = exp(mu) (Q^2)^(sigma/Q) Gamma(k + sigma/Q) / Gamma(k).
            k = 1 / q**2
            scale = a * np.exp(special.gammaln(k) - special.gammaln(k + b / q))
            return stats.gengamma(a=k, c=q / b, scale=scale)
        return _gengamma(a, b, q)
    if name == "gpd":
        return stats.genpareto(b, scale=a)
    if name == "lomax":
        return stats.lomax(b, scale=a)
    if name == "loglogistic":
        return _LogLogistic(b, scale=a)
    (skew,) = c
    if name == "two_piece_lognormal":
        left, right = b * (1 - skew), b * (1 + skew)
        loading = (1 - skew) * np.exp(left**2 / 2) * special.ndtr(-left) + (1 + skew) * np.exp(
            right**2 / 2
        ) * special.ndtr(right)
        return _TwoPieceLaw(np.log(a) - np.log(loading), b, skew, log=True)
    return _TwoPieceLaw(a, b, skew, log=name != "two_piece_normal")


def _row_law(model, X, name, row=0):
    theta = model.predict_parameters(X.iloc[[row]]).to_numpy()[0]
    return _law(name, theta)


# --------------------------------------------------------------------------- #
# A1. The density
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(LAWS))
def test_the_density_is_the_closed_form_of_each_family(name):
    model, X = _fitted(name)
    law = _row_law(model, X, name)
    y = model.predict_quantile(X.iloc[[0] * 7], np.array([1e-4, 0.01, 0.2, 0.5, 0.8, 0.99, 0.9999]))
    got = model.predict_pdf(X.iloc[[0] * 7], y)
    expected = law.logpdf(y)
    np.testing.assert_allclose(
        np.log(got), expected, rtol=0, atol=64 * U * max(1, np.abs(expected).max())
    )


@pytest.mark.parametrize("name", sorted(LAWS))
def test_the_density_over_a_window_is_the_difference_of_the_cdf(name):
    model, X = _fitted(name)
    row = X.iloc[[0]]
    edges = model.predict_quantile(X.iloc[[0] * 4], np.array([0.05, 0.3, 0.7, 0.99]))
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        area, _ = integrate.quad(
            lambda t: model.predict_pdf(row, t)[0], a, b, epsabs=0, epsrel=1e-12, limit=200
        )
        window = model.predict_cdf(row, b)[0] - model.predict_cdf(row, a)[0]
        assert area == pytest.approx(window, rel=1e-10)


def test_amounts_outside_the_support_read_zero_and_the_closed_edges_their_limits():
    gamma, X = _fitted("gamma")
    rows = X.iloc[[0] * 4]
    assert gamma.predict_pdf(rows, np.array([0.0, -1.0, np.inf, -np.inf])).tolist() == [0.0] * 4
    gaussian, _ = _fitted("gaussian")
    assert gaussian.predict_pdf(X.iloc[[0, 0]], np.array([np.inf, -np.inf])).tolist() == [0.0, 0.0]
    gpd, _ = _fitted("gpd")
    scale, _ = gpd.predict_parameters(X.iloc[[0]]).to_numpy()[0]
    assert gpd.predict_pdf(X.iloc[[0]], 0.0)[0] == pytest.approx(1 / scale, rel=4 * U)
    lomax, _ = _fitted("lomax")
    scale, shape = lomax.predict_parameters(X.iloc[[0]]).to_numpy()[0]
    assert lomax.predict_pdf(X.iloc[[0]], 0.0)[0] == pytest.approx(shape / scale, rel=4 * U)
    loglogistic, _ = _fitted("loglogistic")
    assert loglogistic.predict_pdf(X.iloc[[0, 0]], np.array([0.0, -2.0])).tolist() == [0.0, 0.0]


def test_a_row_outside_a_links_domain_reads_nan_with_one_warning():
    model = negative_power_lss()
    with pytest.warns(UserWarning, match="no predicted value of 'mean'"):
        density = model.predict_pdf(CELLS, 2.0)
    with pytest.warns(UserWarning, match="no predicted value of 'mean'"):
        excess = model.predict_mean_excess(CELLS, 2.0)
    for values in (density, excess):
        assert np.isnan(values[3]) and np.all(np.isfinite(values[:3]))
    assert not density.flags.writeable


def test_a_missing_amount_is_refused_by_row():
    model, X = _fitted("gamma")
    with pytest.raises(
        ValueError, match=r"predict_pdf needs an amount for every row; rows \[1, 3\]"
    ):
        model.predict_pdf(X.iloc[:4], np.array([1.0, np.nan, 2.0, np.nan]))
    with pytest.raises(ValueError, match=r"finite amount for every row; rows \[0, 2\] are missing"):
        model.predict_mean_excess(X.iloc[:3], np.array([np.nan, 1.0, np.inf]))


@pytest.mark.parametrize(
    ("name", "amount"),
    [("gaussian", 1e300), ("two_piece_normal", 1e300)],
)
def test_an_amount_beyond_float64_is_refused_by_row_beside_an_ordinary_one(name, amount):
    """R2: a finite amount inside the support never fails with an internal error."""
    model, X = _fitted(name)
    with pytest.raises(ValueError, match=r"cannot evaluate the density at rows \[1\] in float64"):
        model.predict_pdf(X.iloc[:2], np.array([2.0, amount]))
    assert np.isfinite(model.predict_pdf(X.iloc[:1], 2.0)[0])


def test_a_fit_with_frequency_weights_reports_the_unit_law_of_each_row():
    """The density of a frequency-weighted fit, on its own rows and on new ones, is its row law."""
    rng = np.random.default_rng(11)
    n = 600
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = stats.gamma(2, scale=np.exp(0.4 * X["x"].to_numpy())).rvs(random_state=rng)
    family = GammaLS()
    model = _quiet_fit(
        SuperLSS(family, family.mean("x"), family.scale(), weight_semantics="frequency"),
        X,
        y,
        rng.integers(1, 6, size=n).astype(float),
    )
    for rows, amounts in ((X, y), (X.iloc[:7], np.linspace(0.5, 9, 7))):
        mean, cv = model.predict_parameters(rows).to_numpy().T
        expected = stats.gamma.logpdf(amounts, 1 / cv**2, scale=mean * cv**2)
        np.testing.assert_allclose(
            np.log(model.predict_pdf(rows, amounts)),
            expected,
            rtol=0,
            atol=64 * U * max(1, np.abs(expected).max()),
        )


@pytest.mark.parametrize("bounds", ["truncation", "censored"])
def test_a_truncated_or_censored_fit_reports_the_parent_density(bounds):
    rng = np.random.default_rng(12)
    y = stats.gamma(2, scale=3).rvs(800, random_state=rng)
    X = pd.DataFrame({"x": np.zeros(len(y))})
    if bounds == "truncation":
        keep = y < 9.0
        X, y = X[keep], y[keep]
        options = {"truncation": (None, 9.0)}
    else:
        options = {"censored": y > 9.0}
        y = np.minimum(y, 9.0)
    family = GammaLS()
    model = _quiet_fit(SuperLSS(family, family.mean(), family.scale()), X, y, **options)
    # The training rows themselves: their own bounds must not enter the density.
    mean, cv = model.predict_parameters(X).to_numpy().T
    expected = stats.gamma.logpdf(y, 1 / cv**2, scale=mean * cv**2)
    np.testing.assert_allclose(np.log(model.predict_pdf(X, y)), expected, rtol=0, atol=64 * U * 10)


# --------------------------------------------------------------------------- #
# A2. The mean excess through the model
# --------------------------------------------------------------------------- #


def _quad_mean_excess(law, v):
    log_s = law.logsf(v)
    area, _ = integrate.quad(
        lambda s: np.exp(law.logsf(v + s) - log_s), 0, np.inf, epsabs=0, epsrel=1e-12, limit=400
    )
    return area


@pytest.mark.parametrize("name", sorted(LAWS))
def test_the_mean_excess_is_the_integrated_survival_of_each_family(name):
    model, X = _fitted(name)
    law = _row_law(model, X, name)
    v = model.predict_quantile(X.iloc[[0] * 3], np.array([0.1, 0.5, 0.99]))
    expected = [_quad_mean_excess(law, t) for t in v]
    np.testing.assert_allclose(model.predict_mean_excess(X.iloc[[0] * 3], v), expected, rtol=1e-9)


@pytest.mark.parametrize("name", ["gamma", "lognormal", "gpd", "loglogistic"])
def test_below_the_support_the_mean_excess_is_the_mean_less_the_amount(name):
    model, X = _fitted(name)
    rows = X.iloc[[0, 0]]
    v = np.array([-3.0, 0.0])
    np.testing.assert_allclose(
        model.predict_mean_excess(rows, v), model.predict(rows) - v, rtol=16 * U
    )


@pytest.fixture(scope="module")
def row_varying():
    """A gamma whose mean and CV move with ``x``, so every row has its own law."""
    rng = np.random.default_rng(13)
    X = pd.DataFrame({"x": np.linspace(-1, 1, 800)})
    x = X["x"].to_numpy()
    cv = np.exp(-0.5 + 0.3 * x)
    y = stats.gamma(1 / cv**2, scale=np.exp(1 + 0.5 * x) * cv**2).rvs(random_state=rng)
    family = GammaLS()
    return _quiet_fit(SuperLSS(family, family.mean("x"), family.scale("x")), X, y), X


@pytest.mark.parametrize("path", ["scalar", "vector"])
def test_the_public_mean_excess_reads_each_rows_own_law(row_varying, path):
    model, X = row_varying
    rows = X.iloc[[0, 250, 799, 400]]
    v = 4.0 if path == "scalar" else np.array([0.5, 9.0, 2.0, 30.0])
    expected = [
        _quad_mean_excess(_law("gamma", theta), t)
        for theta, t in zip(
            model.predict_parameters(rows).to_numpy(), np.broadcast_to(v, (4,)), strict=True
        )
    ]
    np.testing.assert_allclose(model.predict_mean_excess(rows, v), expected, rtol=1e-9)
    assert len(set(np.round(expected, 6))) == 4


def _heavy(name, sample, family):
    rng = np.random.default_rng(14)
    y = sample.rvs(2000, random_state=rng)
    X = pd.DataFrame({"x": np.zeros(len(y))})
    return _quiet_fit(_estimator(family), X, y), X


@pytest.mark.parametrize(
    ("name", "law", "family"),
    [
        ("lomax", stats.lomax(0.6, scale=2), LomaxLSS()),
        ("loglogistic", stats.fisk(0.7, scale=2), LogLogisticLSS()),
        (
            "gengamma_location",
            _gengamma(0, 1, -1.5),
            GeneralizedGammaLSS(parametrisation="location"),
        ),
    ],
)
def test_an_infinite_mean_reads_infinity_as_predict_does(name, law, family):
    model, X = _heavy(name, law, family)
    rows = X.iloc[[0, 0, 0]]
    assert np.all(np.isinf(model.predict(rows)))
    assert np.all(np.isinf(model.predict_mean_excess(rows, np.array([-1.0, 0.5, 50.0]))))


# --------------------------------------------------------------------------- #
# A3. The curve and its payload
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["gaussian", "gamma", "gpd", "two_piece_normal"])
@pytest.mark.parametrize("kind", ["density", "survival", "mean_excess"])
def test_the_curve_holds_the_predictions_on_its_own_grid(name, kind):
    model, X = _fitted(name)
    row = X.iloc[[0]]
    curve = model.distribution_curve(row, kind=kind, n_points=40)
    rows = X.iloc[[0] * len(curve.y)]
    if kind == "density":
        expected = model.predict_pdf(rows, curve.y)
    elif kind == "survival":
        expected = np.exp(_row_law(model, X, name).logsf(curve.y))
    else:
        expected = model.predict_mean_excess(rows, curve.y)
    if kind == "survival":
        np.testing.assert_allclose(curve.values, expected, rtol=1e-12)
    else:
        np.testing.assert_array_equal(curve.values, expected)
    p_low, p_high = (0.01, 1 - 1e-6) if kind == "survival" else (1e-3, 0.999)
    np.testing.assert_allclose(
        curve.y[[0, -1]],
        model.predict_quantile(X.iloc[[0, 0]], np.array([p_low, p_high])),
        rtol=1e-12,
    )
    positive = name == "gamma" or name == "gpd"
    spacing = np.diff(np.log(curve.y)) if kind == "survival" and positive else np.diff(curve.y)
    np.testing.assert_allclose(spacing, spacing[0], rtol=1e-9)
    assert curve.amount_scale == ("log" if kind == "survival" and positive else "linear")
    assert curve.piece == ("law",) * 40 and curve.threshold is None
    assert curve.mean == model.predict(row)[0]
    assert dict(curve.parameters) == model.predict_parameters(row).iloc[0].to_dict()
    assert curve.family == type(model.family).__name__


@pytest.mark.parametrize("name", ["gaussian", "gamma"])
def test_the_survival_curve_keeps_a_far_tail_that_one_minus_the_cdf_loses(name):
    """R9: the survival reads the family's upper window, not ``1 - predict_cdf``."""
    model, X = _fitted(name)
    law = _row_law(model, X, name)
    far = law.isf(np.array([1e-3, 1e-20, 1e-100, 1e-250]))
    curve = model.distribution_curve(X.iloc[[0]], kind="survival", y=far)
    np.testing.assert_allclose(curve.values, law.sf(far), rtol=1e-11)
    assert curve.amount_scale == ("log" if name == "gamma" else "linear")


def test_a_real_line_survival_keeps_a_linear_axis_on_a_positive_grid():
    model, X = _fitted("gaussian")
    curve = model.distribution_curve(X.iloc[[0]], kind="survival", y=np.array([1.0, 2.0, 4.0]))
    assert curve.amount_scale == "linear"


def test_the_payload_is_json_and_plot_data_returns_it():
    model, X = _fitted("gpd")
    row = X.iloc[[0]]
    curve = model.distribution_curve(row, n_points=5)
    payload = curve.to_json()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert payload["y"] == curve.y.tolist() and payload["values"] == curve.values.tolist()
    assert payload["kind"] == "density" and payload["threshold"] is None
    assert model.plot_data("distribution", row=row, n_points=5) == payload
    survival = model.distribution_curve(row, kind="survival", n_points=5).to_json()
    assert model.plot_data("distribution", row=row, curve="survival", n_points=5) == survival


def test_an_infinite_mean_excess_is_null_in_json():
    model, X = _heavy("lomax", stats.lomax(0.6, scale=2), LomaxLSS())
    payload = model.distribution_curve(X.iloc[[0]], kind="mean_excess", n_points=3).to_json()
    assert payload["values"] == [None] * 3 and payload["mean"] is None
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_distribution_hands_the_curve_to_its_engines_renderer(monkeypatch, engine):
    module, function = {
        "matplotlib": (mpl_renderers, "plot_distribution_curve"),
        "plotly": (plotly_renderers, "plotly_distribution_curve"),
    }[engine]
    monkeypatch.setattr(module, function, lambda curve: ("drawn", curve), raising=False)
    model, X = _fitted("gamma")
    drawn, curve = model.plot_distribution(X.iloc[[0]], kind="survival", engine=engine, n_points=9)
    assert drawn == "drawn" and isinstance(curve, DistributionCurve)
    assert (
        curve.to_json()
        == model.distribution_curve(X.iloc[[0]], kind="survival", n_points=9).to_json()
    )


# --------------------------------------------------------------------------- #
# C. Refusals
# --------------------------------------------------------------------------- #


@functools.cache
def _no_density(name):
    rng = np.random.default_rng(15)
    n = 300
    X = pd.DataFrame({"x": np.zeros(n)})
    if name == "TweedieLSS":
        y = np.where(rng.random(n) < 0.4, 0.0, stats.gamma(2, scale=2).rvs(n, random_state=rng))
        family = TweedieLSS()
        return _quiet_fit(SuperLSS(family, family.mu(), family.phi(), family.p()), X, y), X
    if name == "NegativeBinomialLS":
        return _quiet_fit(_estimator(NegativeBinomialLS()), X, rng.poisson(3.0, n).astype(float)), X
    family = DirichletLSS(("a", "b", "c"))
    y = pd.DataFrame(rng.dirichlet([2.0, 3.0, 4.0], n), columns=("a", "b", "c"))
    model = SuperLSS(family, *(family.alpha(c) for c in "abc"), weight_semantics="frequency")
    return _quiet_fit(model, X, y), X


_CALLS = {
    "predict_pdf": lambda model, rows: model.predict_pdf(rows, 1.0),
    "predict_mean_excess": lambda model, rows: model.predict_mean_excess(rows, 1.0),
    **{
        f"{method}-{kind}": functools.partial(
            lambda model, rows, method, kind: getattr(model, method)(rows, kind=kind),
            method=method,
            kind=kind,
        )
        for method in ("distribution_curve", "plot_distribution")
        for kind in ("density", "survival", "mean_excess")
    },
}


@pytest.mark.parametrize("family", ["TweedieLSS", "NegativeBinomialLS", "DirichletLSS"])
@pytest.mark.parametrize("call", sorted(_CALLS))
@pytest.mark.parametrize("n_rows", [0, 3])
def test_a_family_without_a_density_is_refused_by_every_call(family, call, n_rows):
    model, X = _no_density(family)
    what = "mean excess" if call.endswith("mean_excess") else "density"
    # predict_cdf is named only for a family that has one.
    remedy = "predict_cdf" if family == "TweedieLSS" else "predict_parameters"
    with pytest.raises(NotImplementedError, match=rf"^{family} .*{what}.*{remedy}\(\)") as refused:
        _CALLS[call](model, X.iloc[:n_rows])
    assert "predict_parameters()" in str(refused.value)


@pytest.mark.parametrize("method", ["distribution_curve", "plot_distribution"])
def test_a_curve_needs_one_row_a_known_kind_and_an_increasing_grid(method):
    model, X = _fitted("gamma")
    call = getattr(model, method)
    with pytest.raises(
        ValueError, match=rf"^{method} \w+ one row's law and was given 3 rows.*iloc"
    ):
        call(X.iloc[:3])
    with pytest.raises(
        ValueError, match=r"kind must be 'density', 'survival' or 'mean_excess', not 'cdf'"
    ):
        call(X.iloc[[0]], kind="cdf")
    for grid in ([1.0, 1.0, 2.0], [1.0, np.nan], [3.0]):
        with pytest.raises(ValueError, match="y must be at least two finite amounts in increasing"):
            call(X.iloc[[0]], y=np.array(grid))


# --------------------------------------------------------------------------- #
# F. The experimental label
# --------------------------------------------------------------------------- #


_ENTRY_POINTS = {
    "predict_pdf": lambda model, row: model.predict_pdf(row, 1.0),
    "predict_mean_excess": lambda model, row: model.predict_mean_excess(row, 1.0),
    "distribution_curve": lambda model, row: model.distribution_curve(row, n_points=3),
    "plot_distribution": lambda model, row: model.plot_distribution(row, n_points=3),
}


@pytest.mark.parametrize("name", sorted(_ENTRY_POINTS))
def test_each_entry_point_warns_on_its_first_call(monkeypatch, name):
    monkeypatch.setattr(
        mpl_renderers, "plot_distribution_curve", lambda curve: curve, raising=False
    )
    model, X = _fitted("gamma")
    monkeypatch.setattr("superglm._experimental._announced", False)
    with pytest.warns(ExperimentalWarning, match=rf"^{name} is experimental") as caught:
        _ENTRY_POINTS[name](model, X.iloc[[0]])
    assert caught[0].filename == __file__
